import os
import sys
import torch
from torch import nn
import numpy as np
from functools import partial
import argparse
import importlib
import timm
import copy
import time
from tqdm import tqdm

import utils.datasets as mydatasets
from utils.calibrator import QuantCalibrator
from utils.block_recon import BlockReconstructor
from utils.wrap_net import wrap_modules_in_net, wrap_reparamed_modules_in_net
from utils.test_utils import *
from datetime import datetime
import logging
import torchvision as tv
from torchinfo import summary
import matplotlib.pyplot as plt

while True:
    try:
        timestamp = datetime.now()
        formatted_timestamp = timestamp.strftime("%Y%m%d_%H%M")
        root_path = './checkpoints/quant_result/{}'.format(formatted_timestamp)
        os.makedirs(root_path)
        break
    except FileExistsError:
        time.sleep(10)
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    handlers=[
                        logging.FileHandler('{}/output.log'.format(root_path)),
                        logging.StreamHandler()
                    ])


import builtins
original_print = builtins.print
def custom_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    original_print(*args, **kwargs)
builtins.print = custom_print

def get_args_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model", default="deit_small",
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large',
                                 'deit_tiny', 'deit_small', 'deit_base', 
                                 'swin_tiny', 'swin_small', 'swin_base', 'swin_base_384',
                                 'resnet18', 'mobilenet', 'densenet121', 'alexnet', 'vgg11', 'shufflenet', 'inception_v3'],
                        help="model")
    parser.add_argument('--config', type=str, default="./configs/vit_config.py",
                        help="File path to import Config class from")
    parser.add_argument('--dataset', default="/dataset/imagenet/",
                        help='path to dataset')
    parser.add_argument("--calib-size", default=argparse.SUPPRESS,
                        type=int, help="size of calibration set")
    parser.add_argument("--calib-batch-size", default=argparse.SUPPRESS,
                        type=int, help="batchsize of calibration set")
    parser.add_argument("--val-batch-size", default=200,
                        type=int, help="batchsize of validation set")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="number of data loading workers (default: 8)")
    parser.add_argument("--device", default="cuda", type=str, help="device")
    
    calibrate_mode_group = parser.add_mutually_exclusive_group()
    calibrate_mode_group.add_argument('--calibrate', action='store_true', help="Calibrate the model")
    calibrate_mode_group.add_argument('--load-calibrate-checkpoint', type=str, default=None, help="Path to the calibrated checkpoint.")
    parser.add_argument('--test-calibrate-checkpoint', action='store_true', help='validate the calibrated checkpoint.')

    optimize_mode_group = parser.add_mutually_exclusive_group()
    optimize_mode_group.add_argument('--optimize', action='store_true', help="Optimize the model")
    optimize_mode_group.add_argument('--load-optimize-checkpoint', type=str, default=None, help="Path to the optimized checkpoint.")
    parser.add_argument('--test-optimize-checkpoint', action='store_true', help='validate the optimized checkpoint.')

    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--seed", default=5, type=int, help="seed")
    parser.add_argument('--w_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of weights')
    parser.add_argument('--a_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of activation')
    parser.add_argument('--s_bit', type=int, default=argparse.SUPPRESS, help='bit-precision of post softmax activation')
    return parser


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_cur_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def save_model(model, args, cfg, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    if mode == 'calibrate':
        auto_name = '{}_w{}_a{}_s{}_calibsize_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.s_bit, cfg.calib_size)
    else:
        auto_name = '{}_w{}_a{}_s{}_optimsize_{}.pth'.format(
            args.model, cfg.w_bit, cfg.a_bit, cfg.s_bit, cfg.optim_size)
    save_path = os.path.join(root_path, auto_name)

    logging.info(f"Saving checkpoint to {save_path}")
    torch.save(model.state_dict(), save_path)


def load_model(model, args, device, mode='calibrate'):
    assert mode in ['calibrate', 'optimize']
    ckpt_path = args.load_calibrate_checkpoint if mode == 'calibrate' else args.load_optimize_checkpoint
    for name, module in model.named_modules():
        if hasattr(module, 'mode'):
            module.calibrated = True
            module.mode = 'quant_forward'
        if isinstance(module, nn.Linear) and 'reduction' in name:
            module.bias = nn.Parameter(torch.zeros(module.out_features))
        quantizer_attrs = ['a_quantizer', 'w_quantizer', 'A_quantizer', 'B_quantizer']
        for attr in quantizer_attrs:
            if hasattr(module, attr):
                getattr(module, attr).inited = True
    ckpt = torch.load(ckpt_path)
    result = model.load_state_dict(ckpt, strict=False)
    logging.info(str(result))
    model.to(device)
    model.eval()
    return model


def finish_training(model):
    for name, module in model.named_modules():
        if hasattr(module, 'mode') and hasattr(module, 'reparam_bias'):
            module.reparam_bias()

def train_quantized_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=1, print_freq=100, class_to_idx=None, freeze_scale=0):
    """
    Fine-tune the quantized model after PTQ calibration.
    """
    model.train()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            print(f"Quantized Conv2d layer: {name}, training mode: {module.w_quantizer.training_mode}, ")
    imagenet100_map = load_map('/afs/andrew.cmu.edu/usr14/furix/furix/Public/Labels.txt')
    imagenet1000_map = load_map('/afs/andrew.cmu.edu/usr14/furix/furix/Public/imagenet1000_clsidx_to_labels.txt')
    reverse_imagenet100_map = {v: k for k, v in imagenet100_map.items()}
    reverse_imagenet1000_map = {v: k for k, v in imagenet1000_map.items()}
    n_classes_1000 = 1000
    translation = torch.full((n_classes_1000,), -1, device=device, dtype=torch.long)
    
    # Map ImageNet-1000 indices to ImageNet-100 indices
    for k, v in reverse_imagenet1000_map.items():
        if k in reverse_imagenet100_map:
            translation[v] = class_to_idx[reverse_imagenet100_map[k]]
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-9)

    # For plotting
    step_list = []
    train_loss_list = []
    train_acc_list = []
    epoch_list = []
    val_acc_list = []
    lr_list = []

    global_step = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True) as pbar:
            for i, (inputs, targets) in enumerate(train_loader):

                # for m in model.modules():
                #     if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                #         m.eval()
                #         for param in m.parameters():
                #             param.requires_grad = False
                
                if i >= freeze_scale:
                    for name, module in model.named_modules():
                        if hasattr(module, 'w_quantizer'):
                            if hasattr(module.w_quantizer, 'scale'):
                                module.w_quantizer.scale.requires_grad = False
                            if hasattr(module.w_quantizer, 'zero_point'):
                                module.w_quantizer.zero_point.requires_grad = False
                        if hasattr(module, 'a_quantizer'):
                            if hasattr(module.a_quantizer, 'scale'):
                                module.a_quantizer.scale.requires_grad = False
                            if hasattr(module.a_quantizer, 'zero_point'):
                                module.a_quantizer.zero_point.requires_grad = False
                
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                # Forward pass - get 1000-class predictions
                output_1000 = model(inputs)  # shape: [batch_size, 1000]
                
                # Translate to 100 classes efficiently
                batch_size = output_1000.shape[0]
                output_100 = torch.zeros((batch_size, 100), device=device)
                valid_indices = translation != -1
                output_100[:, translation[valid_indices]] = output_1000[:, valid_indices]
                outputs = output_100

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if (i + 1) % print_freq == 0 or (i + 1) == len(train_loader):
                    current_lr = optimizer.param_groups[0]['lr']
                    avg_loss = running_loss / (i + 1)
                    avg_acc = 100. * correct / total
                    pbar.set_postfix({
                        "Loss": f"{avg_loss:.4f}",
                        "Acc": f"{avg_acc:.2f}%",
                        "LR": f"{current_lr:.6f}"
                    })
                    # Save step-wise stats for plotting
                    step_list.append(global_step / len(train_loader))
                    train_loss_list.append(avg_loss)
                    train_acc_list.append(avg_acc)
                global_step += 1
                pbar.update(1)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Loss: {running_loss/len(train_loader):.4f}, "
              f"Acc: {100.*correct/total:.2f}%")
        scheduler.step()

        # Save for plotting epoch-wise
        epoch_list.append(epoch + 1)
        lr_list.append(optimizer.param_groups[0]['lr'])

        # Validation after each epoch with tqdm
        val_acc = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with tqdm(total=len(val_loader), desc=f"Validation {epoch+1}/{num_epochs}", leave=True) as val_pbar:
                for i, (val_inputs, val_targets) in enumerate(val_loader):
                    val_inputs = val_inputs.to(device)
                    val_targets = val_targets.to(device)
                    with torch.no_grad():
                        val_output_1000 = model(val_inputs)
                        batch_size = val_output_1000.shape[0]
                        val_output_100 = torch.zeros((batch_size, 100), device=device)
                        val_output_100[:, translation[valid_indices]] = val_output_1000[:, valid_indices]
                        val_outputs = val_output_100
                        val_loss_batch = criterion(val_outputs, val_targets)
                    val_loss += val_loss_batch.item()
                    _, val_predicted = val_outputs.max(1)
                    val_total += val_targets.size(0)
                    val_correct += val_predicted.eq(val_targets).sum().item()
                    if (i + 1) % print_freq == 0 or (i + 1) == len(val_loader):
                        val_pbar.set_postfix({
                            "ValLoss": f"{val_loss/(i+1):.4f}",
                            "ValAcc": f"{100.*val_correct/val_total:.2f}%"
                        })
                    val_pbar.update(1)
            val_acc = 100. * val_correct / val_total
            val_acc_list.append(val_acc)
            print(f"Validation [{epoch+1}/{num_epochs}] finished. Loss: {val_loss/len(val_loader):.4f}, Acc: {val_acc:.2f}%")
            model.train()
        else:
            val_acc_list.append(float('nan'))
        model_second_word_ratio(model)

    model.eval()

    # Plotting
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 4, 1)
    plt.plot(step_list, train_loss_list, marker='o', label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Epoch vs Training Loss')
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.plot(step_list, train_acc_list, marker='o', color='g', label='Train Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Epoch vs Training Accuracy')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(epoch_list, val_acc_list, marker='o', color='orange', label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Epoch vs Validation Accuracy')
    plt.legend()

    plt.subplot(1, 4, 4)
    plt.plot(epoch_list, lr_list, marker='o', color='purple', label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Epoch vs Learning Rate')
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png")
    plt.close()

def model_second_word_ratio(model):
    total_mask_sum = 0
    total_mask_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "w_quantizer") and hasattr(module.w_quantizer, "mask"):
            mask = module.w_quantizer.mask
            if mask is not None:
                total_mask_sum += mask.float().sum().item()
                total_mask_count += mask.numel()
    if total_mask_count > 0:
        overall_ratio = total_mask_sum / total_mask_count
        print(f"[Overall Second Word Ratio] {overall_ratio:.4f}")
        logging.info(f"[Overall Second Word Ratio] {overall_ratio:.4f}")
    else:
        print("[Overall Second Word Ratio] No mask found in model.")
        logging.info("[Overall Second Word Ratio] No mask found in model.")

    
def main(args):
    logging.info("{} - start the process.".format(get_cur_time()))
    logging.info(str(args))
    dir_path = os.path.dirname(os.path.abspath(args.config))
    if dir_path not in sys.path:
        sys.path.append(dir_path)
    module_name = os.path.splitext(os.path.basename(args.config))[0]
    imported_module = importlib.import_module(module_name)
    Config = getattr(imported_module, 'Config')
    logging.info("Successfully imported Config class!")
        
    cfg = Config()
    cfg.calib_size = args.calib_size if hasattr(args, 'calib_size') else cfg.calib_size
    cfg.calib_batch_size = args.calib_batch_size if hasattr(args, 'calib_batch_size') else cfg.calib_batch_size
    cfg.w_bit = args.w_bit if hasattr(args, 'w_bit') else cfg.w_bit
    cfg.a_bit = args.a_bit if hasattr(args, 'a_bit') else cfg.a_bit
    cfg.s_bit = args.s_bit if hasattr(args, 's_bit') else cfg.s_bit
    for name, value in vars(cfg).items():
        logging.info(f"{name}: {value}")
        
    if args.device.startswith('cuda:'):
        gpu_id = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
        args.device = 'cuda:0'
    device = torch.device(args.device)
    
    model_zoo = {
        'vit_tiny'  : 'vit_tiny_patch16_224',
        'vit_small' : 'vit_small_patch16_224',
        'vit_base'  : 'vit_base_patch16_224',
        'vit_large' : 'vit_large_patch16_224',

        'deit_tiny' : 'deit_tiny_patch16_224',
        'deit_small': 'deit_small_patch16_224',
        'deit_base' : 'deit_base_patch16_224',

        'swin_tiny' : 'swin_tiny_patch4_window7_224',
        'swin_small': 'swin_small_patch4_window7_224',
        'swin_base' : 'swin_base_patch4_window7_224',
        'swin_base_384': 'swin_base_patch4_window12_384',

        'resnet18': '',
        'mobilenet': '',
    }

    seed_all(args.seed)
    
    logging.info('Building model ...')
    if args.model == 'resnet18':
        model = tv.models.resnet18(weights="DEFAULT")
    elif args.model == 'mobilenet':
        # model = tv.models.mobilenet_v2(weights="DEFAULT")
        model = timm.create_model('mobilenetv1_100', pretrained=True)
    elif args.model == 'densenet121':
        model = tv.models.densenet121(weights="DEFAULT")
    elif args.model == 'alexnet':
        model = tv.models.alexnet(weights="DEFAULT")
    elif args.model == 'vgg11':
        model = tv.models.vgg11(weights="DEFAULT")
    elif args.model == 'inception_v3':
        model = tv.models.inception_v3(weights="DEFAULT")
    elif args.model == 'shufflenet':
        model = tv.models.shufflenet_v2_x1_0(weights="DEFAULT")
    else:
        try:
            model = timm.create_model(model_zoo[args.model], checkpoint_path='./checkpoints/vit_raw/{}.bin'.format(model_zoo[args.model]))
        except:
            model = timm.create_model(model_zoo[args.model], pretrained=True)
    full_model = copy.deepcopy(model)
    full_model.to(device)
    full_model.eval()
    model.to(device)
    model.eval()
    summary(model, input_size=(args.val_batch_size, 3, 224, 224), device=args.device, depth=5, col_names=("input_size", "output_size", "num_params", "kernel_size", "trainable"))
    data_path = args.dataset
    g = mydatasets.TorchvisionImageNetLoaderGenerator(data_path, args.val_batch_size, args.num_workers, kwargs={"model":model})
    
    logging.info('Building validation dataloader ...')
    val_loader = g.val_loader()
    criterion = nn.CrossEntropyLoss().to(device)

    reparam = args.load_calibrate_checkpoint is None and args.load_optimize_checkpoint is None
    reparam = False
    logging.info('Wraping quantiztion modules (reparam: {}) ...'.format(reparam))
    model = wrap_modules_in_net(model, cfg, reparam=reparam)
    model.to(device)
    model.eval()
    
    if not args.load_optimize_checkpoint:
        if args.load_calibrate_checkpoint:
            logging.info(f"Restoring checkpoint from '{args.load_calibrate_checkpoint}'")
            model = load_model(model, args, device, mode='calibrate')
            if args.test_calibrate_checkpoint:
                val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
                
        else:
            logging.info("{} - start calibration".format(get_cur_time()))
            calib_loader = g.calib_loader(num=cfg.calib_size, batch_size=cfg.calib_batch_size, seed=args.seed)
            quant_calibrator = QuantCalibrator(model, calib_loader)
            quant_calibrator.batching_quant_calib()
            model = wrap_reparamed_modules_in_net(model)
            model.to(device)
            logging.info("{} - calibration finished.".format(get_cur_time()))
            if not args.optimize:
                finish_training(model)
            save_model(model, args, cfg, mode='calibrate')
            model_second_word_ratio(model)
            logging.info('Validating after calibration ...')
            val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device, class_to_idx=g.val_set.class_to_idx)
            logging.info('Validating for full model ...')
            val_loss, val_prec1, val_prec5 = validate(val_loader, full_model, criterion, print_freq=args.print_freq, device=device, class_to_idx=g.val_set.class_to_idx)
    
    if args.optimize:
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        logging.info("{} - start block reconstruction".format(get_cur_time()))
        block_reconstructor = BlockReconstructor(model, full_model, calib_loader)
        block_reconstructor.reconstruct_model(quant_act=cfg.train_act, keep_gpu=cfg.keep_gpu)
        finish_training(model)
        logging.info("{} - block reconstruction finished.".format(get_cur_time()))
        save_model(model, args, cfg, mode='optimize')
    if args.load_optimize_checkpoint:
        logging.info('Building calibrator ...')
        calib_loader = g.calib_loader(num=cfg.optim_size, batch_size=cfg.optim_batch_size, seed=args.seed)
        model = load_model(model, args, device, mode='optimize')
    if args.optimize or args.test_optimize_checkpoint:
        logging.info('Validating on calibration set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(calib_loader, model, criterion, print_freq=args.print_freq, device=device)
        logging.info('Validating on test set after block reconstruction ...')
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device)
    if not args.optimize:
        finish_training(model)
        # Fine-tune quantized model
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                print(f"Quantized Conv2d layer: {name}, training mode: {module.w_quantizer.training_mode}, ")
        train_loader = g.train_loader(batch_size=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        train_quantized_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=5, print_freq=1, class_to_idx=g.val_set.class_to_idx, freeze_scale=0)
        save_model(model, args, cfg, mode='calibrate')
        logging.info('Validating after calibration ...')
        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, print_freq=args.print_freq, device=device, class_to_idx=g.val_set.class_to_idx)
    logging.info("{} - finished the process.".format(get_cur_time()))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
    