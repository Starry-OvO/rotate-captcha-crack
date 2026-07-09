import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rotate_captcha_crack.common import device
from rotate_captcha_crack.const import DEFAULT_CLS_NUM
from rotate_captcha_crack.dataset.midware import DEFAULT_NORM, CircularSmoothLabel, Rotator, path_to_tensor
from rotate_captcha_crack.dataset.paths import glob_imgs
from rotate_captcha_crack.dataset.pipe import SeqSupportsPipe
from rotate_captcha_crack.loss import SoftCrossEntropy
from rotate_captcha_crack.lr import LRManager
from rotate_captcha_crack.model import RotNetR
from rotate_captcha_crack.trainer import Trainer
from rotate_captcha_crack.utils import default_num_workers, slice_from_range
from rotate_captcha_crack.visualizer import visualize_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume", "-r", type=int, default=None, help="Resume from which index. -1 leads to the last training process"
    )
    opts = parser.parse_args()

    #################################
    ### Custom configuration area ###
    dataset_root = Path("D:/dataset/unlabeled2017")

    img_paths = list(glob_imgs(dataset_root))
    cls_num = DEFAULT_CLS_NUM
    labelling = CircularSmoothLabel(cls_num)

    train_img_paths = slice_from_range(img_paths, (0.0, 0.98))
    train_dataset = train_img_paths | SeqSupportsPipe() | path_to_tensor | Rotator() | DEFAULT_NORM | labelling | tuple
    val_img_paths = slice_from_range(img_paths, (0.98, 1.0))
    val_dataset = val_img_paths | SeqSupportsPipe() | path_to_tensor | Rotator() | DEFAULT_NORM | labelling | tuple

    num_workers = default_num_workers()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=128,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = RotNetR(cls_num)
    model = model.to(device)

    lr = 0.002
    epochs = 256
    steps = 512

    pgroups = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    for module_name, module in model.backbone.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            fullname = f"{module_name}.{param_name}" if module_name else param_name
            if "bias" in fullname:
                # bias (no decay)
                pgroups[2].append(param)
            elif isinstance(module, bn):
                # weight (no decay)
                pgroups[1].append(param)
            else:
                # weight (with decay)
                pgroups[0].append(param)

    optimizer = torch.optim.AdamW(pgroups[2], lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    optimizer.add_param_group({"params": pgroups[0], "weight_decay": 0.0005})  # add g0 with weight_decay
    optimizer.add_param_group({"params": pgroups[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, pct_start=0.1, epochs=epochs, steps_per_epoch=steps
    )
    lr = LRManager(lr, scheduler, optimizer)
    loss = SoftCrossEntropy()

    trainer = Trainer(model, train_dataloader, val_dataloader, lr, loss, epochs, steps)
    ### Custom configuration area ###
    #################################

    if opts.resume is not None:
        trainer.resume(opts.resume)

    trainer.train()

    visualize_train(trainer.finder.model_dir)
