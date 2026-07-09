import argparse
from pathlib import Path

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from rotate_captcha_crack.common import device
from rotate_captcha_crack.dataset import google_street_view
from rotate_captcha_crack.dataset.midware import DEFAULT_NORM, CircularSmoothLabel, Rotator, path_to_tensor
from rotate_captcha_crack.dataset.pipe import SeqSupportsPipe
from rotate_captcha_crack.lr import LRManager
from rotate_captcha_crack.model import RotNet
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

    img_paths = google_street_view.get_paths(dataset_root)
    cls_num = 360
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

    model = RotNet(cls_num)
    model = model.to(device)

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, total_iters=1)
    lr = LRManager(lr, scheduler, optimizer)
    loss = CrossEntropyLoss()

    epochs = 256
    steps = 512
    trainer = Trainer(model, train_dataloader, val_dataloader, lr, loss, epochs, steps)
    ### Custom configuration area ###
    #################################

    if opts.resume is not None:
        trainer.resume(opts.resume)

    trainer.train()

    visualize_train(trainer.finder.model_dir)
