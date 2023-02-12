import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision import transforms

import rotate_captcha_crack as rcc
from rotate_captcha_crack import CONFIG, RCCLogger, device

batch_size = CONFIG.train.batch_size
epoches = CONFIG.train.epoches
lr = CONFIG.train.lr
lambda_cos = CONFIG.train.loss['lambda_cos']
exponent = CONFIG.train.loss['exponent']
t_0 = CONFIG.train.lr_scheduler['T_0']
t_mult = CONFIG.train.lr_scheduler['T_mult']

start_time = time.time()
start_time_int = int(start_time)

LOG = RCCLogger(start_time_int)

model_dir = Path(f"models/{start_time_int}")
if not model_dir.exists():
    model_dir.mkdir(parents=True)

trans = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
train_dataloader = rcc.dataset.get_dataloader("train", batch_size, device, trans)
val_dataloader = rcc.dataset.get_dataloader("val", batch_size, device, trans)

model = rcc.model.RotationNet()
model = model.to(device)
optmizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optmizer, T_0=t_0, T_mult=t_mult, eta_min=lr / 10e3)
criterion = rcc.loss.RotationLoss()
eval_criterion = rcc.loss.DistanceBetweenAngles()

lr_vec = np.empty(epoches, dtype=np.float64)
train_loss_vec = np.empty(epoches, dtype=np.float64)
eval_loss_vec = np.empty(epoches, dtype=np.float64)
best_eval_loss = sys.maxsize
previous_checkpoint_path = None

for epoch_idx in range(epoches):
    model.train()
    total_train_loss = 0.0
    steps = 0

    for step_idx, (source, target) in enumerate(train_dataloader):
        optmizer.zero_grad()
        predict: Tensor = model(source)
        loss: Tensor = criterion(predict, target)
        loss.backward()
        total_train_loss += loss.cpu().item()
        optmizer.step()
        steps += 1

    scheduler.step()
    lr_vec[epoch_idx] = scheduler.get_last_lr()[0]

    train_loss = total_train_loss / steps
    train_loss_vec[epoch_idx] = train_loss

    model.eval()
    total_eval_loss = 0.0
    batch_count = 0
    with torch.no_grad():
        for source, target in val_dataloader:
            predict: Tensor = model(source)
            eval_loss: Tensor = eval_criterion(predict, target)
            total_eval_loss += eval_loss.cpu().item() * 360
            batch_count += 1

    eval_loss = total_eval_loss / batch_count
    eval_loss_vec[epoch_idx] = eval_loss

    LOG.info(
        f"Epoch#{epoch_idx}. time_cost: {time.time()-start_time:.2f} s. train_loss: {train_loss:.8f}. eval_loss: {eval_loss:.4f} degrees"
    )

    torch.save(model.state_dict(), str(model_dir / "last.pth"))
    if eval_loss < best_eval_loss:
        best_eval_loss = eval_loss
        new_checkpoint_path = str(model_dir / f"{epoch_idx}.pth")
        torch.save(model.state_dict(), new_checkpoint_path)
        if previous_checkpoint_path is not None:
            os.remove(previous_checkpoint_path)

        previous_checkpoint_path = new_checkpoint_path

x = np.arange(epoches, dtype=np.int16)

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, eval_loss_vec)
fig.savefig(str(model_dir / "eval_loss.png"))

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, train_loss_vec)
fig.savefig(str(model_dir / "train_loss.png"))

fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, lr_vec)
fig.savefig(str(model_dir / "lr.png"))
