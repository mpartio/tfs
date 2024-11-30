import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from gnll import GaussianNLLLoss
from betanll import BetaNLLLoss
from hete import HeteroscedasticLoss
from cc2 import CloudCastV2
from tqdm import tqdm
from config import get_args
from util import *
from datetime import datetime

args = get_args()


def roll_forecast(model, x, y, steps):
    assert y.shape[1] == steps

    total_loss = None

    for i in range(steps):
        y_true = y[:, i, ...].unsqueeze(1)
        y_hat = model(x)

        assert y_true.shape == y_hat.shape, "shape mismatch: {} vs {}".format(
            y_true.shape, y_hat.shape
        )
        loss = criterion(y_hat, y_true)
        x = y_hat

        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss

    return total_loss / steps


run_dir = f"runs/{args.run_name}"
os.makedirs(run_dir, exist_ok=True)

training_start = datetime.now()
print("Starting training at", training_start)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device", device)

model = CloudCastV2(dim=args.dim, patch_size=args.patch_size)

print(model)
num_params = count_trainable_parameters(model)
print(f"Number of trainable parameters: {num_params:,}")

found_existing_model = False
try:
    model.load_state_dict(
        torch.load(f"runs/{args.run_name}/model.pth", weights_only=True)
    )
    print("Model loaded from", f"runs/{args.run_name}/model.pth")
    found_existing_model = True
except FileNotFoundError:
    print("No model found, starting from scratch")

model = model.to(device)

train_loader, val_loader = read_data(dataset_size=args.dataset_size)

var_reg_weight = 1e-2
criterion = torch.nn.L1Loss()

lr = 1e-4 if not found_existing_model else 1e-5

# Kun LR=0.000005 niin hiukan syvempi malli alkoi oppia
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    8,
    2,
)

if not found_existing_model:
    lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: min(1, epoch / 8)
    )

config = vars(args)
config["num_params"] = count_trainable_parameters(model)
with open(f"{run_dir}/{training_start.strftime('%Y%m%d%H%M%S')}-config.json", "w") as f:
    json.dump(vars(args), f)


min_loss = None
min_loss_epoch = None

use_amp = False

scaler = torch.amp.GradScaler() if use_amp else None
autocast_context = autocast if use_amp else lambda: torch.enable_grad()

for epoch in range(1, args.epochs + 1):
    epoch_start_time = datetime.now()
    model.train()

    total_train_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        assert torch.min(inputs) >= 0.0 and torch.max(inputs) <= 1.0
        assert torch.min(targets) >= 0.0 and torch.max(targets) <= 1.0
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with autocast_context():  # torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = roll_forecast(model, inputs, targets, 1)
            total_train_loss += loss.item()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

    total_train_loss /= len(train_loader)

    model.eval()

    total_val_loss = 0.0
    with torch.no_grad():

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            loss = criterion(model(inputs), targets)
            total_val_loss += loss.item()
    total_val_loss /= len(val_loader)

    if epoch <= 10:
        if not found_existing_model:
            lambda_scheduler.step()
    else:
        annealing_scheduler.step(epoch)

    print(
        "Epoch [{}/{}], current best: {}. Train Loss: {:.6f} Val Loss: {:.6f}, LRx1M: {:.3f}".format(
            epoch,
            args.epochs,
            min_loss_epoch,
            total_train_loss,
            total_val_loss,
            1e6 * optimizer.param_groups[0]["lr"],
        )
    )

    saved = False

    if min_loss is None:
        min_loss = total_val_loss

    if epoch > 2 and total_val_loss < min_loss:
        min_loss = total_val_loss
        min_loss_epoch = epoch
        saved = True
        torch.save(model.state_dict(), f"{run_dir}/model.pth")

    break_loop = False
    if min_loss_epoch is not None and (epoch >= 8 and epoch - min_loss_epoch > 14):
        print("No improvement in 14 epochs; early stopping")
        break_loop = True

    epoch_end_time = datetime.now()

    epoch_results = {
        "epoch": epoch,
        "train_loss": total_train_loss,
        "val_loss": total_val_loss,
        "distribution_components": {},
        "loss_components": {},
        "gradients": {},
        "epoch_start_time": epoch_start_time.isoformat(),
        "epoch_end_time": epoch_end_time.isoformat(),
        "duration": (epoch_end_time - epoch_start_time).total_seconds(),
        "lr": optimizer.param_groups[0]["lr"],
        "saved": saved,
        "last_epoch": break_loop or epoch == args.epochs,
    }

    with open(
        f"{run_dir}/{training_start.strftime('%Y%m%d%H%M%S')}-{epoch:03d}.json", "w"
    ) as f:
        json.dump(epoch_results, f)

    if break_loop:
        break

print("Training done")
