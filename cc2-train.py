import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from gnll import GaussianNLLLoss
from betanll import BetaNLLLoss
from hete import HeteroscedasticLoss
from cc2 import CloudCastV2
from tqdm import tqdm
from config import get_args
from util import *

args = get_args()


def roll_forecast(model, x, y, steps):
    assert y.shape[1] == steps

    total_loss = None

    for i in range(steps):
        y_true = y[:, i, ...].unsqueeze(1)
        if args.loss_function == "mae" or args.loss_function == "mse":
            y_hat = model(x)
            loss = criterion(y_hat, y_true)
            x = y_hat
        else:
            mean, stde = model(x)
            loss = criterion(mean, stde, y_true)
            x = mean

        if total_loss is None:
            total_loss = loss
        else:
            total_loss += loss

    return total_loss / steps


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device", device)

model = CloudCastV2(dim=args.dim, patch_size=args.patch_size)

print(model)
num_params = count_trainable_parameters(model)
print(f"Number of trainable parameters: {num_params:,}")

found_existing_model = False
try:
    model.load_state_dict(torch.load(args.load_model_from, weights_only=True))
    print("Model loaded")
    found_existing_model = True
except FileNotFoundError:
    print("No model found, starting from scratch")

model.loss_type = args.loss_function
model = model.to(device)

train_loader, val_loader = read_data(dataset=args.datase)

var_reg_weight = 1e-2
if args.loss_function == "gaussian_nll":
    criterion = GaussianNLLLoss(var_reg_weight=var_reg_weight)
elif args.loss_function == "beta_nll":
    criterion = BetaNLLLoss()
elif args.loss_function == "mae":
    criterion = torch.nn.L1Loss()
elif args.loss_function == "mse":
    criterion = torch.nn.MSELoss()
elif args.loss_function == "hete":
    criterion = HeteroscedasticLoss()

assert criterion is not None

lr = 2e-5 if not found_existing_model else 5e-6

# Kun LR=0.000005 niin hiukan syvempi malli alkoi oppia
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
# optimizer = torch.optim.Adam(
#    [
#        {
#            "params": model.mean_head.parameters(),
#            "lr": 1e-6,
#        },
#        {
#            "params": model.var_head.parameters(),
#            "lr": 3e-5,
#        },
#    ],
#   weight_decay=1e-4,
# )

lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=4, factor=0.5
)

if not found_existing_model:
    lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: min(1, epoch / 8)
    )

min_loss = None
min_loss_epoch = None

use_amp = False

scaler = torch.amp.GradScaler() if use_amp else None
autocast_context = autocast if use_amp else lambda: torch.enable_grad()

for epoch in range(1, args.epochs + 1):

    model.train()

    train_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        assert torch.min(inputs) >= 0.0 and torch.max(inputs) <= 1.0
        assert torch.min(targets) >= 0.0 and torch.max(targets) <= 1.0
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        with autocast_context():  # torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = roll_forecast(model, inputs, targets, 1)
            train_loss += loss.item()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

    train_loss /= len(train_loader)
    means, stdes = [], []

    model.eval()

    val_loss = 0.0
    with torch.no_grad():

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            if args.loss_function == "mae" or args.loss_function == "mse":
                loss = criterion(model(inputs), targets)
            else:
                mean, stde = model(inputs)
                loss = criterion(mean, stde, targets)
                means.append(torch.mean(mean).item())
                stdes.append(torch.mean(stde).item())

            val_loss += loss.item()
    val_loss /= len(val_loader)
    means = torch.tensor(means)
    stdes = torch.tensor(stdes)

    mean = torch.mean(means).item()
    stde = torch.mean(stdes).item()

    if epoch <= 10 and not found_existing_model:
        lambda_scheduler.step()
    else:
        lr_scheduler.step(val_loss)  # Reduce LR if validation loss has plateaued

    if args.loss_function == "mae" or args.loss_function == "mse":
        print(
            "Epoch [{}/{}], current best: {}. Train Loss: {:.6f} Val Loss: {:.6f}, LRx1M: {:.3f}".format(
                epoch,
                args.epochs,
                min_loss_epoch,
                train_loss,
                val_loss,
                1e6 * optimizer.param_groups[0]["lr"],
            )
        )
    else:
        a = "Mean%"
        b = "Stde%"
        if args.loss_function == "beta_nll":
            # a = "alpha"
            # b = "beta"
            print("alpha: {:.3f} beta: {:.3f}".format(mean, stde))
            xmean = 100 * (mean / (mean + stde))
            stde = 100 * np.sqrt(
                (mean * stde) / ((mean + stde) ** 2 * (mean + stde + 1))
            )
            mean = xmean
        else:
            mean *= 100
            stde *= 100

        print(
            "Epoch [{}/{}], current best: {}. Train Loss: {:.6f} Val Loss: {:.6f} {}: {:.2f} {}: {:.3f} LRx1M: {:.3f}".format(
                epoch,
                args.epochs,
                min_loss_epoch,
                train_loss,
                val_loss,
                a,
                mean,
                b,
                stde,
                1e6 * optimizer.param_groups[0]["lr"],
            )
        )

    if min_loss is None or val_loss < min_loss:
        min_loss = val_loss
        min_loss_epoch = epoch
        torch.save(model.state_dict(), args.save_model_to)

    if epoch >= 8 and epoch - min_loss_epoch > 12:
        print("No improvement in 12 epochs; early stopping")
        break

    if epoch == 8 and not found_existing_model:
        print("Saving model after warmup")
        torch.save(model.state_dict(), args.save_model_to)

    if args.loss_function == "gaussian_nll" and epoch and epoch % 20 == 0:
        criterion.var_reg_weight *= 2
        min_loss = None
        print("Doubled var_reg_weight to", criterion.var_reg_weight)

print("Training done")
