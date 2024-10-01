import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from mbetanll import MixtureBetaNLLLoss
from cc2bmm import CloudCastV2
from tqdm import tqdm
from config import get_args
from util import *

args = get_args()

args.load_model_from = "models/cc2bmm-model.pth"
args.save_model_to = "models/cc2bmm-model.pth"


def roll_forecast(model, x, y, steps):
    assert y.shape[1] == steps

    total_loss = None

    for i in range(steps):
        y_true = y[:, i, ...].unsqueeze(1)
        assert torch.isnan(x).sum() == 0, "NaN in input x"
        loss = criterion(*model(x), y_true)
        # x = mean

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

train_loader, val_loader = read_data(dataset_size=args.dataset_size)

criterion = MixtureBetaNLLLoss()

lr = 2e-5 if not found_existing_model else 5e-6

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

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

    a1, b1, a2, b2, w1, w2 = [], [], [], [], [], []

    model.eval()

    val_loss = 0.0
    with torch.no_grad():

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            alpha1, beta1, alpha2, beta2, weights = model(inputs)
            loss = criterion(alpha1, beta1, alpha2, beta2, weights, targets)
            a1.append(torch.mean(alpha1).item())
            b1.append(torch.mean(beta1).item())
            a2.append(torch.mean(alpha2).item())
            b2.append(torch.mean(beta2).item())
            w1.append(torch.mean(weights[..., 0]).item())
            w2.append(torch.mean(weights[..., 1]).item())

            val_loss += loss.item()

            assert np.isnan(val_loss) == 0, "NaN in validation loss"
    val_loss /= len(val_loader)
    a1 = torch.tensor(a1).mean().item()
    b1 = torch.tensor(b1).mean().item()
    a2 = torch.tensor(a2).mean().item()
    b2 = torch.tensor(b2).mean().item()
    w1 = torch.tensor(w1).mean().item()
    w2 = torch.tensor(w2).mean().item()

    #    mean = torch.mean(means).item()

    if epoch <= 10 and not found_existing_model:
        lambda_scheduler.step()
    else:
        lr_scheduler.step(val_loss)  # Reduce LR if validation loss has plateaued

    print(
        "Epoch [{}/{}], current best: {}. Train Loss: {:.6f} Val Loss: {:.6f} a1: {:.3f} b1: {:.3f} w1: {:.3f} a2: {:.3f} b2: {:.3f} w2: {:.3f} LRx1M: {:.3f}".format(
            epoch,
            args.epochs,
            min_loss_epoch,
            train_loss,
            val_loss,
            a1,
            b1,
            w1,
            a2,
            b2,
            w2,
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


print("Training done")
