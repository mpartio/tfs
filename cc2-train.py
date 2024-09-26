import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from gnll import GaussianNLLLoss
from betanll import BetaNLLLoss
from hete import HeteroscedasticLoss
from cc2 import *
from tqdm import tqdm
from config import get_args

args = get_args()


def read_data():
    train_loader, val_loader = None, None

    for ds in ("train", "val"):
        data = np.load(f"{ds}-{args.dataset_size}.npz")["arr_0"]
        #        data = data.reshape(int(data.shape[0] / 3), 3, 128, 128, 1)
        #        x_data = data[:, :2, ...]
        #        y_data = data[:, 2:3, ...]

        data = data.reshape(data.shape[0] // 2, 2, data.shape[1], data.shape[2], 1)
        x_data = data[:, 0:1, ...]
        y_data = data[:, 1:2, ...]
        x_data = torch.tensor(x_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.float32)

        x_data = x_data.permute(0, 1, 4, 2, 3)
        y_data = y_data.permute(0, 1, 4, 2, 3)

        print("{} number of samples: {}".format(ds, x_data.shape[0]))
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        if ds == "train":
            train_loader = dataloader
        else:
            val_loader = dataloader

    return train_loader, val_loader


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def roll_forecast(model, x, y, steps):
    assert y.shape[1] == steps

    total_loss = 0.0

    for i in range(steps):
        if args.loss_function == "mae" or args.loss_function == "mse":
            y_hat = model(x)
            loss = criterion(y_hat, y[:, i, ...])
            x = y_hat
        else:
            mean, stde = model(x)
            loss = criterion(mean, stde, y[:, i, ...])
            x = mean

        total_loss += loss.item()

    return total_loss / steps


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device", device)

model = CloudCastV2(dim=192, patch_size=(8, 8))

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

train_loader, val_loader = read_data()

if args.loss_function == "gaussian_nll":
    criterion = GaussianNLLLoss(var_reg_weight=1e-2)
elif args.loss_function == "beta_nll":
    criterion = BetaNLLLoss()
elif args.loss_function == "mae":
    criterion = torch.nn.L1Loss()
elif args.loss_function == "mse":
    criterion = torch.nn.MSELoss()
elif args.loss_function == "hete":
    criterion = HeteroscedasticLoss()

assert criterion is not None

lr = 3e-6 if not found_existing_model else 1e-6

# Kun LR=0.000005 niin hiukan syvempi malli alkoi oppia
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5
)

if not found_existing_model:
    lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: min(1, epoch / 8)
    )

min_loss = 10000
min_loss_epoch = None
last_write_epoch = None

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
            roll_forecast(model, inputs, targets, 1)
            if args.loss_function == "mae" or args.loss_function == "mse":
                loss = criterion(model(inputs), targets)
            else:
                loss = criterion(*model(inputs), targets)
            # loss = criterion(*model(inputs), targets)
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
        print(
            "Epoch [{}/{}], current best: {}. Train Loss: {:.6f} Val Loss: {:.6f} Mean%: {:.2f} Stde%: {:.3f} LRx1M: {:.3f}".format(
                epoch,
                args.epochs,
                min_loss_epoch,
                train_loss,
                val_loss,
                100 * mean,
                100 * stde,
                1e6 * optimizer.param_groups[0]["lr"],
            )
        )

    if val_loss < min_loss:
        min_loss = val_loss
        min_loss_epoch = epoch
        if last_write_epoch is not None and epoch - last_write_epoch > 10:
            print("Saving model")
            torch.save(model.state_dict(), "models/cc2-model.pth")
            last_write_epoch = epoch

    if epoch >= 8 and epoch - min_loss_epoch > 10:
        print("No improvement in 10 epochs; early stopping")
        break

    if epoch == 8:
        print("Saving model after warmup")
        torch.save(model.state_dict(), "models/cc2-model.pth")
        last_write_epoch = epoch

print("Saving model")
torch.save(model.state_dict(), "models/cc2-model.pth")
