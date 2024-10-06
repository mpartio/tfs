import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import randomname
import os
import json
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from mbetanll import MixtureBetaNLLLoss
from ssim import SSIMLoss
from cc2bmm import CloudCastV2
from tqdm import tqdm
from config import get_args
from util import *
from fft_hfc import fft_hfc

args = get_args()

if args.run_name is None:
    args.run_name = randomname.get_name()
    print("No run name specified, generating a new random name:", args.run_name)

run_dir = f"runs/{args.run_name}"
os.makedirs(run_dir, exist_ok=True)

with open(f"{run_dir}/{datetime.now().strftime('%Y%m%d%H%M%S')}-config.json", "w") as f:
    json.dump(vars(args), f)


def mean_and_var(alpha, beta, weights):
    mean = alpha / (alpha + beta)
    var = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))

    weighted_mean = torch.sum(weights * mean, dim=-1, keepdim=True)
    weighted_var = torch.sum(weights * var, dim=-1, keepdim=True)

    # Combine both the internal variance of each Beta distribution
    # (weighted by its corresponding weight) and the variance between
    # the means of the distributions (also weighted by the weights).

    mixture_var = weighted_var + torch.sum(
        weights * (mean - weighted_mean) ** 2, dim=-1, keepdim=True
    )

    return weighted_mean, mixture_var


def combine_losses(beta, ssim, l1):
    bw = 0.05
    sw = 0.5
    lw = 10
    return beta * bw + ssim * sw + l1 * lw


def roll_forecast(model, x, y, steps):
    assert y.shape[1] == steps

    cumulative_loss = 0.0
    cumulative_beta_loss = 0.0
    cumulative_var_reg = 0.0
    cumulative_recon_loss = 0.0
    cumulative_l1_loss = 0.0

    for i in range(steps):
        y_true = y[:, i, ...].unsqueeze(1)
        assert torch.isnan(x).sum() == 0, "NaN in input x"

        alpha, beta, weights = model(x)

        beta_loss = beta_criterion(alpha, beta, weights, y_true)
        mean, var = mean_and_var(alpha, beta, weights)

        recon_loss = ssim_criterion(mean, y_true)
        l1_loss = l1_criterion(mean, y_true)

        total_loss = combine_losses(beta_loss, recon_loss, l1_loss)

        cumulative_loss += total_loss
        cumulative_beta_loss += beta_loss
        cumulative_recon_loss += recon_loss
        cumulative_l1_loss += l1_loss

    assert steps == 1
    cumulative_loss /= steps
    cumulative_beta_loss /= steps
    cumulative_recon_loss /= steps
    cumulative_l1_loss /= steps

    return (
        cumulative_loss,
        cumulative_beta_loss,
        cumulative_recon_loss,
        cumulative_l1_loss,
        alpha,
        beta,
        weights,
        mean,
        var,
    )


training_start = datetime.now()
print("Starting training at", training_start)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device", device)

model = CloudCastV2(dim=args.dim, patch_size=args.patch_size)

print(model)

print("Run name is ", args.run_name)

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

train_loader, val_loader = read_data(
    dataset_size=args.dataset_size, batch_size=args.batch_size, hourly=True
)

beta_criterion = MixtureBetaNLLLoss()
ssim_criterion = SSIMLoss()
l1_criterion = nn.L1Loss()

lr = 2e-5 if not found_existing_model else 2e-6

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
    epoch_start_time = datetime.now()
    model.train()

    train_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        assert torch.min(inputs) >= 0.0 and torch.max(inputs) <= 1.0
        assert torch.min(targets) >= 0.0 and torch.max(targets) <= 1.0
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        with autocast_context():  # torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, _, _, _, _, _, _, _, _ = roll_forecast(model, inputs, targets, 1)
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
    alpha_s, beta_s, weights_s = [], [], []

    model.eval()

    val_loss = 0.0
    total_beta_loss = 0.0
    total_var_reg = 0.0
    total_recon_loss = 0.0
    total_l1_loss = 0.0
    total_fft_hfc_pred = 0.0
    total_fft_hfc_true = 0.0
    total_mean = 0.0
    total_var = 0.0

    with torch.no_grad():

        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            (
                loss,
                beta_loss,
                recon_loss,
                l1_loss,
                alpha,
                beta,
                weights,
                mean,
                var,
            ) = roll_forecast(model, inputs, targets, 1)

            alpha_s.append(torch.mean(alpha, dim=0).cpu().numpy())
            beta_s.append(torch.mean(beta, dim=0).cpu().numpy())
            weights_s.append(torch.mean(weights, dim=0).cpu().numpy())

            val_loss += loss.item()
            total_beta_loss += beta_loss.item()
            total_recon_loss += recon_loss.item()
            total_l1_loss += l1_loss.item()

            total_fft_hfc_pred += fft_hfc(mean.cpu().numpy())
            total_fft_hfc_true += fft_hfc(targets.cpu().numpy())

            total_mean += torch.mean(mean).item()
            total_var += torch.mean(var).item()

            assert np.isnan(val_loss) == 0, "NaN in validation loss"

    val_loss /= len(val_loader)
    total_beta_loss /= len(val_loader)
    total_recon_loss /= len(val_loader)
    total_l1_loss /= len(val_loader)
    total_mean /= len(val_loader)
    total_var /= len(val_loader)

    total_fft_hfc_pred = total_fft_hfc_pred.item() / len(val_loader)
    total_fft_hfc_true = total_fft_hfc_true.item() / len(val_loader)
    fft_hfc_ratio = total_fft_hfc_true / total_fft_hfc_pred

    alpha_s = np.asarray(alpha_s).mean(axis=(0, 1, 2, 3))
    beta_s = np.asarray(beta_s).mean(axis=(0, 1, 2, 3))
    weights_s = np.asarray(weights_s).mean(axis=(0, 1, 2, 3))

    def format_beta(alpha, beta, weights):
        ret = ""
        for i in range(alpha.shape[0]):
            ret += f"a{i}: {alpha[i]:.3f}, b{i}: {beta[i]:.3f}, w{i}: {weights[i]:.3f} "

        return ret

    if epoch <= 10 and not found_existing_model:
        lambda_scheduler.step()
    else:
        lr_scheduler.step(val_loss)  # Reduce LR if validation loss has plateaued

    print(
        "Epoch [{}/{}], current best: {}. Train Loss: {:.6f} Val Loss: {:.6f} {} fft: {:.3f} LRx1M: {:.3f}".format(
            epoch,
            args.epochs,
            min_loss_epoch,
            train_loss,
            val_loss,
            format_beta(alpha_s, beta_s, weights_s),
            fft_hfc_ratio,
            1e6 * optimizer.param_groups[0]["lr"],
        )
    )

    saved = False
    if min_loss is None or val_loss < min_loss:
        min_loss = val_loss
        min_loss_epoch = epoch
        saved = True
        torch.save(model.state_dict(), f"{run_dir}/model.pth")

    epoch_end_time = datetime.now()
    epoch_results = {
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "alpha": alpha_s.tolist(),
        "beta": beta_s.tolist(),
        "weights": weights_s.tolist(),
        "epoch_start_time": epoch_start_time.isoformat(),
        "epoch_end_time": epoch_end_time.isoformat(),
        "duration": (epoch_end_time - epoch_start_time).total_seconds(),
        "lr": optimizer.param_groups[0]["lr"],
        "beta_loss": total_beta_loss,
        "recon_loss": total_recon_loss,
        "l1_loss": total_l1_loss,
        "fft_hfc_pred": total_fft_hfc_pred,
        "fft_hfc_true": total_fft_hfc_true,
        "fft_hfc_ratio": fft_hfc_ratio,
        "mean": total_mean,
        "var": total_var,
        "saved": saved,
    }

    with open(
        f"{run_dir}/{training_start.strftime('%Y%m%d%H%M%S')}-{epoch}.json", "w"
    ) as f:
        json.dump(epoch_results, f)

    if epoch >= 8 and epoch - min_loss_epoch > 12:
        print("No improvement in 12 epochs; early stopping")
        break


print("Training done at", datetime.now())
