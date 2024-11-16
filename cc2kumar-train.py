import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import randomname
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from functools import partial
from cc2kumar import CloudCastV2
from tqdm import tqdm
from config import get_args
from util import *
from plot import plot_training_history
from crps import CRPSKumaraswamyLoss
from diagnostics import DistributionDiagnostics
import matplotlib

matplotlib.use("Agg")

args = get_args()

crps_loss = CRPSKumaraswamyLoss()

if args.run_name is None:
    args.run_name = randomname.get_name()
    print("No run name specified, generating a new random name:", args.run_name)


def roll_forecast(model, x, y, steps):
    assert y.shape[1] == steps

    cumulative_loss = 0.0
    cumulative_crps_loss = 0.0
    cumulative_beta_regularization = 0.0

    for i in range(steps):
        y_true = y[:, i, ...].unsqueeze(1)
        assert torch.isnan(x).sum() == 0, "NaN in input x"

        alpha, beta, weights = model(x)

        _crps_loss = crps_loss(alpha, beta, weights, y_true)
        _beta_reg = beta_regularization(beta)

        total_loss = _crps_loss + _beta_reg

        cumulative_loss += total_loss
        cumulative_crps_loss += _crps_loss
        cumulative_beta_regularization += _beta_reg

    assert steps == 1
    cumulative_loss /= steps
    cumulative_crps_loss /= steps
    cumulative_beta_regularization /= steps

    return {
        "loss": cumulative_loss,
        "crps_loss": cumulative_crps_loss,
        "beta_regularization": cumulative_beta_regularization,
        "alpha": alpha,
        "beta": beta,
        "weights": weights,
        "prediction": None,  # prediction,
    }


# def spatial_smoothness_loss(alpha, beta, weights, lambda_smooth=100, return_raw=False):
#    """
#    Apply smoothness regularization to mean of Kumaraswamy distribution
#    """
#    # Mean of Kumaraswamy is alpha/(alpha + 1)
#    mean_pred = alpha / (alpha + 1)
#
#    # Compute gradients in both directions
#    dy = mean_pred[:, :, 1:, :] - mean_pred[:, :, :-1, :]
#    dx = mean_pred[:, :, :, 1:] - mean_pred[:, :, :, :-1]
#
#    raw_loss = torch.mean(dx**2) + torch.mean(dy**2)
#
#    if return_raw:
#        return (raw_loss, lambda_smooth * raw_loss)
#    else:
#        return lambda_smooth * raw_loss


def beta_regularization(beta, target_beta=1.5, lambda_beta=0.1):
    """
    Penalize beta values that exceed target_beta
    - Soft constraint using ReLU to only penalize when beta > target_beta
    - Quadratic penalty for smoother gradients
    """
    excess_beta = torch.relu(beta - target_beta)
    return lambda_beta * torch.mean(excess_beta**2)


def analyze_parameter_variation(epoch, alpha, beta):
    """Analyze spatial variation in distribution parameters"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Parameter maps
    im1 = axes[0, 0].imshow(alpha[0, 0].cpu())  # Single batch, channel
    axes[0, 0].set_title("Alpha Map")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(beta[0, 0].cpu())
    axes[0, 1].set_title("Beta Map")
    plt.colorbar(im2, ax=axes[0, 1])

    # Local variation maps
    alpha_var = (alpha[0, 0, 1:, :] - alpha[0, 0, :-1, :]) ** 2
    beta_var = (beta[0, 0, 1:, :] - beta[0, 0, :-1, :]) ** 2

    im3 = axes[1, 0].imshow(alpha_var.cpu())
    axes[1, 0].set_title("Alpha Local Variation")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(beta_var.cpu())
    axes[1, 1].set_title("Beta Local Variation")
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.savefig(f"runs/{args.run_name}/parameter_variation_{epoch:03d}.png")
    plt.close()


def train(model, train_loader, val_loader, found_existing_model):

    torch.manual_seed(0)

    lr = 5e-5 if not found_existing_model else 1e-6

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5
    )

    if not found_existing_model:
        lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: min(1, epoch / 8)
        )

    T_0 = 8
    T_mult = 2
    eta_min = 1e-8
    annealing_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0, T_mult
    )

    min_loss = None
    min_loss_epoch = None

    use_amp = False

    scaler = torch.amp.GradScaler() if use_amp else None
    autocast_context = autocast if use_amp else lambda: torch.enable_grad()

    config = vars(args)
    config["num_params"] = count_trainable_parameters(model)

    with open(
        f"{run_dir}/{training_start.strftime('%Y%m%d%H%M%S')}-config.json", "w"
    ) as f:
        json.dump(vars(args), f)

    mlflow.log_params(config)
    mlflow.log_text(str(model), "model_summary.txt")

    grad_magnitudes = {
        "total_grad": 0.0,
        "crps_loss": 0.0,
        "beta_regularization": 0.0,
    }

    diagnostics = DistributionDiagnostics(f"runs/{args.run_name}")

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = datetime.now()
        model.train()

        total_train_loss = 0.0
        total_crps_loss = 0.0
        total_beta_regularization = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            assert torch.min(inputs) >= 0.0 and torch.max(inputs) <= 1.0
            assert torch.min(targets) >= 0.0 and torch.max(targets) <= 1.0
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast_context():  # torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = roll_forecast(model, inputs, targets, 1)
                train_loss = loss["loss"]
                crps_loss = loss["crps_loss"]
                beta_regularization = loss["beta_regularization"]

                total_train_loss += train_loss.item()
                total_crps_loss += crps_loss.item()
                total_beta_regularization += beta_regularization.item()

            # optimizer.zero_grad()

            if use_amp:
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()

                crps_loss.backward(retain_graph=True)

                grad_magnitudes["crps_loss"] = sum(
                    p.grad.abs().sum().item()
                    for p in model.parameters()
                    if p.grad is not None
                )

                optimizer.zero_grad()

                beta_regularization.backward(retain_graph=True)

                grad_magnitudes["beta_regularization"] = sum(
                    p.grad.abs().sum().item()
                    for p in model.parameters()
                    if p.grad is not None
                )

                optimizer.zero_grad()

                train_loss.backward()

                grad_magnitudes["total_grad"] = sum(
                    p.grad.norm().item()
                    for p in model.parameters()
                    if p.grad is not None
                )

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            if batch_idx % 60 == 0:
                with torch.no_grad():
                    samples = sample_kumaraswamy(
                        loss["alpha"], loss["beta"], loss["weights"], num_samples=100
                    )

                    batch_diagnostics = diagnostics.analyze_batch(
                        loss["alpha"],
                        loss["beta"],
                        loss["weights"],
                        samples,
                        targets,
                        inputs,
                    )

                    # Store in training history
                    for key in diagnostics.train_history:
                        if key in ("last_variance_map", "last_targets", "last_inputs"):
                            # diagnostics.train_history[key] = batch_diagnostics[key]
                            continue
                        else:
                            diagnostics.train_history[key].append(
                                batch_diagnostics[key]
                            )

        total_train_loss /= len(train_loader)
        total_crps_loss /= len(train_loader)
        total_beta_regularization /= len(train_loader)

        alpha_s, beta_s, weights_s = [], [], []

        model.eval()

        total_val_loss = 0.0
        total_val_crps_loss = 0.0
        total_val_beta_regularization = 0.0

        with torch.no_grad():

            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                loss = roll_forecast(model, inputs, targets, 1)

                alpha_s.append(torch.mean(loss["alpha"], dim=0).cpu().numpy())
                beta_s.append(torch.mean(loss["beta"], dim=0).cpu().numpy())
                weights_s.append(torch.mean(loss["weights"], dim=0).cpu().numpy())

                total_val_loss += loss["loss"].item()
                total_val_crps_loss += loss["crps_loss"].item()
                total_val_beta_regularization += loss["beta_regularization"].item()

                assert np.isnan(total_val_loss) == 0, "NaN in validation loss"

                if batch_idx == len(val_loader) - 1:
                    analyze_parameter_variation(epoch, loss["alpha"], loss["beta"])

            # Validation diagnostics (compute for all validation batches)
            samples = sample_kumaraswamy(
                loss["alpha"], loss["beta"], loss["weights"], num_samples=100
            )

            batch_diagnostics = diagnostics.analyze_batch(
                loss["alpha"], loss["beta"], loss["weights"], samples, targets, inputs
            )

            # Store in validation history
            for key in diagnostics.val_history:
                if key in ("last_variance_map", "last_targets", "last_inputs"):
                    # diagnostics.val_history[key] = batch_diagnostics[key]
                    continue
                else:
                    diagnostics.val_history[key].append(batch_diagnostics[key])

        n = len(val_loader)
        total_val_loss /= n
        total_val_crps_loss /= n
        total_val_beta_regularization /= n

        alpha_s = np.asarray(alpha_s).mean(axis=(0, 1, 2, 3))
        beta_s = np.asarray(beta_s).mean(axis=(0, 1, 2, 3))
        weights_s = np.asarray(weights_s).mean(axis=(0, 1, 2, 3))

        if epoch <= 10:
            if not found_existing_model:
                lambda_scheduler.step()
        else:
            annealing_scheduler.step(epoch)

        def format_alpha(alpha, beta, weights):
            ret = ""
            for i in range(alpha.shape[0]):
                ret += f"a{i}: {alpha[i]:.5f}, b{i}: {beta[i]:.6f}, w{i}: {weights[i]:.3f} "

            return ret

        print(
            "Epoch [{}/{}], current best: {}. Train Loss: {:.6f} Val Loss: {:.6f} {} LRx1M: {:.3f}".format(
                epoch,
                args.epochs,
                min_loss_epoch,
                total_train_loss,
                total_val_loss,
                format_alpha(alpha_s, beta_s, weights_s),
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
        if min_loss_epoch is not None and (epoch >= 8 and epoch - min_loss_epoch > 12):
            print("No improvement in 12 epochs; early stopping")
            break_loop = True

        epoch_end_time = datetime.now()

        if (
            (epoch == 1 or epoch > 100)
            and epoch % 10 == 0
            or break_loop
            or epoch == args.epochs
        ):
            diagnostics.plot_diagnostics(epoch, phase="both")

        epoch_results = {
            "epoch": epoch,
            "train_loss": total_train_loss,
            "val_loss": total_val_loss,
            "distribution_components": {
                "alpha": alpha_s.tolist(),
                "beta": beta_s.tolist(),
                "weights": weights_s.tolist(),
            },
            "loss_components": {
                "crps": total_val_crps_loss,
                "beta_reg": total_val_beta_regularization,
                "weight_beta_reg": 0.1,
                "weight_beta_target": 1.5,
            },
            "gradients": {
                "crps": grad_magnitudes["crps_loss"],
                "smoothness": grad_magnitudes["beta_regularization"],
                "total": grad_magnitudes["total_grad"],
            },
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

        epoch_results = flatten_dict(epoch_results)
        epoch_results["epoch_start_time"] = epoch_start_time.timestamp()
        epoch_results["epoch_end_time"] = epoch_end_time.timestamp()

        mlflow.log_metrics(epoch_results, step=epoch)

        if break_loop:
            break


def load_model(args):
    model = CloudCastV2(
        dim=args.dim, patch_size=args.patch_size, num_mix=args.num_mixtures
    )

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

    return model, num_params, found_existing_model


run_dir = f"runs/{args.run_name}"
os.makedirs(run_dir, exist_ok=True)

training_start = datetime.now()
print("Starting training at", training_start)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device", device)

model, num_params, found_existing_model = load_model(args)

print("Run name is", args.run_name)

train_loader, val_loader = read_data(
    dataset_size=args.dataset_size, batch_size=args.batch_size, hourly=True
)

mlflow = setup_mlflow("cc2kumar")

with mlflow.start_run(run_name=f"{args.run_name}_{training_start.isoformat()}"):
    if found_existing_model:
        mlflow.log_artifact(f"runs/{args.run_name}/model.pth", "restored_model")

    train(model, train_loader, val_loader, found_existing_model)

    files = plot_training_history(
        read_training_history(args.run_name, latest_only=True)
    )
    for f in files:
        mlflow.log_artifact(f)

    mlflow.log_artifact(f"runs/{args.run_name}/model.pth", "trained_model")

    mlflow.end_run()

print("Training done at", datetime.now())
