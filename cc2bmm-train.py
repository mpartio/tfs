import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import randomname
import os
import json
import mlflow
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
from mbetanll import MixtureBetaNLLLoss
from functools import partial
from cc2bmm import CloudCastV2
from tqdm import tqdm
from config import get_args
from util import *
from plot import plot_training_history
from loss import *  # LossWeightScheduler, adaptive_smoothness_loss, CombinedLoss
import matplotlib

matplotlib.use("Agg")

args = get_args()

if args.run_name is None:
    args.run_name = randomname.get_name()
    print("No run name specified, generating a new random name:", args.run_name)


loss_weight_scheduler = LossWeightScheduler(
    {
        "total_epochs": args.epochs,
        "bnll": {
            "initial": 0.7,
            "final": 0.7,  # Stays constant
        },
        "recon": {
            "initial": 20,
            "final": 15,  # Reduce over time
        },
    }
)


class Dummy:
    """Dummy element that can be called with everything."""

    def __getattribute__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        pass


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


def roll_forecast(model, x, y, steps, combined_loss, epoch):
    assert y.shape[1] == steps

    cumulative_loss = 0.0
    cumulative_bnll_loss = 0.0
    cumulative_recon_loss = 0.0

    for i in range(steps):
        y_true = y[:, i, ...].unsqueeze(1)
        assert torch.isnan(x).sum() == 0, "NaN in input x"

        alpha, beta, weights = model(x)
        prediction = (
            sample_beta(
                alpha,
                beta,
                weights,
                num_samples=1,
            )
            .to(device)
            .requires_grad_(True)
        )

        total_loss, bnll_loss, recon_loss = combined_loss(
            epoch, alpha, beta, weights, prediction, y_true
        )

        cumulative_loss += total_loss
        cumulative_bnll_loss += bnll_loss
        cumulative_recon_loss += recon_loss

    assert steps == 1
    cumulative_loss /= steps
    cumulative_bnll_loss /= steps
    cumulative_recon_loss /= steps

    return {
        "loss": cumulative_loss,
        "bnll_loss": cumulative_bnll_loss,
        "recon_loss": cumulative_recon_loss,
        "alpha": alpha,
        "beta": beta,
        "weights": weights,
        "prediction": prediction,
    }


def train(model, train_loader, val_loader, found_existing_model):

    torch.manual_seed(0)

    lr = 1e-4 if not found_existing_model else 1e-5

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

    config = vars(args)
    config["num_params"] = count_trainable_parameters(model)

    config["loss_weights"] = {}
    for k in ("bnll", "recon"):
        config["loss_weights"][k] = loss_weight_scheduler.config[k]

    with open(
        f"{run_dir}/{training_start.strftime('%Y%m%d%H%M%S')}-config.json", "w"
    ) as f:
        json.dump(vars(args), f)

    mlflow.log_params(config)
    mlflow.log_text(str(model), "model_summary.txt")

    grad_magnitudes = {
        "bnll_loss": 0.0,
        "recon_loss": 0.0,
        "total": 0.0,
    }

    def save_grad_magnitude(name, grad, loss_name):
        grad_magnitudes[loss_name] += grad.abs().sum().item()

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(
                partial(save_grad_magnitude, grad_magnitudes, loss_name="total")
            )

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = datetime.now()
        model.train()

        total_train_loss = 0.0

        combined_loss = CombinedLoss([BNLLLoss(), L1Loss()], loss_weight_scheduler)

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            assert torch.min(inputs) >= 0.0 and torch.max(inputs) <= 1.0
            assert torch.min(targets) >= 0.0 and torch.max(targets) <= 1.0
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast_context():  # torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = roll_forecast(model, inputs, targets, 1, combined_loss, epoch)
                train_loss = loss["loss"]
                total_train_loss += train_loss.item()

            # optimizer.zero_grad()

            if use_amp:
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # BEGIN CALCULATE GRADIENT MAGNITUDE

                # Calculate individual gradient magnitudes
                optimizer.zero_grad()  # Ensure gradients are zeroed at start
                loss["bnll_loss"].backward(retain_graph=True)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_magnitudes["bnll_loss"] += param.grad.abs().sum().item()

                optimizer.zero_grad()  # Zero gradients before next backward pass
                loss["recon_loss"].backward(retain_graph=True)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_magnitudes["recon_loss"] += param.grad.abs().sum().item()

                # END CALCULATE GRADIENT MAGNITUDE

                optimizer.zero_grad()

                train_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

        total_train_loss /= len(train_loader)
        grad_magnitudes = {k: v / len(train_loader) for k, v in grad_magnitudes.items()}

        alpha_s, beta_s, weights_s = [], [], []

        model.eval()

        total_val_loss = 0.0
        total_bnll_loss = 0.0
        total_recon_loss = 0.0

        with torch.no_grad():

            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                loss = roll_forecast(model, inputs, targets, 1, combined_loss, epoch)

                alpha_s.append(torch.mean(loss["alpha"], dim=0).cpu().numpy())
                beta_s.append(torch.mean(loss["beta"], dim=0).cpu().numpy())
                weights_s.append(torch.mean(loss["weights"], dim=0).cpu().numpy())

                total_val_loss += loss["loss"].item()
                total_bnll_loss += loss["bnll_loss"].item()
                total_recon_loss += loss["recon_loss"].item()

                assert np.isnan(total_val_loss) == 0, "NaN in validation loss"

        n = len(val_loader)
        total_val_loss /= n
        total_bnll_loss /= n
        total_recon_loss /= n

        alpha_s = np.asarray(alpha_s).mean(axis=(0, 1, 2, 3))
        beta_s = np.asarray(beta_s).mean(axis=(0, 1, 2, 3))
        weights_s = np.asarray(weights_s).mean(axis=(0, 1, 2, 3))

        plt.imshow(loss["prediction"][0, 0, ...].detach().cpu().numpy())
        plt.title(f"Epoch {epoch}")
        plt.colorbar()
        plt.savefig(
            f"{run_dir}/{training_start.strftime('%Y%m%d%H%M%S')}-{epoch:03d}-val.png"
        )
        plt.close()

        def format_beta(alpha, beta, weights):
            ret = ""
            for i in range(alpha.shape[0]):
                ret += f"a{i}: {alpha[i]:.3f}, b{i}: {beta[i]:.3f}, w{i}: {weights[i]:.3f} "

            return ret

        if epoch <= 10 and not found_existing_model:
            lambda_scheduler.step()
        else:
            lr_scheduler.step(
                total_val_loss
            )  # Reduce LR if validation loss has plateaued

        print(
            "Epoch [{}/{}], current best: {}. Train Loss: {:.6f} Val Loss: {:.6f} {} LRx1M: {:.3f}".format(
                epoch,
                args.epochs,
                min_loss_epoch,
                total_train_loss,
                total_val_loss,
                format_beta(alpha_s, beta_s, weights_s),
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
        if epoch >= 8 and epoch - min_loss_epoch > 12:
            print("No improvement in 12 epochs; early stopping")
            break_loop = True

        epoch_end_time = datetime.now()

        loss_weights = loss_weight_scheduler.get_weights(epoch)

        epoch_results = {
            "epoch": epoch,
            "train_loss": total_train_loss,
            "val_loss": total_val_loss,
            "distribution_components": {
                "alpha": alpha_s.tolist(),
                "beta": beta_s.tolist(),
                "weights": weights_s.tolist(),
            },
            "epoch_start_time": epoch_start_time.isoformat(),
            "epoch_end_time": epoch_end_time.isoformat(),
            "duration": (epoch_end_time - epoch_start_time).total_seconds(),
            "lr": optimizer.param_groups[0]["lr"],
            "loss_components": {
                "total_bnll": total_bnll_loss,
                "unweighted_bnll": total_bnll_loss / loss_weights["bnll"],
                "weight_bnll": loss_weights["bnll"],
                "total_recon": total_recon_loss,
                "unweighted_recon": total_recon_loss / loss_weights["recon"],
                "weight_recon": loss_weights["recon"],
            },
            "gradients": {
                "bnll": grad_magnitudes["bnll_loss"],
                "recon": grad_magnitudes["recon_loss"],
            },
            "saved": saved,
            "last_epoch": break_loop or epoch == args.epochs,
        }

        with open(
            f"{run_dir}/{training_start.strftime('%Y%m%d%H%M%S')}-{epoch:03d}.json", "w"
        ) as f:
            json.dump(epoch_results, f)

        def flatten_dict(d, parent_key="", sep="_"):
            items = []
            for k, v in d.items():
                new_key = parent_key + sep + k if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        items.append((f"{new_key}_{i}", item))
                else:
                    items.append((new_key, v))
            return dict(items)

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

mlflow.set_tracking_uri("https://mlflow.apps.ock.fmi.fi")
mlflow.set_experiment("cc2bmm")

model, num_params, found_existing_model = load_model(args)

print("Run name is", args.run_name)

train_loader, val_loader = read_data(
    dataset_size=args.dataset_size, batch_size=args.batch_size, hourly=True
)

mlflow_enabled = os.environ.get("MLFLOW_DISABLE", None) is None

if not mlflow_enabled:
    mlflow = Dummy()

with mlflow.start_run(run_name=f"{args.run_name}_{training_start.isoformat()}"):
    if found_existing_model:
        mlflow.log_artifact(f"runs/{args.run_name}/model.pth", "restored_model")

    train(model, train_loader, val_loader, found_existing_model)

if mlflow_enabled:
    files = plot_training_history(
        read_training_history(args.run_name, latest_only=True)
    )
    for f in files:
        mlflow.log_artifact(f)
        os.remove(f)

mlflow.log_artifact(f"runs/{args.run_name}/model.pth", "trained_model")

mlflow.end_run()
print("Training done at", datetime.now())
