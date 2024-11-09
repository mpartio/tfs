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
from mgnll import MixtureGaussianNLLLoss
from functools import partial
from cc2gmm import CloudCastV2
from tqdm import tqdm
from config import get_args
from util import *
from plot import plot_training_history

# from loss import * #LossWeightScheduler, adaptive_smoothness_loss, CombinedLoss
import matplotlib


def setup_mlflow():

    mlflow_enabled = os.environ.get("MLFLOW_DISABLE", None) is None

    if not mlflow_enabled:
        mlflow = Dummy()
        print("mlflow disabled")
        return

    try:
        import mlflow
    except ModuleNotFoundError:
        mlflow = Dummy()
        print("mlflow disabled")

    mlflow.set_tracking_uri("https://mlflow.apps.ock.fmi.fi")
    mlflow.set_experiment("cc2gmm")


matplotlib.use("Agg")

args = get_args()

gnll_loss = MixtureGaussianNLLLoss()


class PenaltyLoss:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        pass

    def __call__(self, y_pred):
        penalty_loss = alpha * (
            max(0, y_pred - 1)  # Penalize if y_pred > 1
            + max(0, -y_pred)  # Penalize if y_pred < 0
        )

        return penalty_loss


if args.run_name is None:
    args.run_name = randomname.get_name()
    print("No run name specified, generating a new random name:", args.run_name)


def roll_forecast(model, x, y, steps):
    assert y.shape[1] == steps

    cumulative_loss = 0.0

    for i in range(steps):
        y_true = y[:, i, ...].unsqueeze(1)
        assert torch.isnan(x).sum() == 0, "NaN in input x"

        # Try to predict the delta ie the difference between the current
        # frame and the next frame. This is more stable than predicting the
        # next frame directly.

        y_true = y_true - x

        mean, stde, weights = model(x)

        prediction = (
            sample_gaussian(
                mean,
                stde,
                weights,
                num_samples=1,
            )
            .to(device)
            .requires_grad_(True)
        )

        prediction = torch.unsqueeze(prediction, -1)

        total_loss = gnll_loss(mean, stde, weights, y_true)

        cumulative_loss += total_loss

    assert steps == 1
    cumulative_loss /= steps

    return {
        "loss": cumulative_loss,
        "mean": mean,
        "stde": stde,
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

    with open(
        f"{run_dir}/{training_start.strftime('%Y%m%d%H%M%S')}-config.json", "w"
    ) as f:
        json.dump(vars(args), f)

    mlflow.log_params(config)

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = datetime.now()
        model.train()

        total_train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            assert torch.min(inputs) >= 0.0 and torch.max(inputs) <= 1.0
            assert torch.min(targets) >= 0.0 and torch.max(targets) <= 1.0
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast_context():  # torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = roll_forecast(model, inputs, targets, 1)
                train_loss = loss["loss"]
                total_train_loss += train_loss.item()

            # optimizer.zero_grad()

            if use_amp:
                scaler.scale(train_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()

                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

        total_train_loss /= len(train_loader)

        mean_s, stde_s, weights_s = [], [], []

        model.eval()

        total_val_loss = 0.0

        with torch.no_grad():

            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                loss = roll_forecast(model, inputs, targets, 1)

                mean_s.append(torch.mean(loss["mean"], dim=0).cpu().numpy())
                stde_s.append(torch.mean(loss["stde"], dim=0).cpu().numpy())
                weights_s.append(torch.mean(loss["weights"], dim=0).cpu().numpy())

                total_val_loss += loss["loss"].item()

                assert np.isnan(total_val_loss) == 0, "NaN in validation loss"

        n = len(val_loader)
        total_val_loss /= n

        mean_s = np.asarray(mean_s).mean(axis=(0, 1, 2, 3))
        stde_s = np.asarray(stde_s).mean(axis=(0, 1, 2, 3))
        weights_s = np.asarray(weights_s).mean(axis=(0, 1, 2, 3))

        plt.imshow(loss["prediction"][0, 0, ...].detach().cpu().numpy())
        plt.title(f"Epoch {epoch}")
        plt.colorbar()
        plt.savefig(
            f"{run_dir}/{training_start.strftime('%Y%m%d%H%M%S')}-{epoch:03d}-val.png"
        )
        plt.close()

        if epoch <= 10 and not found_existing_model:
            lambda_scheduler.step()
        else:
            lr_scheduler.step(
                total_val_loss
            )  # Reduce LR if validation loss has plateaued

        def format_mean(mean, stde, weights):
            ret = ""
            for i in range(mean.shape[0]):
                ret += (
                    f"m{i}: {mean[i]:.5f}, s{i}: {stde[i]:.6f}, w{i}: {weights[i]:.3f} "
                )

            return ret

        print(
            "Epoch [{}/{}], current best: {}. Train Loss: {:.6f} Val Loss: {:.6f} {} LRx1M: {:.3f}".format(
                epoch,
                args.epochs,
                min_loss_epoch,
                total_train_loss,
                total_val_loss,
                format_mean(mean_s, stde_s, weights_s),
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

        epoch_results = {
            "epoch": epoch,
            "train_loss": total_train_loss,
            "val_loss": total_val_loss,
            "distribution_components": {
                "mean": mean_s.tolist(),
                "stde": stde_s.tolist(),
                "weights": weights_s.tolist(),
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

setup_mlflow()

model, num_params, found_existing_model = load_model(args)

print("Run name is", args.run_name)

train_loader, val_loader = read_data(
    dataset_size=args.dataset_size, batch_size=args.batch_size, hourly=True
)


with mlflow.start_run(run_name=f"{args.run_name}_{training_start.isoformat()}"):
    if found_existing_model:
        mlflow.log_artifact(f"runs/{args.run_name}/model.pth", "restored_model")

    train(model, train_loader, val_loader, found_existing_model)

files = plot_training_history(
    read_training_history(f"runs/{args.run_name}", latest_only=True)
)
for f in files:
    mlflow.log_artifact(f)

mlflow.log_artifact(f"runs/{args.run_name}/model.pth", "trained_model")
mlflow.end_run()

print("Training done at", datetime.now())
