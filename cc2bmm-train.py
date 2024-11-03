import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import randomname
import os
import json
import math
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

args = get_args()

if args.run_name is None:
    args.run_name = randomname.get_name()
    print("No run name specified, generating a new random name:", args.run_name)


bnll_criterion = MixtureBetaNLLLoss()
l1_criterion = nn.L1Loss()


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


class LossWeightScheduler:
    def __init__(self, config):
        self.config = {
            "bnll": {
                "initial": 0.7,
                "final": 0.7,  # Stays constant
            },
            "smoothness": {
                "initial": 1e-5,  # Start  small
                "final": 1e-4,  # Gradually increase
                "warmup_epochs": 5,  # Optional warmup period
            },
            "reconstruction": {
                "initial": 20,
                "final": 10,  # Reduce over time
            },
        }
        self.total_epochs = config["epochs"]

    def get_weights(self, epoch):
        # Progress from 0 to 1
        progress = epoch / self.total_epochs

        # Smoothness weight with optional warmup
        if epoch < self.config["smoothness"]["warmup_epochs"]:
            # Linear warmup
            warmup_progress = epoch / self.config["smoothness"]["warmup_epochs"]
            smoothness_weight = self.config["smoothness"]["initial"] * warmup_progress
        else:
            # Gradual increase
            smoothness_weight = (
                self.config["smoothness"]["initial"]
                + (
                    self.config["smoothness"]["final"]
                    - self.config["smoothness"]["initial"]
                )
                * progress
            )

        # Exponential decay for reconstruction weight
        recon_weight = self.config["reconstruction"]["final"] + (
            self.config["reconstruction"]["initial"]
            - self.config["reconstruction"]["final"]
        ) * math.exp(-3 * progress)

        return {
            "bnll": self.config["bnll"]["initial"],
            "smoothness": smoothness_weight,
            "reconstruction": recon_weight,
        }


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


def adaptive_smoothness_loss(alphas, betas, weights, input_image, num_mixtures):
    losses = []

    scale = 5.0

    def compute_edge_weights(input_image):
        padded = F.pad(input_image, (0, 1, 0, 1), mode="replicate")

        # Compute gradients of input image
        dy = padded[:, :, 1:, :-1] - padded[:, :, :-1, :-1]  # vertical differences
        dx = padded[:, :, :-1, 1:] - padded[:, :, :-1, :-1]  # horizontal differences

        # Compute gradient magnitude
        gradient_magnitude = torch.sqrt(dx**2 + dy**2)

        # Normalize to [0, 1] range
        # edge_weights = gradient_magnitude / (gradient_magnitude.max() + 1e-8)

        edge_weights = torch.sigmoid(gradient_magnitude * scale)
        # Optional: Apply some thresholding to make it more binary
        # edge_weights = torch.sigmoid((edge_weights - threshold) * scale)

        return edge_weights

    if input_image.dim() == 5:
        input_image = input_image.squeeze(-1)  # Removes the last dimension

    # Compute input image gradients to detect edges
    edge_weights = compute_edge_weights(input_image)

    for params in [alphas, betas, weights]:
        for k in range(num_mixtures):
            param_map = params[:, :, :, :, k]
            padded = F.pad(param_map, (0, 1, 0, 1), mode="replicate")

            dy = padded[:, :, 1:, :-1] - padded[:, :, :-1, :-1]
            dx = padded[:, :, :-1, 1:] - padded[:, :, :-1, :-1]

            # Allow larger parameter changes at cloud edges
            weighted_dy = dy * (1.0 - edge_weights[:, :, :, :])
            weighted_dx = dx * (1.0 - edge_weights[:, :, :, :])

            losses.append(
                torch.mean(torch.abs(weighted_dx)) + torch.mean(torch.abs(weighted_dy))
            )

    return sum(losses) / (3 * num_mixtures)


def advanced_adaptive_smoothness_loss(
    alphas, betas, weights, input_image, num_mixtures
):
    def compute_edge_weights(input_image):
        padded = F.pad(input_image, (0, 1, 0, 1), mode="replicate")

        # Compute gradients of input image
        dy = padded[:, :, 1:, :-1] - padded[:, :, :-1, :-1]  # vertical differences
        dx = padded[:, :, :-1, 1:] - padded[:, :, :-1, :-1]  # horizontal differences

        # Compute gradient magnitude
        gradient_magnitude = torch.sqrt(dx**2 + dy**2)

        # Normalize to [0, 1] range
        edge_weights = gradient_magnitude / (gradient_magnitude.max() + 1e-8)

        # Optional: Apply some thresholding to make it more binary
        # edge_weights = torch.sigmoid((edge_weights - threshold) * scale)

        return edge_weights

    if input_image.dim() == 5:
        input_image = input_image.squeeze(-1)  # Removes the last dimension

    # Compute multi-scale edge weights
    edge_weights_small = compute_edge_weights(input_image)
    edge_weights_medium = compute_edge_weights(
        F.avg_pool2d(input_image, 3, stride=1, padding=1)
    )
    edge_weights_large = compute_edge_weights(
        F.avg_pool2d(input_image, 5, stride=1, padding=2)
    )

    # Combine edge weights from different scales
    edge_weights = (edge_weights_small + edge_weights_medium + edge_weights_large) / 3.0

    edge_weights = edge_weights.permute(0, 2, 3, 1)
    # Different weights for different parameter types
    alpha_weight = 1.0
    beta_weight = 1.0
    mixture_weight = 0.5  # Less smoothing for mixture weights

    losses = []
    param_weights = [alpha_weight, beta_weight, mixture_weight]

    for params, weight in zip([alphas, betas, weights], param_weights):
        for k in range(num_mixtures):
            param_map = params[:, :, :, :, k]
            padded = F.pad(param_map, (0, 1, 0, 1), mode="replicate")

            dy = padded[:, :, 1:, :-1] - padded[:, :, :-1, :-1]
            dx = padded[:, :, :-1, 1:] - padded[:, :, :-1, :-1]

            weighted_dy = dy * (1.0 - edge_weights[:, :, :, :])
            weighted_dx = dx * (1.0 - edge_weights[:, :, :, :])

            losses.append(
                weight
                * (
                    torch.mean(torch.abs(weighted_dx))
                    + torch.mean(torch.abs(weighted_dy))
                )
            )

    return sum(losses) / (3 * num_mixtures)


def sample_beta(alpha, beta, weights, num_samples=1):
    """
    Sample from a mixture of Beta distributions.

    Args:
        alpha (torch.Tensor): Tensor of alpha parameters, shape [batch_size, num_mix, ...].
        beta (torch.Tensor): Tensor of beta parameters, shape [batch_size, num_mix, ...].
        weights (torch.Tensor): Tensor of weights for each Beta distribution, shape [batch_size, num_mix, ...].
        num_samples (int): Number of samples to draw for each distribution.

    Returns:
        torch.Tensor: Sampled values, shape [batch_size, num_samples, ...].
    """
    batch_size, steps, H, W, num_mix = alpha.shape

    # Flatten batch and steps dimensions for easier handling
    alpha = alpha.view(batch_size * steps, num_mix, H, W)
    beta = beta.view(batch_size * steps, num_mix, H, W)
    weights = weights.reshape(batch_size * steps, num_mix, H, W)

    # Sample from each Beta distribution
    beta_distributions = torch.distributions.Beta(alpha, beta)

    # Shape: [batch_size * steps, num_mix, H, W]
    samples = beta_distributions.sample()

    # Weight the samples according to the weights of the Beta mixture
    # Reshape weights to match sample dimensions and apply softmax to ensure they sum to 1
    # Shape: [batch_size, num_mix, ...]
    weights = weights.softmax(dim=1)
    samples = (samples * weights).sum(dim=1)  # Weighted sum over num_mix

    # Reshape back to [batch_size, steps, H, W]
    samples = samples.view(batch_size, steps, H, W, 1)

    return samples


def roll_forecast(model, x, y, steps, loss_weights):
    assert y.shape[1] == steps

    cumulative_loss = 0.0
    cumulative_bnll_loss = 0.0
    cumulative_smoothness_loss = 0.0
    cumulative_recon_loss = 0.0

    for i in range(steps):
        y_true = y[:, i, ...].unsqueeze(1)
        assert torch.isnan(x).sum() == 0, "NaN in input x"

        alpha, beta, weights = model(x)

        bnll_loss = bnll_criterion(alpha, beta, weights, y_true)
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

        apsl_loss = adaptive_smoothness_loss(alpha, beta, weights, x, args.num_mixtures)
        # recon_loss = ssim_criterion(prediction, y_true.cpu())
        recon_loss = l1_criterion(prediction, y_true)

        total_loss = (
            loss_weights["bnll"] * bnll_loss
            + loss_weights["smoothness"] * apsl_loss
            + 10 * loss_weights["reconstruction"] * recon_loss
        )
        cumulative_loss += total_loss
        cumulative_bnll_loss += bnll_loss
        cumulative_recon_loss += recon_loss
        cumulative_smoothness_loss += apsl_loss

    assert steps == 1
    cumulative_loss /= steps
    cumulative_bnll_loss /= steps
    cumulative_recon_loss /= steps
    cumulative_smoothness_loss /= steps

    return {
        "loss": cumulative_loss,
        "bnll_loss": cumulative_bnll_loss,
        "recon_loss": cumulative_recon_loss,
        "smoothness_loss": cumulative_smoothness_loss,
        "alpha": alpha,
        "beta": beta,
        "weights": weights,
        "prediction": prediction,
    }


def train(model, train_loader, val_loader, found_existing_model):

    lr = 1e-4 if not found_existing_model else 1e-5

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5
    )

    if not found_existing_model:
        lambda_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: min(1, epoch / 8)
        )

    # Scheduler to adjust loss weights dynamically
    scheduler = LossWeightScheduler({"epochs": args.epochs})

    min_loss = None
    min_loss_epoch = None

    use_amp = False

    scaler = torch.amp.GradScaler() if use_amp else None
    autocast_context = autocast if use_amp else lambda: torch.enable_grad()

    config = vars(args)
    config["num_params"] = count_trainable_parameters(model)

    config["loss_weights"] = {}
    for k in ("bnll", "reconstruction", "smoothness"):
        config["loss_weights"][k] = scheduler.config[k]

    with open(
        f"{run_dir}/{training_start.strftime('%Y%m%d%H%M%S')}-config.json", "w"
    ) as f:
        json.dump(vars(args), f)

    grad_magnitudes = {
        "bnll_loss": 0.0,
        "recon_loss": 0.0,
        "smoothness_loss": 0.0,
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

        loss_weights = scheduler.get_weights(epoch)

        total_train_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            assert torch.min(inputs) >= 0.0 and torch.max(inputs) <= 1.0
            assert torch.min(targets) >= 0.0 and torch.max(targets) <= 1.0
            inputs, targets = inputs.to(device), targets.to(device)

            with autocast_context():  # torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = roll_forecast(model, inputs, targets, 1, loss_weights)
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

                optimizer.zero_grad()  # Zero gradients before next backward pass
                loss["smoothness_loss"].backward(retain_graph=True)
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_magnitudes["smoothness_loss"] += (
                            param.grad.abs().sum().item()
                        )

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
        total_smoothness_loss = 0.0

        with torch.no_grad():

            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                loss = roll_forecast(model, inputs, targets, 1, loss_weights)

                alpha_s.append(torch.mean(loss["alpha"], dim=0).cpu().numpy())
                beta_s.append(torch.mean(loss["beta"], dim=0).cpu().numpy())
                weights_s.append(torch.mean(loss["weights"], dim=0).cpu().numpy())

                total_val_loss += loss["loss"].item()
                total_bnll_loss += loss["bnll_loss"].item()
                total_recon_loss += loss["recon_loss"].item()
                total_smoothness_loss += loss["smoothness_loss"].item()

                assert np.isnan(total_val_loss) == 0, "NaN in validation loss"

        n = len(val_loader)
        total_val_loss /= n
        total_bnll_loss /= n
        total_recon_loss /= n
        total_smoothness_loss /= n

        alpha_s = np.asarray(alpha_s).mean(axis=(0, 1, 2, 3))
        beta_s = np.asarray(beta_s).mean(axis=(0, 1, 2, 3))
        weights_s = np.asarray(weights_s).mean(axis=(0, 1, 2, 3))

        plt.imshow(loss["prediction"][0, 0, ...].detach().cpu().numpy())
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
        if epoch > 2 and (min_loss is None or total_val_loss < min_loss):
            min_loss = total_val_loss
            min_loss_epoch = epoch
            saved = True
            torch.save(model.state_dict(), f"{run_dir}/model.pth")

        break_loop = False
        if epoch >= 8 and epoch - min_loss_epoch > 12:
            print("No improvement in 12 epochs; early stopping")
            break_loop = True

        epoch_end_time = datetime.now()
        epoch_results = {
            "epoch": epoch,
            "train_loss": total_train_loss,
            "val_loss": total_val_loss,
            "alpha": alpha_s.tolist(),
            "beta": beta_s.tolist(),
            "weights": weights_s.tolist(),
            "epoch_start_time": epoch_start_time.isoformat(),
            "epoch_end_time": epoch_end_time.isoformat(),
            "duration": (epoch_end_time - epoch_start_time).total_seconds(),
            "lr": optimizer.param_groups[0]["lr"],
            "bnll_loss": total_bnll_loss * loss_weights["bnll"],
            "recon_loss": total_recon_loss * loss_weights["reconstruction"],
            "smoothness_loss": total_smoothness_loss * loss_weights["smoothness"],
            "raw_bnll_loss": total_bnll_loss,
            "raw_recon_loss": total_recon_loss,
            "raw_smoothness_loss": total_smoothness_loss,
            "bnll_weight": loss_weights["bnll"],
            "recon_weight": loss_weights["reconstruction"],
            "smoothness_weight": loss_weights["smoothness"],
            # "sampled_mean": total_mean,
            "saved": saved,
            "bnll_loss_grad_magnitude": grad_magnitudes["bnll_loss"],
            "recon_loss_grad_magnitude": grad_magnitudes["recon_loss"],
            "smoothness_loss_grad_magnitude": grad_magnitudes["smoothness_loss"],
            "last_epoch": break_loop,
        }

        with open(
            f"{run_dir}/{training_start.strftime('%Y%m%d%H%M%S')}-{epoch:03d}.json", "w"
        ) as f:
            json.dump(epoch_results, f)

        n = len(epoch_results["alpha"])
        for i in range(n):
            epoch_results[f"alpha_{i}"] = epoch_results["alpha"][i]
            epoch_results[f"beta_{i}"] = epoch_results["beta"][i]
            epoch_results[f"weight_{i}"] = epoch_results["weights"][i]

        epoch_results["epoch_start_time"] = epoch_start_time.timestamp()
        epoch_results["epoch_end_time"] = epoch_end_time.timestamp()

        del epoch_results["alpha"]
        del epoch_results["beta"]
        del epoch_results["weights"]

        mlflow.log_metrics(epoch_results)

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

with mlflow.start_run(run_name=f"{args.run_name}_{starting_time.isoformat()}"):
    mlflow.log_params(vars(args))
    mlflow.log_param("num_params", num_params)
    train(model, train_loader, val_loader, found_existing_model)

if mlflow_enabled:
    files = plot_training_history(args.run_dir, latest_only=True)
    for f in files:
        mlflow.log_artifact(f)
        os.remove(f)

mlflow.end_run()
print("Training done at", datetime.now())
