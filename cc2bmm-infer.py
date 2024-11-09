import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
from glob import glob
from scipy.stats import beta as stats_beta
from config import get_args
from cc2bmm import CloudCastV2

args = get_args()

if args.run_name is None:
    print("Please provide a run name")
    sys.exit(1)


def plot_beta(alpha, beta, weights):
    plt.close()
    plt.clf()
    plt.figure()
    # Define the range of x values (0 to 1, since it's a Beta distribution)
    x = np.linspace(0, 1, 100)

    num_mix = alpha.shape[0]

    mixture_pdf = 0

    for i in range(num_mix):
        a = alpha[i]
        b = beta[i]
        w = weights[i]

        b_pdf = stats_beta.pdf(x, a, b)
        plt.plot(x, b_pdf, label=f"Beta({a:.3f}, {b:.3f})")

        mixture_pdf += w * b_pdf

    plt.plot(x, mixture_pdf, label="Mixture")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.title("Mixture of Two Beta Distributions")
    plt.savefig(f"runs/{args.run_name}/betamix.png")


def plot_beta_2d(alpha, beta, weights):
    num_mix = alpha.shape[-1]

    fig, axes = plt.subplots(num_mix, 3, figsize=(15, 10))
    axes = np.atleast_2d(axes)

    for i in range(num_mix):
        im1 = axes[i, 0].imshow(alpha[..., i], cmap="viridis")
        axes[i, 0].set_title(f"Alpha {i}")
        fig.colorbar(im1, ax=axes[i, 0])

        im2 = axes[i, 1].imshow(beta[..., i], cmap="viridis")
        axes[i, 1].set_title(f"Beta {i}")
        fig.colorbar(im2, ax=axes[i, 1])

        im3 = axes[i, 2].imshow(weights[..., i], cmap="viridis")
        axes[i, 2].set_title(f"Weight {i}")
        fig.colorbar(im3, ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig(f"runs/{args.run_name}/betamix2d.png")


def sample(alpha, beta, weights):
    num_mix = alpha.shape[-1]

    samples = np.zeros(alpha.shape[:-1])

    for i in range(num_mix):
        a = alpha[..., i]
        b = beta[..., i]
        w = weights[..., i]

        sample = np.random.beta(a, b, size=alpha.shape[:-1])
        samples += w * sample

    return samples


configs = glob(f"runs/{args.run_name}/*-config.json")

if len(configs) == 0:
    print("No config found from run", args.run_name)
    sys.exit(1)

with open(configs[-1], "r") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device", device)

model = CloudCastV2(
    dim=config["dim"], patch_size=config["patch_size"], num_mix=config["num_mixtures"]
)

try:
    model.load_state_dict(
        torch.load(f"runs/{args.run_name}/model.pth", weights_only=True)
    )
    print("Model loaded successfully from ", f"runs/{args.run_name}/model.pth")
except FileNotFoundError:
    print("No model found, exiting")
    sys.exit(1)

model = model.to(device)

pred = np.load("data/pred2.npz")["arr_0"]  # (5, 128, 128, 1)

pred = torch.tensor(pred, dtype=torch.float32)
model.eval()

input_data = pred[0, ...].unsqueeze(0).unsqueeze(0).to(device)
target_image = pred[-1, ...].numpy().squeeze()

assert torch.min(input_data) >= 0.0 and torch.max(input_data) <= 1.0

with torch.no_grad():  # We don't need to calculate gradients for prediction
    alpha, beta, weights = model(input_data)

    sampled = sample(alpha.cpu().numpy(), beta.cpu().numpy(), weights.cpu().numpy())
    mean = weights * (alpha / (alpha + beta))
    predicted_mean = torch.sum(mean, dim=-1, keepdim=True)

predicted_mean = predicted_mean.cpu()
predicted_image = predicted_mean.squeeze().numpy()

sampled_image = sampled.squeeze()

print(
    "target    --> min: {:.4f} mean: {:.4f} stde: {:.4f} max: {:.4f}".format(
        np.min(target_image),
        np.mean(target_image),
        np.std(target_image),
        np.max(target_image),
    )
)

print(
    "predicted --> min: {:.4f} mean: {:.4f} stde: {:.4f} max: {:.4f}".format(
        np.min(sampled_image),
        np.mean(sampled_image),
        np.std(sampled_image),
        np.max(sampled_image),
    )
)

print(
    "sampled   --> min: {:.4f} mean: {:.4f} stde: {:.4f} max: {:.4f}".format(
        np.min(predicted_image),
        np.mean(predicted_image),
        np.std(predicted_image),
        np.max(predicted_image),
    )
)

fig, axs = plt.subplots(1, 4, figsize=(12, 5))

for i in range(0, 2):
    axs[i].imshow(pred[i + 1, ...].squeeze(), cmap="gray")
    axs[i].set_title(f"Input Channel {i}" if i < 1 else "Target Image")
    axs[i].axis("off")  # Hide axes

axs[-2].imshow(sampled_image, cmap="gray")
axs[-2].set_title("Sampled Image")
axs[-2].axis("off")  # Hide axes

axs[-1].imshow(predicted_image, cmap="gray")
axs[-1].set_title("Mean Image")
axs[-1].axis("off")  # Hide axes

plt.tight_layout()
plt.savefig(f"runs/{args.run_name}/cc2bmm-prediction.png")

alpha = alpha.cpu().numpy().squeeze(axis=(0, 1))
beta = beta.cpu().numpy().squeeze(axis=(0, 1))
weights = weights.cpu().numpy().squeeze(axis=(0, 1))

for i in range(alpha.shape[-1]):
    print(
        "alpha{}: {:.3f}, beta{}: {:.3f}, w{}: {:.3f}".format(
            i,
            np.mean(alpha[..., i]),
            i,
            np.mean(beta[..., i]),
            i,
            np.mean(weights[..., i]),
        )
    )

plot_beta_2d(alpha, beta, weights)

alpha = np.mean(alpha, axis=(0, 1))
beta = np.mean(beta, axis=(0, 1))
weights = np.mean(weights, axis=(0, 1))

plot_beta(alpha, beta, weights)
