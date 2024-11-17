import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
from datetime import datetime
from glob import glob
from config import get_args
from cc2kumar import CloudCastV2
from util import fast_sample_kumaraswamy, beta_function

args = get_args()

if args.run_name is None:
    print("Please provide a run name")
    sys.exit(1)


now = datetime.now().strftime("%Y%m%d%H%M%S")


def kumaraswamy_pdf(x, alpha, beta):
    """
    Compute the PDF of the Kumaraswamy distribution at value x with parameters alpha and beta.

    Parameters:
    - x: The value(s) at which to evaluate the PDF
    - alpha: Shape parameter alpha.
    - beta: Shape parameter beta.

    Returns:
    - PDF values for each value in x.
    """
    pdf = alpha * beta * x ** (alpha - 1) * (1 - x**alpha) ** (beta - 1)
    return pdf


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

        b_pdf = kumaraswamy_pdf(x, a, b)
        plt.plot(x, b_pdf, label=f"Kumaraswamy({a:.3f}, {b:.3f})")

        mixture_pdf += w * b_pdf

    plt.plot(x, mixture_pdf, label="Mixture")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.title("Mixture of Two Kumaraswamy Distributions")
    plt.savefig(f"runs/{args.run_name}/{now}_kumarmix.png")


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
    plt.savefig(f"runs/{args.run_name}/{now}_kumarmix2d.png")


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

with torch.no_grad():
    alpha, beta, weights = model(input_data)

    sampled = fast_sample_kumaraswamy(alpha, beta, weights)
    mean = weights * (beta * beta_function(1 + 1 / alpha, beta))
    second_moment = beta * beta_function(1 + 2 / alpha, beta)
    variance = second_moment - mean**2
    stde = torch.sqrt(variance)
    # predicted_mean = torch.sum(mean, dim=-1, keepdim=True)

predicted_mean = mean.squeeze().cpu().numpy().mean(axis=-1)
predicted_stde = stde.squeeze().cpu().numpy().mean(axis=-1)
predicted_alpha = alpha.squeeze().cpu().numpy()
predicted_beta = beta.squeeze().cpu().numpy()

sampled_image = sampled.cpu().numpy().squeeze()

print(
    "target    --> min: {:.4f} mean: {:.4f} stde: {:.4f} max: {:.4f}".format(
        np.min(target_image),
        np.mean(target_image),
        np.std(target_image),
        np.max(target_image),
    )
)

print(
    "sampled --> min: {:.4f} mean: {:.4f} stde: {:.4f} max: {:.4f}".format(
        np.min(sampled_image),
        np.mean(sampled_image),
        np.std(sampled_image),
        np.max(sampled_image),
    )
)

print(
    "mean   --> min: {:.4f} mean: {:.4f} stde: {:.4f} max: {:.4f}".format(
        np.min(predicted_mean),
        np.mean(predicted_mean),
        np.std(predicted_mean),
        np.max(predicted_mean),
    )
)

num_mix = predicted_alpha.shape[-1]

fig, axs = plt.subplots(2, 5, figsize=(12, 5))

for i in range(0, 2):
    im = axs[0][i].imshow(pred[i + 1, ...].squeeze(), cmap="gray")
    axs[0][i].set_title(f"Input" if i < 1 else "Target Image")
    axs[0][i].axis("off")  # Hide axes
    plt.colorbar(im, ax=axs[0, i])

im = axs[0][2].imshow(sampled_image, cmap="gray")
axs[0][2].set_title("Sampled Image")
axs[0][2].axis("off")  # Hide axes
plt.colorbar(im, ax=axs[0, 2])

im = axs[0][3].imshow(predicted_mean, cmap="gray")
axs[0][3].set_title("Mean")
axs[0][3].axis("off")  # Hide axes
plt.colorbar(im, ax=axs[0, 3])

im = axs[0][4].imshow(predicted_stde, cmap="gray")
axs[0][4].set_title("Stde")
axs[0][4].axis("off")  # Hide axes
plt.colorbar(im, ax=axs[0, 4])

for i in range(num_mix):
    im = axs[1][i * 2].imshow(predicted_alpha[...,i], cmap="gray")
    axs[1][i * 2].set_title(f"Alpha{i}")
    axs[1][i * 2].axis("off")  # Hide axes
    plt.colorbar(im, ax=axs[1, i * 2])

    im = axs[1][i * 2 + 1].imshow(predicted_beta[...,i], cmap="gray")
    axs[1][i * 2 + 1].set_title(f"Beta{i}")
    axs[1][i * 2 + 1].axis("off")  # Hide axes
    plt.colorbar(im, ax=axs[1, i * 2 + 1])

bins = np.linspace(0, 1, 20)
hist1, _ = np.histogram(sampled_image.flatten(), bins=bins, density=True)
hist2, _ = np.histogram(target_image.flatten(), bins=bins, density=True)

bar_width = 0.05
axs[1][-1].bar(bins[:-1] - bar_width / 2, hist1, width=bar_width, alpha=0.7, color="r", label="Sampled")
axs[1][-1].bar(bins[:-1] + bar_width / 2, hist2, width=bar_width, alpha=0.7, color="b", label="Target")
axs[1][-1].legend()

plt.tight_layout()
plt.savefig(f"runs/{args.run_name}/{now}_cc2kumar-prediction.png")

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
