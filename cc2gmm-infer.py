import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
from scipy.stats import norm
from glob import glob
from config import get_args
from cc2gmm import CloudCastV2
from datetime import datetime

args = get_args()

if args.run_name is None:
    print("Please provide a run name")
    sys.exit(1)

now = datetime.now().strftime("%Y%m%d%H%M%S")


def plot_pdf(mean, stde, weights):
    plt.close()
    plt.clf()
    plt.figure()
    # Define the range of x values (0 to 1, since it's a Beta distribution)
    #    x = np.linspace(0, 1, 100)
    x = np.linspace(min(mean - 3 * stde), max(mean + 3 * stde), 500)

    num_mix = mean.shape[0]

    mixture_pdf = np.zeros_like(x)

    for i in range(num_mix):
        m = mean[i]
        s = stde[i]
        w = weights[i]

        g_pdf = norm.pdf(x, m, s)
        plt.plot(x, g_pdf, label=f"Gaussian(mean={m:.3f}, stde={s:.3f})")

        mixture_pdf += w * g_pdf

    plt.plot(x, mixture_pdf, label="Mixture")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.title("Mixture of Gaussian Distributions")
    plt.savefig(f"runs/{args.run_name}/{now}_gaussian_mix.png")


def plot_2d(mean, stde, weights):
    num_mix = mean.shape[-1]

    fig, axes = plt.subplots(num_mix, 3, figsize=(15, 10))
    axes = np.atleast_2d(axes)

    for i in range(num_mix):
        im1 = axes[i, 0].imshow(mean[..., i], cmap="viridis")
        axes[i, 0].set_title(f"Mean {i}")
        fig.colorbar(im1, ax=axes[i, 0])

        im2 = axes[i, 1].imshow(stde[..., i], cmap="viridis")
        axes[i, 1].set_title(f"Stde {i}")
        fig.colorbar(im2, ax=axes[i, 1])

        im3 = axes[i, 2].imshow(weights[..., i], cmap="viridis")
        axes[i, 2].set_title(f"Weight {i}")
        fig.colorbar(im3, ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig(f"runs/{args.run_name}/{now}_gaussian_mix2d.png")


def sample(mean, stde, weights):
    num_mix = mean.shape[-1]
    samples = np.zeros(mean.shape[:-1])

    for i in range(num_mix):
        m = mean[..., i]
        s = stde[..., i]
        w = weights[..., i]

        sample = np.random.normal(m, s, size=mean.shape[:-1])
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
    print("Model loaded successfully from", f"runs/{args.run_name}/model.pth")
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
    mean, stde, weights = model(input_data)

    input_data = input_data.cpu().numpy().squeeze()
    mean = mean.cpu().numpy()
    stde = stde.cpu().numpy()
    weights = weights.cpu().numpy()

    sampled = sample(mean, stde, weights)

    print(
        "mean min: {:.5f} mean max: {:.5f} mean stde: {:.5f}".format(
            np.min(mean), np.max(mean), np.std(mean)
        )
    )
    predicted_mean = input_data + np.sum(weights * mean, axis=-1)
    predicted_sample = input_data + sampled

input_data = input_data.squeeze()
predicted_mean = predicted_mean.squeeze()
predicted_sample = predicted_sample.squeeze()

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
        np.min(predicted_mean),
        np.mean(predicted_mean),
        np.std(predicted_mean),
        np.max(predicted_mean),
    )
)

print(
    "sampled   --> min: {:.4f} mean: {:.4f} stde: {:.4f} max: {:.4f}".format(
        np.min(predicted_sample),
        np.mean(predicted_sample),
        np.std(predicted_sample),
        np.max(predicted_sample),
    )
)

fig, axs = plt.subplots(1, 4, figsize=(12, 5))

for i in range(0, 2):
    axs[i].imshow(pred[i + 1, ...].squeeze(), cmap="gray")
    axs[i].set_title(f"Input Channel {i}" if i < 1 else "Target Image")
    axs[i].axis("off")  # Hide axes

axs[-2].imshow(predicted_sample, cmap="gray")
axs[-2].set_title("Sampled Image")
axs[-2].axis("off")  # Hide axes

axs[-1].imshow(predicted_mean, cmap="gray")
axs[-1].set_title("Mean Image")
axs[-1].axis("off")  # Hide axes

plt.tight_layout()
plt.savefig(f"runs/{args.run_name}/{now}_cc2gmm-prediction.png")


for i in range(model.num_mix):
    print(
        "mean{}: {:.4f}, stde{}: {:.4f}, w{}: {:.3f}".format(
            i,
            np.mean(mean[..., i]),
            i,
            np.mean(stde[..., i]),
            i,
            np.mean(weights[..., i]),
        )
    )

if len(mean.shape) == 2:
    mean = mean[..., np.newaxis]
    stde = stde[..., np.newaxis]
    weights = weights[..., np.newaxis]

# squeeze out the batch dimension and pick the first channel

mean = np.squeeze(mean, axis=0)[0, ...]
stde = np.squeeze(stde, axis=0)[0, ...]
weights = np.squeeze(weights, axis=0)[0, ...]

plot_2d(mean, stde, weights)

mean = np.mean(mean, axis=(0, 1))
stde = np.mean(stde, axis=(0, 1))
weights = np.mean(weights, axis=(0, 1))

plot_pdf(mean, stde, weights)
