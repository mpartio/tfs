import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import beta as stats_beta
from config import get_args
from cc2bmm import CloudCastV2

args = get_args()

args.load_model_from = "models/cc2bmm-model.pth"
args.save_model_to = "models/cc2bmm-model.pth"


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
    # Calculate the PDF of the mixture

    #    mixture_pdf = weights *
    #    mixture_pdf = w1 * beta1_pdf + w2 * beta2_pdf

    plt.plot(x, mixture_pdf, label="Mixture")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("PDF")
    plt.title("Mixture of Two Beta Distributions")
    plt.savefig("betamix.png")


def plot_beta_2d(alpha, beta, weights):
    num_mix = alpha.shape[-1]

    fig, axes = plt.subplots(num_mix, 3, figsize=(15, 10))

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
    plt.savefig("betamix2d.png")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device", device)

model = CloudCastV2(dim=args.dim, patch_size=args.patch_size)

try:
    model.load_state_dict(torch.load(args.load_model_from, weights_only=True))
    print("Model loaded successfully from ", args.load_model_from)
except FileNotFoundError:
    print("No model found, exiting")
    sys.exit(1)

model = model.to(device)

pred = np.load("pred.npz")["arr_0"]  # (3, 128, 128, 1)
pred = torch.tensor(pred, dtype=torch.float32)

model.eval()

input_data = pred[:1, ...].unsqueeze(0).to(device)
target_image = pred[1, ...].numpy().squeeze()

assert torch.min(input_data) >= 0.0 and torch.max(input_data) <= 1.0

with torch.no_grad():  # We don't need to calculate gradients for prediction
    alpha, beta, weights = model(input_data)

    mean = weights * (alpha / (alpha + beta))
    predicted_mean = torch.sum(mean, dim=-1, keepdim=True)

print(predicted_mean.shape)
predicted_mean = predicted_mean.cpu()
predicted_image = predicted_mean.squeeze().numpy()

print(
    "target        --> min: {:.4f} mean: {:.4f} max: {:.4f}".format(
        np.min(target_image), np.mean(target_image), np.max(target_image)
    )
)

print(
    "predicted raw --> min: {:.4f} mean: {:.4f} max: {:.4f}".format(
        np.min(predicted_image), np.mean(predicted_image), np.max(predicted_image)
    )
)

predicted_image = np.clip(predicted_image, 0.0, 1.0)

print(
    "predicted     --> min: {:.4f} mean: {:.4f} max: {:.4f}".format(
        np.min(predicted_image), np.mean(predicted_image), np.max(predicted_image)
    )
)

fig, axs = plt.subplots(1, 3, figsize=(12, 5))

for i in range(0, 2):
    axs[i].imshow(pred[i + 1, ...].squeeze(), cmap="gray")
    axs[i].set_title(f"Input Channel {i}" if i < 1 else "Target Image")
    axs[i].axis("off")  # Hide axes

axs[-1].imshow(predicted_image, cmap="gray")
axs[-1].set_title("Predicted Image")
axs[-1].axis("off")  # Hide axes

plt.tight_layout()
plt.savefig("prediction.png")

alpha = alpha.cpu().numpy().squeeze()
beta = beta.cpu().numpy().squeeze()
weights = weights.cpu().numpy().squeeze()

for i in range(alpha.shape[-1]):
    print(
        "alpha{}: {:.3f}, beta{}: {:.3f}, w{}: {:.3f}".format(
            i, np.mean(alpha[..., i]), i, np.mean(beta[..., i]), i, np.mean(weights[..., i])
        )
    )

plot_beta_2d(alpha, beta, weights)

alpha = np.mean(alpha, axis=(0, 1))
beta = np.mean(beta, axis=(0, 1))
weights = np.mean(weights, axis=(0, 1))

plot_beta(alpha, beta, weights)
