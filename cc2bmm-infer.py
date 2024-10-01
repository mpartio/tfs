import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import beta
from config import get_args
from cc2bmm import CloudCastV2

args = get_args()

args.load_model_from = "models/cc2bmm-model.pth"
args.save_model_to = "models/cc2bmm-model.pth"

def plot_beta(alpha1, beta1, alpha2, beta2, w1, w2):
    plt.close()
    plt.clf()
    plt.figure()
    # Define the range of x values (0 to 1, since it's a Beta distribution)
    x = np.linspace(0, 1, 100)

    # Calculate the PDF of each Beta distribution
    beta1_pdf = beta.pdf(x, alpha1, beta1)
    beta2_pdf = beta.pdf(x, alpha2, beta2)
    
    # Calculate the PDF of the mixture
    mixture_pdf = w1 * beta1_pdf + w2 * beta2_pdf

    plt.plot(x, beta1_pdf, label=f'Beta({alpha1:.3f}, {beta1:.3f})')
    plt.plot(x, beta2_pdf, label=f'Beta({alpha2:.3f}, {beta2:.3f})')
    plt.plot(x, mixture_pdf, label='Mixture')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.title('Mixture of Two Beta Distributions')
    plt.savefig("betamix.png")


def plot_beta_2d(alpha1, beta1, alpha2, beta2, w1, w2):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    im1 = axes[0, 0].imshow(alpha1, cmap='viridis')
    axes[0, 0].set_title('Alpha 1')
    fig.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(beta1, cmap='viridis')
    axes[0, 1].set_title('Beta 1')
    fig.colorbar(im2, ax=axes[0, 1])

    im3 = axes[0, 2].imshow(w1, cmap='viridis')
    axes[0, 2].set_title('Weight 1')
    fig.colorbar(im3, ax=axes[0, 2])

    im4 = axes[1, 0].imshow(alpha2, cmap='viridis')
    axes[1, 0].set_title('Alpha 2')
    fig.colorbar(im4, ax=axes[1, 0])

    im5 = axes[1, 1].imshow(beta2, cmap='viridis')
    axes[1, 1].set_title('Beta 2')
    fig.colorbar(im5, ax=axes[1, 1])

    im6 = axes[1, 2].imshow(w2, cmap='viridis')
    axes[1, 2].set_title('Weight 2')
    fig.colorbar(im6, ax=axes[1, 2])

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
model.loss_type = args.loss_function
# Make a prediction

pred = np.load("pred.npz")["arr_0"]  # (3, 128, 128, 1)
pred = torch.tensor(pred, dtype=torch.float32)
#pred = pred.permute(0, 3, 1, 2)

model.eval()

input_data = pred[:1, ...].unsqueeze(0).to(device)
target_image = pred[1, ...].numpy().squeeze()

assert torch.min(input_data) >= 0.0 and torch.max(input_data) <= 1.0


with torch.no_grad():  # We don't need to calculate gradients for prediction
    alpha1, beta1, alpha2, beta2, weights = model(input_data)

    mean1 = alpha1 / (alpha1 + beta1)
    mean2 = alpha2 / (alpha2 + beta2)
    w1 = weights[...,0].unsqueeze(-1)
    w2 = weights[...,1].unsqueeze(-1)
    
    predicted_mean = w1 * mean1 + w2 * mean2

#    alpha1 = np.mean(alpha1.cpu().numpy())
#    beta1 = np.mean(beta1.cpu().numpy())
#    alpha2 = np.mean(alpha2.cpu().numpy())
#    beta2 = np.mean(beta2.cpu().numpy())
#    w1 = np.mean(weights[...,0].unsqueeze(-1).cpu().numpy())
#    w2 = np.mean(weights[...,1].unsqueeze(-1).cpu().numpy())

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

for i in range(0,2):
    axs[i].imshow(pred[i+1, ...].squeeze(), cmap="gray")
    axs[i].set_title(f"Input Channel {i}" if i < 1 else "Target Image")
    axs[i].axis("off")  # Hide axes

axs[-1].imshow(predicted_image, cmap="gray")
axs[-1].set_title("Predicted Image")
axs[-1].axis("off")  # Hide axes

plt.tight_layout()
plt.savefig("prediction.png")

alpha1 = alpha1.cpu().numpy().squeeze()
beta1 = beta1.cpu().numpy().squeeze()
alpha2 = alpha2.cpu().numpy().squeeze()
beta2 = beta2.cpu().numpy().squeeze()
w1 = weights[...,0].unsqueeze(-1).cpu().numpy().squeeze()
w2 = weights[...,1].unsqueeze(-1).cpu().numpy().squeeze()

plot_beta_2d(alpha1, beta1, alpha2, beta2, w1, w2)

alpha1 = np.mean(alpha1)
beta1 = np.mean(beta1)
alpha2 = np.mean(alpha2)
beta2 = np.mean(beta2)
w1 = np.mean(w1)
w2 = np.mean(w2)

plot_beta(alpha1, beta1, alpha2, beta2, w1, w2)

print("alpha1: {:.4f}, beta1: {:.4f}, alpha2: {:.4f}, beta2: {:.4f}, w1: {:.4f}, w2: {:.4f}".format(alpha1, beta1, alpha2, beta2, w1, w2))
