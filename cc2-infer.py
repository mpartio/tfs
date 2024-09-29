import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
from config import get_args
from cc2 import CloudCastV2

args = get_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device", device)

model = CloudCastV2(dim=args.dim, patch_size=args.patch_size)

try:
    model.load_state_dict(torch.load(args.load_model_from, weights_only=True))
except FileNotFoundError:
    print("No model found, exiting")
    sys.exit(1)

model = model.to(device)
model.loss_type = args.loss_function
# Make a prediction

pred = np.load("pred.npz")["arr_0"]  # (3, 128, 128, 1)
pred = torch.tensor(pred, dtype=torch.float32)
pred = pred.permute(0, 3, 1, 2)

model.eval()

input_data = pred[:1, ...].unsqueeze(0).to(device)
target_image = pred[1, ...].numpy().squeeze()

assert torch.min(input_data) >= 0.0 and torch.max(input_data) <= 1.0

print(input_data.shape, target_image.shape)
with torch.no_grad():  # We don't need to calculate gradients for prediction
    if args.loss_function == "mse" or args.loss_function == "mae":
        predicted_mean = model(input_data)
    elif args.loss_function == "beta_nll":
        predicted_alpha, predicted_beta = model(input_data)
        predicted_mean = (predicted_alpha / (predicted_alpha + predicted_beta))
    elif args.loss_function == "gaussian_nll" or args.loss_function == "hete" or args.loss_function == "crps":
        predicted_mean, predicted_std = model(input_data)

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

# Plot the target (ground truth) image
axs[-1].imshow(predicted_image, cmap="gray")
axs[-1].set_title("Predicted Image")
axs[-1].axis("off")  # Hide axes


# Show the plots
plt.tight_layout()
plt.savefig("prediction.png")
