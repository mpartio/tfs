import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import json
from glob import glob
from config import get_args
from cc2 import CloudCastV2
from datetime import datetime

args = get_args()

if args.run_name is None:
    print("Please provide a run name")
    sys.exit(1)

now = datetime.now().strftime("%Y%m%d%H%M%S")

configs = glob(f"runs/{args.run_name}/*-config.json")

if len(configs) == 0:
    print("No config found from run", args.run_name)
    sys.exit(1)

with open(configs[-1], "r") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device", device)

model = CloudCastV2(dim=config["dim"], patch_size=config["patch_size"])

try:
    model.load_state_dict(
        torch.load(f"runs/{args.run_name}/model.pth", weights_only=True)
    )
    print("Model loaded successfully from ", f"runs/{args.run_name}/model.pth")
except FileNotFoundError:
    print("No model found, exiting")
    sys.exit(1)

model = model.to(device)

pred = np.load("pred.npz")["arr_0"]  # (3, 128, 128, 1)
pred = torch.tensor(pred, dtype=torch.float32) / 100.0

model.eval()

input_data = pred[:1, ...].unsqueeze(0).unsqueeze(-1).to(device)
target_image = pred[1, ...].numpy().squeeze()

assert torch.min(input_data) >= 0.0 and torch.max(input_data) <= 1.0

print("input shape", input_data.shape)
with torch.no_grad():  # We don't need to calculate gradients for prediction
    predicted_image = model(input_data)

predicted_image = predicted_image.cpu().squeeze().numpy()

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

# Plot the target (ground truth) image
axs[-1].imshow(predicted_image, cmap="gray")
axs[-1].set_title("Predicted Image")
axs[-1].axis("off")  # Hide axes


# Show the plots
plt.tight_layout()
plt.savefig(f"runs/{args.run_name}/{now}_prediction.png")
