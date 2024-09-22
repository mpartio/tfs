import torch
import numpy as np
import sys
from cc2 import CloudCastV2
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device", device)


def load_data():
    data = np.load("test-20k.npz")["arr_0"]  # (B, 128, 128, 1)

    data = data.reshape(data.shape[0] // 2, 2, 128, 128, 1)
    x_data = data[:, 0:1, ...]
    y_data = data[:, 1:2, ...]
    x_data = torch.tensor(x_data, dtype=torch.float32)
    y_data = torch.tensor(y_data, dtype=torch.float32)

    x_data = x_data.permute(0, 1, 4, 2, 3)
    y_data = y_data.permute(0, 1, 4, 2, 3)

    dataset = TensorDataset(x_data, y_data)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=False)

    return dataloader


model = CloudCastV2(dim=192, patch_size=(8, 8))

try:
    model.load_state_dict(torch.load("models/cc2-model.pth", weights_only=True))
except FileNotFoundError:
    print("No model found, exiting")
    sys.exit(1)

model = model.to(device)
model.loss_type = "mse"

loader = load_data()
model.eval()

criterion = torch.nn.L1Loss()

losses = []
with torch.no_grad():
    for inputs, targets in tqdm(loader):
        inputs = inputs.to(device)
        outputs = model(inputs).detach().cpu()
        losses.append(criterion(outputs, targets))

print("Mean absolute error over {} forecasts: {:.4f}".format(5*len(loader), torch.mean(torch.tensor(losses))))
