import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def read_data(n_hist=1, n_futu=1, dataset_size="10k", batch_size=8):
    train_loader, val_loader = None, None

    n_hist = 1
    n_futu = 1

    for ds in ("train", "val"):
        s_len = n_hist + n_futu

        data = np.load(f"{ds}-{dataset_size}.npz")["arr_0"]
        d_len = (data.shape[0] // s_len) * s_len
        data = data[:d_len, ...]
        data = data.reshape(d_len // s_len, s_len, data.shape[1], data.shape[2], 1)
        x_data = data[:, :n_hist, ...]
        y_data = data[:, n_hist : n_hist + n_futu, ...]

        x_data = torch.tensor(x_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.float32)

        print("{} number of samples: {}".format(ds, x_data.shape[0]))
        dataset = TensorDataset(x_data, y_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if ds == "train":
            train_loader = dataloader
        else:
            val_loader = dataloader

    return train_loader, val_loader


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
