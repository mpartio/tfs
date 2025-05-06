import torch


def calculate_psd(data: torch.Tensor):
    _nx, _ny = data.shape[-2:]
    _window_x = torch.hann_window(
        _nx, periodic=False, device=data.device, dtype=data.dtype
    )
    _window_y = torch.hann_window(
        _ny, periodic=False, device=data.device, dtype=data.dtype
    )
    _window = _window_x.unsqueeze(-1) * _window_y.unsqueeze(0)
    data = data * _window

    nx, ny = data.shape[-2:]
    # Using data.shape[-2] for 'nx' in dx calculation as in original
    dx = 2.5 * (949.0 - 1.0) / nx
    dy = 2.5 * (1069.0 - 1.0) / ny

    f_transform = torch.fft.fft2(data, dim=(-2, -1))
    f_transform = torch.fft.fftshift(f_transform, dim=(-2, -1))

    psd = torch.abs(f_transform) ** 2

    # Using data.shape[-2] for 'nx' in kx as in original
    kx = torch.fft.fftfreq(nx, d=dx, device=data.device, dtype=data.dtype)
    # Using data.shape[-1] for 'ny' in ky, and d=dx as in original
    ky = torch.fft.fftfreq(ny, d=dy, device=data.device, dtype=data.dtype)

    kx = torch.fft.fftshift(kx)
    ky = torch.fft.fftshift(ky)

    # Positive frequencies, excluding zero.
    # nx corresponds to data.shape[-2]
    # ny corresponds to data.shape[-1]
    scale_x = 1.0 / kx[nx // 2 + 1 :]
    scale_y = 1.0 / ky[ny // 2 + 1 :]

    scale_x = torch.flip(scale_x, dims=[0])
    scale_y = torch.flip(scale_y, dims=[0])

    # Ellipsis '...' handles potential batch dimensions
    psd_quadrant = psd[..., nx // 2 + 1 :, ny // 2 + 1 :]
    psd_quadrant = torch.flip(psd_quadrant, dims=[-2, -1])
    psd_quadrant = torch.mean(psd_quadrant, dim=0).sum(dim=-1)

    return scale_x, scale_y, psd_quadrant


def psd(all_truth: torch.tensor, all_predictions: torch.tensor):

    truth = all_truth[0]

    if truth.ndim > 3:
        truth = truth.reshape(-1, truth.shape[-2], truth.shape[-1])

    sx, sy, psd_q = calculate_psd(truth)

    observed_psd = {"sx": sx, "sy": sy, "psd": psd_q}
    predicted_psds = []

    for i in range(len(all_predictions)):
        prediction = all_predictions[i]
        if prediction.ndim > 3:
            prediction = prediction.reshape(
                -1, prediction.shape[-2], prediction.shape[-1]
            )

        sx, sy, psd_q = calculate_psd(prediction)
        predicted_psds.append({"sx": sx, "sy": sy, "psd": psd_q})

    return observed_psd, predicted_psds
