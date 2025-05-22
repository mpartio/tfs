import torch
import matplotlib.pyplot as plt


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


def plot_psd(
    run_name: list[str],
    obs_psd: dict,
    pred_psds: list[dict],
    save_path: str = "runs/verification/psd.png",
):

    plt.figure()
    plt.xlabel("Horizontal Scale (km)", fontsize=12)
    plt.ylabel(
        "PSD", fontsize=12
    )  # Add units if clear, e.g., '(Cloud Cover Fraction)$^2$ / (km$^{-2}$)'
    #    plt.xscale("log")
    #    plt.yscale("log")
    plt.title("Power Spectral Density Comparison", fontsize=14)
    plt.grid(True, which="both", ls="-", alpha=0.7)  # Grid for major and minor ticks

    # scales = obs_psd["scales"]
    sx = obs_psd["sx"]
    psd = obs_psd["psd"]
    plt.loglog(sx, psd, label="Observed", linewidth=1, color="black")

    for i in range(len(run_name)):
        sx = pred_psds[i]["sx"]
        # sort_indices = np.argsort(scales)[::-1] # Sort scales descending
        psd = pred_psds[i]["psd"]
        plt.loglog(sx, psd, label=run_name[i], linewidth=2)

    plt.gca().invert_xaxis()

    plt.legend(fontsize=10)
    plt.savefig(save_path)

    print(f"Plot saved to {save_path}")

    plt.close()
