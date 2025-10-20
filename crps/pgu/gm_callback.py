# stage_grad_stats_callback.py
import torch
from lightning.pytorch.callbacks import Callback

EPS = 1e-12


def _flatten_grads(module):
    """Collect and flatten all parameter grads from a module."""
    grads = []
    for p in module.parameters(recurse=True):
        if p.grad is not None:
            g = p.grad.detach()
            grads.append(g.reshape(-1))
    if not grads:
        return None
    return torch.cat(grads, dim=0)


def _mean(x):
    return x.mean()


def _std(x):
    return x.std(unbiased=False)


def _rms(x):
    return (x.pow(2).mean() + EPS).sqrt()


def _max_abs(x):
    return x.abs().max()


def _frac_nan(x):
    return torch.isnan(x).float().mean()


class GradientMonitorCallback(Callback):
    """
    Logs gradient statistics to MLflow (or logger) during training.
    Mirrors your activation logger style.

    modules_to_track: dict(name -> nn.Module)
    Example:
      StageGradStatsCallback({
         "patch_embed": model.patch_embed,
         "downsample": model.downsample,
         "encoder1.0": model.encoder1[0],
         "encoder2.0": model.encoder2[0],
         "decoder1.0": model.decoder1[0],
         "decoder2.0": model.decoder2[0],
         "patch_expand": model.patch_expand,
         "final_expand": model.final_expand,
      }, log_every_n_steps=200)
    """

    def __init__(self, modules_to_track: dict, log_every_n_steps: int = 100):
        super().__init__()
        self.modules = modules_to_track
        self.log_every_n_steps = log_every_n_steps
        self.buffer = {}  # name -> stats dict
        self.global_grad_norm = None

    def on_after_backward(self, trainer, pl_module):
        # Per-stage grads
        self.buffer.clear()
        for name, mod in self.modules.items():
            g = _flatten_grads(mod)
            if g is None:
                continue
            stats = {}
            stats["grad_mean"] = float(_mean(g))
            stats["grad_std"] = float(_std(g))
            stats["grad_rms"] = float(_rms(g))
            stats["grad_max_abs"] = float(_max_abs(g))
            stats["grad_frac_nan"] = float(_frac_nan(g))
            self.buffer[name] = stats

        # Global grad norm (all params)
        total_sq = 0.0
        for p in pl_module.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.detach().float().pow(2).sum())
        self.global_grad_norm = float(total_sq**0.5)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.log_every_n_steps != 0:
            return
        # Emit
        for name, d in self.buffer.items():
            for k, v in d.items():
                pl_module.log(
                    f"grads/{name}_{k}",
                    v,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    sync_dist=False,
                )
        if self.global_grad_norm is not None:
            pl_module.log(
                "grads/global_grad_norm",
                self.global_grad_norm,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=False,
            )
        # clear
        self.buffer.clear()
        self.global_grad_norm = None
