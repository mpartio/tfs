import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


def scheduled_sampling_inputs(
    previous_state: torch.Tensor,
    current_state: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    t: int,
    step: int,
    max_step: int,
    steepness: float = 10.0,
) -> torch.Tensor:
    """
    Mixes ground-truth and model predictions for two-lag inputs at rollout step t.

    Bengio et al: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks

    Args:
        previous_state: Tensor of shape [B,1,C,H,W], model's last prediction or ground-truth mix.
        current_state:  Tensor of shape [B,1,C,H,W], model's current prediction or ground-truth mix.
        x:             Original input sequence Tensor [B,T,C,H,W].
        y:             Ground-truth future targets Tensor [B,n_step,C,H,W].
        t:             Current rollout step (0-based).
        step:          Current step index.
        max_step:      Total number of steps for scheduling.
        steepness:     Controls sigmoid slope (higher = sharper transition).

    Returns:
        input_state:  Mixed input Tensor to feed into model, shape [B,2,C,H,W].
    """
    B = previous_state.size(0)
    device = previous_state.device

    # Compute probability of using model prediction (p_pred from 0 -> 1)
    p_pred = torch.sigmoid(
        torch.tensor((step / max_step) * steepness - steepness / 2, device=device)
    )

    # Sample independent Bernoulli for each lag
    mask_prev = torch.bernoulli(p_pred * torch.ones([B, 1, 1, 1, 1], device=device))
    mask_curr = torch.bernoulli(p_pred * torch.ones([B, 1, 1, 1, 1], device=device))

    # Determine ground-truth frames for this step
    if t == 0:
        gt_prev = x[:, -2:-1, ...]  # second-to-last input
        gt_curr = x[:, -1:, ...]  # last input
    elif t == 1:
        gt_prev = x[:, -1:, ...]
        gt_curr = y[:, 0:1, ...]
    else:
        gt_prev = y[:, t - 2 : t - 1, ...]
        gt_curr = y[:, t - 1 : t, ...]

    # Mix: mask=1 -> use prediction; mask=0 -> use ground truth
    input_prev = mask_prev * previous_state + (1 - mask_prev) * gt_prev
    input_curr = mask_curr * current_state + (1 - mask_curr) * gt_curr

    return (
        torch.cat([input_prev, input_curr], dim=1),
        p_pred.item(),
        mask_prev.float().mean(),
        mask_curr.float().mean(),
    )


def roll_forecast(
    model: nn.Module,
    data: torch.Tensor,
    forcing: torch.Tensor,
    n_step: int,
    loss_fn,
    use_scheduled_sampling: bool,
    step: int | None = None,
    max_step: int | None = None,
    pl_module: pl.LightningModule | None = None,
):
    # torch.Size([32, 2, 1, 128, 128]) torch.Size([32, 1, 1, 128, 128])
    x, y = data
    B, T, C_data, H, W = x.shape
    _, T_y, _, _, _ = y.shape

    assert T_y == n_step, "y does not match n_steps: {} vs {}".format(T_y, n_step)

    # Initial state is the last state from input sequence
    current_state = x[:, -1, ...].unsqueeze(1)  # Shape: [B, 1, C, H, W]
    previous_state = x[:, -2, ...].unsqueeze(1)

    # Initialize empty lists for multi-step evaluation
    all_losses = []
    all_predictions = []
    all_tendencies = []

    # Loop through each rollout step
    for t in range(n_step):
        step_forcing = forcing[:, t : t + 2, ...]

        if use_scheduled_sampling:
            input_state, p_pred, mask_prev, mask_curr = scheduled_sampling_inputs(
                previous_state,
                current_state,
                x,
                y,
                t,
                step,
                max_step,
            )

            pl_module.log(
                "ss_p_pred",
                p_pred,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            pl_module.log(
                "ss_mask_prev",
                mask_prev,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
            pl_module.log(
                "ss_mask_curr",
                mask_curr,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

        else:
            input_state = torch.cat([previous_state, current_state], dim=1)

        tendency = model(input_state, step_forcing, t)

        # Add the predicted tendency to get the next state
        next_state = current_state + tendency

        # Store the prediction
        all_predictions.append(next_state)
        all_tendencies.append(tendency)

        # Update current state for next iteration
        previous_state = current_state
        current_state = next_state

        # Compute loss for this step
        if loss_fn is not None:
            # Calculate ground truth delta for this step
            if t == 0:
                # First step: y - last_x
                y_true = y[:, t : t + 1, ...] - x[:, -1, ...].unsqueeze(1)
            else:
                # Second, third, ... step: y - y_prev
                y_true = y[:, t : t + 1, ...] - y[:, t - 1 : t, ...]

            loss = loss_fn(y_true, tendency)
            all_losses.append(loss)

    # Stack predictions into a single tensor
    tendencies = torch.cat(all_tendencies, dim=1)
    predictions = torch.cat(all_predictions, dim=1)
    predictions = torch.clamp(predictions, 0, 1)

    loss = None

    if len(all_losses) > 0:
        # aggregate step losses
        loss = {"loss": []}
        for l in all_losses:
            for k, v in l.items():
                if k == "loss":
                    loss["loss"].append(v)
                else:
                    try:
                        loss[k].append(v)
                    except KeyError:
                        loss[k] = [v]

        loss["loss"] = torch.mean(torch.stack(loss["loss"]))

        for k, v in loss.items():
            if k == "loss":
                continue
            loss[k] = torch.stack(loss[k])

    assert tendencies.ndim == 5
    return loss, tendencies, predictions
