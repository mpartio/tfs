import torch
import torch.nn.functional as F


def roll_forecast(model, data, forcing, n_step, loss_fn):
    # torch.Size([32, 2, 1, 128, 128]) torch.Size([32, 1, 1, 128, 128])
    x, y = data
    B, T, C_data, H, W = x.shape
    _, T_y, _, _, _ = y.shape

    assert T_y == n_step, "y does not match n_steps: {} vs {}".format(T_y, n_step)

    step_weights = torch.tensor([1, 1.2, 1.5, 1.8, 2.2, 3.0])[:n_step].to(x.device)
    tendency_weights = torch.tensor([2, 2.2, 2.5, 2.8, 3.2, 4.0])[:n_step].to(x.device)

    # Initial state is the last state from input sequence
    current_state = x[:, -1, ...].unsqueeze(1)  # Shape: [B, 1, C, H, W]
    previous_state = x[:, -2, ...].unsqueeze(1)

    # Initialize empty lists for multi-step evaluation
    step_losses = []
    tendency_losses = []
    all_predictions = []
    all_tendencies = []

    # Loop through each rollout step
    for t in range(n_step):
        step_forcing = forcing[:, t : t + 2, ...]
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

            step_loss = loss_fn(tendency, y_true)
            step_losses.append(step_loss)

            tendency_importance = 1.0 + 5.0 * torch.abs(y_true)
            tendency_loss = torch.mean(tendency_importance * (tendency - y_true) ** 2)
            tendency_losses.append(tendency_loss)

    # Stack predictions into a single tensor
    tendencies = torch.cat(all_tendencies, dim=1)
    predictions = torch.cat(all_predictions, dim=1)
    predictions = torch.clamp(predictions, 0, 1)

    loss = None

    if loss_fn is not None:
        step_losses = step_weights * torch.stack(step_losses)
        tendency_losses = tendency_weights * torch.stack(tendency_losses)

        step_losses_individual = step_losses.detach()
        tendency_losses_individual = tendency_losses.detach()

        loss = torch.sum(step_losses + tendency_losses)

        assert torch.isfinite(loss).all() , "Loss is non-finite"

        loss = {
            "loss": loss,
            "step_losses": step_losses_individual,
            "tendency_losses": tendency_losses_individual,
        }

    assert tendencies.ndim == 5
    return loss, tendencies, predictions
