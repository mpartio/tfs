import torch
import torch.nn.functional as F


def roll_forecast(model, data, forcing, n_step, loss_fn):
    # torch.Size([32, 2, 1, 128, 128]) torch.Size([32, 1, 1, 128, 128])
    x, y = data  # x: B, T, C, H, W y: B, T, C, H, W
    B, T, C_data, H, W = x.shape
    _, T_y, _, _, _ = y.shape

    assert C_data == 1
    assert T_y == n_step, "y does not match n_steps: {} vs {}".format(T_y, n_step)

    M = model.model.num_members

    assert y.ndim == 5

    previous_state = (
        x[:, -2, ...].unsqueeze(1).unsqueeze(1).expand(B, M, 1, C_data, H, W)
    )

    current_state = (
        x[:, -1, ...].unsqueeze(1).unsqueeze(1).expand(B, M, 1, C_data, H, W)
    )

    forcing = forcing.unsqueeze(1).repeat(1, M, 1, 1, 1, 1)

    assert previous_state.ndim == current_state.ndim == 6

    losses = []
    all_predictions = []
    all_tendencies = []

    for t in range(n_step):
        step_forcing = forcing[:, :, t : t + 2, ...]
        input_state = torch.cat([previous_state, current_state], dim=2)

        tendency = model(input_state, step_forcing, t)  # B, M, T, C, H, W

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

            loss = loss_fn(tendency.squeeze(2), y_true.squeeze(1))
            losses.append(loss)

    # Stack predictions into a single tensor
    tendencies = torch.cat(all_tendencies, dim=2)
    predictions = torch.cat(all_predictions, dim=2)
    predictions = torch.clamp(predictions, 0, 1)

    # Average the losses if we have multiple steps
    if len(losses) > 0:
        losses = torch.stack(losses)
        loss = losses.mean()
    else:
        loss = None

    assert predictions.ndim == tendencies.ndim == 6
    return {"loss": loss, "step_losses": losses}, tendencies, predictions
