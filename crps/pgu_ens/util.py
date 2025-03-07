import torch
import torch.nn.functional as F


def roll_forecast(model, data, forcing, n_step, loss_fn):
    # torch.Size([32, 2, 1, 128, 128]) torch.Size([32, 1, 1, 128, 128])
    x, y = data
    B, T, C_data, H, W = x.shape
    _, T_y, _, _, _ = y.shape

    assert T_y == n_step, "y does not match n_steps: {} vs {}".format(T_y, n_step)

    tendencies = model(data, forcing, n_step)  # B, M, T, C, H, W

    M = tendencies.shape[1]

    assert tendencies.ndim == 6
    assert y.ndim == 5

    current_state = (
        x[:, -1, ...].unsqueeze(1).unsqueeze(1).expand(B, M, 1, C_data, H, W)
    )

    # For single-step rollout
    if n_step == 1:
        # Calculate the ground truth delta between last input state and target
        y_true = y - x[:, -1, ...].unsqueeze(1)

        if loss_fn is None:
            loss = None
        else:
            # loss_fn expects no time dimension
            loss = loss_fn(y_true.squeeze(2), tendencies.squeeze(2))

        # Generate the actual prediction by adding tendency to last input state

        predictions = current_state + tendencies
        predictions = torch.clamp(predictions, 0, 1)

        assert (
            tendencies.shape == predictions.shape
        ), "tendencies and predictions don't match: {} vs {}".format(
            tendencies.shape, predictions.shape
        )
        return loss, tendencies, predictions

    # Initialize empty lists for multi-step evaluation
    losses = []
    all_predictions = []

    # Loop through each rollout step
    for t in range(n_step):
        tendency = tendencies[:, t : t + 1, ...]

        # Add the predicted tendency to get the next state
        next_state = current_state + tendency

        # Store the prediction
        all_predictions.append(next_state)

        # Update current state for next iteration
        current_state = next_state

        # Compute loss for this step
        if loss_fn is not None:
            if t == 0:
                # First step: y - last_x
                # Note: y does not have M dimension
                y_delta = y[:, t : t + 1, ...] - x[:, -1, ...].unsqueeze(1)
            else:
                # Second, third, ... step: y - y_prev
                y_delta = y[:, t : t + 1, ...] - y[:, t - 1 : t, ...]

            step_loss = loss_fn(tendency, y_delta)
            losses.append(step_loss)

    # Stack predictions into a single tensor
    predictions = torch.cat(all_predictions, dim=1)
    predictions = torch.clamp(predictions, 0, 1)

    # Average the losses if we have multiple steps
    if len(losses) > 0:
        loss = torch.stack(losses).mean()
    else:
        loss = None

    assert tendencies.ndim == 6
    return {"loss": loss}, tendencies, predictions
