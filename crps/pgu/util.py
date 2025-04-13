import torch
import torch.nn.functional as F


def roll_forecast(model, data, forcing, n_step, loss_fn):
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

            loss = loss_fn(y_true, tendency, t)
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
