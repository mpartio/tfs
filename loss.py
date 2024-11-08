import math
import torch
import torch.nn.functional as F
from mbetanll import MixtureBetaNLLLoss


class BNLLLoss:
    def __init__(self):
        self.loss_fn = MixtureBetaNLLLoss()
        self.__name__ = "bnll"

    def __call__(self, alphas, betas, weights, prediction, target):
        return self.loss_fn(alphas, betas, weights, target)


class L1Loss:
    def __init__(self):
        self.loss_fn = torch.nn.L1Loss()
        self.__name__ = "recon"

    def __call__(self, alphas, betas, weights, prediction, target):
        assert target.shape == prediction.shape, "Shapes must match: {} vs {}".format(
            target.shape, prediction.shape
        )
        return self.loss_fn(prediction, target)


class CombinedLoss:
    def __init__(self, losses, weight_scheduler):
        self.weight_scheduler = weight_scheduler
        self.losses = losses
        self.cumulative_losses = {loss.__name__: 0 for loss in losses}
        self.cumulative_losses["total"] = 0

    def __call__(self, epoch, *args):
        weights = self.weight_scheduler.get_weights(epoch)

        step_losses = []

        total_loss = 0

        for loss_fn in self.losses:
            component_loss = loss_fn(*args)
            total_loss += component_loss * weights[loss_fn.__name__]

            step_losses.append(component_loss)

            self.cumulative_losses[loss_fn.__name__] += component_loss
            self.cumulative_losses["total"] += total_loss

        return [total_loss, *step_losses]

    def get_cumulative_losses(self):
        return self.cumulative_losses


class LossWeightScheduler:
    def __init__(self, config):

        #        self.config = {
        #            "bnll": {
        #                "initial": 0.7,
        #                "final": 0.7,  # Stays constant
        #            },
        #            "smoothness": {
        #                "initial": 1e-5,  # Start  small
        #                "final": 1e-4,  # Gradually increase
        #                "warmup_epochs": 5,  # Optional warmup period
        #            },
        #            "reconstruction": {
        #                "initial": 20,
        #                "final": 15,  # Reduce over time
        #            },
        #        }
        #        self.total_epochs = config["epochs"]

        self.config = config

    def get_weights(self, epoch):
        # Progress from 0 to 1
        progress = epoch / self.config["total_epochs"]

        # Smoothness weight with optional warmup
        #        if epoch < self.config["smoothness"]["warmup_epochs"]:
        #            # Linear warmup
        #            warmup_progress = epoch / self.config["smoothness"]["warmup_epochs"]
        #            smoothness_weight = self.config["smoothness"]["initial"] * warmup_progress
        #        else:
        # Gradual increase
        #            smoothness_weight = (
        #                self.config["smoothness"]["initial"]
        #                + (
        #                    self.config["smoothness"]["final"]
        #                    - self.config["smoothness"]["initial"]
        #                )
        #                * progress
        #            )

        # Exponential decay for reconstruction weight
        recon_weight = self.config["recon"]["final"] + (
            self.config["recon"]["initial"] - self.config["recon"]["final"]
        ) * math.exp(-3 * progress)

        return {
            "bnll": self.config["bnll"]["initial"],
            #            "smoothness": smoothness_weight,
            "recon": recon_weight,
        }


def compute_edge_weights(input_image, scale=5.0):
    padded = F.pad(input_image, (0, 1, 0, 1), mode="replicate")

    # Compute gradients of input image
    dy = padded[:, :, 1:, :-1] - padded[:, :, :-1, :-1]  # vertical differences
    dx = padded[:, :, :-1, 1:] - padded[:, :, :-1, :-1]  # horizontal differences

    # Compute gradient magnitude
    gradient_magnitude = torch.sqrt(dx**2 + dy**2)

    # Normalize to [0, 1] range
    # edge_weights = gradient_magnitude / (gradient_magnitude.max() + 1e-8)

    edge_weights = torch.sigmoid(gradient_magnitude * scale)
    # Optional: Apply some thresholding to make it more binary
    # edge_weights = torch.sigmoid((edge_weights - threshold) * scale)

    return edge_weights


class AdvancedSmoothnessLoss:
    def __init__(self):
        self.name = "asml"

    def __call__(self, alphas, betas, weights, input_image, num_mixtures):
        losses = []

        if input_image.dim() == 5:
            input_image = input_image.squeeze(-1)  # Removes the last dimension

        # Compute input image gradients to detect edges
        edge_weights = compute_edge_weights(input_image)

        for params in [alphas, betas, weights]:
            for k in range(num_mixtures):
                param_map = params[:, :, :, :, k]
                padded = F.pad(param_map, (0, 1, 0, 1), mode="replicate")

                dy = padded[:, :, 1:, :-1] - padded[:, :, :-1, :-1]
                dx = padded[:, :, :-1, 1:] - padded[:, :, :-1, :-1]

                # Allow larger parameter changes at cloud edges
                weighted_dy = dy * (1.0 - edge_weights[:, :, :, :])
                weighted_dx = dx * (1.0 - edge_weights[:, :, :, :])

                losses.append(
                    torch.mean(torch.abs(weighted_dx))
                    + torch.mean(torch.abs(weighted_dy))
                )

        return sum(losses) / (3 * num_mixtures)


class AdvancedAdaptiveSmoothnessLoss:
    def __init__(self):
        self.name = "aasml"

    def __call__(self, alphas, betas, weights, input_image, num_mixtures):
        if input_image.dim() == 5:
            input_image = input_image.squeeze(-1)  # Removes the last dimension

        # Compute multi-scale edge weights
        edge_weights_small = compute_edge_weights(input_image)
        edge_weights_medium = compute_edge_weights(
            F.avg_pool2d(input_image, 3, stride=1, padding=1)
        )
        edge_weights_large = compute_edge_weights(
            F.avg_pool2d(input_image, 5, stride=1, padding=2)
        )

        # Combine edge weights from different scales
        edge_weights = (
            edge_weights_small + edge_weights_medium + edge_weights_large
        ) / 3.0

        edge_weights = edge_weights.permute(0, 2, 3, 1)
        # Different weights for different parameter types
        alpha_weight = 1.0
        beta_weight = 1.0
        mixture_weight = 0.5  # Less smoothing for mixture weights

        losses = []
        param_weights = [alpha_weight, beta_weight, mixture_weight]

        for params, weight in zip([alphas, betas, weights], param_weights):
            for k in range(num_mixtures):
                param_map = params[:, :, :, :, k]
                padded = F.pad(param_map, (0, 1, 0, 1), mode="replicate")

                dy = padded[:, :, 1:, :-1] - padded[:, :, :-1, :-1]
                dx = padded[:, :, :-1, 1:] - padded[:, :, :-1, :-1]

                weighted_dy = dy * (1.0 - edge_weights[:, :, :, :])
                weighted_dx = dx * (1.0 - edge_weights[:, :, :, :])

                losses.append(
                    weight
                    * (
                        torch.mean(torch.abs(weighted_dx))
                        + torch.mean(torch.abs(weighted_dy))
                    )
                )

        return sum(losses) / (3 * num_mixtures)
