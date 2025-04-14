import torch
import xarray as xr
import os
import importlib
import sys
import argparse
import lightning as L
import torch.nn as nn
from glob import glob
from dataloader.cc2CRPS_data import cc2DataModule

package = os.environ.get("MODEL_FAMILY", "pgu")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rollout_length", type=int, default=1)

    # Training params
    parser.add_argument("--batch_size", type=int)

    # Data params
    parser.add_argument("--apply_smoothing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--data_path", type=str, required=True)

    # Compute environment
    parser.add_argument("--num_devices", type=int)
    parser.add_argument("--run_name", type=str, required=True)

    args = parser.parse_args()

    return args


def dynamic_import(items):
    for item in items:
        path_name = ".".join(item.split(".")[:-1])
        item_name = item.split(".")[-1]
        print("Importing {}".format(item))

        _module = importlib.import_module(path_name)
        globals()[item_name] = getattr(_module, item_name)


imports = [
    "common.util.string_to_type",
    "common.util.get_latest_run_dir",
]

dynamic_import(imports)

sys.path.append(os.path.abspath(package))


imports = [
    "util.roll_forecast",
    "config.get_config",
    "config.TrainingConfig",
]

imports = [f"{package}.{x}" for x in imports]

dynamic_import(imports)
dynamic_import([f"{package}.cc2.cc2CRPS"])
model_class = string_to_type(f"{package}.cc2.cc2CRPS")


def prepare_data(config):

    # assert not os.path.exists(outfile), "Outfile {} exists".format(outfile)

    datamodule = cc2DataModule(config, val_split=0)
    dataloader = datamodule.train_dataloader(shuffle=False)

    return dataloader


def prepare_model_old_school(args):
    latest_dir = get_latest_run_dir(f"runs/{args.run_name}")

    assert latest_dir is not None, "run directory not found for {}".format(
        args.run_name
    )

    config = TrainingConfig.load(f"{latest_dir}/run-info.json")
    config.data_path = args.data_path
    config.rollout_length = args.rollout_length

    if config.run_name != args.run_name:
        print(
            "Warning: config run name ({}) is different from args run name ({})".format(
                config.run_name, args.run_name
            )
        )

    model = cc2CRPSModel(config)
    file_path = f"{latest_dir}/models"

    checkpoints = glob(f"{file_path}/*.ckpt")
    assert checkpoints, "No model checkpoints found in directory {}".format(file_path)
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    ckpt = torch.load(latest_ckpt, weights_only=False)

    state_dict = ckpt["state_dict"]

    if "var_embed" in state_dict.keys():
        del state_dict["var_embed"]

    model.load_state_dict(state_dict)

    return config, model


def prepare_model(args):
    latest_dir = get_latest_run_dir(f"runs/{args.run_name}")

    assert latest_dir is not None, "run directory not found for {}".format(
        args.run_name
    )

    config = TrainingConfig.load(f"{latest_dir}/run-info.json")
    config.data_path = args.data_path
    config.rollout_length = args.rollout_length

    file_path = f"{latest_dir}/models"

    checkpoints = glob(f"{file_path}/*.ckpt")
    assert checkpoints, "No model checkpoints found in directory {}".format(file_path)
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    cc2CRPSModel.load_from_checkpoint(checkpoint_path=latest_ckpt)
    return config, model


class cc2CRPSModel(model_class, L.LightningModule):
    def __init__(self, config):
        L.LightningModule.__init__(self)
        model_class.__init__(
            self,
            config,
        )
        self.config = config
        self.predictions = []
        self.truth = []

    def test_step(self, batch, batch_idx):
        data, forcing = batch

        _, tendencies, predictions = roll_forecast(
            self,
            data,
            forcing,
            args.rollout_length,
            loss_fn=None,
        )

        # We want to include the analysis time also
        analysis_time = data[0][:, -1, ...].unsqueeze(1)
        predictions = torch.concatenate((analysis_time, predictions), dim=1)
        self.predictions.append(predictions)
        truth = torch.concatenate((analysis_time, data[1]), dim=1)
        self.truth.append(truth)

        return {
            "tendencies": tendencies,
            "predictions": predictions,
            "source": data[0],
            "truth": data[1],
        }

    def on_test_end(self):
        output_dir = f"{self.config.run_dir}/test-output/"
        output_dir = f"runs/{args.run_name}/{self.config.run_number}/test-output/"
        os.makedirs(output_dir, exist_ok=True)

        predictions = torch.concatenate(self.predictions)
        truth = torch.concatenate(self.truth)
        torch.save(predictions, f"{output_dir}/predictions.pt")
        torch.save(truth, f"{output_dir}/truth.pt")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Truth shape: {truth.shape}")
        print(f"Wrote files predictions.pt and truth.pt to {output_dir}")


args = get_args()
config, model = prepare_model_old_school(args)

dataloader = prepare_data(config)

torch.set_float32_matmul_precision("high")

trainer = L.Trainer(
    accelerator="auto",
    devices="auto",
    num_nodes=1,
    logger=False,
    enable_checkpointing=False,
)

trainer.test(model, dataloaders=dataloader)
