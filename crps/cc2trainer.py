import torch
import os
import randomname
import traceback
import pytorch_lightning as pl
import lightning as L
import json
import time
from lightning.pytorch.cli import LightningCLI
from dataloader.cc2CRPS_data import cc2DataModule
from common.util import get_next_run_number, get_rank
from pytorch_lightning.utilities.rank_zero import rank_zero_info

coord_file = "ddp_coordination_info.json"


def write_coordination_info(coord_file, run_name, run_number, run_dir):
    # Write coordination info to file for other processes
    coord_info = {
        "run_name": run_name,
        "run_number": run_number,
        "run_dir": run_dir,
    }
    with open(coord_file, "w") as f:
        json.dump(coord_info, f)


def setup_run_dir(coord_file, ckpt_path):
    # Generate random name if not already set
    run_name = os.environ.get("CC2_RUN_NAME", randomname.get_name())
    os.environ["CC2_RUN_NAME"] = run_name

    # Get base run directory
    run_dir = os.path.join("runs", run_name)
    os.makedirs(run_dir, exist_ok=True)

    version = get_next_run_number(run_dir)

    if ckpt_path is not None:
        # If a checkpoint path is provided, use the directory of the checkpoint
        version -= 1

    versioned_dir = os.path.join(run_dir, str(version))
    figures_dir = os.path.join(versioned_dir, "figures")
    checkpoints_dir = os.path.join(versioned_dir, "checkpoints")
    os.makedirs(versioned_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Make these accessible to callbacks
    os.environ["CC2_RUN_NUMBER"] = str(version)
    os.environ["CC2_RUN_DIR"] = versioned_dir

    write_coordination_info(coord_file, run_name, version, versioned_dir)

    return run_name, version, versioned_dir


class cc2trainer(LightningCLI):
    def before_instantiate_classes(self):
        if self.subcommand == "fit" and get_rank() == 0:
            run_name, run_number, run_dir = setup_run_dir(
                coord_file, self.config.fit.get("ckpt_path")
            )

        if self.subcommand == "test":
            # no before_test() hook
            ckpt_path = self.config.test.get("ckpt_path")
            assert ckpt_path is not None

            # runs/x-y/1/checkpoints/file.ckpt
            # or
            # /data/runs/era5-big-skip-rollout-1/1/checkpoints/last.ckpt

            parts = ckpt_path.split("/")

            run_dir = "/".join(parts[:-2])
            run_name = parts[-4]
            run_number = parts[-3]

            os.environ["CC2_RUN_NAME"] = run_name
            os.environ["CC2_RUN_NUMBER"] = run_number
            os.environ["CC2_RUN_DIR"] = run_dir

    def before_fit(self):
        if self.trainer.global_rank == 0:
            # Update loggers with version
            for i, logger in enumerate(self.trainer.loggers):
                if isinstance(logger, L.pytorch.loggers.csv_logs.CSVLogger):
                    new_logger = pl.loggers.CSVLogger(
                        save_dir=os.environ["CC2_RUN_DIR"],
                        name="logs" if hasattr(logger, "name") else None,
                        version=None,
                    )
                    self.trainer.loggers[i] = new_logger

            # Update checkpoint callbacks
            for callback in self.trainer.callbacks:
                if callback.__class__.__name__ == "ModelCheckpoint":
                    callback.dirpath = f"{os.environ['CC2_RUN_DIR']}/checkpoints"

            rank_zero_info(f"Run name: {os.environ['CC2_RUN_NAME']}")
            rank_zero_info(f"Version: {os.environ['CC2_RUN_NUMBER']}")
            rank_zero_info(f"Versioned directory: {os.environ['CC2_RUN_DIR']}")

        else:
            max_wait = 20  # seconds
            start_time = time.time()
            while not os.path.exists(coord_file):
                time.sleep(0.2)
                if time.time() - start_time > max_wait:
                    raise TimeoutError(
                        f"Coordination file not created after {max_wait} seconds"
                    )

                # Read coordination info
                with open(coord_file, "r") as f:
                    coord_info = json.load(f)

                os.environ["CC2_RUN_NAME"] = coord_info["run_name"]
                os.environ["CC2_RUN_NUMBER"] = str(coord_info["run_number"])
                os.environ["CC2_RUN_DIR"] = coord_info["run_dir"]


torch.set_float32_matmul_precision("high")

cli = cc2trainer(
    model_class=None,  # read from yaml
    datamodule_class=cc2DataModule,
    save_config_kwargs={"overwrite": True, "multifile": False},
)
