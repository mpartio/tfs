import torch
import os
import randomname
import traceback
import pytorch_lightning as pl
import lightning as L
from lightning.pytorch.cli import LightningCLI
from dataloader.cc2CRPS_data import cc2DataModule
from common.util import get_next_run_number
from pytorch_lightning.utilities.rank_zero import rank_zero_info


class cc2trainer(LightningCLI):
    def before_fit(self):
        # Generate random name if not already set
        run_name = os.environ.get("CC2_RUN_NAME", randomname.get_name())
        os.environ["CC2_RUN_NAME"] = run_name

        # Get base run directory
        run_dir = os.path.join("runs", run_name)
        os.makedirs(run_dir, exist_ok=True)

        version = get_next_run_number(run_dir)

        if self.config.fit.get("ckpt_path") is not None:
            # If a checkpoint path is provided, use the directory of the checkpoint
            version -= 1

        # Create versioned directory
        versioned_dir = os.path.join(run_dir, str(version))
        os.makedirs(versioned_dir, exist_ok=True)

        # Update loggers with version
        for i, logger in enumerate(self.trainer.loggers):
            if isinstance(logger, L.pytorch.loggers.csv_logs.CSVLogger):
                new_logger = pl.loggers.CSVLogger(
                    save_dir=versioned_dir,
                    name="logs" if hasattr(logger, "name") else None,
                    version=None,
                )
                self.trainer.loggers[i] = new_logger

        # Create subdirectories
        figures_dir = os.path.join(versioned_dir, "figures")
        checkpoints_dir = os.path.join(versioned_dir, "checkpoints")
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(checkpoints_dir, exist_ok=True)

        # Store for reference
        self.run_name = run_name
        self.version = version
        self.versioned_dir = versioned_dir

        # Make these accessible to callbacks
        os.environ["CC2_RUN_NUMBER"] = str(version)
        os.environ["CC2_RUN_DIR"] = versioned_dir

        # Update checkpoint callbacks
        for callback in self.trainer.callbacks:
            if callback.__class__.__name__ == "ModelCheckpoint":
                callback.dirpath = checkpoints_dir

        rank_zero_info(f"Run name: {run_name}")
        rank_zero_info(f"Version: {version}")
        rank_zero_info(f"Versioned directory: {versioned_dir}")


torch.set_float32_matmul_precision("high")

cli = cc2trainer(
    model_class=None,  # read from yaml
    datamodule_class=cc2DataModule,
    save_config_kwargs={"overwrite": True, "multifile": False},
)
