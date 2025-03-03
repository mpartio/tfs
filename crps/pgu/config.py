import argparse
import json
import os
import randomname
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, List
from common.util import get_latest_run_dir, get_rank


@dataclass
class TrainingConfig:
    input_resolution: tuple = (128, 128)

    # Rollout params
    rollout_length: int = 1
    history_length: int = 2

    # Model architecture
    hidden_dim: int = 96
    encoder_depth: int = 2
    decoder_depth: int = 2
    window_size: int = 8
    num_heads: int = 12
    num_data_channels: int = 1
    num_forcing_channels: int = 9
    patch_size: int = 4

    # Training params
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_iterations: int = 100000
    warmup_iterations: int = 1000
    precision: str = "16-mixed"

    # Compute environment
    num_nodes: int = 1
    num_devices: int = 1
    strategy: str = "auto"

    # Current training state
    current_iteration: int = 0

    data_path: str = "../data/nwcsaf-128x128-hourly-anemoi.zarr"
    limit_data_to: int = None

    run_name: str = None
    run_dir: str = None
    run_number: int = None

    apply_smoothing: bool = False

    def apply_args(self, args: argparse.Namespace):
        for k, v in vars(args).items():
            if v is not None and hasattr(self, k):
                if k == "run_name":
                    k = "_run_name"

                print(f"Setting {k} to {v}")
                setattr(self, k, v)

    @classmethod
    def load(cls, config_path):
        import json

        with open(config_path, "r") as f:
            config_dict = json.load(f)
            return cls(**config_dict["config"])


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_resolution", type=int, nargs=2)
    parser.add_argument("--num_members", type=int)

    parser.add_argument("--rollout_length", type=int)
    parser.add_argument("--history_length", type=int)

    # Model params
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--num_data_channels", type=int)
    parser.add_argument("--num_forcing_channels", type=int)
    parser.add_argument("--patch_size", type=int)

    # Training params
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_iterations", type=int)
    parser.add_argument("--warmup_iterations", type=int)
    parser.add_argument("--precision", type=str)

    # Compute environment
    parser.add_argument("--num_devices", type=int)
    parser.add_argument("--num_nodes", type=int)
    parser.add_argument("--strategy", type=str)

    parser.add_argument("--run_name", type=str)
    parser.add_argument("--data_path", type=str)

    parser.add_argument("--apply_smoothing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--limit_data_to", type=int)

    parser.add_argument("--only_config", action="store_true")
    parser.add_argument("--generate_run_name", action="store_true")

    args = parser.parse_args()

    return args


def get_config():
    args = get_args()

    # Load base config if specified
    if args.run_name:
        run_dir = get_latest_run_dir(f"runs/{args.run_name}")
        config = TrainingConfig.load(f"{run_dir}/run-info.json")
    else:
        config = TrainingConfig()

    rank = get_rank()
    if args.generate_run_name:
        if rank == 0:
            run_name = randomname.get_name()
            with open("generated_run_name.txt", "w") as f:
                f.write(run_name)
        else:
            run_name = None
            wait_time = 0
            while run_name is None and wait_time < 10:
                try:
                    with open("generated_run_name.txt", "r") as f:
                        run_name = f.read().strip()
                except FileNotFoundError:
                    time.sleep(0.5)
                    wait_time += 0.5

            assert run_name is not None, "Run name not found"

        config.run_name = run_name

    # Override with command line arguments
    for k, v in vars(args).items():
        if v is not None and k not in ("only_config", "generate_run_name"):
            setattr(config, k, v)
            if rank == 0:
                print(k, "to", v)

    assert config.warmup_iterations < config.num_iterations
    return config


if __name__ == "__main__":
    c = get_config()
    print(c)
