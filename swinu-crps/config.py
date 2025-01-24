import argparse
import json
import os
import randomname
import sys
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class TrainingConfig:
    input_resolution: tuple = (128, 128)
    num_members: int = 3

    # Rollout params
    rollout_length: int = 1
    history_length: int = 2

    # Model architecture
    hidden_dim: int = 96
    num_heads: int = 4
    num_layers: int = 8
    window_size: int = 8

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
    current_loss: Optional[float] = None

    run_name: str = randomname.get_name()
    run_dir: str = f"runs/{run_name}"

    data_path: str = "../data/nwcsaf-128x128.zarr"

    def save(self):
        path = f"{self.run_dir}/train-config.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

#    @classmethod
#    def load(cls, path: str):
#        with open(path, "r") as f:
#            return cls(**json.load(f))

    def apply_args(self, args: argparse.Namespace):
        for k, v in vars(args).items():
            if v is not None and hasattr(self, k):
                print(f"Setting {k} to {v}")
                setattr(self, k, v)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_resolution", type=int, nargs=2)
    parser.add_argument("--num_members", type=int)

    parser.add_argument("--rollout_length", type=int)
    parser.add_argument("--history_length", type=int)

    # Model params
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--window_size", type=int)

    # Training params
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_iterations", type=int)
    parser.add_argument("--warmup_iterations", type=int)

    # Compute environment
    parser.add_argument("--num_devices", type=int)
    parser.add_argument("--num_nodes", type=int)
    parser.add_argument("--strategy", type=str)

    parser.add_argument("--run_name", type=str)
    parser.add_argument("--data_path", type=str)

    args = parser.parse_args()

    return args


def get_config():
    args = get_args()

    # Load base config if specified
    if args.run_name:
        config = TrainingConfig.load(f"{args.run_name}/runs/train-config.json")
    else:
        config = TrainingConfig()

    # Override with command line arguments
    for k, v in vars(args).items():
        if v is not None and k != "config":
            setattr(config, k, v)

    return config
