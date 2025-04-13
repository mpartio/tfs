import argparse
import json
import os
import randomname
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict
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
    patch_size: int = 4
    add_skip_connection: bool = False

    # Training params
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_iterations: int = 100000
    warmup_iterations: int = 1000
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 1

    # Compute environment
    num_nodes: int = 1
    num_devices: int = 1
    strategy: str = "auto"

    # Data params
    apply_smoothing: bool = False
    limit_data_to: int = None
    prognostic_params: tuple = ("tcc",)
    forcing_params: tuple = ("insolation",)
    data_path: str = "../data/nwcsaf-128x128-hourly-anemoi.zarr"
    normalization: Dict[str, str] = field(default_factory=dict)
    disable_normalization: bool = False

    # Current training state
    current_iteration: int = 0

    run_name: str = None
    run_dir: str = None
    run_number: int = None

    def apply_args(self, args: argparse.Namespace):
        for k, v in vars(args).items():
            if v is not None and hasattr(self, k):
                print(f"Setting {k} to {v}")
                setattr(self, k, v)

    @classmethod
    def load(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)["config"]
            for k in ("only_config", "generate_run_name", "start_from"):
                config.pop(k, None)

            instance = cls(**config)
            # When read from json, default type for these is list
            instance.prognostic_params = tuple(instance.prognostic_params)
            instance.forcing_params = tuple(instance.forcing_params)
            instance.input_resolution = tuple(instance.input_resolution)

            return instance

    @classmethod
    def generate_run_name(cls):
        return randomname.get_name()


def get_args():
    def parse_key_value_pair(string):
        try:
            key, value = string.split("=", 1)
            return key, value
        except ValueError:
            msg = f"'{string}' is not a valid key=value pair"
            raise argparse.ArgumentTypeError(msg)

    parser = argparse.ArgumentParser()

    parser.add_argument("--input_resolution", type=int, nargs=2)
    parser.add_argument("--rollout_length", type=int)
    parser.add_argument("--history_length", type=int)

    # Model params
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_heads", type=int)
    parser.add_argument("--window_size", type=int)
    parser.add_argument("--patch_size", type=int)
    parser.add_argument("--encoder_depth", type=int)
    parser.add_argument("--decoder_depth", type=int)
    parser.add_argument("--add_skip_connection", action=argparse.BooleanOptionalAction)

    # Training params
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--num_iterations", type=int)
    parser.add_argument("--warmup_iterations", type=int)
    parser.add_argument("--precision", type=str)
    parser.add_argument("--accumulate_grad_batches", type=int)

    # Data params
    parser.add_argument("--apply_smoothing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--limit_data_to", type=int)
    parser.add_argument("--prognostic_params", type=str, nargs="+")
    parser.add_argument("--forcing_params", type=str, nargs="+")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--normalization", type=parse_key_value_pair, action="append")
    parser.add_argument(
        "--disable_normalization", action=argparse.BooleanOptionalAction
    )

    # Compute environment
    parser.add_argument("--num_devices", type=int)
    parser.add_argument("--num_nodes", type=int)
    parser.add_argument("--strategy", type=str)

    parser.add_argument("--run_name", type=str)

    parser.add_argument("--only_config", action="store_true")
    parser.add_argument("--generate_run_name", action="store_true")
    parser.add_argument("--start_from", type=str)

    args = parser.parse_args()

    args_dict = vars(args)
    if args_dict["normalization"] is not None:
        args_dict["normalization"] = dict(args_dict["normalization"])

    if args.prognostic_params:
        args.prognostic_params = tuple(args.prognostic_params)

    if args.forcing_params:
        args.forcing_params = tuple(args.forcing_params)

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
            run_name = TrainingConfig.generate_run_name()
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
        if v is not None and k not in (
            "only_config",
            "generate_run_name",
            "start_from",
        ):
            setattr(config, k, v)
            if rank == 0:
                print(k, "to", v)

    assert config.warmup_iterations < config.num_iterations
    return config


if __name__ == "__main__":
    c = get_config()
    print(c)
