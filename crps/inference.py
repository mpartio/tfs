import torch
import xarray as xr
import rioxarray
import os
import importlib
import sys
import argparse
from anemoi.datasets import open_dataset
from dataloader.cc2CRPS_data import gaussian_smooth

package = os.environ.get("MODEL_FAMILY", "pgu_ens")

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--rollout_length", type=int, default=1)

    # Training params
    parser.add_argument("--batch_size", type=int)

    # Data params
    parser.add_argument("--apply_smoothing", action=argparse.BooleanOptionalAction)
    parser.add_argument("--prognostic_path", type=str, required=True)
    parser.add_argument("--forcing_path", type=str, required=True)

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
    "common.util.read_checkpoint",
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


def prepare_data():

    assert not os.path.exists(outfile), "Outfile {} exists".format(outfile)

    print("Reading prognostic data from {}".format(args.prognostic_path))
    x = open_dataset(args.prognostic_path)

    print("Reading forcings data from {}".format(args.forcing_path))
    forcing = open_dataset(args.forcing_path)


#    data = torch.tensor(ds.effective_cloudiness.values).unsqueeze(0)

    if conf.apply_smoothing:
        print("Applying smoothing")
        data = gaussian_smooth(data)

    return (x, None), forcing


def prepare_model(args):
    latest_dir = get_latest_run_dir(f"runs/{args.run_name}")

    assert latest_dir is not None, "run directory not found for {}".format(
        args.run_name
    )

    config = TrainingConfig.load(f"{latest_dir}/run-info.json")
    model = cc2CRPSModel(config)
    model = read_checkpoint(f"{latest_dir}/models", model)
    model.eval()

    return config, model


args = get_args()
config, model = prepare_model(args)

data, forcing = prepare_data()

with torch.no_grad():
    _, _, predictions = roll_forecast(model, data, forcing, args.rollout_length, None)

# torch.Size([1, 5, 1, 1, 64, 64])

analysis_time = torch.tensor(x[0, -1].unsqueeze(0))
predictions = predictions.squeeze()

predictions = torch.concat((analysis_time, predictions))

out_ds = xr.Dataset(
    coords={"y": ds.y, "x": ds.x, "time": ds.time.values[1:]},
    data_vars={
        "longitude": (["y", "x"], ds.longitude.values),
        "latitude": (["y", "x"], ds.latitude.values),
        "effective_cloudiness": (["time", "y", "x"], predictions),
    },
    attrs={"spatial_ref": ds.spatial_ref.attrs["crs_wkt"]},
)

out_ds = out_ds.chunk(
    {
        "time": 1,
        "y": out_ds.y.shape[0],
        "x": out_ds.x.shape[0],
    }
)

out_ds.to_zarr(outfile)

print(out_ds)
print("Wrote to file", outfile)
