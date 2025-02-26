import torch
import xarray as xr
import config
import rioxarray
import os
from cc2CRPS import cc2CRPS
from cc2util import get_latest_run_dir, read_checkpoint, roll_forecast
from cc2CRPS_data import gaussian_smooth

outfile = "predictions.zarr"

def prepare_data():

    assert not os.path.exists(outfile), "Outfile {} exists".format(outfile)

    print("Opening file {}".format(conf.data_path))

    ds = xr.open_zarr(conf.data_path)

    data = torch.tensor(ds.effective_cloudiness.values).unsqueeze(0)

    if conf.apply_smoothing:
        print("Applying smoothing")
        data = gaussian_smooth(data)

    x = data[:, :2]
    y = data[:, 2:]

    return ds, x, y

def prepare_model():
    assert conf.run_name is not None, "--run_name is missing"
    run_dir = get_latest_run_dir(f"runs/{conf.run_name}")

    assert run_dir is not None, "run_dir not found for {}".format(conf.run_name)

    model = cc2CRPS(conf)

    model = read_checkpoint(f"{run_dir}/models", model)
    model.eval()

    return model


conf = config.get_config()

model = prepare_model()
ds, x, _ = prepare_data()

with torch.no_grad():
    _, _, predictions = roll_forecast(model, x, None, 5, None)

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
