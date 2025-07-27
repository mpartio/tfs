from cc2CRPS_data import cc2DataModule
import matplotlib.pyplot as plt
import torch as t
import zarr
import os
import sys
import platform


def print_env():
    print(f"Python version: {sys.version}")
    print(f"Torch version: {t.__version__}")
    print(f"Zarr version: {zarr.__version__}")
    print(f"Number of CPUs: {os.cpu_count()}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    print(f"Filesystem info:")

    try:
        import subprocess

        print(subprocess.check_output(["df", "-T", "."]).decode())
    except:
        print("Couldn't get filesystem type")

    # Also check ulimit
    try:
        import resource

        print(f"Max open files: {resource.getrlimit(resource.RLIMIT_NOFILE)}")
    except:
        print("Couldn't get ulimit info")


print_env()

zarr_path = "/anemoi-data/cerra-475x535-1994-1998-v3.zarr" if len(sys.argv) == 1 else sys.argv[1]
cc2Data = cc2DataModule(
    data_path=zarr_path,
    input_resolution=[475, 535],
    prognostic_params=["tcc"],
    forcing_params=["insolation"],
    batch_size=1,
)
cc2Data.setup("fit")

df = cc2Data.train_dataloader()

for i, batch in enumerate(df):
    data, forcing = batch
    x, y = data
    print("Batch", i, "X shape", x.shape, "Y shape", y.shape)

    x_f = x[0].squeeze()
    y_f = y[0].squeeze()

    print(
        "X min={:.3f} mean={:.3f} max={:.3f}".format(
            t.min(x_f), t.mean(x_f), t.max(x_f)
        )
    )
    print(
        "Y min={:.3f} mean={:.3f} max={:.3f}".format(
            t.min(y_f), t.mean(y_f), t.max(y_f)
        )
    )

    print("Forcing min={:.3f} mean={:.3f} max={:.3f}".format(t.min(forcing), t.mean(forcing), t.max(forcing)))

    fig, ax = plt.subplots(2, 3, figsize=(10, 8))
    ax[0][0].imshow(x_f[0])
    ax[0][1].imshow(x_f[1])
    ax[0][2].imshow(y_f)

    forcing = forcing[0].squeeze()
    ax[1][0].imshow(forcing[0])
    ax[1][1].imshow(forcing[1])
    ax[1][2].imshow(forcing[2])


    plt.savefig("figures/test-dataloader.png")
    print("Saved figure: figures/test-dataloader.png")
    break
