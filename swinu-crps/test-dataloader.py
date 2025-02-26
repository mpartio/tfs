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
        print(subprocess.check_output(['df', '-T', '.']).decode())
    except:
        print("Couldn't get filesystem type")

    # Also check ulimit
    try:
        import resource
        print(f"Max open files: {resource.getrlimit(resource.RLIMIT_NOFILE)}")
    except:
        print("Couldn't get ulimit info")

print_env()

zarr_path = "../data/nwcsaf-128x128-hourly-anemoi.zarr"
cc2Data = cc2DataModule(zarr_path=zarr_path, batch_size=5, n_x=2, n_y=1)

df = cc2Data.train_dataloader()

for batch in df:
    x, y = batch
    print(x.shape)
    print(y.shape)

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

    fig, ax = plt.subplots(1, 3, figsize=(10, 4))
    ax[0].imshow(x_f[0])
    ax[1].imshow(x_f[1])
    ax[2].imshow(y_f)

    plt.savefig("figures/test-dataloader.png")
    print("Saved figure: figures/test-dataloader.png")
    break
