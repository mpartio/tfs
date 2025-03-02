#!/usr/bin/env python

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd  # For timestamp formatting (if needed)
import sys
import pyproj

def main():
    # Read the two datasets (adjust filenames as needed)
    ds_ground = xr.open_dataset(sys.argv[1])  # 7 time slices
    ds_pred   = xr.open_dataset(sys.argv[2])    # 6 time slices

    times_ground = ds_ground.time.values
    times_pred   = ds_pred.time.values

    ncols = 4
    nrows = 2

    crs_pyproj = pyproj.CRS.from_wkt(ds_pred.spatial_ref)
    proj_dict = crs_pyproj.to_dict()

    cartopy_crs = ccrs.LambertConformal(
        central_longitude=proj_dict["lon_0"],
        central_latitude=proj_dict["lat_0"],
        standard_parallels=(proj_dict["lat_1"], proj_dict["lat_2"]),
    )

    # Create a figure that's wider and taller so plots are bigger
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(20, 10),
        subplot_kw={'projection': cartopy_crs}
    )

    # Common plotting kwargs
    # vmin and vmax force the data range to [0, 1].
    plot_kwargs = dict(
        cmap='viridis',
        vmin=0, vmax=1
    )

    im_gt = None  # We'll store the "mappable" from the last iteration for colorbar
    for i in range(ncols):
        ax = axes[0, i]
        ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5)

        data = ds_ground.effective_cloudiness.isel(time=(i+1))
        im_gt = ax.pcolormesh(ds_ground.x, ds_ground.y, data, **plot_kwargs)

        # Format the datetime to shorten the title
        # E.g. "2025-02-10 11:00"
        time_str = pd.to_datetime(times_ground[i+1]).strftime('%Y-%m-%d %H:%M')
        ax.set_title(time_str, fontsize=9)

    # ----------------------------------------------------------------------
    # Row 2: Predictions (6 slices), first subplot is empty
    # ----------------------------------------------------------------------

    im_pred = None
    for i in range(ncols):
        ax = axes[1, i]
        ax.add_feature(cfeature.BORDERS, edgecolor='black', linewidth=0.5)
        ax.add_feature(cfeature.COASTLINE, edgecolor='black', linewidth=0.5)

        data = ds_pred.effective_cloudiness.isel(time=i)
        im_pred = ax.pcolormesh(ds_pred.x, ds_pred.y, data, **plot_kwargs)

        time_str = pd.to_datetime(times_pred[i]).strftime('%Y-%m-%d %H:%M')
        ax.set_title(time_str, fontsize=9)

    # ----------------------------------------------------------------------
    # Row labels (left side)
    # ----------------------------------------------------------------------
    # Place text along the left side to label rows
    #fig.text(0.04, 0.73, 'Ground Truth', va='center', rotation='vertical',
    #         fontsize=12, fontweight='bold')
    #fig.text(0.04, 0.28, 'Prediction',   va='center', rotation='vertical',
    #         fontsize=12, fontweight='bold')

    plt.subplots_adjust(
        left=0.03, right=0.93, top=0.97, bottom=0.05, hspace=0.06, wspace=0.03
    )

    cbar_ax = fig.add_axes([0.94, 0.07, 0.015, 0.912])  # [left, bottom, width, height]
    cbar = fig.colorbar(im_pred, cax=cbar_ax)
    cbar.set_label("Cloud fraction", size=15)
    cbar.ax.tick_params(labelsize=14)
#    cbar.set_ticks(np.linspace(0,1,11))

    # ----------------------------------------------------------------------
    # Colorbars
    # ----------------------------------------------------------------------
    # 1) Ground truth colorbar
    #    We'll attach it to the entire first row: axes[0, :]
    #cbar_gt = fig.colorbar(im_gt, ax=axes[0, -1], orientation='vertical', fraction=0.02, pad=0.02)
    #cbar_gt.set_label("Effective Cloudiness")

    # 2) Prediction colorbar
    #    We'll attach it to the entire second row (except the empty axis).
    #    For a simpler approach, we can just attach to axes[1, 1:].
    #cbar_pred = fig.colorbar(im_pred, ax=axes[1, -1], orientation='vertical') #, fraction=0.02, pad=0.02)
    #cbar_pred.set_label("Effective Cloudiness")

    # Tighten layout, save, and show
   # plt.tight_layout()
    plt.savefig("figures/predictions.png", dpi=300) #, bbox_inches='tight')


if __name__ == "__main__":
    main()
