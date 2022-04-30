from typing import Tuple, List

import numpy as np
import pandas
from matplotlib import pyplot as plt
import seaborn as sns


def plot_heatmap(
        data: pandas.DataFrame,
        parent_fig,
        title: str = "Heat Plot",
        x_label: str = "X Label",
        y_label: str = "Y Label",
        c_bar_label: str = "Color Bar Label",
        rotate_top_labels: bool = True,
        input_heatmap_options=None,
        invert_y_axis: bool = False,
):
    """Plot a heatmap with the given data."""
    # Set default figure options and add any overridden options.
    heatmap_options = {
        "square": False,
        "linewidths": 1
    }
    if input_heatmap_options is not None:
        heatmap_options |= input_heatmap_options

    # Get the columns in the data frame.
    columns = list(data.columns)

    # Create two subplots--one to hold the heatmap and the other to hold the color map.
    (heatmap_ax, color_map_ax) = parent_fig.subplots(
        nrows=1, ncols=2, gridspec_kw={"width_ratios": [2 * len(columns), 1]}
    )

    # Render the data as a heatmap.
    sns.heatmap(data, ax=heatmap_ax, cbar_ax=color_map_ax, **heatmap_options)

    # Move the x-axis ticks to the top of the image.
    heatmap_ax.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)

    # Rotate the tick labels and set their alignment.
    if rotate_top_labels:
        plt.setp(heatmap_ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")
    plt.setp(heatmap_ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")

    if invert_y_axis:
        # Invert the y-axis such that the first element appears on top.
        heatmap_ax.invert_yaxis()

    # Decorate the figure with the appropriate titles and axis labels.
    parent_fig.suptitle(title, y=1.00)
    heatmap_ax.set_xlabel(x_label, labelpad=12)
    heatmap_ax.set_ylabel(y_label, labelpad=12)
    color_map_ax.set_ylabel(c_bar_label, labelpad=12)
    heatmap_ax.xaxis.set_label_position("top")


def plot_corr_heatmap(
        data: pandas.DataFrame,
        parent_fig,
        title: str = "Heat Plot",
        x_label: str = "X Label",
        y_label: str = "Y Label",
        c_bar_label: str = "Correlation Coefficient",
        include_mask: bool = False,
        input_heatmap_options=None
):
    """Plot a heatmap with the given data."""
    # Default options in heatmaps, including a mask that hides half of the values.
    heatmap_options = {
        "vmin": -1.0,
        "center": 0.0,
        "vmax": 1.0,
        "annot": True,
        "fmt": ".2f"
    }

    # Mask mirrored values if requested.
    if include_mask:
        mask = np.zeros_like(data)
        mask[np.triu_indices_from(mask, 1)] = True
        heatmap_options["mask"] = mask

    # Merge the default correlation heatmap.
    if input_heatmap_options is not None:
        heatmap_options |= input_heatmap_options

    # Call the heatmap plot method.
    plot_heatmap(
        data, parent_fig, title, x_label, y_label, c_bar_label,
        input_heatmap_options=heatmap_options, invert_y_axis=True
    )


def plot_pcolormesh(
        data: pandas.DataFrame,
        parent_fig,
        title: str = "P Color Mesh",
        x_label: str = "X Label",
        y_label: str = "Y Label",
        c_bar_label: str = "Color Bar Label",
        data_dimensions: Tuple[int, int] = None,
        input_pcolormesh_options=None
):
    """Plot a pcolormesh with the given data."""
    # Set the default options.
    pcolormesh_options = {
        "cmap": sns.color_palette("rocket_r", as_cmap=True)
    }
    if input_pcolormesh_options is not None:
        pcolormesh_options |= input_pcolormesh_options

    # Create two subplots--one to hold the pcolormesh and the other to hold the color map.
    (pcolormesh_ax, color_map_ax) = parent_fig.subplots(
        nrows=1, ncols=2, gridspec_kw={"width_ratios": [50, 1]}
    )

    # Plot the data frame as a color mesh plot.
    color_mesh = pcolormesh_ax.pcolormesh(data, **pcolormesh_options)
    color_bar = plt.colorbar(color_mesh, cax=color_map_ax)

    # Decorate the figure with the appropriate titles and axis labels.
    pcolormesh_ax.set_title(title, pad=10)
    pcolormesh_ax.set_xlabel(x_label)
    pcolormesh_ax.set_ylabel(y_label, labelpad=10)
    color_map_ax.set_ylabel(c_bar_label, labelpad=10)

    # Overwrite the dimensions with the provided values.
    if data_dimensions is not None:
        pcolormesh_ax.set_xlim(xmax=data_dimensions[1])
        pcolormesh_ax.set_ylim(ymax=data_dimensions[0])

    # Invert the y-axis such that the first element appears on top.
    pcolormesh_ax.invert_yaxis()

    # Remove all of the axis spines.
    pcolormesh_ax.spines["top"].set_visible(False)
    pcolormesh_ax.spines["right"].set_visible(False)
    pcolormesh_ax.spines["bottom"].set_visible(False)
    pcolormesh_ax.spines["left"].set_visible(False)
    color_bar.outline.set_visible(False)

