from typing import Tuple

import pandas
from matplotlib import pyplot as plt
import seaborn as sns


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
        "cmap": sns.color_palette("rocket_r", as_cmap=True),
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
