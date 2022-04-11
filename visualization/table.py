import matplotlib
import numpy as np
import pandas
import seaborn as sns
from matplotlib import pyplot as plt


def render_frequency_heatmap(
        data: pandas.DataFrame,
        ax,
        title="",
        xlabel="Time (ms)",
        ylabel="File number",
        cmap=sns.color_palette("rocket_r", as_cmap=True)
):
    """Render a heatmap for the given frequency table."""
    # Preprocess the data to be included in the plot.
    modified_df = data.rename(columns=lambda name: int(name.split('_')[-1])).transpose()

    # Display borders and ticks.
    with sns.axes_style("ticks"):
        # Plot the data frame as a color mesh plot.
        c = ax.pcolormesh(modified_df, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.invert_yaxis()
        color_bar = plt.colorbar(c, ax=ax, pad=0.015)
        color_bar.ax.set_ylabel("Frequency", rotation=270, labelpad=18)


def render_model_data_table(data: pandas.DataFrame, ax, title=""):
    """Render the given data frame as a table."""
    ax.table(
        cellText=data.values, colLabels=data.columns, loc="upper center", edges="open", cellLoc="left", colLoc="left",
        bbox=[0.0, 0.45, 1, 0.5],
        colWidths=[2.5 / len(data.columns)] * len(data.columns)
    )
    ax.axis("off")
    ax.set_title(title)


def render_correlation_heatmap(data: pandas.DataFrame, ax):
    """Render a heatmap for the given correlation table."""
    mask = np.zeros_like(data)
    mask[np.triu_indices_from(mask)] = True

    ax = sns.heatmap(
        data, annot=True, square=True, fmt=".2f", center=0.0, vmin=-1.0, vmax=1.0, mask=mask, ax=ax
    )
    ax.set_title("Title")

    # Set the labels.
    ax.set_xlabel("Target", labelpad=10)
    ax.set_ylabel("Subject", labelpad=10)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Display axis labels in monospace.
    ax.set_xticks(ax.get_xticks(), fontname="monospace")
    ax.set_yticks(ax.get_yticks(), fontname="monospace")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")





def render_heatmap(
        data: pandas.DataFrame,
        vmin: float = None,
        vmax: float = None,
        center: float = None,
        square: bool = True,
        annot: bool = True,
        title: str = ""
):
    """Render the given data frame as a heatmap."""

    plt.figure(dpi=300)

    mask = np.zeros_like(data)
    mask[np.triu_indices_from(mask)] = True

    ax = sns.heatmap(
        data, annot=annot, square=square, fmt=".2f", center=center, vmin=vmin, vmax=vmax, mask=mask
    )
    ax.set_title(title)

    # Set the labels.
    ax.set_xlabel("Target", labelpad=10)
    ax.set_ylabel("Subject", labelpad=10)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # Display axis labels in monospace.
    ax.set_xticks(ax.get_xticks(), fontname="monospace")
    ax.set_yticks(ax.get_yticks(), fontname="monospace")

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")

    plt.tight_layout()
    # plt.savefig('example_1.pdf', format='pdf', bbox_inches='tight')
    plt.show()

