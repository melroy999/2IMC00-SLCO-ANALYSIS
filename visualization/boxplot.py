import numpy as np
import pandas
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_violin(
        data: pandas.DataFrame,
        parent_fig,
        title: str = "Violin Plot",
        x_label: str = "X Label",
        input_boxplot_options=None,
):
    """Plot a violin plot with the given data."""
    # Create a single axis.
    ax = parent_fig.subplots()

    # Plot the data frame as a color mesh plot.
    data = data.melt()["value"]
    ax.violinplot(data, vert=False, quantiles=[0.25, 0.75], showmedians=True)

    # Decorate the figure with the appropriate titles and axis labels.
    ax.set_title(title)
    ax.set_xlabel(x_label)

    # Remove all of the axis spines except for the bottom one.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.get_yaxis().set_visible(False)


def plot_violin_group(
        data: pandas.DataFrame,
        parent_fig,
        title: str = "Violin Plot",
        x_label: str = "X Label",
        y_label: str = "Y Label"
):
    """Plot a violin plot with the given data and categories."""
    # Use the white grid style.
    sns.set_style("whitegrid")

    # Create a single axis.
    ax = parent_fig.subplots()

    # Plot the data frame as a color mesh plot.
    ax = sns.violinplot(ax=ax, data=data, inner="quartile", orient="h", scale="width", cut=0, color=".8")
    # ax = sns.swarmplot(ax=ax, data=data, edgecolor="gray", orient="h", size=3)
    ax = sns.stripplot(ax=ax, data=data, edgecolor="gray", orient="h", size=3, jitter=0.25)

    # Decorate the figure with the appropriate titles and axis labels.
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, labelpad=10)

    # Invert the y-axis such that the first element appears on top.
    ax.invert_yaxis()

    # Remove all of the axis spines except for the bottom one.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


# https://stackoverflow.com/a/54654448
# Edited to adjust y-axis instead.
class GridShader:
    def __init__(self, ax, first=True, **kwargs):
        self.spans = []
        self.sf = first
        self.ax = ax
        self.kw = kwargs
        self.ax.autoscale(False, axis="y")
        self.cid = self.ax.callbacks.connect('ylim_changed', self.shade)
        self.shade()

    def clear(self):
        for span in self.spans:
            span.remove()

    def shade(self, evt=None):
        self.clear()
        y_ticks = self.ax.get_yticks() - 0.5
        y_lim = self.ax.get_ylim()
        y_ticks = y_ticks[(y_ticks > y_lim[1]) & (y_ticks < y_lim[0])]
        locs = np.concatenate(([[y_lim[1]], y_ticks, [y_lim[0]]]))

        start = locs[1 - int(self.sf)::2]
        end = locs[2 - int(self.sf)::2]

        for s, e in zip(start, end):
            self.spans.append(self.ax.axhspan(s, e, zorder=0, **self.kw))


def plot_comparison_boxplot(
        left_data: pandas.DataFrame,
        right_data: pandas.DataFrame,
        parent_fig,
        title_left: str = "Box Plot",
        title_right: str = "Box Plot",
        x_label_left: str = "X Label Left",
        x_label_right: str = "X Label Right",
        y_label: str = "Y Label",
        input_boxplot_options_left=None,
        input_boxplot_options_right=None,
):
    """Plot the two boxplot targets (left and right) beside each other to facilitate comparisons."""
    with plt.rc_context({"axes.autolimit_mode": "round_numbers"}):
        # Use the white grid style.
        sns.set_style("whitegrid")

        # Create the display order.
        ordering = list(sorted(set(left_data["variable"])))

        # Set the default options for each boxplot.
        boxplot_options_left = {
            "size": 3,
            "jitter": 0.25,
            "order": ordering
        }
        if input_boxplot_options_left is not None:
            boxplot_options_left |= input_boxplot_options_left
        boxplot_options_right = {
            "size": 3,
            "jitter": 0.25,
            "order": ordering
        }
        if input_boxplot_options_right is not None:
            boxplot_options_right |= input_boxplot_options_right

        # Create the two subplots.
        ax_left, ax_right = parent_fig.subplots(nrows=1, ncols=2, sharey="row", gridspec_kw={"width_ratios": [1, 1]})

        # Plot the boxplot frames.
        ax_left = sns.stripplot(ax=ax_left, x="value", y="variable", data=left_data, **boxplot_options_left)
        ax_right = sns.stripplot(ax=ax_right, x="value", y="variable", data=right_data, **boxplot_options_right)

        # Set margins that are appropriate to the target data.
        ax_left.xaxis.set_major_locator(MaxNLocator(5, min_n_ticks=6, prune="lower"))
        left_x_ticks = ax_left.get_xticks()
        if len(left_x_ticks) != 6:
            left_x_ticks = left_x_ticks[1:]
            ax_left.set_xticks(left_x_ticks)
        left_x_min = left_x_ticks[0] - 0.125 * (left_x_ticks[1] - left_x_ticks[0])
        left_x_max = left_x_ticks[-1] + 0.125 * (left_x_ticks[1] - left_x_ticks[0])
        if left_data["value"].min() < left_x_min or left_data["value"].max() > left_x_max:
            raise Exception("Bounds to not conform the the data.")
        ax_left.set_xlim([left_x_min, left_x_max])
        ax_left.ticklabel_format(style='sci', scilimits=(0, 1), axis="x")

        ax_right.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_right.set_xlim([-0.025, 1.025])

        # Make decision structure blocks bold.
        for label in ax_left.get_yticklabels():
            # noinspection PyProtectedMember
            if len(label._text.split(".")) == 2:
                label.set_weight("bold")

        # De-spine and add shaded guidelines.
        sns.despine(parent_fig, left=True, bottom=True, trim=True)
        GridShader(ax_left, facecolor="lightgrey", alpha=0.7)
        GridShader(ax_right, facecolor="lightgrey", alpha=0.7)

        # Decorate the figures with the appropriate titles and axis labels.
        ax_left.set_title(title_left)
        ax_left.set_xlabel(x_label_left)
        ax_left.set_ylabel(y_label, labelpad=10)
        ax_right.set_title(title_right)
        ax_right.set_xlabel(x_label_right)
        ax_right.set_ylabel("")
