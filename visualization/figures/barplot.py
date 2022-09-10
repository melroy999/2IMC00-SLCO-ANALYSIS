import numpy as np
import pandas

from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator, ScalarFormatter, PercentFormatter, LogLocator, SymmetricalLogLocator


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


# https://stackoverflow.com/a/42156450
class ScalarFormatterForceFormat(ScalarFormatter):
    def _set_format(self):
        self.format = "%1.1f"


def create_ordering_mask(v: str):
    """Create a mask that can be used during a sorting operation to not include the target state."""
    if " " in v:
        name, identity = v.split(" ", 1)
        name = name[:name.rfind(".")]
        return f"{name} {identity}"
    else:
        return v


def get_default_message_ordering(message_table):
    """Get the default message ordering."""
    return list(sorted(set(message_table), key=create_ordering_mask))


def set_numeric_margins(ax, _):
    """Set the default margins for a numeric value type."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=6))
    x_ticks = ax.get_xticks()
    ax.set_xlim([0, x_ticks[-1]])


def set_percentage_margins(ax, _):
    """Set the default margins for a percentage value type."""
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim([0, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))


def set_logarithmic_margins(ax, values):
    """Set the default margins for a logarithmic scale."""
    ax.set_xscale("symlog")
    loc_major = SymmetricalLogLocator(linthresh=1.0e1, base=10)
    loc_major.set_params(numticks=5)
    ax.xaxis.set_major_locator(loc_major)

    x_ticks = ax.get_xticks()
    max_value = values.max()
    if x_ticks[-1] < max_value:
        scale_factor = x_ticks[-1] / x_ticks[-2]
        ax.set_xlim([0, x_ticks[-1] * scale_factor])


def highlight_decision_node(ax):
    """Highlight decision node ticks in the plot."""
    for label in ax.get_yticklabels():
        # noinspection PyProtectedMember
        if len(label._text.split(".")) <= 2:
            label.set_weight("bold")


def plot_two_column_barplot(
        left_data: pandas.DataFrame,
        right_data: pandas.DataFrame,
        parent_fig,
        title_left: str = "Bar Plot",
        title_right: str = "Bar Plot",
        x_label_left: str = "X Label Left",
        x_label_right: str = "X Label Right",
        y_label: str = None,
        hue: str = None,
        get_ordering_key=create_ordering_mask,
        left_margin_function=set_numeric_margins,
        right_margin_function=set_percentage_margins,
        ticks_highlighting_function=highlight_decision_node,
        input_boxplot_options_left=None,
        input_boxplot_options_right=None,
):
    """Plot the two boxplot targets (left and right) beside each other."""
    with plt.rc_context({"axes.autolimit_mode": "round_numbers"}):
        # Use the white grid style.
        sns.set_style("whitegrid")

        # Create the display order.
        ordering = list(sorted(set(left_data["message"]), key=get_ordering_key))

        # Set the default options for each boxplot.
        boxplot_options_left = {
            "order": ordering,
            "linewidth": 0,
        }
        if input_boxplot_options_left is not None:
            boxplot_options_left |= input_boxplot_options_left

        boxplot_options_right = {
            "order": ordering,
            "linewidth": 0,
        }
        if input_boxplot_options_right is not None:
            boxplot_options_right |= input_boxplot_options_right

        # Create the two subplots.
        ax_left, ax_right = parent_fig.subplots(nrows=1, ncols=2, sharey="row", gridspec_kw={"width_ratios": [1, 1]})

        # Plot the boxplot frames.
        ax_left = sns.barplot(ax=ax_left, x="value", y="message", hue=hue, data=left_data, **boxplot_options_left)
        ax_left.legend([], [], frameon=False)
        ax_right = sns.barplot(ax=ax_right, x="value", y="message", hue=hue, data=right_data, **boxplot_options_right)
        ax_left.legend([], [], frameon=False)

        # Set margins that are appropriate to the target data of the left plot.
        if left_margin_function is not None:
            left_margin_function(ax_left, left_data["value"])

        # Set margins that are appropriate to the target data of the right plot.
        if right_margin_function is not None:
            right_margin_function(ax_right, right_data["value"])

        # Make decision structure blocks bold.
        if ticks_highlighting_function is not None:
            ticks_highlighting_function(ax_left)

        # Add shaded guidelines.
        GridShader(ax_left, facecolor="lightgrey", alpha=0.7)
        GridShader(ax_right, facecolor="lightgrey", alpha=0.7)

        # Decorate the figures with the appropriate titles and axis labels.
        ax_left.set_title(title_left)
        ax_left.set_xlabel(x_label_left)
        ax_left.set_ylabel(y_label, labelpad=10)
        ax_right.set_title(title_right)
        ax_right.set_xlabel(x_label_right)
        ax_right.set_ylabel("")
