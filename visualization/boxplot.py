import numpy as np
import pandas
import seaborn as sns


import seaborn.categorical


# Overwrite the violin plotter.
# class _BlackViolinPlotter(seaborn.categorical._ViolinPlotter):
#     def __init__(self, *args, **kwargs):
#         super(_BlackViolinPlotter, self).__init__(*args, **kwargs)
#         self.gray = "black"
#
#
# seaborn.categorical._ViolinPlotter = _BlackViolinPlotter
from matplotlib import pyplot as plt


def adjacent_values(values, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, values[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, values[0], q1)
    return lower_adjacent_value, upper_adjacent_value


def plot_violin(
        data: pandas.DataFrame,
        parent_fig,
        title: str = "Boxplot",
        x_label: str = "X Label",
        input_boxplot_options=None,
):
    """Plot a boxplot with the given data."""
    # Create a single axis.
    ax = parent_fig.subplots()

    # Plot the data frame as a color mesh plot.
    # sns.violinplot(x="value", data=data.melt(), **boxplot_options)
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
        title: str = "Boxplot",
        x_label: str = "X Label",
        y_label: str = "Y Label"
):
    """Plot a boxplot with the given data and categories."""
    # Remove all NaN tags.
    dataset = [
        data[column_name][data[column_name].notnull()] for column_name in data
    ]

    # Create a single axis.
    ax = parent_fig.subplots()

    # Plot the data frame as a color mesh plot.
    ax.violinplot(dataset=dataset, widths=0.8, vert=False, showmedians=True)

    # Decorate the figure with the appropriate titles and axis labels.
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_yticks([i + 1 for i, _ in enumerate(dataset)])
    ax.set_yticklabels([d.name for d in dataset])

    # Invert the y-axis such that the first element appears on top.
    ax.invert_yaxis()

    # Remove all of the axis spines except for the bottom one.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
