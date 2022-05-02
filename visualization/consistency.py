from typing import Dict, Tuple

import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "font.family": "serif",
#     "text.usetex": True,
#     "pgf.rcfonts": False,
# })
import numpy as np
import pandas as pd
import seaborn

from matplotlib import pyplot as plt

from analysis.util import create_correlation_table, create_normalized_table, create_difference_sum_table
from visualization.boxplot import plot_violin, plot_violin_group
from visualization.heatmap import plot_pcolormesh, plot_corr_heatmap, plot_heatmap


def plot_throughput_sum(model_data: Dict, run_id: int, dimensions: Tuple[int, int]):
    """Plot a color mesh depicting the per file global throughput data for the given model run."""
    root_fig = plt.figure(figsize=(10, 3.5), dpi=300)
    plot_data = model_data["log_frequency"]["files"]["sum"][run_id].transpose()
    plot_pcolormesh(
        plot_data,
        root_fig,
        title=f"Logging Throughput Sum ({model_data['model']['id']}, Run \\#{run_id})",
        x_label="Timestamp (ms)",
        y_label="File Number",
        c_bar_label="Frequency",
        data_dimensions=dimensions
    )
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    # import tikzplotlib
    # tikzplotlib.save("figure.pgf", dpi=300)
    # plt.savefig("histogram.pgf")

    plt.show()


def plot_throughput_difference(model_data: Dict, run_id: int, dimensions: Tuple[int, int]):
    """Plot a color mesh depicting the sum difference to the row minimum throughput for the given model run."""
    root_fig = plt.figure(figsize=(10, 3.5), dpi=300)
    plot_data = model_data["log_frequency"]["files"]["difference"][run_id].transpose()
    plot_pcolormesh(
        plot_data,
        root_fig,
        title=f"Logging Throughput Difference ({model_data['model']['id']}, Run \\#{run_id})",
        x_label="Timestamp (ms)",
        y_label="File Number",
        c_bar_label="Sum Difference To Row Minimum",
        data_dimensions=dimensions
    )
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_throughput_report(model_data: Dict, run_id: int, dimensions: Tuple[int, int]):
    """Plot a logging throughput report for the given model run."""
    plot_throughput_sum(model_data, run_id, dimensions)
    plot_throughput_difference(model_data, run_id, dimensions)


def plot_throughput_reports(model_data: Dict):
    """Plot a logging throughput report for the given model results."""
    max_nr_of_timestamps = max(x.shape[0] for x in model_data["log_frequency"]["files"]["sum"])
    max_nr_of_files = max(x.shape[1] for x in model_data["log_frequency"]["files"]["sum"])
    data_dimensions = (max_nr_of_files, max_nr_of_timestamps)

    for i, _ in enumerate(model_data["log_frequency"]["files"]["sum"]):
        plot_throughput_report(model_data, i, data_dimensions)


def plot_thread_message_count(model_data: Dict):
    """Plot the message count of each thread in each run."""
    # Calculate the sums.
    frequency_table = model_data["log_frequency"]["global"]["table"].sum()
    thread_targets = model_data["log_frequency"]["global"]["targets"]["thread"]
    target_data = {thread: frequency_table[target_columns].values for thread, target_columns in thread_targets.items()}
    plot_data = pd.DataFrame.from_dict(target_data)

    # Determine the dimensions.
    n, m = plot_data.shape
    scale_factor = 0.7
    root_fig = plt.figure(figsize=(scale_factor * n + 4, scale_factor * m), dpi=300)

    # Set heatmap options.
    heatmap_options = {
        "annot": plot_data,
        "fmt": ".2e",
    }

    # Plot the data.
    plot_heatmap(
        plot_data,
        root_fig,
        title=f"Log Message Count ({model_data['model']['id']})",
        y_label="Test Run",
        x_label="State Machine",
        c_bar_label="Message Count",
        rotate_top_labels=False,
        input_heatmap_options=heatmap_options
    )

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_thread_message_ratio(model_data: Dict):
    """Plot the message ratio of each thread in each run."""
    # Calculate the sums.
    frequency_table = model_data["log_frequency"]["global"]["table"].sum()
    thread_targets = model_data["log_frequency"]["global"]["targets"]["thread"]
    target_data = {thread: frequency_table[target_columns].values for thread, target_columns in thread_targets.items()}
    plot_data = create_normalized_table(pd.DataFrame.from_dict(target_data))

    # Determine the dimensions.
    n, m = plot_data.shape
    scale_factor = 0.7
    root_fig = plt.figure(figsize=(scale_factor * n + 3, scale_factor * m), dpi=300)

    # Set heatmap options.
    heatmap_options = {
        "annot": plot_data,
        "fmt": ".3f",
    }

    # Plot the data.
    plot_heatmap(
        plot_data,
        root_fig,
        title=f"Log Message Ratio ({model_data['model']['id']})",
        y_label="Test Run",
        x_label="State Machine",
        c_bar_label="Message Ratio",
        rotate_top_labels=False,
        input_heatmap_options=heatmap_options
    )

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_thread_workload_balance_report(model_data: Dict):
    """Plot information on the amount of messages processed by each thread and the balance between threads in runs."""
    plot_thread_message_count(model_data)
    plot_thread_message_ratio(model_data)


def plot_message_correlation(
        plot_data: pd.DataFrame,
        title,
        x_label="Test Run",
        y_label="Test Run",
        c_bar_label="Correlation Coefficient (Spearman)",
        include_mask=True
):
    """Plot a frequency correlation plot for the given model results."""
    correlation_table = create_correlation_table(plot_data, method="spearman")
    n = len(correlation_table)

    scale_factor = 0.7
    root_fig = plt.figure(figsize=(scale_factor * n + 2, scale_factor * n + 1), dpi=300)
    plot_corr_heatmap(
        correlation_table,
        root_fig,
        title=title,
        x_label=x_label,
        y_label=y_label,
        c_bar_label=c_bar_label,
        include_mask=include_mask
    )

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_frequency_normalized_difference_sum(
        plot_data: pd.DataFrame,
        title,
        x_label="Test Run",
        y_label="Test Run",
        c_bar_label="Normalized Difference Sum (NDS)",
        include_mask=True
):
    """Plot a normalized difference sum plot for the given model results."""
    difference_sum_table = create_difference_sum_table(plot_data)
    n = len(difference_sum_table)

    # Overwrite options.
    heatmap_options = {
        "vmin": 0,
        "center": None,
        "vmax": 2,
        "cmap": seaborn.color_palette("rocket_r", as_cmap=True)
    }

    scale_factor = 0.7
    root_fig = plt.figure(figsize=(scale_factor * n + 2, scale_factor * n + 1), dpi=300)
    plot_corr_heatmap(
        difference_sum_table,
        root_fig,
        title=title,
        x_label=x_label,
        y_label=y_label,
        c_bar_label=c_bar_label,
        include_mask=include_mask,
        input_heatmap_options=heatmap_options
    )

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_frequency_interval_correlation_boxplot(
        plot_data: pd.DataFrame,
        title,
        x_label="Correlation Coefficient (Spearman)"
):
    """Plot the correlation coefficients between interval results in the given model results as a box plot."""
    correlation_table = create_correlation_table(plot_data, method="spearman")
    root_fig = plt.figure(figsize=(8, 1.5), dpi=300)

    plot_violin(
        correlation_table,
        root_fig,
        title=title,
        x_label=x_label,
    )

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_frequency_interval_normalized_difference_sum_boxplot(
        plot_data: pd.DataFrame,
        title,
        x_label="Normalized Difference Sum (NDS)"
):
    """Plot the correlation coefficients between interval results in the given model results as a box plot."""
    sum_difference_table = create_difference_sum_table(plot_data)
    root_fig = plt.figure(figsize=(8, 1.5), dpi=300)

    plot_violin(
        sum_difference_table,
        root_fig,
        title=title,
        x_label=x_label,
    )

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_message_frequency_correlation(model_data: Dict):
    """Plot a message frequency correlation plot for the given model results."""
    plot_data = model_data["message_frequency"]["global"]["table"]
    plot_message_correlation(plot_data, f"Message Frequency Correlation ({model_data['model']['id']})")


def plot_message_frequency_normalized_difference_sum(model_data: Dict):
    """Plot a normalized difference sum plot for the given model results."""
    plot_data = model_data["message_frequency"]["global"]["table"]
    plot_frequency_normalized_difference_sum(plot_data, f"Message Frequency NDS ({model_data['model']['id']})")


def create_grouped_target_frame(plot_data: pd.DataFrame, model_data: Dict, include_individual: bool):
    """Create a data frame in which target columns are selected and all data is appropriately masked with NaN values."""
    melted_table = plot_data.melt(ignore_index=False)

    # Create an ordering for the columns such that each result is only counted once.
    ordering = {column_name: i for i, column_name in enumerate(plot_data)}

    # Target the entire scope of results.
    def filter_targets(row):
        if ordering[row.name] < ordering[row["variable"]]:
            return row["value"]
        else:
            return np.NaN
    melted_table[f"Global"] = melted_table.apply(filter_targets, axis=1)

    # Target all results in which comparisons are made between two intervals of the same run.
    def filter_targets(row):
        if row.name.split("_")[1] == row["variable"].split("_")[1] and ordering[row.name] < ordering[row["variable"]]:
            return row["value"]
        else:
            return np.NaN
    melted_table[f"Internal"] = melted_table.apply(filter_targets, axis=1)

    # Target all results in which comparisons are made between two intervals of different runs.
    def filter_targets(row):
        if row.name.split("_")[1] != row["variable"].split("_")[1] and ordering[row.name] < ordering[row["variable"]]:
            return row["value"]
        else:
            return np.NaN
    melted_table[f"External"] = melted_table.apply(filter_targets, axis=1)

    # Make targets for each individual run.
    if include_individual:
        for i, ordering in model_data["message_order"]["intervals"]["targets"]["runs"].items():
            # Copy the column and overwrite all values that are not internal run comparisons to NaN.
            def filter_targets(row):
                if row.name.split("_")[1] == str(i) and row["variable"].split("_")[1] == str(i) \
                        and ordering[row.name] < ordering[row["variable"]]:
                    return row["value"]
                else:
                    return np.NaN
            melted_table[f"Run \\#{i}"] = melted_table.apply(filter_targets, axis=1)

    # Remove unused columns.
    melted_table.drop(columns=["variable", "value"], inplace=True)
    return melted_table


def plot_interval_group_boxplot(
        target_data: pd.DataFrame,
        model_data: Dict,
        title,
        x_label="Correlation Coefficient (Spearman)",
        y_label="Scope",
        include_individual: bool = False
):
    """Plot the correlation coefficients between interval results as a violin plot with informative groups."""
    target_data = create_grouped_target_frame(target_data, model_data, include_individual)
    n = len(target_data.columns)

    # Plot the data.
    root_fig = plt.figure(figsize=(8, 1 + n), dpi=300)

    plot_violin_group(
        target_data,
        root_fig,
        title=title,
        x_label=x_label,
        y_label=y_label
    )

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_message_frequency_interval_correlation_boxplot(model_data: Dict):
    """Plot the correlation coefficients between interval results in the given model results as a box plot."""
    # Render a plot for each target group (Global, internal and external comparisons).
    plot_data = model_data["message_frequency"]["intervals"]["table"]
    correlation_table = create_correlation_table(plot_data, method="spearman")
    plot_interval_group_boxplot(
        correlation_table,
        model_data,
        f"Message Frequency Correlation Between Target Intervals ({model_data['model']['id']})"
    )


def plot_message_frequency_interval_normalized_difference_sum_boxplot(model_data: Dict):
    """Plot the correlation coefficients between interval results in the given model results as a box plot."""
    # Render a plot for each target group (Global, internal and external comparisons).
    plot_data = model_data["message_frequency"]["intervals"]["table"]
    difference_sum_table = create_difference_sum_table(plot_data)
    plot_interval_group_boxplot(
        difference_sum_table,
        model_data,
        f"Message Frequency NDS Between Target Intervals ({model_data['model']['id']})",
        x_label="Normalized Difference Sum (NDS)"
    )


def plot_message_frequency_similarity_report(model_data: Dict):
    """Plot a message frequency similarity report for the given model results."""
    # Plot box plots for global-scoped correlation coefficients and NDSs.
    plot_message_frequency_correlation(model_data)
    plot_message_frequency_normalized_difference_sum(model_data)

    # Plot box plots for interval-scoped correlation coefficients and NDSs.
    plot_message_frequency_interval_correlation_boxplot(model_data)
    plot_message_frequency_interval_normalized_difference_sum_boxplot(model_data)


def plot_message_order_correlation(model_data: Dict):
    """Plot a message order correlation plot for the given model results."""
    plot_data = model_data["message_order"]["global"]["frequency_table"]
    plot_message_correlation(plot_data, f"Message Order Correlation ({model_data['model']['id']})")


def plot_message_order_normalized_difference_sum(model_data: Dict):
    """Plot a normalized difference sum plot for the given model results."""
    plot_data = model_data["message_order"]["global"]["frequency_table"]
    plot_frequency_normalized_difference_sum(plot_data, f"Message Order NDS ({model_data['model']['id']})")


def plot_message_order_interval_correlation_boxplot(model_data: Dict, include_individual: bool = False):
    """Plot the correlation coefficients between interval results in the given model results as a box plot."""
    # Render a plot for each target group (Global, internal and external comparisons).
    plot_data = model_data["message_order"]["intervals"]["frequency_table"]
    correlation_table = create_correlation_table(plot_data, method="spearman")
    plot_interval_group_boxplot(
        correlation_table,
        model_data,
        f"Message Order Correlation Between Target Intervals ({model_data['model']['id']})"
    )


def plot_message_order_interval_normalized_difference_sum_boxplot(model_data: Dict, include_individual: bool = False):
    """Plot the correlation coefficients between interval results in the given model results as a box plot."""
    # Render a plot for each target group (Global, internal and external comparisons).
    plot_data = model_data["message_order"]["intervals"]["frequency_table"]
    difference_sum_table = create_difference_sum_table(plot_data)
    plot_interval_group_boxplot(
        difference_sum_table,
        model_data,
        f"Message Order NDS Between Target Intervals ({model_data['model']['id']})",
        x_label="Normalized Difference Sum (NDS)"
    )


def plot_message_order_similarity_report(model_data: Dict):
    """Plot a message order similarity report for the given model results."""
    # Plot box plots for global-scoped correlation coefficients and NDSs.
    plot_message_order_correlation(model_data)
    plot_message_order_normalized_difference_sum(model_data)

    # Plot box plots for interval-scoped correlation coefficients and NDSs.
    plot_message_order_interval_correlation_boxplot(model_data)
    plot_message_order_interval_normalized_difference_sum_boxplot(model_data)
