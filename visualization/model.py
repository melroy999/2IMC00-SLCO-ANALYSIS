from typing import Dict, Tuple

import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     "font.family": "serif",
#     "text.usetex": True,
#     "pgf.rcfonts": False,
# })
import pandas as pd
import seaborn

from matplotlib import pyplot as plt

from analysis.util import create_correlation_table, create_normalized_table, create_difference_sum_table
from visualization.heatmap import plot_pcolormesh, plot_corr_heatmap, plot_heatmap


def plot_message_frequency_correlation(model_data: Dict):
    """Plot a message frequency correlation plot for the given model results."""
    plot_data = model_data["message_frequency"]["global"]["table"]
    correlation_table = create_correlation_table(plot_data, method="spearman")
    n = len(correlation_table)

    scale_factor = 0.7
    root_fig = plt.figure(figsize=(scale_factor * n + 2, scale_factor * n + 1), dpi=300)
    plot_corr_heatmap(
        correlation_table,
        root_fig,
        title=f"Message Frequency Correlation ({model_data['model']['id']})",
        x_label="Test Run",
        y_label="Test Run",
        c_bar_label="Correlation Coefficient (Spearman)",
        include_mask=True
    )

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_message_frequency_normalized_difference_sum(model_data: Dict):
    """Plot a normalized difference sum plot for the given model results."""
    plot_data = model_data["message_frequency"]["global"]["table"]
    difference_sum_table = create_difference_sum_table(plot_data)
    n = len(difference_sum_table)

    # Overwrite options.
    heatmap_options = {
        "vmin": 0,
        "center": None,
        "vmax": 2,
        "cmap": seaborn.cm.rocket_r
    }

    scale_factor = 0.7
    root_fig = plt.figure(figsize=(scale_factor * n + 2, scale_factor * n + 1), dpi=300)
    plot_corr_heatmap(
        difference_sum_table,
        root_fig,
        title=f"Message Frequency NDS ({model_data['model']['id']})",
        x_label="Test Run",
        y_label="Test Run",
        c_bar_label="Normalized Difference Sum (NDS)",
        include_mask=True,
        input_heatmap_options=heatmap_options
    )

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_message_frequency_similarity_report(model_data: Dict):
    """Plot a message frequency similarity report for the given model results."""
    plot_message_frequency_correlation(model_data)
    plot_message_frequency_normalized_difference_sum(model_data)


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
    root_fig = plt.figure(figsize=(scale_factor * n + 3, scale_factor * m), dpi=300)

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