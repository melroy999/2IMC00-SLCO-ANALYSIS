from typing import Dict

from matplotlib import pyplot as plt
import seaborn as sns

from analysis.util import create_aggregate_table
from visualization.table import render_correlation_heatmap, render_frequency_heatmap


def get_throughput_report(model_results: Dict):
    """Report on the logging throughput of each run."""
    # Set render settings.
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams["figure.dpi"] = 300

    # Get information about the model.
    model_name = model_results["model"]["data"]["name"]
    model_id = model_results["model"]["id"]

    # Make a report for each run.
    for run in model_results["logging"]["aggregate_data"]["log_data"]["runs"]:
        # Display borders and ticks.
        with sns.axes_style("ticks"):
            # Create the figure layout.
            fig, (global_log_frequency_ax, thread_log_frequency_diff_ax) = plt.subplots(2, 1)

            # Render the global file log frequencies.
            render_frequency_heatmap(
                run["aggregate_data"]["file_log_frequencies"],
                global_log_frequency_ax,
                title=f"\"{model_name}\" Logging Throughput Frequency"
            )

            # Join all of the threads tables.
            thread_names = sorted(run["aggregate_data"]["threads"])
            threads_log_frequencies = create_aggregate_table(
                [run["aggregate_data"]["threads"][v] for v in thread_names]
            )

            # Calculate the row means for each file's runs and the associated absolute difference.
            run_range = [i for i, _ in enumerate(thread_names)]
            column_names = list(run["aggregate_data"]["threads"][thread_names[0]])
            for column_name in column_names:
                # Find the target columns.
                target_columns = [f"{column_name}_{i}" for i in run_range]

                # Get the row-wise mean values for the current column.
                row_mean_values = threads_log_frequencies[target_columns].mean(axis=1)

                # Calculate the differences.
                frequency_differences = threads_log_frequencies[target_columns].subtract(row_mean_values, axis=0).abs()

                # Add the difference sum value to the table.
                threads_log_frequencies[f"diff_sum_{column_name}"] = frequency_differences.sum(axis=1)

            # Render the differences.
            thread_frequency_diff_sums = threads_log_frequencies[
                [f"diff_sum_{column_name}" for column_name in column_names]
            ]
            render_frequency_heatmap(
                thread_frequency_diff_sums,
                thread_log_frequency_diff_ax,
                title=f"\"{model_name}\" Thread Logging Throughput Frequency Sum Differences to Mean Throughput Value"
            )

            plt.tight_layout()
            plt.show()


def get_similarity_report(model_results: Dict, similarity_measurements: Dict):
    return

    # TODO: Create the layout to be used for the report.
    # Set the figure size this way such that the color bar is always made the right size.
    plt.rcParams["figure.figsize"] = (6.4, 16)
    plt.rcParams["figure.dpi"] = 300

    fig, (pearson_corr_ax, spearman_corr_ax, kendall_corr_ax, sum_diff_ax) = plt.subplots(4, 1)

    # Find the target tables.
    aggregate_message_frequency = similarity_measurements["aggregate"]["message_frequency"]
    aggregate_message_frequency_pearson_corr = aggregate_message_frequency["corr"]["pearson"]
    aggregate_message_frequency_spearman_corr = aggregate_message_frequency["corr"]["spearman"]
    aggregate_message_frequency_kendall_corr = aggregate_message_frequency["corr"]["kendall"]
    aggregate_message_frequency_sum_diff = aggregate_message_frequency["diff"]["sum"]

    # Report the similarity between counting-based data and logging-based data measurements.
    render_correlation_heatmap(aggregate_message_frequency_pearson_corr, pearson_corr_ax)
    render_correlation_heatmap(aggregate_message_frequency_spearman_corr, spearman_corr_ax)
    render_correlation_heatmap(aggregate_message_frequency_kendall_corr, kendall_corr_ax)
    render_correlation_heatmap(aggregate_message_frequency_kendall_corr, sum_diff_ax)

    plt.tight_layout()
    plt.show()


    pass


