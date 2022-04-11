from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from analysis.util import create_aggregate_table
from visualization.table import render_correlation_heatmap, render_frequency_heatmap, render_model_data_table


def get_throughput_report(model_results: Dict):
    """Report on the logging throughput of each run."""
    # Set render settings.
    plt.rcParams["figure.figsize"] = (8, 9)
    plt.rcParams["figure.dpi"] = 300

    # Use the seaborn theme.
    sns.set_theme()

    # Get information about the model.
    model_name = model_results["model"]["data"]["name"]
    model_id = model_results["model"]["id"]

    # Make a report for each run.
    for run_id, run in enumerate(model_results["logging"]["aggregate_data"]["log_data"]["runs"]):
        # Display borders and ticks.
        with sns.axes_style("ticks"):
            # Create the figure layout.
            outer_nested_mosaic = [
                ["Global"],
                ["Thread"],
                ["Table"]
            ]
            fig, axd = plt.subplot_mosaic(
                outer_nested_mosaic, empty_sentinel="."
            )

            # Render the global file log frequencies.
            render_frequency_heatmap(
                run["aggregate_data"]["file_log_frequencies"],
                axd["Global"],
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
                row_mean_values = threads_log_frequencies[target_columns].min(axis=1)

                # Calculate the differences.
                frequency_differences = threads_log_frequencies[target_columns].subtract(row_mean_values, axis=0)

                # Add the difference sum value to the table.
                threads_log_frequencies[f"diff_sum_{column_name}"] = frequency_differences.sum(axis=1)

            # Render the differences.
            thread_frequency_diff_sums = threads_log_frequencies[
                [f"diff_sum_{column_name}" for column_name in column_names]
            ]
            render_frequency_heatmap(
                thread_frequency_diff_sums,
                axd["Thread"],
                title=f"\"{model_name}\" Logging Throughput Frequency (Difference Sum to Row Min)"
            )

            # Calculate the differences model-wide such that they can be included in the data table.
            sum_difference_table = run["log_frequency"].subtract(run["log_frequency"].min(axis=1), axis=0)
            n = len(run["log_frequency"].index)
            thread_to_sum_difference_rate = {
                k: sum_difference_table[
                    f"{k.lower().replace('-', '_')}_frequency"
                ].sum() / n for k, v in model_results["logging"]["entries"][run_id]["log_data"]["threads"].items()
            }
            thread_to_max_difference_rate = {
                k: sum_difference_table[
                    f"{k.lower().replace('-', '_')}_frequency"
                ].max() for k, v in model_results["logging"]["entries"][run_id]["log_data"]["threads"].items()
            }

            # Add a table showing average rates for each thread and percentage of activity.
            model_information_data = {
                k: [
                    k,
                    v["global"]["lines"],
                    f"{v['global']['rate']:.2f}",
                    f"{v['global']['activity']:.2%}",
                    f"{sum(thread_to_sum_difference_rate.values()):.2f}" if k == "Global"
                    else f"{thread_to_sum_difference_rate[k]:.2f}",
                    f"{max(thread_to_max_difference_rate.values()):.0f}" if k == "Global"
                    else f"{thread_to_max_difference_rate[k]:.0f}"
                ] for k, v in list(model_results["logging"]["entries"][run_id]["log_data"]["threads"].items()) + [
                    ("Global", model_results["logging"]["entries"][run_id]["log_data"]["global"])
                ]
            }

            # Render a title that adds identification information.
            model_information_table = pd.DataFrame.from_dict(
                model_information_data, orient="index", columns=[
                    "", "#Messages", "Rate (#/ms)", "Activity", "Diff Rate (#/ms)", "Max Diff"
                ]
            )
            render_model_data_table(
                model_information_table,
                axd["Table"],
                title=f"{model_id} (Run = {run_id}, #Files = {len(run['files']['entries'])})"
            )

            plt.tight_layout()
            plt.show()
            plt.close("all")


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
