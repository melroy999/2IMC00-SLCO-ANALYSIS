from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from analysis.util import create_aggregate_table
from visualization.table import render_heatmap, render_frequency_heatmap, render_model_data_table


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
                ["Table"],
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
            thread_to_sum_difference_rate = {
                k: sum_difference_table[
                    f"{k.lower().replace('-', '_')}_frequency"
                ].mean() for k, v in model_results["logging"]["entries"][run_id]["log_data"]["threads"].items()
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
                    "", "#Messages", "Rate (#/ms)", "Activity", "Mean Diff (#/ms)", "Max Diff (#/ms)"
                ]
            )
            render_model_data_table(
                model_information_table,
                axd["Table"],
                title=f"{model_id} (Run = {run_id}, #Files = {len(run['files']['entries'])})"
            )

            plt.tight_layout()
            plt.show()


def get_similarity_report(model_results: Dict, similarity_measurements: Dict):
    # Gather the data (Spearman + Normalized Sum Difference) that needs to be plotted.
    aggregate_message_frequency = similarity_measurements["aggregate"]["message_frequency"]
    aggregate_message_frequency_spearman_corr = aggregate_message_frequency["corr"]["spearman"]
    aggregate_message_frequency_sum_diff = aggregate_message_frequency["diff"]["sum"]

    agg_succession_freq = similarity_measurements["logging"]["aggregate_data"]["message_data"]["succession_frequency"]
    aggregate_succession_frequency_spearman_corr = agg_succession_freq["corr"]["spearman"]
    aggregate_succession_frequency_sum_diff = agg_succession_freq["diff"]["sum"]
    # aggregate_transition_succession_frequency = aggregate_message_data["transition_succession_frequency"]

    plot_data = [
        {
            "data": aggregate_message_frequency_spearman_corr,
            "title": "Similarity Between Logging and Counting-Based Frequencies\n (Spearman Correlation Coefficient)",
            "vmin": -1.0,
            "vmax": 1.0,
            "center": 0.0,
            "cmap": None,
            "size": (6.4, 6.4)
        },
        {
            "data": aggregate_message_frequency_sum_diff,
            "title": "Similarity Between Logging and Counting-Based Frequencies\n (Absolute Sum Difference)",
            "vmin": 0.0,
            "vmax": 2.0,
            "center": None,
            "cmap": sns.color_palette("rocket_r", as_cmap=True),
            "size": (6.4, 6.4)
        },
        {
            "data": aggregate_succession_frequency_spearman_corr,
            "title": "Similarity Between Succession Frequencies\n (Spearman Correlation Coefficient)",
            "vmin": -1.0,
            "vmax": 1.0,
            "center": 0.0,
            "cmap": None,
            "size": (4, 4)
        },
        {
            "data": aggregate_succession_frequency_sum_diff,
            "title": "Similarity Between Succession Frequencies\n (Absolute Sum Difference)",
            "vmin": 0.0,
            "vmax": 2.0,
            "center": None,
            "cmap": sns.color_palette("rocket_r", as_cmap=True),
            "size": (4, 4)
        },
    ]
    for target_data in plot_data:
        # Set render settings.
        plt.rcParams["figure.figsize"] = target_data["size"]
        plt.rcParams["figure.dpi"] = 300

        # Display borders and ticks.
        with sns.axes_style("ticks"):
            # Render the data as a heatmap.
            render_heatmap(
                target_data["data"],
                None,
                title=target_data["title"],
                v_min=target_data["vmin"],
                v_max=target_data["vmax"],
                center=target_data["center"],
                cmap=target_data["cmap"]
            )

            plt.tight_layout()
            plt.show()

    return
