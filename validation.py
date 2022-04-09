from typing import Dict, List, Tuple

import pandas

from analysis.similarity import get_similarity_measurements
from analyzation import create_correlation_table, create_difference_sum_table, create_aggregate_table


def validate_source_model(data: Dict) -> None:
    """Validate that all of the given results have the same source model and abbreviations."""
    results = data["logging"]["entries"] + data["counting"]["entries"]
    if not all(v["model"] == results[0]["model"] for v in results):
        raise Exception("The source model is not the same for all results.")


def get_similarity_metrics(data: pandas.DataFrame, index_contains: str = None, method="pearson") -> Tuple:
    """Get the similarity metrics for the given data frame."""
    if index_contains is not None:
        # Filter the index of the data.
        data = data[data.index.str.contains(index_contains)]

    # TODO: what other data could be useful to get to see if outliers are causing the issue?
    #   - Correlation is sensitive to outliers. Could quantiles be used on the difference table to detect these?
    #   - Chi-squared is proposed, but the data will not always fit the requirements.

    correlation_table = create_correlation_table(data, method=method)
    difference_sum_table = create_difference_sum_table(data)
    min_correlation = correlation_table.min().min()
    difference_sum = difference_sum_table.max().max()
    return min_correlation, difference_sum


def get_local_data_integrity_table(global_data: Dict, thread_data: List[Dict]) -> Tuple:
    """Get the similarity metrics for the given combination of global data and thread data."""
    # Select all frequency tables (global + all threads).
    frequency_tables = [global_data["frequency_table"]]
    frequency_tables += [data["frequency_table"].add_prefix(f"thread_{i}_") for i, data in enumerate(thread_data)]
    aggregate_frequency_table = create_aggregate_table(frequency_tables, add_index_suffix=False)
    return get_similarity_metrics(aggregate_frequency_table)


def validate_local_data_integrity(data: Dict):
    # Target the global and thread data.
    global_log_data = data["log_data"]["global"]
    thread_log_data = data["log_data"]["threads"]

    # Create an order to iterate over the threads with.
    thread_names = list(thread_log_data.keys())

    # Keep the results in a dictionary that will be converted to a data frame.
    similarity_data = dict()

    # Analyze the global-level data first.
    similarity_data["*"] = get_local_data_integrity_table(
        global_log_data["global"], [thread_log_data[thread_name]["global"] for thread_name in thread_names]
    )

    # Compare data file-by-file.
    # TODO: temporarily disabled.
    # for i, _ in enumerate(global_log_data["files"]):
    #     similarity_data[f"file_{i}"] = get_local_data_integrity_table(
    #         global_log_data["files"][i], [thread_log_data[thread_name]["files"][i] for thread_name in thread_names]
    #     )
    #
    # # Create a table containing all the resulting data.
    # similarity_table = pandas.DataFrame.from_dict(
    #     data=similarity_data, orient="index", columns=["correlation", "difference_sum"]
    # )
    # print(similarity_table)

    return similarity_data["*"]





def get_message_frequency_similarity_between_runs(data: pandas.DataFrame) -> pandas.DataFrame:
    """Validate that the message frequency follows a similar trend within each run for the given aggregate table."""
    # Calculate the correlation coefficient and normalized sum difference tables.
    similarity_measurements = get_similarity_measurements(data)

    # Create a summary table which indicates whether a threshold has been violated or not.
    similarity_summary = similarity_measurements["summary"]
    result = pandas.DataFrame.from_dict(
        data={
            "pearson": [
                similarity_summary["corr"]["pearson"]["min"],
                "> 0.8",
                similarity_summary["corr"]["pearson"]["min"] < 0.8
            ],
            "spearman": [
                similarity_summary["corr"]["spearman"]["min"],
                "> 0.8",
                similarity_summary["corr"]["spearman"]["min"] < 0.8
            ],
            "kendall": [
                similarity_summary["corr"]["kendall"]["min"],
                "> 0.8",
                similarity_summary["corr"]["kendall"]["min"] < 0.8
            ],
            "diff_sum": [
                similarity_summary["diff"]["sum"]["max"],
                "< 0.2",
                similarity_summary["diff"]["sum"]["max"] > 0.2
            ]
        }, orient="index", columns=["value", "preferred", "violation"]
    )

    if any(result["violation"]):
        # TODO: handle a violation more appropriately--a heatmap of the gathered data might be useful.
        print("One or more of the metrics violate the threshold.")
        print(result)

    pass


def validate_data_integrity(data: Dict) -> None:
    """
    Find signs that may indicate that the data may be affected by an outside influence and hence less representable.
    """
    logging_agg_frequency = data["logging"]["aggregate_data"]["event_count"].add_prefix(f"logging_")
    counting_agg_frequency = data["counting"]["aggregate_data"]["event_count"].add_prefix(f"counting_")
    full_agg_frequency = create_aggregate_table([logging_agg_frequency, counting_agg_frequency], add_index_suffix=False)
    s_agg_frequency = full_agg_frequency[full_agg_frequency.index.str.contains(".S")]

    # validate_message_frequency_similarity_between_runs(logging_agg_frequency)
    # validate_message_frequency_similarity_between_runs(counting_agg_frequency)
    get_message_frequency_similarity_between_runs(full_agg_frequency)
    get_message_frequency_similarity_between_runs(s_agg_frequency)

    # Get a list of all logging based entries.
    logging_results = data["logging"]["entries"]

    # Validate targets in individual results and combine them into a data frame.
    similarity_data = {f"run_{i}": validate_local_data_integrity(result) for i, result in enumerate(logging_results)}
    similarity_table = pandas.DataFrame.from_dict(
        data=similarity_data, orient="index", columns=["correlation", "difference_sum"]
    )

    print(f"\nSimilarity metric extremes for logging-based measurements.")
    print(similarity_table)


    print()
    # TODO: What do we want to validate here?
    #   - Validate that time gaps within the data do not influence the behavior and results.
    #   - Repeat the same for thread-grouped data.

    # for result in logging_results:
    #     # Target the global log data.
    #     global_log_data = result["log_data"]["global"]["global"]
    #
    #     # Measure the difference metric.
    #     count_data = np.array(global_log_data["count"])
    #     count_table = pd.DataFrame(count_data, columns=["count"])
    #     count_table["expected"] = global_log_data["rate"]
    #     difference_table = create_difference_table(count_table)
    #
    #     pass

    pass


def validate_logging_representability(data: Dict) -> None:
    """
    Find signs that the data collected through locking is dissimilar to that found through counting.
    """
    # Create an aggregate table containing both the logging and counting aggregate data.
    aggregate_count_comparison_table = create_aggregate_table([
        data["logging"]["aggregate_data"]["event_count"].add_prefix(f"logging_"),
        data["counting"]["aggregate_data"]["event_count"].add_prefix(f"counting_")
    ], add_index_suffix=False)

    # Create correlation and difference tables and search for issues by looking at the minimum values.
    results = dict()
    results["*"] = get_similarity_metrics(aggregate_count_comparison_table)

    # Find the same metrics but grouped by opening, success and failure events.
    results["*.O"] = get_similarity_metrics(aggregate_count_comparison_table, index_contains=".O")
    results["*.S"] = get_similarity_metrics(aggregate_count_comparison_table, index_contains=".S")
    results["*.F"] = get_similarity_metrics(aggregate_count_comparison_table, index_contains=".F")

    # Return the results as a data table.
    result = pandas.DataFrame.from_dict(
        data=results, orient="index", columns=["correlation", "difference_sum"]
    )
    # TODO: test per interval.

    print("\nSimilarity between global logging and counting based message count measurements.")
    print(result)

    # TODO: What do we want to validate here?
    #   - Compare the correlation and difference metrics between logging and counting results.
    #   - Repeat the same comparison above but filtered by event type--i.e. opening, failure and success.
