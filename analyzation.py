from typing import Dict, List, Callable

import numpy as np
import pandas
import pandas as pd

from plotting import create_global_log_file_throughput_plot, create_succession_heat_map_plot, \
    create_thread_grouped_log_file_throughput_plot, create_concurrency_heat_map_plot


def create_desc_statistics_table(data: pandas.DataFrame):
    """Create a table that contains the descriptive statistics of the given data."""
    descriptive_data = pd.DataFrame(index=data.index, columns=["min", "mean", "median", "max", "std"])
    descriptive_data[f"min"] = data.min(axis=1)
    descriptive_data[f"mean"] = data.mean(axis=1)
    descriptive_data[f"median"] = data.median(axis=1)
    descriptive_data[f"max"] = data.max(axis=1)
    descriptive_data[f"std"] = data.std(axis=1)
    return descriptive_data


def create_aggregate_table(
        results: List[Dict], target: Callable = lambda v: v, add_index_suffix: bool = True
) -> pandas.DataFrame:
    """Create a data frame that holds aggregate data for the target in question."""
    # Add a suffix if requested.
    if add_index_suffix:
        results = [target(v).add_suffix(f"_{i}") for i, v in enumerate(results)]

    # Use the selector to target the tables that need to be merged. Replace missing values with zeros.
    return pd.concat(results, axis=1).fillna(.0).sort_index()


def create_normalized_table(data: pandas.DataFrame, target_columns: List[str]) -> pandas.DataFrame:
    """Normalize the given data frame by scaling each column such that its sum is one."""
    normalized_data = pd.DataFrame(index=data.index, columns=target_columns)
    for target_column in target_columns:
        normalized_data[target_column] = data[target_column] / data[target_column].sum()
    return normalized_data


def create_correlation_table(
        data: pandas.DataFrame, target_columns: List[str] = None, method="pearson"
) -> pandas.DataFrame:
    """Measure the correlation coefficient for all column pairs in the given data frame over the target columns."""
    if target_columns is None:
        target_columns = list(data)
    return data[target_columns].corr(method=method)


def get_difference_sum(a, b) -> float:
    """Get the difference between the two arrays."""
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def get_difference_max(a, b):
    """Get the maximum difference between the two arrays."""
    return max(abs(val1 - val2) for val1, val2 in zip(a, b))


def create_measurement_table(
        data: pandas.DataFrame, measure: Callable, target_columns: List[str] = None, normalize: bool = True
):
    """Create a table that applies the measure to all column pairs in the given table."""
    if target_columns is None:
        # Set the target columns to be all of the columns.
        target_columns = list(data)
    if normalize:
        # Ensure that all of the data is of the same scale.
        data = create_normalized_table(data, target_columns)

    # Create a pairwise table.
    return pd.DataFrame(index=target_columns, columns=target_columns, data=np.array(
        [[measure(data[i], data[j]) for j in target_columns] for i in target_columns]
    ))


def create_difference_sum_table(data: pandas.DataFrame, target_columns: List[str] = None) -> pandas.DataFrame:
    """
    Calculate the total difference for all column pairs in the given data frame over the target columns.

    Note that due to the normalization, the values within the difference table will always be between 0 and 2.
        - Suppose that the length of both arrays is n.
        - Due to normalization, it holds that:
            sum(a[i] for i in range(0, n)) = 1
            sum(b[i] for i in range(0, n)) = 1
        - The difference d between the two arrays is defined as:
            d = sum(abs(a[i] - b[i]) for i in range(0, n))
        - Given that a and b are both frequencies, all values are positive. Moreover, they are bound by one.
            all(0 <= a[i] <= 1 for i in range(0, n))
            all(0 <= b[i] <= 1 for i in range(0, n))
        - Due to the bounds on elements in a and b, it holds that the difference is always lower than the sum.
            abs(a[i] - b[i]) <= a[i] + b[i]
            d = sum(abs(a[i] - b[i]) for i in range(0, n)) <= sum(a[i] + b[i] for i in range(0, n))
        - Hence, given the normalized sum, we can conclude that:
            d <= sum(a[i] + b[i] for i in range(0, n)) = sum(a[i] for i in range(0, n)) + sum(b[i] for i in range(0, n))
            d <= 1 + 1
            d <= 2
        - The difference between two elements cannot be lower than zero, hence:
            0 <= d <= 2
        - To prove that d cannot be lower than 2, a counter-proof is given.
            a = [0, 1], b = [1, 0], d(a, b) = 2
            a = [1, 0, ...], b = [0, 1/(n - 1), ...], for any length n, d(a, b) = 2
    """
    # Call the measurement table function with the difference function.
    return create_measurement_table(data, get_difference_sum, target_columns, True)


def create_difference_max_table(data: pandas.DataFrame, target_columns: List[str] = None) -> pandas.DataFrame:
    """
    Calculate the max difference for all column pairs in the given data frame over the target columns.
    """
    # Call the measurement table function with the difference 9th percentile function.
    return create_measurement_table(data, get_difference_max, target_columns, True)


def select_logging_event_count(data):
    """Select the event count within the given dictionary structure."""
    return data["message_data"]["global_data"]["event_count"]


def select_logging_succession_table(data):
    """Select the succession table within the given dictionary structure."""
    return data["message_data"]["global_data"]["succession_table"]


def select_logging_transition_succession_table(data):
    """Select the transition succession table within the given dictionary structure."""
    return data["message_data"]["global_data"]["transition_succession_table"]


def create_logging_aggregate_data(results: List[Dict]) -> Dict:
    """Create aggregate data for the given collection of results that are taken from log-based measurements."""
    # A dictionary containing all aggregate data.
    result = dict()

    # Concatenate the event and succession count data of the different runs.
    result["event_count"] = create_aggregate_table(results, select_logging_event_count)
    result["succession_count"] = create_aggregate_table(results, select_logging_succession_table)
    result["transition_succession_count"] = create_aggregate_table(results, select_logging_transition_succession_table)

    # Create descriptive statistics tables.
    result["event_count_desc"] = create_desc_statistics_table(result["event_count"])
    result["succession_count_desc"] = create_desc_statistics_table(result["succession_count"])
    result["transition_succession_count_desc"] = create_desc_statistics_table(result["transition_succession_count"])

    # Create correlation coefficient tables to check for differences in the data's trends.
    result["event_count_corr"] = create_correlation_table(result["event_count"])
    result["succession_count_corr"] = create_correlation_table(result["succession_count"])
    result["transition_succession_count_corr"] = create_correlation_table(result["transition_succession_count"])

    # Create a difference table.
    result["event_count_corr"] = create_difference_sum_table(result["event_count"])
    result["succession_count_corr"] = create_difference_sum_table(result["succession_count"])
    result["transition_succession_count_corr"] = create_difference_sum_table(result["transition_succession_count"])

    return result


def select_counting_event_count(data):
    """Select the event count within the given dictionary structure."""
    return data["event_count"]


def create_counting_aggregate_data(results: List[Dict]) -> Dict:
    """Create aggregate data for the given collection of results that are taken from count-based measurements."""
    # A dictionary containing all aggregate data.
    result = dict()

    # Concatenate the event and succession count data of the different runs.
    result["event_count"] = create_aggregate_table(results, select_counting_event_count)

    # Create descriptive statistics tables.
    result["event_count_desc"] = create_desc_statistics_table(result["event_count"])

    # Create correlation coefficient tables to check for differences in the data's trends.
    result["event_count_corr"] = create_correlation_table(result["event_count"])

    return result


def analyze_data(data: Dict):
    """Analyze the given data."""
    create_global_log_file_throughput_plot(data)
    create_thread_grouped_log_file_throughput_plot(data)
    create_succession_heat_map_plot(data)
    create_succession_heat_map_plot(data, target="transition_succession_graph")
    create_concurrency_heat_map_plot(data)
