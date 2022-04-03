from typing import Dict, List, Callable

import numpy as np
import pandas
import pandas as pd

from plotting import create_global_log_file_throughput_plot, create_succession_heat_map_plot, \
    create_thread_grouped_log_file_throughput_plot, create_concurrency_heat_map_plot


def create_aggregate_table(results: List[Dict], target: Callable) -> pandas.DataFrame:
    """Create a data frame that holds aggregate data for the target in question."""
    # Find a list of columns to use.
    column_names = list(target(results[0]))

    # Use the selector to target the tables that need to be merged. Replace missing values with zeros.
    aggregate_data = pd.concat(
        [target(v).add_suffix(f"_{i}") for i, v in enumerate(results)], axis=1
    ).fillna(.0).sort_index()

    # Add basic statistic columns for each original column.
    for column_name in column_names:
        # Gather the list of columns to target.
        target_columns = [f"{column_name}_{i}" for i, v in enumerate(results)]

        # Add columns for the statistics.
        aggregate_data[f"{column_name}_min"] = aggregate_data[target_columns].min(axis=1)
        aggregate_data[f"{column_name}_mean"] = aggregate_data[target_columns].mean(axis=1)
        aggregate_data[f"{column_name}_median"] = aggregate_data[target_columns].median(axis=1)
        aggregate_data[f"{column_name}_max"] = aggregate_data[target_columns].max(axis=1)
        aggregate_data[f"{column_name}_std"] = aggregate_data[target_columns].std(axis=1)

    return aggregate_data


def create_correlation_table(data: pandas.DataFrame, target_columns: List[str], method="pearson") -> pandas.DataFrame:
    """Measure the correlation coefficient for all column pairs in the given data frame over the target columns."""
    return data[target_columns].corr(method=method)


def create_manhattan_table(data: pandas.DataFrame, target_columns: List[str]) -> pandas.DataFrame:
    """Calculate the Manhattan distance for all column pairs in the given data frame over the target columns."""
    # Ensure that all of the data is of the same scale.
    normalized_data = pd.DataFrame(index=data.index, columns=target_columns)
    for target_column in target_columns:
        normalized_data[target_column] = data[target_column] / data[target_column].sum()

    def manhattan(a, b):
        return sum(abs(val1 - val2) for val1, val2 in zip(a, b))

    test_data = np.array(
        [[manhattan(normalized_data[i], normalized_data[j]) for j in target_columns] for i in target_columns]
    )

    return pd.DataFrame(data=test_data, index=target_columns, columns=target_columns)


def create_logging_aggregate_data(results: List[Dict]) -> Dict:
    """Create aggregate data for the given collection of results that are taken from log-based measurements."""
    # A dictionary containing all aggregate data.
    result = dict()

    # Concatenate the event and succession count data of the different runs.
    def event_count_target(data):
        return data["message_data"]["global_data"]["event_count"]
    event_count = result["event_count"] = create_aggregate_table(
        results, event_count_target
    )

    def succession_count_target(data):
        return data["message_data"]["global_data"]["succession_table"]
    succession_count = result["succession_count"] = create_aggregate_table(
        results, succession_count_target
    )

    def transition_succession_count_target(data):
        return data["message_data"]["global_data"]["transition_succession_table"]
    transition_succession_count = result["transition_succession_count"] = create_aggregate_table(
        results, transition_succession_count_target
    )

    # Create correlation coefficient tables to check for differences in the data's trends.
    a = result["event_count_corr"] = create_correlation_table(
        event_count, list(event_count)[:len(results)]
    )
    b = result["succession_count_corr"] = create_correlation_table(
        succession_count, list(succession_count)[:len(results)]
    )
    c = result["transition_succession_count_corr"] = create_correlation_table(
        transition_succession_count, list(transition_succession_count)[:len(results)]
    )

    return result


def create_counting_aggregate_data(results: List[Dict]) -> Dict:
    """Create aggregate data for the given collection of results that are taken from count-based measurements."""
    # A dictionary containing all aggregate data.
    result = dict()

    # Concatenate the event and succession count data of the different runs.
    def event_count_target(data):
        return data["event_count"]
    event_count = result["event_count"] = create_aggregate_table(
        results, event_count_target
    )

    # Create correlation coefficient tables to check for differences in the data's trends.
    a = result["event_count_corr"] = create_correlation_table(
        event_count, list(event_count)[:len(results)]
    )

    return result


def analyze_data(data: Dict):
    """Analyze the given data."""
    create_global_log_file_throughput_plot(data)
    create_thread_grouped_log_file_throughput_plot(data)
    create_succession_heat_map_plot(data)
    create_succession_heat_map_plot(data, target="transition_succession_graph")
    create_concurrency_heat_map_plot(data)
