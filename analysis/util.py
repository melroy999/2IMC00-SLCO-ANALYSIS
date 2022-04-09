from typing import List, Callable, Dict

import numpy as np
import pandas as pd


def create_desc_statistics_table(data: pd.DataFrame):
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
) -> pd.DataFrame:
    """Create a data frame that holds aggregate data for the target in question."""
    # Add a suffix if requested.
    if add_index_suffix:
        results = [target(v).add_suffix(f"_{i}") for i, v in enumerate(results)]

    # Use the selector to target the tables that need to be merged. Replace missing values with zeros.
    return pd.concat(results, axis=1).fillna(.0).sort_index()


def create_normalized_table(data: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
    """Normalize the given data frame by scaling each column such that its sum is one."""
    normalized_data = pd.DataFrame(index=data.index, columns=target_columns)
    for target_column in target_columns:
        normalized_data[target_column] = data[target_column] / data[target_column].sum()
    return normalized_data


def create_correlation_table(
        data: pd.DataFrame, target_columns: List[str] = None, method="pearson"
) -> pd.DataFrame:
    """Measure the correlation coefficient for all column pairs in the given data frame over the target columns."""
    if target_columns is None:
        target_columns = list(data)
    return data[target_columns].corr(method=method)


def get_difference_sum(a, b) -> float:
    """Get the difference between the two arrays."""
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def create_measurement_table(
        data: pd.DataFrame, measure: Callable, target_columns: List[str] = None, normalize: bool = True
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


def create_difference_sum_table(data: pd.DataFrame, target_columns: List[str] = None) -> pd.DataFrame:
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
