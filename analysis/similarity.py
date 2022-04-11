from typing import Dict

import numpy as np
import pandas
import pandas as pd

from analysis.util import create_correlation_table, create_difference_sum_table
from visualization.model import get_similarity_report, get_throughput_report


def get_similarity_measurements_summary_statistics(data: pd.DataFrame) -> Dict[str, float]:
    """Get the summary statistics of the given data frame."""
    flattened_data = data.to_numpy().flatten()
    return {
        "min": np.min(flattened_data),
        "mean": np.mean(flattened_data),
        # "median": np.median(flattened_data),
        "max": np.max(flattened_data),
        "std": np.std(flattened_data)
    }


def get_similarity_measurements_summary(data: pd.DataFrame, similarity_measurements=None):
    """Calculate the similarity metrics for the given data table and report ."""
    if similarity_measurements is None:
        # Calculate the similarity measurements if not already given.
        similarity_measurements = get_similarity_measurements(data, include_summary=False)

    # Report the mean, median and the extremes of each measurement.
    return {
        "corr": {
            # "pearson": get_similarity_measurements_summary_statistics(similarity_measurements["corr"]["pearson"]),
            "spearman": get_similarity_measurements_summary_statistics(similarity_measurements["corr"]["spearman"]),
            # "kendall": get_similarity_measurements_summary_statistics(similarity_measurements["corr"]["kendall"])
        },
        "diff": {
            "sum": get_similarity_measurements_summary_statistics(similarity_measurements["diff"]["sum"])
        }
    }


def get_similarity_measurements(data: pd.DataFrame, include_summary=True, preprocess=True) -> Dict:
    """Calculate the similarity metrics for the given data table."""
    if preprocess:
        # Remove rows that only have zeroes.
        preprocessed_data = data.loc[(data != 0).any(axis=1)]
    else:
        preprocessed_data = data

    # Calculate the correlation coefficient and normalized sum difference tables.
    similarity_measurements = {
        "corr": {
            # "pearson": create_correlation_table(preprocessed_data, method="pearson"),
            "spearman": create_correlation_table(preprocessed_data, method="spearman"),
            # "kendall": create_correlation_table(preprocessed_data, method="kendall")
        },
        "diff": {
            "sum": create_difference_sum_table(preprocessed_data)
        }
    }

    # Add a summary if requested.
    if include_summary:
        similarity_measurements["summary"] = get_similarity_measurements_summary(
            preprocessed_data, similarity_measurements
        )
    return similarity_measurements


# This method is too slow for practical use.
# def get_similarity_measurements_recursively(node):
#     """Recursively find all data frames with more than one column and find their similarity measurements."""
#     if isinstance(node, list):
#         # Apply the function to all elements in the list, but do not include elements returning nothing.
#         result = [v for v in [get_similarity_measurements_recursively(v) for v in node] if v is not None]
#         return None if len(result) == 0 else result
#     elif isinstance(node, dict):
#         # Apply the function to all values in the dictionary, but do not include elements returning nothing.
#         result = {k: v for k, v in {
#             k: get_similarity_measurements_recursively(v) for k, v in node.items()
#         }.items() if v is not None}
#         return None if len(result) == 0 else result
#     elif isinstance(node, pandas.DataFrame):
#         # Get similarity data from the data frame but only if there is more than one column.
#         return None if len(node) < 2 else get_similarity_measurements(node)

def recursively_generate_similarity_measurements(data_node, target_node):
    """Recursively find the similarity measurements for the defined targets."""
    if isinstance(data_node, pandas.DataFrame):
        # Return the similarity measurements for the given data frame.
        return get_similarity_measurements(data_node)

    if isinstance(target_node, list):
        # Apply the function to all elements in the list, but do not include elements returning nothing.
        result = [
            v for v in [
                recursively_generate_similarity_measurements(v, u) for v, u in zip(data_node, target_node)
            ] if v is not None
        ]
        return None if len(result) == 0 else result
    elif isinstance(target_node, dict):
        # Apply the function to all values in the dictionary, but do not include elements returning nothing.
        result = {k: v for k, v in {
            k: recursively_generate_similarity_measurements(data_node[k], v) for k, v in target_node.items()
        }.items() if v is not None}
        return None if len(result) == 0 else result


def perform_model_similarity_analysis(model_results: Dict):
    """Perform a similarity analysis on the given model and report the results."""
    # List the targets for the similarity analysis and get the associated results.
    similarity_data_targets = {
        "logging": {
            "aggregate_data": {
                "message_data": {
                    "succession_frequency": None,
                    "transition_succession_frequency": None,
                    "intervals": {
                        "aggregate_data": {
                            "message_frequency": None,
                            "succession_frequency": None,
                            "transition_succession_frequency": None,
                        }
                    }
                },
                "log_data": {
                    "log_frequencies": None,
                    "thread_log_frequencies": None,
                }
            }
        },
        "aggregate": {
            "message_frequency": None
        }
    }
    similarity_measurements = recursively_generate_similarity_measurements(model_results, similarity_data_targets)

    get_throughput_report(model_results)
    get_similarity_report(model_results, similarity_measurements)

    # TODO: decide what to do with rows with only zeroes.
    # TODO: report elaborate similarity analysis on global message frequency (counting vs logging).
    # TODO: report summarized similarity analysis on intervals.
    # TODO: report elaborate similarity analysis on log frequency (global vs threads).
