import json
from os import path, listdir
from typing import Dict, Tuple, List

import pandas as pd

from analyzation import analyze_data, create_logging_aggregate_data, create_counting_aggregate_data, \
    create_correlation_table, create_manhattan_table
from preprocessing.counting import preprocess_counting_data
from preprocessing.logging import preprocess_logging_data


def import_data(target: str) -> Dict:
    """Import the json data associated with the given run of the target model."""
    with open(path.join(target, "results.json"), 'r') as f:
        return json.load(f)


def import_logging_run_result(result_entry_path: str) -> Dict:
    """Import and preprocess the results found in the given run of the target model using the log based measurements."""
    model_data = import_data(result_entry_path)
    return preprocess_logging_data(model_data)


def import_counting_run_result(result_entry_path: str) -> Dict:
    """
    Import and preprocess the results found in the given run of the target model using the counting-based measurements.
    """
    model_data = import_data(result_entry_path)
    return preprocess_counting_data(model_data)


def import_logging_results(model_path: str) -> Tuple[List, Dict]:
    """Import all results found during log-based measurements."""
    # Find the folder that contains the log-based model measurements and the list of associated result entries.
    target_path = path.join(model_path, "logging")
    result_entries = sorted(listdir(target_path))

    # Import all of the result entries.
    run_results = [import_logging_run_result(path.join(target_path, result_entry)) for result_entry in result_entries]

    # Create the aggregate data.
    aggregate_data = create_logging_aggregate_data(run_results)

    # Return the results and aggregate data.
    return run_results, aggregate_data


def import_counting_results(model_path: str) -> Tuple[List, Dict]:
    """Import all results found during count-based measurements."""
    # Find the folder that contains the log-based model measurements and the list of associated result entries.
    target_path = path.join(model_path, "counting")
    result_entries = sorted(listdir(target_path))

    # Import all of the result entries.
    run_results = [import_counting_run_result(path.join(target_path, result_entry)) for result_entry in result_entries]

    # Create the aggregate data.
    aggregate_data = create_counting_aggregate_data(run_results)

    # Return the results and aggregate data.
    return run_results, aggregate_data


def analyze_model(target_model: str):
    """Analyze the results found for each run of the given model."""
    # Find the folder that contains the model data and the list of associated result entries.
    model_path = path.join("results", target_model)
    logging_result_entries, logging_aggregate_data = import_logging_results(model_path)
    counting_result_entries, counting_aggregate_data = import_counting_results(model_path)

    compare_logging_and_counting = pd.concat(
        [
            logging_aggregate_data["event_count"].add_prefix(f"logging_"),
            counting_aggregate_data["event_count"].add_prefix(f"counting_")
        ], axis=1
    ).fillna(.0).sort_index()

    target_columns = [f"logging_count_{i}" for i, _ in enumerate(logging_result_entries)] + \
                     [f"counting_count_{i}" for i, _ in enumerate(logging_result_entries)]
    filtered_data = compare_logging_and_counting[target_columns]
    filtered_data = filtered_data[filtered_data.index.str.contains(".S")]
    for target_column in target_columns:
        filtered_data[f"{target_column}_n"] = filtered_data[target_column] / filtered_data[target_column].sum()
    correlation_table = create_correlation_table(filtered_data, target_columns)
    manhattan_table = create_manhattan_table(filtered_data, [f"{v}_n" for v in target_columns])
    # TODO: check if there are runs that stand out and may possibly be influenced by outside interference.

    # TODO: create aggregate data.

    # Analyze the data.
    for data in logging_result_entries:
        analyze_data(data)


if __name__ == '__main__':
    analyze_model("Elevator[CL=3,LBS=4194304,LFS=100MB,T=60s,URP]")

# TODO:
#   - An important observation to make is that the counting and logging methods do not produce the same results.
#       - However, when filtering on successful events only, the results are once again quite similar.
