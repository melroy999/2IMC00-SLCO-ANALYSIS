import json
from os import path, listdir
from typing import Dict, Tuple, List

import pandas as pd

from analysis.similarity import perform_model_similarity_analysis
from preprocessing.aggregate import combine_aggregate_data, create_logging_aggregate_data, \
    create_counting_aggregate_data
from preprocessing.counting import preprocess_counting_data
from preprocessing.logging import preprocess_logging_data
from validation import validate_source_model, validate_data_integrity, validate_logging_representability


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


def import_model_results(target_model: str) -> Dict:
    """Import all results of the given model."""
    # Find the folder that contains the model data and the list of associated result entries.
    model_path = path.join("results", target_model)
    logging_result_entries, logging_aggregate_data = import_logging_results(model_path)
    counting_result_entries, counting_aggregate_data = import_counting_results(model_path)
    aggregate_data = combine_aggregate_data(logging_aggregate_data, counting_aggregate_data)

    # Get the model information of one of the models.
    model_data = (logging_result_entries + logging_result_entries)[0]["model"]

    # Create a dictionary containing all preprocessed data.
    result = {
        "model": {
            "id": target_model,
            "data": model_data
        },
        "logging": {
            "entries": logging_result_entries,
            "aggregate_data": logging_aggregate_data
        },
        "counting": {
            "entries": counting_result_entries,
            "aggregate_data": counting_aggregate_data
        },
        "aggregate": aggregate_data
    }

    # Validate the source of the models to ensure that they can be compared with each other.
    validate_source_model(result)

    # Remove model information from the entries.
    for entry in logging_result_entries + counting_result_entries:
        del entry["model"]

    # # Check if the data's integrity shows signs of outside influence or other issues.
    # validate_data_integrity(result)
    #
    # # Validate whether the logging and counting data follow a similar trend.
    # validate_logging_representability(result)

    # Return the results as a dictionary.
    return result


def analyze_model(target_model: str):
    """Analyze the results found for each run of the given model."""
    # Find the folder that contains the model data and the list of associated result entries.
    print(f"Analyzing model {target_model}")
    data = import_model_results(target_model)
    perform_model_similarity_analysis(data)

    print()


if __name__ == '__main__':
    analyze_model("Elevator[CL=3,LBS=4194304,LFS=100MB,T=60s,URP]")
    analyze_model("Elevator[CL=3,LBS=4194304,LFS=100MB,T=60s]")

# TODO:
#   - An important observation to make is that the counting and logging methods do not produce the same results.
#       - However, when filtering on successful events only, the results are once again quite similar.
