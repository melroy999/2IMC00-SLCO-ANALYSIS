import json
from os import path, listdir
from typing import Dict, List

from preprocessing.model import preprocess_model_results


def import_data(target: str) -> Dict:
    """Import the json data associated with the given run of the target model."""
    with open(path.join(target, "results.json"), 'r') as f:
        return json.load(f)


def import_logging_results(model_path: str) -> List:
    """Import all results found during log-based measurements."""
    # Find the folder that contains the log-based model measurements and the list of associated result entries.
    target_path = path.join(model_path, "logging")
    result_entries = sorted(listdir(target_path))

    # Import and return all of the result entries.
    return [import_data(path.join(target_path, result_entry)) for result_entry in result_entries]


def import_counting_results(model_path: str) -> List:
    """Import all results found during count-based measurements."""
    # Find the folder that contains the log-based model measurements and the list of associated result entries.
    target_path = path.join(model_path, "counting")
    result_entries = sorted(listdir(target_path))

    # Import and return all of the result entries.
    return [import_data(path.join(target_path, result_entry)) for result_entry in result_entries]


def import_model_results(target_model: str) -> Dict:
    """Import all results of the given model."""
    # Find the folder that contains the model data and the list of associated result entries.
    model_path = path.join("results", target_model)
    model_results = {
        "logging": import_logging_results(model_path),
        "counting": import_counting_results(model_path)
    }

    # Preprocess the results.
    data = preprocess_model_results(model_results, target_model)

    return data


def analyze_model(target_model: str):
    """Analyze the results found for each run of the given model."""
    # Find the folder that contains the model data and the list of associated result entries.
    print(f"Analyzing model {target_model}")
    data = import_model_results(target_model)


if __name__ == '__main__':
    analyze_model("Elevator[CL=3,LBS=4194304,LFS=100MB,T=60s,URP]")
    analyze_model("Elevator[CL=3,LBS=4194304,LFS=100MB,T=60s]")

# TODO:
#   - An important observation to make is that the counting and logging methods do not produce the same results.
#       - However, when filtering on successful events only, the results are once again quite similar.
