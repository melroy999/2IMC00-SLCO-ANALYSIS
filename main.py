import json
from os import path, listdir
from typing import Dict

import numpy as np
import pandas as pd

from analyzation import analyze_data
from preprocessing import preprocess_data


def import_data(target: str) -> Dict:
    """Import the json data associated with the given run of the target model."""
    with open(path.join(target, "results.json"), 'r') as f:
        return json.load(f)


def import_run_result(result_entry_path: str) -> Dict:
    """Import and preprocess the results found in the given run of the target model."""
    model_data = import_data(result_entry_path)
    return preprocess_data(model_data)


def analyze_model(target_model: str):
    """Analyze the results found for each run of the given model."""
    # Find the folder that contains the model data and the list of associated result entries.
    model_path = path.join("results", target_model)
    result_entries = sorted(listdir(model_path))

    # Import all of the result entries.
    run_results = [import_run_result(path.join(model_path, result_entry)) for result_entry in result_entries]

    # TODO: check if there are runs that stand out and may possibly be influenced by outside interference.

    # TODO: create aggregate data.

    # Concatenate the count data of the different runs.
    aggregate_count_data = pd.concat(
        [data["message_data"]["global"]["count"].add_suffix(f"_{i}") for i, data in enumerate(run_results)], axis=1
    )
    aggregate_count_data.sort_index(inplace=True)

    # Create statistical data.
    aggregate_count_data["count_mean"] = aggregate_count_data[
        [f"count_{i}" for i, _ in enumerate(run_results)]
    ].mean(axis=1)
    aggregate_count_data["count_std"] = aggregate_count_data[
        [f"count_{i}" for i, _ in enumerate(run_results)]
    ].std(axis=1)
    aggregate_count_data["count_s_mean"] = aggregate_count_data[
        [f"count_s_{i}" for i, _ in enumerate(run_results)]
    ].mean(axis=1)
    aggregate_count_data["count_s_std"] = aggregate_count_data[
        [f"count_s_{i}" for i, _ in enumerate(run_results)]
    ].std(axis=1)

    correlation_table = np.array([
        [
            aggregate_count_data[f"count_{i}"].corr(aggregate_count_data[f"count_{j}"]) for j, _ in enumerate(run_results)
        ] for i, _ in enumerate(run_results)
    ])

    # Analyze the data.
    for data in run_results:
        analyze_data(data)

if __name__ == '__main__':
    target_folders = {
        "Elevator[CL=3,LBS=4194304,LFS=100MB,T=60s]"
    }

    for target_folder in target_folders:
        analyze_model(target_folder)
