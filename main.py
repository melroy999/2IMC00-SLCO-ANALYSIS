import json
from os import path
from typing import Dict, List

import tikzplotlib
import matplotlib.pyplot as plt
import numpy as np


def create_global_log_file_throughput_plot(data: Dict):
    # Convert the files entries to a 2d array.
    files = data["log_data"]["global"]["files"]
    array_data = [f["count"] for f in files]
    numpy_array = np.array(array_data)

    # Plot the figure as a color mesh plot.
    plt.figure(figsize=(10, 5), dpi=300)
    ax = plt.axes()
    c = ax.pcolormesh(numpy_array, cmap="Blues")
    ax.set_title(f"`{data['model']['name']}' Logging Throughput")
    plt.xlabel("Time (ms)")
    plt.ylabel("File number")
    plt.colorbar(c, ax=ax)
    plt.gca().invert_yaxis()
    # tikzplotlib.save("plot.tex", dpi=300)
    plt.show()


def create_thread_grouped_log_file_throughput_plot(data: Dict):
    threads = data["log_data"]["threads"]

    # Put all threads into one figure.
    fig, axes = plt.subplots(len(threads), sharex="all", sharey="all", figsize=(10, 4 * len(threads)), dpi=300)

    # Convert all the data to 2d numpy arrays and keep the max value range.
    array_data_map = dict()
    max_value = 0
    for thread in threads:
        files = threads[thread]["files"]
        array_data = [f["count"] for f in files]
        array_data_map[thread] = np.array(array_data)
        max_value = max(max_value, np.amax(array_data_map[thread]))

    # Plot each thread individually.
    c = None
    for i, thread in enumerate(threads):
        # Plot the figure as a color mesh plot.
        ax = axes[i]
        c = ax.pcolormesh(array_data_map[thread], cmap="Blues", vmin=0, vmax=max_value)
        ax.set_title(f"{thread}")
        ax.invert_yaxis()

    plt.xlabel("Time (ms)")
    fig.supylabel("File number")

    fig.suptitle(f"`{data['model']['name']}' Logging Throughput")
    fig.colorbar(c, ax=axes.ravel().tolist())

    plt.show()


def import_data(target: str) -> Dict:
    """Import the json data associated with the model."""
    with open(path.join("results", target, "results.json"), 'r') as f:
        return json.load(f)


def preprocess_log_data_file_entry(file_data: Dict, array_size: int):
    """Preprocess a file entry within the JSON hierarchy using the given parameters."""
    # Having the data as a mapping can be inconvenient--instead, it would be preferable to have fixed size arrays.
    # Adjust all entries to the starting time.
    start_time = int(file_data["start"])

    # Create an empty array with a length equal to the maximum measurement duration encountered.
    entries = [0] * array_size
    for timestamp, count in file_data["count"].items():
        entries[int(timestamp) - start_time] = count

    # Replace the count data with the generated array.
    file_data["count"] = entries


def preprocess_log_data_entry(entry_data: Dict):
    """Preprocess a files/global dictionary pair contained within the JSON hierarchy."""
    max_file_duration = max(entry["duration"] for entry in entry_data["files"])
    for i, _ in enumerate(entry_data["files"]):
        # Find the target file and the target ranges.
        preprocess_log_data_file_entry(entry_data["files"][i], max_file_duration)
    preprocess_log_data_file_entry(entry_data["global"], entry_data["global"]["duration"])


def preprocess_log_data(log_data: Dict):
    """Preprocess the log data entry in the JSON hierarchy."""
    preprocess_log_data_entry(log_data["global"])
    for entry in log_data["threads"].values():
        preprocess_log_data_entry(entry)


def preprocess_data(data: Dict):
    """Preprocess the json data."""
    # Preprocess the log data such that it can be used more easily in figures and graphs.
    preprocess_log_data(data["log_data"])

    return data


def analyze_model(target: str):
    """Analyze the given model."""
    model_data = import_data(target)
    preprocessed_data = preprocess_data(model_data)
    create_global_log_file_throughput_plot(preprocessed_data)
    create_thread_grouped_log_file_throughput_plot(preprocessed_data)


if __name__ == '__main__':
    analyze_model("Elevator[T=60s]_2022-03-26T23.45.19.982103100Z")
