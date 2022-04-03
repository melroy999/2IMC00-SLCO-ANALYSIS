import random
from typing import Dict

import networkx as nx
import pandas as pd


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


def create_succession_graph(edges: Dict[str, int]) -> nx.DiGraph:
    """Create a succession graph with the given list of edges."""
    graph = nx.DiGraph()

    # Create the edges.
    for edge, count in edges.items():
        source, target = edge.split("~")
        graph.add_edge(source, target, weight=count)
    return graph


def preprocess_message_data_graph(entry_data: Dict):
    """Convert the mapping from node pairs to count to a graph object."""
    # Convert the dictionary of node pairs to counts to a pandas table for correlation measurements.
    entry_data["succession_table"] = pd.DataFrame.from_dict(
        entry_data["succession_graph"], orient="index", columns=["count"]
    )
    entry_data["transition_succession_table"] = pd.DataFrame.from_dict(
        entry_data["transition_succession_graph"], orient="index", columns=["count"]
    )

    # Convert the dictionary of node pairs to counts to a succession graph object.
    entry_data["succession_graph"] = create_succession_graph(entry_data["succession_graph"])
    entry_data["transition_succession_graph"] = create_succession_graph(entry_data["transition_succession_graph"])


def preprocess_message_data_count(entry_data: Dict):
    """Convert the count dictionaries to pandas data frames."""
    entry_data["event_count"] = pd.DataFrame.from_dict(entry_data["event_count"], orient="index", columns=["count"])


def preprocess_message_data(message_data: Dict):
    """Preprocess the message data entry in the JSON hierarchy."""
    preprocess_message_data_graph(message_data["global_data"])
    preprocess_message_data_count(message_data["global_data"])
    for v in message_data["intervals"]:
        preprocess_message_data_graph(v["data"])
        preprocess_message_data_count(v["data"])


def preprocess_data(data: Dict):
    """Preprocess the json data."""
    # Preprocess the log data such that it can be used more easily in figures and graphs.
    preprocess_log_data(data["log_data"])
    preprocess_message_data(data["message_data"])
    return data
