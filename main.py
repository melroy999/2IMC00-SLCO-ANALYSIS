import json
from os import path
from typing import Dict, Tuple

import networkx as nx

from plotting import create_global_log_file_throughput_plot, create_thread_grouped_log_file_throughput_plot, \
    render_graph, create_succession_heat_map_plot


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


def obscure_graph_node_name(node: str):
    """Obscure the individual identity of the target node and replace it with an asterisks."""
    node_thread, node_id, node_type = node.split(".")
    return f"{node_thread}.{node_id[0]}*.{node_type}"


def obscure_succession_graph_nodes(graph: nx.DiGraph) -> nx.DiGraph:
    """Create a graph that obscures the individual identities of the contained nodes."""
    node_name_mapping = {n: obscure_graph_node_name(n) for n in graph.nodes}
    obscured_graph = nx.DiGraph()
    for u, v in graph.edges:
        source = node_name_mapping[u]
        target = node_name_mapping[v]
        if obscured_graph.has_edge(source, target):
            obscured_graph[source][target]["weight"] += graph[u][v]["weight"]
        else:
            obscured_graph.add_edge(source, target, weight=graph[u][v]["weight"])
    return obscured_graph


def preprocess_message_data_succession_graph(entry_data: Dict):
    """Convert the mapping from node pairs to count to a graph object."""
    edges = entry_data["graph"]
    graph = nx.DiGraph()

    # Create the edges.
    for edge, count in edges.items():
        source, target = edge.split("~")
        graph.add_edge(source, target, weight=count)

    graph = obscure_succession_graph_nodes(graph)
    entry_data["succession_graph"] = graph
    del entry_data["graph"]


def preprocess_message_data(message_data: Dict):
    """Preprocess the message data entry in the JSON hierarchy."""
    preprocess_message_data_succession_graph(message_data["global"])


def preprocess_data(data: Dict):
    """Preprocess the json data."""
    # Preprocess the log data such that it can be used more easily in figures and graphs.
    preprocess_log_data(data["log_data"])
    preprocess_message_data(data["message_data"])

    return data


def analyze_model(target: str):
    """Analyze the given model."""
    model_data = import_data(target)
    preprocessed_data = preprocess_data(model_data)
    create_global_log_file_throughput_plot(preprocessed_data)
    create_thread_grouped_log_file_throughput_plot(preprocessed_data)
    create_succession_heat_map_plot(preprocessed_data)


if __name__ == '__main__':
    analyze_model("Elevator[T=60s]_2022-03-26T23.45.19.982103100Z")
