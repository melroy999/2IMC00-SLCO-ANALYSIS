import copy
from typing import Dict, Tuple

import numpy as np

import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from networkx.drawing.nx_pydot import graphviz_layout


def create_global_log_file_throughput_plot(data: Dict):
    """Plot the file throughput data for the global data entry."""
    # Set the figure size.
    plt.figure(figsize=(10, 3), dpi=300)

    # Convert the files entries to a 2d array.
    files = data["log_data"]["global"]["files"]
    array_data = [f["count"] for f in files]
    numpy_array = np.array(array_data)

    # Plot the figure as a color mesh plot.
    ax = plt.axes()
    c = ax.pcolormesh(numpy_array, cmap="Blues")
    ax.set_title(f"\"{data['model']['name']}\" Logging Throughput (Global)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("File number")
    ax.invert_yaxis()
    plt.colorbar(c, ax=ax, pad=0.015)

    plt.tight_layout()
    plt.show()


def create_thread_grouped_log_file_throughput_plot(data: Dict):
    """Plot the file throughput data for the thread data entries."""
    threads = data["log_data"]["threads"]

    # Put all threads into one figure.
    fig, axes = plt.subplots(len(threads), figsize=(10, 3 * len(threads)), dpi=300)

    # Convert all the data to 2d numpy arrays and keep the max value range.
    array_data_map = dict()
    max_value = 0
    for thread in threads:
        files = threads[thread]["files"]
        array_data = [f["count"] for f in files]
        array_data_map[thread] = np.array(array_data)
        max_value = max(max_value, np.amax(array_data_map[thread]))

    # Plot each thread individually.
    for i, thread in enumerate(threads):
        # Plot the figure as a color mesh plot.
        ax = axes[i]
        c = ax.pcolormesh(array_data_map[thread], cmap="Blues", vmin=0, vmax=max_value)
        ax.set_title(f"\"{data['model']['name']}\" Logging Throughput ({thread})")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("File number")
        ax.invert_yaxis()
        fig.colorbar(c, ax=ax, pad=0.015)

    plt.tight_layout()
    plt.show()


type_order_map = {
        "D*.O": 0,
        "T*.O": 1,
        "T*.S": 2,
        "T*.F": 3,
        "D*.S": 4,
        "D*.F": 5,
    }


def get_succession_heat_map_plot_node_order_key(node: str) -> Tuple[str, int]:
    """Get a key that can be used to reorder the list of nodes to the desired specifications."""
    node_thread, node_type = node.split(".", 1)
    return node_thread, type_order_map[node_type]


def get_succession_plot_color_map():
    """Get the color map to be used in the succession heat map plot."""
    cmap = copy.copy(plt.get_cmap("cividis"))
    new_colors = cmap(np.linspace(0, 1, 2**16))
    black = np.array([0, 0, 0, 1])
    new_colors[:1, :] = black
    return ListedColormap(new_colors)


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


def create_succession_heat_map_plot(data: Dict):
    """Plot the succession graph as a heat map."""
    # Get the appropriate graph data and obscure unneeded data.
    graph = data["message_data"]["global"]["succession_graph"]
    graph = obscure_succession_graph_nodes(graph)

    # Create an order in which to view the nodes.
    sorted_nodes = sorted(graph.nodes, key=lambda x: get_succession_heat_map_plot_node_order_key(x))
    graph_data = nx.to_numpy_array(graph, nodelist=sorted_nodes, dtype=np.dtype([("weight", int)]), weight=None)

    # Create an image plot.
    fig, ax = plt.subplots(1, figsize=(9, 9), dpi=300)
    im = ax.imshow(graph_data["weight"], cmap=get_succession_plot_color_map())

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(sorted_nodes)), labels=sorted_nodes, fontname="monospace")
    ax.set_yticks(np.arange(len(sorted_nodes)), labels=sorted_nodes, fontname="monospace")
    ax.set_xlabel("Successor", labelpad=10)
    ax.set_ylabel("Source", labelpad=10)
    ax.xaxis.set_label_position('top')

    # Make the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-45, ha="right", rotation_mode="anchor")

    # Add the title and color bar.
    ax.set_title(f"\"{data['model']['name']}\" Succession Heat Map", pad=20)
    color_bar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.025, orientation="horizontal")
    color_bar.ax.set_xlabel("Count", labelpad=10)

    plt.tight_layout()
    plt.show()




def render_graph(graph: nx.DiGraph):
    """Visualize the given graph with the given data."""
    if len(graph.nodes) == 0:
        return

    plt.figure(figsize=(28, 28))

    # Determine the location and create an axis to put the title on.
    pos = graphviz_layout(graph, prog="circo")
    ax = plt.gca()

    # Draw the graph with the given colors.
    nx.draw(graph, pos, ax=ax, with_labels=True)

    plt.show()
