import networkx as nx
from matplotlib import pyplot as plt


def plot_graph(graph: nx.MultiDiGraph):
    """Plot the given graph."""
    pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_color='r', node_size=100, alpha=1)
    ax = plt.gca()
    for e in graph.edges:
        ax.annotate(
            "",
            xy=pos[e[0]], xycoords='data',
            xytext=pos[e[1]], textcoords='data',
            arrowprops=dict(
                arrowstyle="->", color="0.5",
                shrinkA=5, shrinkB=5,
                patchA=None, patchB=None,
                connectionstyle="arc3,rad=rrr".replace('rrr', str(0.3 * e[2])),
            ),
        )
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    test_graph = nx.MultiDiGraph()
    test_graph.add_edge("A", "B")
    test_graph.add_edge("A", "B")
    test_graph.add_edge("A", "A")
    test_graph.add_edge("B", "C")
    test_graph.add_edge("C", "C")
    test_graph.add_edge("C", "A")
    plot_graph(test_graph)

