from typing import Dict

import networkx as nx
import pandas as pd

from visualization.results import plot_concurrency_heatmap, plot_local_concurrency_heatmap, \
    plot_global_concurrency_heatmap, plot_transition_frequency_boxplot


def analyze_counting_model_message_frequency_results(model_data: Dict):
    """Analyze the message frequency data attained during the counting-based model runs."""
    # Use a boxplot, since we do not have enough entries for a violin plot.


def analyze_logging_model_message_frequency_results(model_data: Dict):
    """Analyze the message frequency data attained during the logging-based model runs."""
    # Preprocess the table.
    frequency_data = model_data["message_frequency"]["intervals"]["table"]
    plot_transition_frequency_boxplot(frequency_data, model_data)


def analyze_model_message_frequency_results(model_data: Dict):
    """Analyze the message frequency data attained during the model runs."""
    analyze_counting_model_message_frequency_results(model_data)
    analyze_logging_model_message_frequency_results(model_data)


def analyze_model_concurrency(model_data: Dict):
    """Analyze the concurrency within the message order data attained during the model runs."""
    # Create a table that contains all data on concurrency.
    adjacency_table = model_data["message_order"]["intervals"]["adjacency_table"]
    adjacency_table_entries = adjacency_table.melt(ignore_index=False)
    target_state_machines = [v[0] for v in model_data["model"]["thread_to_state_machine"].values()]
    concurrency_table = pd.DataFrame(
        columns=[f"{v}{t}" for t in ["", "_parallel"] for v in target_state_machines],
        index=target_state_machines
    ).fillna(0.0)

    # Fill the concurrency table with data.
    for index, row in adjacency_table_entries.iterrows():
        # Dissect the source and target identities.
        source_sm, source_transition, source_type = index.split(".")
        target_sm, target_transition, target_type = row["target_message"].split(".")

        # Count the number of transitions from the source to the target state machine instance.
        concurrency_table.at[source_sm, f"{target_sm}"] += row["value"]

        # Identify if the pair depicts concurrency. The following edges are deemed to be concurrent:
        #   - x.T*.O --> y.T*.[O, S, F],  x != y
        #   - x.T*.[S, F] --> y.T*.[S, F],  x != y
        if source_sm != target_sm and (source_type == "O" or target_type == "S" or target_type == "F"):
            concurrency_table.at[source_sm, f"{target_sm}_parallel"] += row["value"]

    # Add parallel percentages.
    percentage_table = pd.DataFrame()
    global_percentage_table = pd.DataFrame()
    for sm in target_state_machines:
        # Per cell.
        percentage_table[sm] = concurrency_table[f"{sm}_parallel"] / concurrency_table[sm]
        # Global.
        global_percentage_table[sm] = concurrency_table[f"{sm}_parallel"] / concurrency_table.sum().sum()

    # Plot.
    plot_local_concurrency_heatmap(concurrency_table[target_state_machines], percentage_table, model_data)
    plot_global_concurrency_heatmap(global_percentage_table, model_data)


def analyze_model_message_order_results(model_data: Dict):
    """Analyze the message order data attained during the model runs."""
    analyze_model_concurrency(model_data)


def analyze_model_results(model_data: Dict):
    """Analyze the consistency between the reported model result runs."""
    analyze_model_message_frequency_results(model_data)
    analyze_model_message_order_results(model_data)
