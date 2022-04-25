from collections import defaultdict
from typing import Dict, List

import networkx as nx


def get_thread_to_state_machine_table(model_data: Dict, message_results: List[Dict]) -> Dict[str, str]:
    """Find which thread is associated to which state machine."""
    # Create a table that connects transition and decision node abbreviations to the state machine they are part of.
    abbreviation_to_state_machine = dict()
    for class_data in model_data["classes"].values():
        for state_machine_name, state_machine_data in class_data["state_machines"].items():
            for decision_structure_data in state_machine_data["decision_structures"].values():
                abbreviation_to_state_machine[decision_structure_data["id"]] = state_machine_name
                for transition_data in decision_structure_data["transitions"].values():
                    abbreviation_to_state_machine[transition_data["id"]] = state_machine_name

    # Use the abbreviation codes in the message data to find which thread is associated to each state machine.
    thread_to_state_machine = dict()
    for message_data in message_results:
        for message in message_data["global_data"]["event_count"]:
            # Split the message and find the state machine associated to the given abbreviation.
            thread_name, abbreviation, _ = message.split(".")
            state_machine_name = abbreviation_to_state_machine[abbreviation]
            if thread_name in thread_to_state_machine:
                # It is presumed that threads are only associated to one state machine. Hence, verify.
                if thread_to_state_machine[thread_name] != state_machine_name:
                    raise Exception(f"Thread \"{thread_name}\" is associated to multiple state machines.")
            else:
                thread_to_state_machine[thread_name] = state_machine_name

    # Return the discovered mapping.
    return thread_to_state_machine


def get_graph_representation(model: Dict) -> nx.MultiDiGraph:
    """Convert the given state machine model to a graph object."""
    # Add all the nodes and include the associated decision structure data.
    graph = nx.MultiDiGraph()
    for decision_structure_name, decision_structure_data in model["decision_structures"].items():
        graph.add_node(decision_structure_name, data=decision_structure_data)

    # Add all transitions.
    for decision_structure_name, decision_structure_data in model["decision_structures"].items():
        for transition_name, transition_data in decision_structure_data["transitions"].items():
            graph.add_edge(
                transition_data["source"], transition_data["target"], transition_data["id"], data=transition_data
            )
    return graph


def restructure_model_data(logging_results: List[Dict], target_model: str) -> Dict:
    """Restructure the model information such that it becomes more convenient to work with."""
    # Select the model results.
    model_data = logging_results[0]["model"]

    # Find which thread is associated to which state machine.
    thread_to_state_machine = get_thread_to_state_machine_table(
        model_data, [v["message_data"] for v in logging_results]
    )

    # Return a revised structuring for the model information.
    restructured_model_data = {
        "name": model_data["name"],
        "id": target_model,
    } | {
        "classes": {
            class_name: {
                "name": class_data["name"]
            } | {
                "state_machines": {
                    state_machine_name: state_machine_data
                    for state_machine_name, state_machine_data in class_data["state_machines"].items()
                }
            } for class_name, class_data in model_data["classes"].items()
        }
    }

    # Create a mapping that maps abbreviations to the associated model components.
    abbreviation_to_target = dict()
    class_data: Dict
    for class_data in restructured_model_data["classes"].values():
        for state_machine_name, state_machine_data in class_data["state_machines"].items():
            for decision_structure_data in state_machine_data["decision_structures"].values():
                abbreviation_to_target[decision_structure_data["id"]] = decision_structure_data
                for transition_data in decision_structure_data["transitions"].values():
                    abbreviation_to_target[transition_data["id"]] = transition_data
    restructured_model_data["abbreviation_to_target"] = abbreviation_to_target

    # Add unique names for state machines if multiple threads point to the same state machine.
    state_machine_unique_name = dict()
    state_machine_to_threads = defaultdict(list)
    for thread_name, state_machine_name in thread_to_state_machine.items():
        state_machine_to_threads[state_machine_name].append(thread_name)
    for state_machine_name, thread_names in state_machine_to_threads.items():
        if len(thread_names) > 1:
            for i, thread_name in enumerate(thread_names):
                state_machine_unique_name[thread_name] = f"{state_machine_name}${i}"
        else:
            state_machine_unique_name[thread_names[0]] = state_machine_name

    # Create a mapping that maps threads to the associated model components.
    state_machine_name_to_object = dict()
    for class_data in restructured_model_data["classes"].values():
        for state_machine_name, state_machine_data in class_data["state_machines"].items():
            state_machine_name_to_object[state_machine_name] = state_machine_data
    restructured_model_data["thread_to_state_machine"] = {
        thread_name: (state_machine_unique_name[thread_name], state_machine_name_to_object[state_machine_name])
        for thread_name, state_machine_name in thread_to_state_machine.items()
    }

    # Create a graph object for each of the state machines.
    for class_data in restructured_model_data["classes"].values():
        for state_machine_data in class_data["state_machines"].values():
            state_machine_data["graph"] = get_graph_representation(state_machine_data)

    return restructured_model_data
