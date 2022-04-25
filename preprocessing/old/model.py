from typing import Dict


def preprocess_model_data(data: Dict) -> Dict:
    """Preprocess the json data."""
    # Create a mapping between object abbreviations and the objects themselves.
    data["abbreviation_mapping"] = dict()
    for class_info in data["classes"].values():
        for state_machine_info in class_info["state_machines"].values():
            for decision_structures_info in state_machine_info["decision_structures"].values():
                data["abbreviation_mapping"][decision_structures_info["id"]] = decision_structures_info
                for transition_info in decision_structures_info["transitions"].values():
                    data["abbreviation_mapping"][transition_info["id"]] = transition_info

    return data
