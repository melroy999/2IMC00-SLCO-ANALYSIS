import json
from os import path, listdir
from typing import Dict, List

from analysis.telephony import run_telephony_analysis
from preprocessing.model import preprocess_model_results
from analysis.default import run_default_analysis


def import_data(target: str) -> Dict:
    """Import the json data associated with the given run of the target model."""
    with open(path.join(target, "results.json"), "r") as f:
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


def import_model_results(target_model: str, include_logging: bool = False) -> Dict:
    """Import all results of the given model."""
    # Find the folder that contains the model data and the list of associated result entries.
    model_path = path.join("results", target_model)
    model_results = {
        "counting": import_counting_results(model_path),
        "logging": []
    }

    if include_logging:
        model_results |= {
            "logging": import_logging_results(model_path)
        }

    # Preprocess the results.
    return preprocess_model_results(model_results, target_model, include_logging)


def get_target_file(model_name: str, configuration: List[str]) -> str:
    """Get the file associated with the given options."""
    return f"{model_name}[{','.join(sorted(v for v in configuration if v != ''))}]"


def load_default_target_models(
        model_name: str,
        include_logging: bool = False,
        analyze_deterministic_structures: bool = True,
        analyze_locking_mechanism: bool = True

) -> Dict[str, Dict]:
    """Import all of the default data required to analyse the given model."""
    default_model_data = import_model_results(get_target_file(model_name, ["T=30s"]), include_logging=include_logging)

    # Create a master table of all desired options.
    model_entries = {
        "Default": default_model_data
    }
    if analyze_locking_mechanism:
        model_entries |= {
            "Element": default_model_data,
            "Variable": import_model_results(get_target_file(model_name, ["LA", "T=30s"])),
            "Statement": import_model_results(get_target_file(model_name, ["SLL", "T=30s"])),
        }
    if analyze_deterministic_structures:
        model_entries |= {
            "Random + Det": import_model_results(get_target_file(model_name, ["URP", "T=30s"])),
            "Sequential + Det": default_model_data,
            "Random": import_model_results(get_target_file(model_name, ["NDS", "URP", "T=30s"])),
            "Sequential": import_model_results(get_target_file(model_name, ["NDS", "T=30s"])),
        }

    return model_entries


def run_default_analysis_suite(
        model_name: str,
        include_logging: bool = False,
        analyze_deterministic_structures: bool = True,
        analyze_locking_mechanism: bool = True
):
    """Run the default analysis suite for the model with the given name."""
    model_entries = load_default_target_models(
        model_name, include_logging, analyze_deterministic_structures, analyze_locking_mechanism
    )
    run_default_analysis(
        model_name, model_entries, include_logging, analyze_deterministic_structures, analyze_locking_mechanism
    )


def load_telephony_target_models(
        model_name: str,
        include_logging: bool = False,
        analyze_deterministic_structures: bool = True,
        analyze_locking_mechanism: bool = True,
        no_deterministic_structures: bool = True,
) -> Dict[str, Dict]:
    """Import all of the data required to analyse the telephony model."""
    if no_deterministic_structures:
        default_model_data = import_model_results(
            get_target_file(model_name, ["NDS", "URP", "T=30s"]), include_logging=include_logging
        )

        # Create a master table of all desired options.
        model_entries = {
            "Default": default_model_data
        }
        if analyze_locking_mechanism:
            model_entries |= {
                "Element": default_model_data,
                "Variable": import_model_results(get_target_file(model_name, ["LA", "NDS", "URP", "T=30s"])),
                "Statement": import_model_results(get_target_file(model_name, ["SLL", "NDS", "URP", "T=30s"])),
            }
        if analyze_deterministic_structures:
            model_entries |= {
                "Random + Det": import_model_results(get_target_file(model_name, ["URP", "T=30s"])),
                "Sequential + Det": import_model_results(get_target_file(model_name, ["T=30s"])),
                "Random": default_model_data,
                "Sequential": import_model_results(get_target_file(model_name, ["NDS", "T=30s"])),
            }
    else:
        default_model_data = import_model_results(
            get_target_file(model_name, ["URP", "T=30s"]), include_logging=include_logging
        )

        # Create a master table of all desired options.
        model_entries = {
            "Default": default_model_data
        }
        if analyze_locking_mechanism:
            model_entries |= {
                "Element": default_model_data,
                "Variable": import_model_results(get_target_file(model_name, ["LA", "URP", "T=30s"])),
                "Statement": import_model_results(get_target_file(model_name, ["SLL", "URP", "T=30s"])),
            }
        if analyze_deterministic_structures:
            model_entries |= {
                "Random + Det": default_model_data,
                "Sequential + Det": import_model_results(get_target_file(model_name, ["T=30s"])),
                "Random": import_model_results(get_target_file(model_name, ["NDS", "URP", "T=30s"])),
                "Sequential": import_model_results(get_target_file(model_name, ["NDS", "T=30s"])),
            }
    return model_entries


def run_telephony_analysis_suite(
        model_name: str = "Telephony",
        include_logging: bool = False,
        analyze_deterministic_structures: bool = True,
        analyze_locking_mechanism: bool = True,
        no_deterministic_structures: bool = True
):
    """Run an altered analysis suite for the telephony model."""
    model_entries = load_telephony_target_models(
        model_name, include_logging, analyze_deterministic_structures, analyze_locking_mechanism,
        no_deterministic_structures=no_deterministic_structures
    )
    run_telephony_analysis(
        model_name, model_entries, include_logging, analyze_deterministic_structures, analyze_locking_mechanism,
        no_deterministic_structures=no_deterministic_structures
    )


def analyze_counter_distributor_models():
    """Analyze the counter distributor model instances."""
    run_default_analysis_suite("CounterDistributor", include_logging=True)


def analyze_elevator_models():
    """Analyze the elevator model instances."""
    run_default_analysis_suite("Elevator", include_logging=True)


def analyze_telephony_models():
    """Analyze the telephony model instances."""
    run_telephony_analysis_suite(include_logging=True, no_deterministic_structures=False)


def analyze_toads_and_frogs_models():
    """Analyze the toads and frogs model instances."""
    run_default_analysis_suite("ToadsAndFrogs", include_logging=True)


def analyze_tokens_models():
    """Analyze the tokens model instances."""
    run_default_analysis_suite("Tokens", include_logging=True)


if __name__ == "__main__":
    # analyze_counter_distributor_models()
    # analyze_elevator_models()
    analyze_telephony_models()
    # analyze_toads_and_frogs_models()
    # analyze_tokens_models()
