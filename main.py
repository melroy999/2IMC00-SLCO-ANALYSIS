import json
from os import path, listdir
from typing import Dict, List

from preprocessing.model import preprocess_model_results
from visualization.plots import plot_transition_frequency_comparison_boxplot, \
    plot_state_machine_frequency_comparison_boxplot, plot_throughput_reports, plot_frequency_results_table


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
        "logging": import_logging_results(model_path)
    }

    # Preprocess the results.
    return preprocess_model_results(model_results, target_model, include_logging)


def analyze_counting_logging_distribution(model_data, target_model_name: str):
    """Analyze the transition distributions of the counting and logging measurements of a model."""
    counting_columns = model_data["message_frequency"]["global"]["targets"]["counting"]
    logging_columns = model_data["message_frequency"]["global"]["targets"]["logging"]
    counting_frequency_data = model_data["message_frequency"]["global"]["table"][counting_columns]
    logging_frequency_data = model_data["message_frequency"]["global"]["table"][logging_columns]

    categories = {
        "Counting": counting_frequency_data,
        "Logging": logging_frequency_data,
    }

    plot_transition_frequency_comparison_boxplot(
        categories,
        model_data,
        target_model_name,
        log_scale=True,
        legend_title="Measurement Type", file_name="counting_logging_transition_frequency_comparison"
    )

    plot_state_machine_frequency_comparison_boxplot(
        categories,
        model_data,
        target_model_name,
        log_scale=True,
        legend_title="Measurement Type", file_name="counting_state_machine_transition_frequency_comparison"
    )


def analyze_elevator_models():
    """Analyze the elevator model instances."""
    model_data = import_model_results("Elevator[T=60s]_old", include_logging=True)

    target_columns = model_data["message_frequency"]["global"]["targets"]["counting"]
    frequency_data = model_data["message_frequency"]["global"]["table"][target_columns]

    plot_frequency_results_table(frequency_data, model_data, "Elevator")

    categories = {
        "A": frequency_data,
        "B": frequency_data,
        "C": frequency_data,
        "D": frequency_data
    }

    plot_transition_frequency_comparison_boxplot(
        categories, model_data, "Elevator", file_name="counting_transition_frequency_comparison"
    )

    plot_state_machine_frequency_comparison_boxplot(
        categories, model_data, "Elevator", file_name="counting_state_machine_frequency_comparison"
    )

    analyze_counting_logging_distribution(model_data, "Elevator")

    plot_throughput_reports(model_data, "Elevator")


def analyze_synthetic_test_tokens_models():
    """Analyze the elevator model instances."""
    model_data = import_model_results("SyntheticTestTokens[T=60s]_old", include_logging=True)

    target_columns = model_data["message_frequency"]["global"]["targets"]["counting"]
    frequency_data = model_data["message_frequency"]["global"]["table"][target_columns]

    plot_frequency_results_table(frequency_data, model_data, "SyntheticTestTokens")

    categories = {
        "A": frequency_data,
        "B": frequency_data,
        "C": frequency_data,
        "D": frequency_data
    }

    plot_transition_frequency_comparison_boxplot(
        categories, model_data, "SyntheticTestTokens", file_name="counting_transition_frequency_comparison"
    )

    plot_state_machine_frequency_comparison_boxplot(
        categories, model_data, "SyntheticTestTokens", file_name="counting_state_machine_frequency_comparison"
    )

    analyze_counting_logging_distribution(model_data, "SyntheticTestTokens")

    plot_throughput_reports(model_data, "SyntheticTestTokens")


if __name__ == "__main__":
    analyze_elevator_models()
    analyze_synthetic_test_tokens_models()
