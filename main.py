import json
from os import path, listdir
from typing import Dict, List

from preprocessing.model import preprocess_model_results
from visualization.plots import plot_transition_frequency_comparison_boxplot, \
    plot_state_machine_frequency_comparison_boxplot, plot_throughput_reports, plot_frequency_results_table, \
    plot_throughput_information_table


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
        "Measurement Type",
        log_scale=True,
        file_name="counting_logging_transition_frequency_comparison"
    )

    plot_state_machine_frequency_comparison_boxplot(
        categories,
        model_data,
        target_model_name,
        "Measurement Type",
        log_scale=True,
        file_name="counting_state_machine_transition_frequency_comparison"
    )


def analyze_decision_structure_performance(
        random_pick_structures_model_data,
        sequential_structures_model_data,
        random_pick_model_data,
        sequential_model_data,
        target_model_name: str
):
    """Analyze the transition distributions of the different locking modes."""
    counting_columns = random_pick_structures_model_data["message_frequency"]["global"]["targets"]["counting"]
    random_pick_structures_frequency_data = \
        random_pick_structures_model_data["message_frequency"]["global"]["table"][counting_columns]
    counting_columns = sequential_structures_model_data["message_frequency"]["global"]["targets"]["counting"]
    sequential_structures_frequency_data = \
        sequential_structures_model_data["message_frequency"]["global"]["table"][counting_columns]
    counting_columns = random_pick_model_data["message_frequency"]["global"]["targets"]["counting"]
    random_pick_frequency_data = random_pick_model_data["message_frequency"]["global"]["table"][counting_columns]
    counting_columns = sequential_model_data["message_frequency"]["global"]["targets"]["counting"]
    sequential_frequency_data = sequential_model_data["message_frequency"]["global"]["table"][counting_columns]

    categories = {
        "Sequential + Det": sequential_structures_frequency_data,
        "Random + Det": random_pick_structures_frequency_data,
        "Sequential": sequential_frequency_data,
        "Random": random_pick_frequency_data,
    }

    plot_transition_frequency_comparison_boxplot(
        categories,
        random_pick_model_data,
        target_model_name,
        "Decision Mode",
        log_scale=True,
        file_name="decision_mode_transition_frequency_comparison"
    )

    plot_state_machine_frequency_comparison_boxplot(
        categories,
        random_pick_model_data,
        target_model_name,
        "Decision Mode",
        log_scale=True,
        file_name="decision_mode_state_machine_transition_frequency_comparison"
    )


def analyze_locking_mechanism_performance(
        element_model_data,
        variable_model_data,
        statement_model_data,
        no_locks_model_data,
        target_model_name: str
):
    """Analyze the transition distributions of the different locking modes."""
    counting_columns = element_model_data["message_frequency"]["global"]["targets"]["counting"]
    element_frequency_data = element_model_data["message_frequency"]["global"]["table"][counting_columns]
    counting_columns = variable_model_data["message_frequency"]["global"]["targets"]["counting"]
    variable_frequency_data = variable_model_data["message_frequency"]["global"]["table"][counting_columns]
    counting_columns = statement_model_data["message_frequency"]["global"]["targets"]["counting"]
    statement_frequency_data = statement_model_data["message_frequency"]["global"]["table"][counting_columns]
    counting_columns = no_locks_model_data["message_frequency"]["global"]["targets"]["counting"]
    no_locks_frequency_data = no_locks_model_data["message_frequency"]["global"]["table"][counting_columns]

    categories = {
        "Element": element_frequency_data,
        "Variable": variable_frequency_data,
        "Statement": statement_frequency_data,
        "No Locks": no_locks_frequency_data,
    }

    plot_transition_frequency_comparison_boxplot(
        categories,
        element_model_data,
        target_model_name,
        "Locking Mode",
        log_scale=True,
        file_name="locking_mode_transition_frequency_comparison"
    )

    plot_state_machine_frequency_comparison_boxplot(
        categories,
        element_model_data,
        target_model_name,
        "Locking Mode",
        log_scale=True,
        file_name="locking_mode_state_machine_transition_frequency_comparison"
    )


def run_test_suite(model_name: str, include_logging=False):
    """Run the default test suite for the model with the given name."""
    default_model_data = import_model_results(f"{model_name}[T=60s]", include_logging=include_logging)

    # TODO: link to the right models.
    # Create a master table of all desired options.
    model_entries = {
        "Default": default_model_data,
        "Element": default_model_data,
        "Variable": default_model_data,
        "Statement": default_model_data,
        "No Locks": default_model_data,
        "Random + Det": default_model_data,
        "Sequential + Det": default_model_data,
        "Random": default_model_data,
        "Sequential": default_model_data,
    }

    # Plot all the frequency tables.
    # "Element" and "Sequential + Det" are default options, and as such, do not require an additional table.
    plot_frequency_results_table(
        model_entries["Default"],
        model_name,
        category="Default",
        caption_addendum="The Java code has been generated using the default settings."
    )

    # Plot the tables for the locking mechanism.
    for category in ["Variable", "Statement", "No Locks"]:
        plot_frequency_results_table(
            model_entries[category],
            model_name,
            category=category,
            caption_addendum=f"The Java code has been generated with the `{category}' locking mode enabled."
        )

    # Plot the tables for the decision structures.
    for category in ["Random + Det", "Random", "Sequential"]:
        plot_frequency_results_table(
            model_entries[category],
            model_name,
            category=category,
            caption_addendum=f"The Java code has been generated with the `{category}' decision mode enabled."
        )

    # Create plots that compare the different locking mechanism configurations.
    analyze_locking_mechanism_performance(
        model_entries["Element"],
        model_entries["Variable"],
        model_entries["Statement"],
        model_entries["No Locks"],
        model_name
    )

    # Create plots that compare the different decision structure configurations.
    analyze_decision_structure_performance(
        model_entries["Random + Det"],
        model_entries["Sequential + Det"],
        model_entries["Random"],
        model_entries["Sequential"],
        model_name
    )

    if include_logging:
        # Plot a default logging frequency table.
        plot_frequency_results_table(
            model_entries["Default"],
            model_name,
            target_type="logging",
            category="Logging",
            caption_addendum="The Java code has been generated using the default settings and the results have been "
                             "attained through logging-based measurements."
        )

        # Plot a table with throughput summary data.
        plot_throughput_information_table(model_entries["Default"], model_name)

        # Plot comparisons between counting and logging data.
        analyze_counting_logging_distribution(model_entries["Default"], model_name)

        # Use the default model to plot log message throughput information.
        plot_throughput_reports(model_entries["Default"], model_name)


def analyze_elevator_models():
    """Analyze the elevator model instances."""
    run_test_suite("Elevator", include_logging=True)


def analyze_synthetic_test_tokens_models():
    """Analyze the elevator model instances."""
    run_test_suite("SyntheticTestTokens", include_logging=True)


if __name__ == "__main__":
    analyze_elevator_models()
    analyze_synthetic_test_tokens_models()
