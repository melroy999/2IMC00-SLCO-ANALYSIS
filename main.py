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
        "logging": []
    }

    if include_logging:
        model_results |= {
            "logging": import_logging_results(model_path)
        }

    # Preprocess the results.
    return preprocess_model_results(model_results, target_model, include_logging)


def analyze_counting_logging_distribution(
        model_data,
        target_model_name: str,
        category: str = None,
        caption_addendum: str = None
):
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
        file_name="counting_logging_transition_frequency_comparison",
        category=category,
        caption_addendum=caption_addendum
    )

    plot_state_machine_frequency_comparison_boxplot(
        categories,
        model_data,
        target_model_name,
        "Measurement Type",
        file_name="counting_logging_state_machine_transition_frequency_comparison",
        category=category,
        caption_addendum=caption_addendum
    )


def analyze_decision_structure_performance(
        random_pick_structures_model_data,
        sequential_structures_model_data,
        random_pick_model_data,
        sequential_model_data,
        target_model_name: str,
        include_sequential: bool = True
):
    """Analyze the transition distributions of the different locking modes."""
    counting_columns = random_pick_structures_model_data["message_frequency"]["global"]["targets"]["counting"]
    random_pick_structures_frequency_data = \
        random_pick_structures_model_data["message_frequency"]["global"]["table"][counting_columns]

    counting_columns = random_pick_model_data["message_frequency"]["global"]["targets"]["counting"]
    random_pick_frequency_data = random_pick_model_data["message_frequency"]["global"]["table"][counting_columns]

    if include_sequential:
        counting_columns = sequential_structures_model_data["message_frequency"]["global"]["targets"]["counting"]
        sequential_structures_frequency_data = \
            sequential_structures_model_data["message_frequency"]["global"]["table"][counting_columns]

        counting_columns = sequential_model_data["message_frequency"]["global"]["targets"]["counting"]
        sequential_frequency_data = sequential_model_data["message_frequency"]["global"]["table"][counting_columns]

        categories = {
            "Sequential + Det": sequential_structures_frequency_data,
            "Random + Det": random_pick_structures_frequency_data,
            "Sequential": sequential_frequency_data,
            "Random": random_pick_frequency_data,
        }
    else:
        categories = {
            "Random + Det": random_pick_structures_frequency_data,
            "Random": random_pick_frequency_data,
        }

    plot_transition_frequency_comparison_boxplot(
        categories,
        random_pick_model_data,
        target_model_name,
        "Decision Mode",
        file_name="decision_mode_transition_frequency_comparison"
    )

    plot_state_machine_frequency_comparison_boxplot(
        categories,
        random_pick_model_data,
        target_model_name,
        "Decision Mode",
        file_name="decision_mode_state_machine_transition_frequency_comparison"
    )


def analyze_locking_mechanism_performance(
        element_model_data,
        variable_model_data,
        statement_model_data,
        no_locks_model_data,
        target_model_name: str,
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
        # TODO: Removed, due to the results not providing worthwhile information.
        # "No Locks": no_locks_frequency_data,
    }

    plot_transition_frequency_comparison_boxplot(
        categories,
        element_model_data,
        target_model_name,
        "Locking Mode",
        file_name="locking_mode_transition_frequency_comparison"
    )

    plot_state_machine_frequency_comparison_boxplot(
        categories,
        element_model_data,
        target_model_name,
        "Locking Mode",
        file_name="locking_mode_state_machine_transition_frequency_comparison"
    )


def get_target_file(model_name: str, configuration: List[str]) -> str:
    """Get the file associated with the given options."""
    return f"{model_name}[{','.join(sorted(v for v in configuration if v != ''))}]"


def run_default_analysis_suite(
        model_name: str,
        include_logging: bool = False,
        analyze_deterministic_structures: bool = True,
        analyze_locking_mechanism: bool = True
):
    """Run the default test suite for the model with the given name."""
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
            "No Locks": import_model_results(get_target_file(model_name, ["NL", "T=30s"])),
        }
    if analyze_deterministic_structures:
        model_entries |= {
            "Random + Det": import_model_results(get_target_file(model_name, ["URP", "T=30s"])),
            "Sequential + Det": default_model_data,
            "Random": import_model_results(get_target_file(model_name, ["NDS", "URP", "T=30s"])),
            "Sequential": import_model_results(get_target_file(model_name, ["NDS", "T=30s"])),
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
    if analyze_locking_mechanism:
        for category in ["Variable", "Statement", "No Locks"]:
            plot_frequency_results_table(
                model_entries[category],
                model_name,
                category=category,
                caption_addendum=f"The Java code has been generated with the `{category}' locking mode enabled."
            )

        # Create plots that compare the different locking mechanism configurations.
        analyze_locking_mechanism_performance(
            model_entries["Element"],
            model_entries["Variable"],
            model_entries["Statement"],
            model_entries["No Locks"],
            model_name
        )

    # Plot the tables for the decision structures.
    if analyze_deterministic_structures:
        for category in ["Random + Det", "Random", "Sequential"]:
            plot_frequency_results_table(
                model_entries[category],
                model_name,
                category=category,
                caption_addendum=f"The Java code has been generated with the `{category}' decision mode enabled."
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


def run_telephony_analysis_suite(
        model_name: str = "Telephony",
        include_logging: bool = False,
        analyze_deterministic_structures: bool = True,
        analyze_locking_mechanism: bool = True
):
    """Run the default test suite for the telephony model."""
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
            "Variable": import_model_results(get_target_file(model_name, ["URP", "LA", "T=30s"])),
            "Statement": import_model_results(get_target_file(model_name, ["URP", "SLL", "T=30s"])),
            "No Locks": import_model_results(get_target_file(model_name, ["URP", "NL", "T=30s"])),
        }
    if analyze_deterministic_structures:
        model_entries |= {
            "Random + Det": default_model_data,
            "Sequential + Det": import_model_results(get_target_file(model_name, ["T=30s"])),
            "Random": import_model_results(get_target_file(model_name, ["NDS", "URP", "T=30s"])),
            "Sequential": import_model_results(get_target_file(model_name, ["NDS", "T=30s"])),
        }

    # Plot the tables for the locking mechanism.
    if analyze_locking_mechanism:
        # Create plots that compare the different locking mechanism configurations.
        analyze_locking_mechanism_performance(
            model_entries["Element"],
            model_entries["Variable"],
            model_entries["Statement"],
            model_entries["No Locks"],
            model_name
        )

    # Plot the tables for the decision structures.
    if analyze_deterministic_structures:
        # Create plots that compare the different decision structure configurations.
        analyze_decision_structure_performance(
            model_entries["Random + Det"],
            model_entries["Sequential + Det"],
            model_entries["Random"],
            model_entries["Sequential"],
            model_name
        )

    if include_logging:
        # Plot comparisons between counting and logging data.
        analyze_counting_logging_distribution(model_entries["Default"], model_name)


def analyze_counter_distributor_models():
    """Analyze the counter distributor model instances."""
    run_default_analysis_suite("CounterDistributor", include_logging=True)


def analyze_elevator_models():
    """Analyze the elevator model instances."""
    run_default_analysis_suite("Elevator", include_logging=True)


def analyze_telephony_models():
    """Analyze the telephony model instances."""
    run_telephony_analysis_suite("Telephony", include_logging=True)


def analyze_toads_and_frogs_models():
    """Analyze the telephony model instances."""
    run_default_analysis_suite("ToadsAndFrogs", include_logging=True)

    # Plot comparisons between counting and logging data.
    target_model = import_model_results(get_target_file("ToadsAndFrogs", ["URP", "T=30s"]), include_logging=True)
    analyze_counting_logging_distribution(
        target_model, "ToadsAndFrogs", category="Random",
        caption_addendum=f"The Java code has been generated with the `Random' decision mode enabled."
    )


def analyze_tokens_models():
    """Analyze the tokens model instances."""
    run_default_analysis_suite("Tokens", include_logging=True)


if __name__ == "__main__":
    # analyze_counter_distributor_models()
    # analyze_elevator_models()
    analyze_telephony_models()
    # analyze_toads_and_frogs_models()
    # analyze_tokens_models()
