from typing import Dict

from visualization.plots import plot_frequency_results_table, plot_throughput_information_table, \
    plot_throughput_reports, plot_transition_frequency_comparison_boxplot, \
    plot_state_machine_frequency_comparison_boxplot


def run_default_logging_measurement_analysis(
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

    # Plot a default logging frequency table.
    plot_frequency_results_table(
        model_data,
        target_model_name,
        target_type="logging",
        category="Logging",
        caption_addendum="The Java code has been generated using the default settings and the results have been "
                         "attained through logging-based measurements."
    )


def run_default_deterministic_structures_analysis(
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

    for category, model_entry in {
        "Random + Det": random_pick_structures_model_data,
        "Random": random_pick_model_data,
        "Sequential": sequential_model_data
    }.items():
        plot_frequency_results_table(
            model_entry,
            target_model_name,
            category=category,
            caption_addendum=f"The Java code has been generated with the `{category}' decision mode enabled."
        )


def run_default_locking_mechanism_analysis(
        element_model_data,
        variable_model_data,
        statement_model_data,
        target_model_name: str,
):
    """Analyze the transition distributions of the different locking modes."""
    counting_columns = element_model_data["message_frequency"]["global"]["targets"]["counting"]
    element_frequency_data = element_model_data["message_frequency"]["global"]["table"][counting_columns]
    counting_columns = variable_model_data["message_frequency"]["global"]["targets"]["counting"]
    variable_frequency_data = variable_model_data["message_frequency"]["global"]["table"][counting_columns]
    counting_columns = statement_model_data["message_frequency"]["global"]["targets"]["counting"]
    statement_frequency_data = statement_model_data["message_frequency"]["global"]["table"][counting_columns]

    categories = {
        "Element": element_frequency_data,
        "Variable": variable_frequency_data,
        "Statement": statement_frequency_data
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

    for category, model_entry in {
        "Variable": variable_model_data,
        "Statement": statement_model_data
    }.items():
        plot_frequency_results_table(
            model_entry,
            target_model_name,
            category=category,
            caption_addendum=f"The Java code has been generated with the `{category}' locking mode enabled."
        )


def run_default_analysis(
        model_name: str,
        model_entries: Dict[str, Dict],
        include_logging: bool = False,
        analyze_deterministic_structures: bool = True,
        analyze_locking_mechanism: bool = True
):
    """Run the default analytics for the model with the given name."""
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
        # Create plots that compare the different locking mechanism configurations.
        run_default_locking_mechanism_analysis(
            model_entries["Element"],
            model_entries["Variable"],
            model_entries["Statement"],
            model_name
        )

    # Plot the tables for the decision structures.
    if analyze_deterministic_structures:
        # Create plots that compare the different decision structure configurations.
        run_default_deterministic_structures_analysis(
            model_entries["Random + Det"],
            model_entries["Sequential + Det"],
            model_entries["Random"],
            model_entries["Sequential"],
            model_name
        )

    if include_logging:
        # Plot comparisons between counting and logging data.
        run_default_logging_measurement_analysis(model_entries["Default"], model_name)

        # Plot a table with throughput summary data.
        plot_throughput_information_table(model_entries["Default"], model_name)

        # Use the default model to plot log message throughput information.
        plot_throughput_reports(model_entries["Default"], model_name)
