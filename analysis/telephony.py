from typing import Dict

from visualization.plots import plot_frequency_results_table, plot_throughput_information_table, \
    plot_throughput_reports, plot_transition_frequency_comparison_boxplot, \
    plot_state_machine_frequency_comparison_boxplot


def run_telephony_logging_measurement_analysis(
        model_data,
        target_model_name: str,
        category: str = None,
        caption_addendum: str = None
):
    """
    Analyze the transition distributions of the counting and logging measurements of a model for the telephony model.
    """
    counting_columns = model_data["message_frequency"]["global"]["targets"]["counting"]
    logging_columns = model_data["message_frequency"]["global"]["targets"]["logging"]
    counting_frequency_data = model_data["message_frequency"]["global"]["table"][counting_columns]
    logging_frequency_data = model_data["message_frequency"]["global"]["table"][logging_columns]

    categories = {
        "Counting": counting_frequency_data,
        "Logging": logging_frequency_data,
    }

    plot_state_machine_frequency_comparison_boxplot(
        categories,
        model_data,
        target_model_name,
        "Measurement Type",
        file_name="counting_logging_state_machine_transition_frequency_comparison",
        category=category,
        caption_addendum=caption_addendum
    )

    # The model contains too much data to present in one table or imagine. Hence, split by state machine.
    for state_machine in ["User_0", "User_1", "User_2", "User_3"]:
        # Filter out the right data.
        filtered_categories = {key: df[df.index.str.contains(state_machine)] for key, df in categories.items()}

        plot_transition_frequency_comparison_boxplot(
            filtered_categories,
            model_data,
            target_model_name,
            "Measurement Type",
            file_name=f"counting_logging_transition_frequency_comparison",
            category=f"{category}, {state_machine}" if category is not None else state_machine,
            caption_addendum=caption_addendum
        )

        # Plot a default logging frequency table.
        plot_frequency_results_table(
            model_data,
            target_model_name,
            target_type="logging",
            category=f"Logging",
            caption_addendum=f"The Java code has been generated with the `{category}' decision enabled and "
                             f"the results have been attained through logging-based measurements.",
            target_state_machine=state_machine
        )


def run_telephony_deterministic_structures_analysis(
        random_pick_structures_model_data,
        sequential_structures_model_data,
        random_pick_model_data,
        sequential_model_data,
        target_model_name: str,
        include_sequential: bool = True
):
    """Analyze the transition distributions of the different locking modes for the telephony model."""
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

    plot_state_machine_frequency_comparison_boxplot(
        categories,
        random_pick_model_data,
        target_model_name,
        "Decision Mode",
        file_name="decision_mode_state_machine_transition_frequency_comparison"
    )

    # The model contains too much data to present in one table or imagine. Hence, split by state machine.
    for state_machine in ["User_0", "User_1", "User_2", "User_3"]:
        # Filter out the right data.
        filtered_categories = {key: df[df.index.str.contains(state_machine)] for key, df in categories.items()}

        plot_transition_frequency_comparison_boxplot(
            filtered_categories,
            random_pick_model_data,
            target_model_name,
            "Decision Mode",
            file_name=f"decision_mode_transition_frequency_comparison",
            category=state_machine,
        )

        for category, model_entry in {
            "Sequential + Det": sequential_structures_model_data,
            "Random": random_pick_model_data,
            "Sequential": sequential_model_data
        }.items():
            plot_frequency_results_table(
                model_entry,
                target_model_name,
                category=category,
                caption_addendum=f"The Java code has been generated with the `{category}' decision mode enabled.",
                target_state_machine=state_machine
            )


def run_telephony_locking_mechanism_analysis(
        element_model_data,
        variable_model_data,
        statement_model_data,
        target_model_name: str,
        base_category: str,
):
    """Analyze the transition distributions of the different locking modes for the telephony model."""
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

    plot_state_machine_frequency_comparison_boxplot(
        categories,
        element_model_data,
        target_model_name,
        "Locking Mode",
        file_name="locking_mode_state_machine_transition_frequency_comparison",
        category=base_category
    )

    # The model contains too much data to present in one table or imagine. Hence, split by state machine.
    for state_machine in ["User_0", "User_1", "User_2", "User_3"]:
        # Filter out the right data.
        filtered_categories = {key: df[df.index.str.contains(state_machine)] for key, df in categories.items()}

        plot_transition_frequency_comparison_boxplot(
            filtered_categories,
            element_model_data,
            target_model_name,
            "Locking Mode",
            file_name=f"locking_mode_transition_frequency_comparison",
            category=f"{base_category}, {state_machine}",
        )

        for category, model_entry in {
            "Variable": variable_model_data,
            "Statement": statement_model_data
        }.items():
            plot_frequency_results_table(
                model_entry,
                target_model_name,
                category=f"{base_category}, {category}",
                caption_addendum=f"The Java code has been generated with the `{base_category}' "
                                 f"decision mode and `{category}' locking mode enabled.",
                target_state_machine=state_machine
            )


# TODO: Ensure that all of the captions are correct.

def run_telephony_analysis(
        model_name: str,
        model_entries: Dict[str, Dict],
        include_logging: bool = False,
        analyze_deterministic_structures: bool = True,
        analyze_locking_mechanism: bool = True,
        no_deterministic_structures: bool = True
):
    """Run altered analytics for the telephony model."""
    # Determine which setting has been used as the default.
    base_category = "Random" if no_deterministic_structures else "Random + Det"

    # Plot a separate frequency table for all state machines to ensure results fit on a page.
    for state_machine in ["User_0", "User_1", "User_2", "User_3"]:
        plot_frequency_results_table(
            model_entries["Default"],
            model_name,
            category=base_category,
            caption_addendum=f"The Java code has been generated with the `{base_category}' decision mode enabled.",
            target_state_machine=state_machine
        )

    # Plot the tables for the locking mechanism.
    if analyze_locking_mechanism:
        # Create plots that compare the different locking mechanism configurations.
        run_telephony_locking_mechanism_analysis(
            model_entries["Element"],
            model_entries["Variable"],
            model_entries["Statement"],
            model_name,
            base_category
        )

    # Plot the tables for the decision structures.
    if analyze_deterministic_structures:
        # Create plots that compare the different decision structure configurations.
        run_telephony_deterministic_structures_analysis(
            model_entries["Random + Det"],
            model_entries["Sequential + Det"],
            model_entries["Random"],
            model_entries["Sequential"],
            model_name,
            include_sequential=False
        )

    if include_logging:
        # Plot comparisons between counting and logging data.
        run_telephony_logging_measurement_analysis(model_entries["Default"], model_name, category=base_category)

        # Plot a table with throughput summary data.
        plot_throughput_information_table(model_entries["Default"], model_name, category=base_category)

        # Use the default model to plot log message throughput information.
        plot_throughput_reports(model_entries["Default"], model_name, category=base_category)
