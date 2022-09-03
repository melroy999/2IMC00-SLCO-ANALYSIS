import json
from heapq import heappush, heappushpop
from os import path, listdir
from typing import Dict, List, Set, Tuple

import sympy as sympy

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


def run_default_test_suite(
        model_name: str,
        include_logging: bool = False,
        analyze_deterministic_structures: bool = True,
        analyze_locking_mechanism: bool = True
):
    """Run the default test suite for the model with the given name."""
    default_model_data = import_model_results(f"{model_name}[T=30s]", include_logging=include_logging)

    # Create a master table of all desired options.
    model_entries = {
        "Default": default_model_data
    }
    if analyze_locking_mechanism:
        model_entries |= {
            "Element": default_model_data,
            "Variable": import_model_results(f"{model_name}[LA,T=30s]"),
            "Statement": import_model_results(f"{model_name}[SLL,T=30s]"),
            "No Locks": import_model_results(f"{model_name}[NL,T=30s]"),
        }
    if analyze_deterministic_structures:
        model_entries |= {
            "Random + Det": import_model_results(f"{model_name}[T=30s,URP]"),
            "Sequential + Det": default_model_data,
            "Random": import_model_results(f"{model_name}[NDS,T=30s,URP]"),
            "Sequential": import_model_results(f"{model_name}[NDS,T=30s]"),
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


def analyze_counter_distributor_models():
    """Analyze the counter distributor model instances."""
    run_default_test_suite("CounterDistributor", include_logging=True)


def analyze_elevator_models():
    """Analyze the elevator model instances."""
    run_default_test_suite("Elevator", include_logging=True)


def analyze_telephony_models():
    """Analyze the telephony model instances."""
    run_default_test_suite("Telephony", include_logging=True)


def analyze_toads_and_frogs_models():
    """Analyze the telephony model instances."""
    run_default_test_suite("ToadsAndFrogs", include_logging=True)


def analyze_tokens_models():
    """Analyze the tokens model instances."""
    run_default_test_suite("Tokens", include_logging=True)


# def full_period_generator(_z_prev: int, _a: int, _c: int, _m: int) -> Tuple[bool, List[int]]:
#     _sequence: List[int] = []
#     _visited: Set[int] = set()
#     for i in range(_m):
#         z = (_a * _z_prev + _c) % _m
#         _sequence.append(_z_prev)
#         _visited.add(_z_prev)
#         _z_prev = z
#         if _z_prev in _visited:
#             if len(_sequence) != _m - 1 or _z_prev != _sequence[0]:
#                 return False, _sequence
#             else:
#                 return True, _sequence


if __name__ == "__main__":
    analyze_counter_distributor_models()
    analyze_elevator_models()
    analyze_telephony_models()
    analyze_toads_and_frogs_models()
    analyze_tokens_models()

    # prime_numbers = list(sympy.sieve.primerange(1, 1000))
    # j = 0
    # results = []
    # for a in [1] + prime_numbers:
    #     for c in prime_numbers:
    #         for m in [sympy.nextprime(1000)]:
    #             result, sequence = full_period_generator(1, a, c, m)
    #             if result:
    #                 j += 1
    #                 # print(1, a, c, m, rating, sequence)
    #                 modulo_sequence = [v % 10 for v in sequence]
    #                 results.append((1, a, c, m, modulo_sequence))
    #
    # print(results[5665])
    # print(results[2008])
    # print(results[6890])
    #
    # print(full_period_generator(1, 641, 719, 1009))
    # print(full_period_generator(42, 193, 953, 1009))
    # print(full_period_generator(308, 811, 31, 1009))
