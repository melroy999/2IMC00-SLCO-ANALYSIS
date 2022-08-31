from typing import Dict, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from visualization.figures.barplot import plot_two_column_barplot, set_numeric_margins, \
    set_percentage_margins, set_logarithmic_margins, create_ordering_mask

# Constants.
from visualization.figures.heatmap import plot_pcolormesh
from visualization.figures.table import render_table

plot_width = 10


def remove_subject_label_suffix(x):
    """Remove the message type from the subject label."""
    return ".".join(x.split(".")[:-1])


def get_revised_index_names(data: pd.DataFrame, abbreviation: Dict) -> Dict:
    """Rename the subject labels to be more informative by including states and more descriptive indices."""
    renaming_dict = dict()
    for v in data.index:
        if "." not in v:
            renaming_dict[v] = v
        else:
            sm_name, identifier = v.split(".")
            target = abbreviation[identifier]
            if identifier.startswith("D"):
                renaming_dict[v] = f"{sm_name}.{target['source']}"
            else:
                renaming_dict[v] = \
                    f"{sm_name}.{target['source']}.{target['target']} {target['name'].split('|')[0].strip()}"
    return renaming_dict


def extract_frequency_data(frequency_data: pd.DataFrame, model_data: Dict, _type: str = None):
    """
    Extract the data required to plot the number of transitions and the number of successful transitions side by side
    in a boxplot.
    """
    # Plot the number of successful transitions and the percentage of successful transitions side by side in a boxplot.
    opening_frequency_data = frequency_data[frequency_data.index.str.endswith("O")]
    success_frequency_data = frequency_data[frequency_data.index.str.endswith("S")]

    # Change the indices to no longer include opening/success suffixes.
    opening_frequency_data = opening_frequency_data.rename(index=remove_subject_label_suffix)
    success_frequency_data = success_frequency_data.rename(index=remove_subject_label_suffix)

    # Preprocess the index names to be more meaningful (add states and different transition identifiers).
    opening_frequency_data = opening_frequency_data.rename(
        index=get_revised_index_names(opening_frequency_data, model_data["model"]["abbreviation_to_target"])
    )
    success_frequency_data = success_frequency_data.rename(
        index=get_revised_index_names(success_frequency_data, model_data["model"]["abbreviation_to_target"])
    )

    # Transpose and flatten the two tables such that they can be plotted as box plots.
    opening_frequency_data = opening_frequency_data.transpose().melt()
    success_frequency_data = success_frequency_data.transpose().melt()

    # Rename for consistency.
    opening_frequency_data.rename(columns={"variable": "message"}, inplace=True)
    success_frequency_data.rename(columns={"variable": "message"}, inplace=True)

    # Set a type if given.
    if _type is not None:
        opening_frequency_data["type"] = _type
        success_frequency_data["type"] = _type

    # Return the two columns.
    return opening_frequency_data, success_frequency_data


def extract_success_data(frequency_data: pd.DataFrame, model_data: Dict, _type: str = None):
    """
    Extract the data required to plot the number of successful transitions and the percentage of successful transitions
    side by side in a boxplot.
    """
    # Plot the number of successful transitions and the percentage of successful transitions side by side in a boxplot.
    opening_frequency_data = frequency_data[frequency_data.index.str.endswith("O")]
    success_frequency_data = frequency_data[frequency_data.index.str.endswith("S")]

    # Change the indices to no longer include opening/success suffixes.
    opening_frequency_data = opening_frequency_data.rename(index=remove_subject_label_suffix)
    success_frequency_data = success_frequency_data.rename(index=remove_subject_label_suffix)
    success_ratio_data = success_frequency_data / opening_frequency_data

    # Preprocess the index names to be more meaningful (add states and different transition identifiers).
    success_frequency_data = success_frequency_data.rename(
        index=get_revised_index_names(success_frequency_data, model_data["model"]["abbreviation_to_target"])
    )
    success_ratio_data = success_ratio_data.rename(
        index=get_revised_index_names(success_ratio_data, model_data["model"]["abbreviation_to_target"])
    )

    # Transpose and flatten the two tables such that they can be plotted as box plots.
    success_frequency_data = success_frequency_data.transpose().melt()
    success_ratio_data = success_ratio_data.transpose().melt()

    # Rename for consistency.
    success_frequency_data.rename(columns={"variable": "message"}, inplace=True)
    success_ratio_data.rename(columns={"variable": "message"}, inplace=True)

    # Set a type if given.
    if _type is not None:
        success_frequency_data["type"] = _type
        success_ratio_data["type"] = _type

    # Return the two columns.
    return success_frequency_data, success_ratio_data


def plot_transition_frequency_boxplot(frequency_data: pd.DataFrame, model_data: Dict, title: str):
    """Plot the transition frequencies recorded in the model."""
    # Plot the number of successful transitions and the percentage of successful transitions side by side in a boxplot.
    success_frequency_data, success_ratio_data = extract_success_data(frequency_data, model_data)

    # Create two sub-figures.
    root_fig = plt.figure(figsize=(plot_width, 6), dpi=300)
    plot_two_column_barplot(
        success_frequency_data, success_ratio_data, root_fig,
        title_left="Success Count",
        title_right="Success/Total Ratio",
        y_label="Subject",
        x_label_left="Count",
        x_label_right="Percentage",
        left_margin_function=set_numeric_margins,
        right_margin_function=set_percentage_margins
    )
    root_fig.suptitle(title, y=1.00)

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    plt.show()

    # with plt.rc_context({
    #     # "pgf.texsystem": "pdflatex",
    #     # 'font.family': 'serif',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # }):
    #     import tikzplotlib
    #     # tikzplotlib.save("comparison.tex", dpi=300)
    #     plt.savefig("comparison.pgf")
    #
    # plt.show()


def plot_transition_frequency_comparison_boxplot(
        frequency_data: Dict[str, pd.DataFrame],
        model_data: Dict,
        title: str,
        y_scale: int = 10,
        log_scale: bool = False,
        legend_title: str = "Configuration"
):
    """Plot the transition frequencies recorded in the given models."""
    # Plot the number of successful transitions and the percentage of successful transitions side by side in a boxplot.
    target_data_entries = [extract_success_data(v, model_data, i) for i, v in frequency_data.items()]
    success_frequency_data = pd.concat([v[0] for v in target_data_entries])
    success_ratio_data = pd.concat([v[1] for v in target_data_entries])

    # Create two sub-figures.
    root_fig = plt.figure(figsize=(plot_width, y_scale), dpi=300)
    plot_two_column_barplot(
        success_frequency_data, success_ratio_data, root_fig,
        title_left="Success Count",
        title_right="Success/Total Ratio",
        y_label="Subject",
        x_label_left="Count (Logarithmic Scale)" if log_scale else "Count",
        x_label_right="Percentage",
        hue="type",
        left_margin_function=set_logarithmic_margins if log_scale else set_numeric_margins,
        right_margin_function=set_percentage_margins
    )
    root_fig.suptitle(title, y=1.00)

    # Put the legend outside of the plot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=legend_title)

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    with plt.rc_context({
        'text.usetex': True,
        'pgf.rcfonts': False,
    }):
        plt.savefig("comparison.pgf")

    plt.show()


def get_state_machine_mask(v):
    """Mask the given string to exclude the decision node identifier contained therein."""
    state_machine, _, _type = v.split(".")
    return f"{state_machine}.{_type}"


def extract_state_machine_level_data(frequency_data):
    """Create a dataframe that contains the frequency data grouped by state machine."""
    # Filter out all decision node data.
    decision_node_frequency_data = frequency_data[frequency_data.index.str.contains(".T")]
    state_machine_frequency_data = decision_node_frequency_data.groupby(get_state_machine_mask, as_index=True).sum()
    return state_machine_frequency_data


def plot_state_machine_frequency_boxplot(frequency_data: pd.DataFrame, model_data: Dict, title: str):
    """Plot the state machine frequencies recorded in the model."""
    # Plot the number of transitions and the successful transitions side by side in a boxplot grouped by state machine.
    sm_frequency_data = extract_state_machine_level_data(frequency_data)
    frequency_data, success_frequency_data = extract_frequency_data(sm_frequency_data, model_data)

    # Create two sub-figures.
    root_fig = plt.figure(figsize=(plot_width, 3), dpi=300)
    plot_two_column_barplot(
        frequency_data, success_frequency_data, root_fig,
        title_left="Total Count",
        title_right="Success Count",
        y_label="State Machine",
        x_label_left="Count",
        x_label_right="Count",
        left_margin_function=set_numeric_margins,
        right_margin_function=set_numeric_margins
    )
    root_fig.suptitle(title, y=1.00)

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    plt.show()


def plot_state_machine_frequency_comparison_boxplot(
        frequency_data: Dict[str, pd.DataFrame],
        model_data: Dict,
        title: str,
        y_scale: int = 4,
        log_scale: bool = False,
        legend_title: str = "Configuration"
):
    """Plot the state machine frequencies recorded in the given models."""
    # Plot the number of transitions and the successful transitions side by side in a boxplot grouped by state machine.
    sm_frequency_data = {i: extract_state_machine_level_data(v) for i, v in frequency_data.items()}
    target_data_entries = [extract_frequency_data(v, model_data, i) for i, v in sm_frequency_data.items()]
    frequency_data = pd.concat([v[0] for v in target_data_entries])
    success_frequency_data = pd.concat([v[1] for v in target_data_entries])

    # Create two sub-figures.
    root_fig = plt.figure(figsize=(plot_width, y_scale), dpi=300)
    plot_two_column_barplot(
        frequency_data, success_frequency_data, root_fig,
        title_left="Total Count",
        title_right="Success Count",
        y_label="State Machine",
        x_label_left="Count (Logarithmic Scale)" if log_scale else "Count",
        x_label_right="Count (Logarithmic Scale)" if log_scale else "Count",
        hue="type",
        left_margin_function=set_logarithmic_margins if log_scale else set_numeric_margins,
        right_margin_function=set_logarithmic_margins if log_scale else set_numeric_margins,
    )
    root_fig.suptitle(title, y=1.00)

    # Put the legend outside of the plot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=legend_title)

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    plt.show()


def plot_throughput_sum(model_data: Dict, run_id: int, dimensions: Tuple[int, int], max_value: int):
    """Plot a color mesh depicting the per file global throughput data for the given model run."""
    root_fig = plt.figure(figsize=(10, 3.5), dpi=300)
    plot_data = model_data["log_frequency"]["files"]["sum"][run_id].transpose()
    plot_pcolormesh(
        plot_data,
        root_fig,
        title=f"Logging Throughput Sum ({model_data['model']['id']}, Run \\#{run_id})",
        x_label="Timestamp (ms)",
        y_label="File Number",
        c_bar_label="Frequency",
        data_dimensions=dimensions,
        input_pcolormesh_options={
            "vmax": max_value
        }
    )
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_throughput_difference(model_data: Dict, run_id: int, dimensions: Tuple[int, int], max_value: int):
    """Plot a color mesh depicting the sum difference to the row minimum throughput for the given model run."""
    root_fig = plt.figure(figsize=(10, 3.5), dpi=300)
    plot_data = model_data["log_frequency"]["files"]["difference"][run_id].transpose()
    plot_pcolormesh(
        plot_data,
        root_fig,
        title=f"Logging Throughput Difference ({model_data['model']['id']}, Run \\#{run_id})",
        x_label="Timestamp (ms)",
        y_label="File Number",
        c_bar_label="Sum Difference To Row Minimum",
        data_dimensions=dimensions,
        input_pcolormesh_options={
            "vmax": max_value
        }
    )
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_throughput_report(
        model_data: Dict, run_id: int, dimensions: Tuple[int, int], max_sum: int, max_difference: int
):
    """Plot a logging throughput report for the given model run."""
    plot_throughput_sum(model_data, run_id, dimensions, max_sum)
    plot_throughput_difference(model_data, run_id, dimensions, max_difference)


def plot_throughput_reports(model_data: Dict):
    """Plot a logging throughput report for the given model results."""
    max_nr_of_timestamps = max(x.shape[0] for x in model_data["log_frequency"]["files"]["sum"])
    max_nr_of_files = max(x.shape[1] for x in model_data["log_frequency"]["files"]["sum"])
    data_dimensions = (max_nr_of_files, max_nr_of_timestamps)

    max_sum = int(max(np.nanmax(v.to_numpy()) for v in model_data["log_frequency"]["files"]["sum"]))
    max_difference = int(max(np.nanmax(v.to_numpy()) for v in model_data["log_frequency"]["files"]["difference"]))

    for i, _ in enumerate(model_data["log_frequency"]["files"]["sum"]):
        plot_throughput_report(model_data, i, data_dimensions, max_sum, max_difference)


def format_index(v):
    """Format the given target."""
    if v.count(".") < 2:
        return f"\\\\[-8pt]\\textbf{{{v}}}"
    else:
        return f"\\hspace{{3mm}}{v}"


def plot_frequency_results_table(input_data: pd.DataFrame, model_data: Dict):
    """Plot the results rendered within the given table."""
    # Create a copy, such that the original remains unaltered.
    input_data = input_data.copy()

    # Calculate the mean and standard deviation for every column.
    frequency_data, _ = extract_frequency_data(input_data, model_data)
    success_frequency_data, success_ratio_data = extract_success_data(input_data, model_data)

    frequency_statistics = frequency_data.groupby(["message"]).agg(
        mean=("value", "mean"), std=("value", "std")
    )
    frequency_statistics.rename(columns={"mean": "$\\mu(t)$", "std": "$\\sigma(t)$"}, inplace=True)
    success_frequency_statistics = success_frequency_data.groupby(["message"]).agg(
        mean=("value", "mean"), std=("value", "std")
    )
    success_frequency_statistics.rename(columns={"mean": "$\\mu(st)$", "std": "$\\sigma(st)$"}, inplace=True)
    success_ratio_statistics = success_ratio_data.groupby(["message"]).agg(
        mean=("value", "mean"), std=("value", "std")
    )
    success_ratio_statistics.rename(columns={"mean": "$\\mu(sr)$", "std": "$\\sigma(sr)$"}, inplace=True)

    result_table = pd.concat(
        [frequency_statistics, success_frequency_statistics, success_ratio_statistics], axis=1, join="inner"
    )
    result_table = result_table.reindex(list(sorted(set(result_table.index), key=create_ordering_mask)))
    result_table.index = result_table.index.map(format_index)

    latex_code = render_table(result_table)
    latex_code = latex_code.replace("{}", "Target")
    latex_code_lines = latex_code.splitlines(keepends=True)
    latex_code_lines[3] = latex_code_lines[2]
    latex_code_lines[2] = \
        f"\\multicolumn{{{latex_code_lines[3].count('&') + 1}}}{{c}}{{Performance results for target model `\\texttt{{CounterDistributorExact.i}}' $(n={len(input_data.columns)}, t=30)$}} \\\\[2mm]\n"
    latex_code_lines[5] = latex_code_lines[5][8:]
    latex_code = ''.join(latex_code_lines)

    latex_code = encapsulate_tabular_with_table(latex_code)

    print(latex_code)


def encapsulate_tabular_with_table(latex_code: str) -> str:
    """Encapsulate the given tabular object with a table object."""
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{ %",
        latex_code.strip(),
        "}",
        "\\caption{}",
        "\\label{}",
        "\\end{table}"
    ]
    return "\n".join(lines)
