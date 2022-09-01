from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from visualization.figures.barplot import plot_two_column_barplot, set_numeric_margins, \
    set_percentage_margins, set_logarithmic_margins, create_ordering_mask

from visualization.figures.heatmap import plot_pcolormesh
from visualization.figures.table import render_tabular
from visualization.saving import save_plot_as_pgf, save_plot_as_png, save_table_as_tex, save_png_figure_as_tex, \
    save_pgf_figure_as_tex

# Constants.
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


def plot_transition_frequency_boxplot(
        frequency_data: pd.DataFrame,
        model_data: Dict,
        target_model: str,
        file_name: str = None
):
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

    n = len(frequency_data.columns)
    root_fig.suptitle(f"Successful Transition Executions ({target_model}, n={n}, t=30)", y=1.00)

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    if file_name is not None:
        caption = \
            "A bar plot that reports the number of successful executions and the percentage of successful " \
            f"executions for each decision node and transition in the target model \\texttt{{{target_model}}}. " \
            f"{get_sample_statement(n)}"

        target_pgf_figure = save_plot_as_pgf(target_model, file_name)
        save_plot_as_png(target_model, file_name)
        save_pgf_figure_as_tex(
            target_model,
            target_pgf_figure,
            caption,
            f"figure:{file_name}_{target_model.lower()}",
            f"{file_name}"
        )
    else:
        plt.show()

    plt.close("all")


def plot_transition_frequency_comparison_boxplot(
        frequency_data: Dict[str, pd.DataFrame],
        model_data: Dict,
        target_model: str,
        legend_title: str,
        file_name: str = None,
        y_scale: int = 10,
        log_scale: bool = False,

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

    n = min(len(df.columns) for df in frequency_data.values())
    root_fig.suptitle(f"Successful Transition Executions ({target_model}, n={n}, t=30)", y=1.00)

    # Put the legend outside of the plot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=legend_title)

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    if file_name is not None:
        log_scale_comment = ""
        if log_scale:
            log_scale_comment = "Observe that the success count is depicted in a " \
                                "logarithmic scale, due to the wide range of measured values. "
        caption = \
            "A bar plot that reports the number of successful executions and the percentage of successful " \
            f"executions for each decision node and transition in the target model \\texttt{{{target_model}}}, where " \
            f"the results are grouped by {legend_title.lower()}. {log_scale_comment}{get_sample_statement(n)}"

        target_pgf_figure = save_plot_as_pgf(target_model, file_name)
        save_plot_as_png(target_model, file_name)
        save_pgf_figure_as_tex(
            target_model,
            target_pgf_figure,
            caption,
            f"figure:{file_name}_{target_model.lower()}",
            f"{file_name}"
        )
    else:
        plt.show()

    plt.close("all")


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


def plot_state_machine_frequency_boxplot(
        frequency_data: pd.DataFrame,
        model_data: Dict,
        target_model: str,
        file_name: str = None
):
    """Plot the state machine frequencies recorded in the model."""
    # Plot the number of transitions and the successful transitions side by side in a boxplot grouped by state machine.
    sm_frequency_data = extract_state_machine_level_data(frequency_data)
    total_frequency_data, success_frequency_data = extract_frequency_data(sm_frequency_data, model_data)

    # Create two sub-figures.
    root_fig = plt.figure(figsize=(plot_width, 3), dpi=300)
    plot_two_column_barplot(
        total_frequency_data, success_frequency_data, root_fig,
        title_left="Total Count",
        title_right="Success Count",
        y_label="State Machine",
        x_label_left="Count",
        x_label_right="Count",
        left_margin_function=set_numeric_margins,
        right_margin_function=set_numeric_margins
    )

    n = len(frequency_data.columns)
    root_fig.suptitle(f"Transition Executions ({target_model}, n={n}, t=30)", y=1.00)

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    if file_name is not None:
        caption = \
            "A bar plot that reports the number of total and successful transition executions for each state " \
            f"machine in the target model \\texttt{{{target_model}}}. {get_sample_statement(n)}"

        target_pgf_figure = save_plot_as_pgf(target_model, file_name)
        save_plot_as_png(target_model, file_name)
        save_pgf_figure_as_tex(
            target_model,
            target_pgf_figure,
            caption,
            f"figure:{file_name}_{target_model.lower()}",
            f"{file_name}"
        )
    else:
        plt.show()

    plt.close("all")


def plot_state_machine_frequency_comparison_boxplot(
        frequency_data: Dict[str, pd.DataFrame],
        model_data: Dict,
        target_model: str,
        legend_title: str,
        file_name: str = None,
        y_scale: int = 4,
        log_scale: bool = False
):
    """Plot the state machine frequencies recorded in the given models."""
    # Plot the number of transitions and the successful transitions side by side in a boxplot grouped by state machine.
    sm_frequency_data = {i: extract_state_machine_level_data(v) for i, v in frequency_data.items()}
    target_data_entries = [extract_frequency_data(v, model_data, i) for i, v in sm_frequency_data.items()]
    total_frequency_data = pd.concat([v[0] for v in target_data_entries])
    success_frequency_data = pd.concat([v[1] for v in target_data_entries])

    # Create two sub-figures.
    root_fig = plt.figure(figsize=(plot_width, y_scale), dpi=300)
    plot_two_column_barplot(
        total_frequency_data, success_frequency_data, root_fig,
        title_left="Total Count",
        title_right="Success Count",
        y_label="State Machine",
        x_label_left="Count (Logarithmic Scale)" if log_scale else "Count",
        x_label_right="Count (Logarithmic Scale)" if log_scale else "Count",
        hue="type",
        left_margin_function=set_logarithmic_margins if log_scale else set_numeric_margins,
        right_margin_function=set_logarithmic_margins if log_scale else set_numeric_margins,
    )

    n = min(len(df.columns) for df in frequency_data.values())
    root_fig.suptitle(f"Transition Executions ({target_model}, n={n}, t=30)", y=1.00)

    # Put the legend outside of the plot.
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=legend_title)

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    if file_name is not None:
        log_scale_comment = ""
        if log_scale:
            log_scale_comment = "Observe that the total count and success count are depicted in a " \
                                "logarithmic scale, due to the wide range of measured values. "
        caption = \
            "A bar plot that reports the number of total and successful transition executions for each state " \
            f"machine in the target model \\texttt{{{target_model}}}, where the results are grouped by " \
            f"{legend_title.lower()}. {log_scale_comment}{get_sample_statement(n)}"

        target_pgf_figure = save_plot_as_pgf(target_model, file_name)
        save_plot_as_png(target_model, file_name)
        save_pgf_figure_as_tex(
            target_model,
            target_pgf_figure,
            caption,
            f"figure:{file_name}_{target_model.lower()}",
            f"{file_name}"
        )
    else:
        plt.show()

    plt.close("all")


def plot_throughput_sum(
        model_data: Dict,
        run_id: int,
        dimensions: Tuple[int, int],
        max_value: int,
        target_model: str,
        file_name: str = None
):
    """Plot a color mesh depicting the per file global throughput data for the given model run."""
    root_fig = plt.figure(figsize=(10, 3.5), dpi=300)
    plot_data = model_data["log_frequency"]["files"]["sum"][run_id].transpose()
    plot_pcolormesh(
        plot_data,
        root_fig,
        title=f"Logging Throughput Sum ({target_model}, Run {run_id}, t=30)",
        x_label="Timestamp (ms)",
        y_label="File Number",
        c_bar_label="Message Count",
        data_dimensions=dimensions,
        input_pcolormesh_options={
            "vmax": max_value
        }
    )
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    if file_name is not None:
        target_png_figure = save_plot_as_png(target_model, f"{file_name}_{run_id}")
        save_png_figure_as_tex(
            target_model,
            target_png_figure,
            f"A heatmap plot that reports on the total number of log messages per time unit (milliseconds) for each "
            f"log file generated during run {run_id} of target model \\texttt{{{target_model}}}. Each file has a "
            f"maximum size of 100MB. {get_sample_statement(None)}",
            f"figure:{file_name}_{target_model.lower()}_{run_id}",
            f"{file_name}_{run_id}"
        )
    else:
        plt.show()

    plt.close("all")


def plot_throughput_difference(
        model_data: Dict,
        run_id: int,
        dimensions: Tuple[int, int],
        max_value: int,
        target_model: str,
        file_name: str = None
):
    """Plot a color mesh depicting the sum difference to the row minimum throughput for the given model run."""
    root_fig = plt.figure(figsize=(10, 3.5), dpi=300)
    plot_data = model_data["log_frequency"]["files"]["difference"][run_id].transpose()
    plot_pcolormesh(
        plot_data,
        root_fig,
        title=f"Logging Throughput Difference ({target_model}, Run {run_id}, t=30)",
        x_label="Timestamp (ms)",
        y_label="File Number",
        c_bar_label="Sum Difference To Row Minimum",
        data_dimensions=dimensions,
        input_pcolormesh_options={
            "vmax": max_value
        }
    )
    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

    if file_name is not None:
        target_png_figure = save_plot_as_png(target_model, f"{file_name}_{run_id}")
        save_png_figure_as_tex(
            target_model,
            target_png_figure,

            f"A heatmap plot that reports on the sum difference to the row minimum of the state machines' log message "
            f"counts per time unit (milliseconds) for each log file generated during run {run_id} of target model "
            f"\\texttt{{{target_model}}}. Each file has a maximum size of 100MB. "
            f"{get_sample_statement(None)}",
            f"figure:{file_name}_{target_model.lower()}_{run_id}",
            f"{file_name}_{run_id}"
        )
    else:
        plt.show()

    plt.close("all")


def plot_throughput_report(
        model_data: Dict, run_id: int, dimensions: Tuple[int, int], max_sum: int, max_difference: int, target_model: str
):
    """Plot a logging throughput report for the given model run."""
    plot_throughput_sum(model_data, run_id, dimensions, max_sum, target_model, "throughput_sum")
    plot_throughput_difference(model_data, run_id, dimensions, max_difference, target_model, "throughput_difference")


def plot_throughput_reports(model_data: Dict, target_model: str):
    """Plot a logging throughput report for the given model results."""
    max_nr_of_timestamps = max(x.shape[0] for x in model_data["log_frequency"]["files"]["sum"])
    max_nr_of_files = max(x.shape[1] for x in model_data["log_frequency"]["files"]["sum"])
    data_dimensions = (max_nr_of_files, max_nr_of_timestamps)

    max_sum = int(max(np.nanmax(v.to_numpy()) for v in model_data["log_frequency"]["files"]["sum"]))
    max_difference = int(max(np.nanmax(v.to_numpy()) for v in model_data["log_frequency"]["files"]["difference"]))

    for i, _ in enumerate(model_data["log_frequency"]["files"]["sum"]):
        plot_throughput_report(model_data, i, data_dimensions, max_sum, max_difference, target_model)


def format_frequency_table_index(v):
    """Format the given target."""
    if v.count(".") < 2:
        return f"\\\\[-8pt]\\textbf{{{v}}}"
    else:
        return f"\\hspace{{3mm}}{v}"


def get_frequency_statistics_table(input_data: pd.DataFrame, model_data: Dict):
    """Get a summary table containing the statistics for the given data frame."""
    # Create a copy, such that the original remains unaltered.
    input_data = input_data.copy()

    # Calculate the mean and standard deviation for every column.
    frequency_data, _ = extract_frequency_data(input_data, model_data)
    success_frequency_data, success_ratio_data = extract_success_data(input_data, model_data)

    frequency_statistics = frequency_data.groupby(["message"]).agg(
        mean=("value", "mean"), std=("value", "std")
    )
    frequency_statistics.rename(columns={"mean": "$\\mu(e)$", "std": "$\\sigma(e)$"}, inplace=True)
    success_frequency_statistics = success_frequency_data.groupby(["message"]).agg(
        mean=("value", "mean"), std=("value", "std")
    )
    success_frequency_statistics.rename(columns={"mean": "$\\mu(se)$", "std": "$\\sigma(se)$"}, inplace=True)
    success_ratio_statistics = success_ratio_data.groupby(["message"]).agg(
        mean=("value", "mean"), std=("value", "std")
    )
    success_ratio_statistics.rename(columns={"mean": "$\\mu(sr)$", "std": "$\\sigma(sr)$"}, inplace=True)

    result_table = pd.concat(
        [frequency_statistics, success_frequency_statistics, success_ratio_statistics], axis=1, join="inner"
    )
    result_table = result_table.reindex(list(sorted(set(result_table.index), key=create_ordering_mask)))
    result_table.index = result_table.index.map(format_frequency_table_index)

    return result_table


def reformat_tabular_header(
        tabular_code: str, title: str, target_header: str = "Target", remove_first_spacing: bool = True
):
    """Reformat the header of the given tabular."""
    tabular_code = tabular_code.replace("{}", target_header)
    tabular_code_lines = tabular_code.splitlines(keepends=True)
    tabular_code_lines[3] = tabular_code_lines[2]
    tabular_code_lines[2] = \
        f"\\multicolumn{{{tabular_code_lines[3].count('&') + 1}}}{{c}}{{{title}}} \\\\[2mm]\n"
    if remove_first_spacing:
        tabular_code_lines[5] = tabular_code_lines[5][8:]
    return ''.join(tabular_code_lines)


def plot_frequency_results_table(
        input_data: pd.DataFrame,
        model_data: Dict,
        target_model: str,
        category: str = None,
        caption_addendum: str = None
):
    """Plot the results rendered within the given table."""
    # Gather all the summary statistics.
    result_table = get_frequency_statistics_table(input_data, model_data)

    # Render the table and make corrections to its formatting.
    tabular_code = render_tabular(result_table)
    model_details = f"n={len(input_data.columns)}, t=30"
    if category is not None:
        model_details += f", {category}"

    title = f"Performance results for target model `\\texttt{{{target_model}}}' $({model_details})$"
    tabular_code = reformat_tabular_header(tabular_code, title)

    # Render the table as a file with an appropriate caption and label.
    caption = \
        f"A table containing statistics on the number of executions $(e)$, number of successful executions $(se)$ " \
        f"and success ratio $(sr)$ measured during the execution of the target model \\texttt{{{target_model}}}. " \
        f"{get_sample_statement(len(input_data.columns))}"
    if caption_addendum is not None:
        caption += f" {caption_addendum}"

    label = f"table:frequency_results_{target_model.lower()}"
    if category is not None:
        label += f"_{category.lower()}"

    file_name = f"frequency_results"
    if category is not None:
        file_name += f"_{category.lower()}"

    save_table_as_tex(target_model, tabular_code, caption, label, file_name)


def format_throughput_table_index(v):
    """Format the given target."""
    if v in ["total", "difference"]:
        return f"\\\\[-8pt]\\textit{{{v}}}"
    else:
        return f"{v}"


def get_throughput_frequency_statistics_table(model_data: Dict):
    """Get a table containing statistics on the throughput characteristics of the log files."""
    throughput_data = model_data["log_frequency"]["global"]
    throughput_frequency_data = throughput_data["table"].copy()
    category_mapping = dict()
    for sm, runs in throughput_data["targets"]["thread"].items():
        category_mapping |= {v: sm for v in runs}
    for i, columns in throughput_data["targets"]["run"].items():
        sum_column_value = throughput_frequency_data[f"total_frequency_{i}"] = \
            throughput_frequency_data[columns].sum(axis=1)
        category_mapping[f"total_frequency_{i}"] = "total"

        # Create a table that holds the difference of each column to the minimum column value for each row per file.
        # Given that the min is used, no absolute values need to be calculated.
        # Hence, this metric is equivalent to subtracting the minimum times the number of threads from the file sum.
        nr_of_threads = len(throughput_data["targets"]["thread"])
        min_column_value = throughput_frequency_data[columns].min(axis=1)
        throughput_frequency_data[f"difference_frequency_{i}"] = \
            sum_column_value.subtract(min_column_value.multiply(nr_of_threads))
        category_mapping[f"difference_frequency_{i}"] = "difference"

    throughput_frequency_data = throughput_frequency_data.melt()
    throughput_frequency_data["variable"] = throughput_frequency_data["variable"].map(category_mapping)

    frequency_summary_statistics = throughput_frequency_data.groupby(["variable"]).agg(
        count=("value", "count"),
        min=("value", "min"),
        max=("value", "max"),
        median=("value", "median"),
        mean=("value", "mean"),
        std=("value", "std")
    )
    frequency_summary_statistics.rename(columns={
        "count": "$N$",
        "min": "$min(e)$",
        "max": "$max(e)$",
        "median": "$median(e)$",
        "mean": "$\\mu(e)$",
        "std": "$\\sigma(e)$"
    }, inplace=True)

    # Reorder the entries.
    ordering = sorted(throughput_data["targets"]["thread"].keys()) + ["total", "difference"]
    frequency_summary_statistics = frequency_summary_statistics.reindex(ordering)
    frequency_summary_statistics.index = frequency_summary_statistics.index.map(format_throughput_table_index)
    return frequency_summary_statistics, len(throughput_data["targets"]["run"])


def plot_throughput_information_table(
        model_data: Dict,
        target_model: str
):
    """Plot a table that gives summary data on the throughput of the logging files."""
    # Create a summary table.
    result_table, n = get_throughput_frequency_statistics_table(model_data)

    # Render the table and make corrections to its formatting.
    tabular_code = render_tabular(result_table)
    model_details = f"n={n}, t=30"

    title = f"Log message throughput statistics for target model `\\texttt{{{target_model}}}' $({model_details})$"
    tabular_code = reformat_tabular_header(tabular_code, title, remove_first_spacing=False)

    # Render the table as a file with an appropriate caption and label.
    caption = \
        f"A table containing statistics on the log messages per milliseconds $(e)$ measured during the execution of " \
        f"the target model \\texttt{{{target_model}}}. The total and difference entries at the end of the table " \
        f"depict the row sum and the sum difference to the row minimum respectively. {get_sample_statement(n)}"

    label = f"table:throughput_statistics_{target_model.lower()}"
    file_name = f"throughput_statistics"

    save_table_as_tex(target_model, tabular_code, caption, label, file_name, include_resize_box=False)


def get_sample_statement(n: Optional[int]) -> str:
    if n is None:
        return "The results have been measured over a time span of 30 seconds."
    else:
        return f"The results have been measured over a time span of 30 seconds, where each entry is represented by " \
           f"measurements taken over {n} trials."
