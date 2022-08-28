from typing import Dict

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from visualization.categorical import plot_comparison_boxplot
from visualization.heatmap import plot_heatmap


def remove_subject_label_suffix(x):
    """Remove the message type from the subject label."""
    return ".".join(x.split(".")[:-1])


def get_revised_index_names(data: pd.DataFrame, abbreviation: Dict) -> Dict:
    """Rename the subject labels to be more informative by including states and more descriptive indices."""
    renaming_dict = dict()
    for v in data.index:
        sm_name, identifier = v.split(".")
        target = abbreviation[identifier]
        if identifier.startswith("D"):
            renaming_dict[v] = f"{sm_name}.{target['source']}"
        else:
            renaming_dict[v] = f"{sm_name}.{target['source']}.{target['target']} {target['name'].split('|')[0].strip()}"
    return renaming_dict


# previous_results = dict()


def plot_transition_frequency_boxplot(frequency_data: pd.DataFrame, model_data: Dict, _type: str = "Logging Intervals"):
    """Plot the transition frequencies recorded in the model."""
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
    # previous_results[model_data['model']['id']] = (success_frequency_data, success_ratio_data)

    # Create two sub-figures.
    root_fig = plt.figure(figsize=(8, 6), dpi=300)
    plot_comparison_boxplot(
        success_frequency_data, success_ratio_data, root_fig,
        title_left="Success Count",
        title_right="Success/Total Ratio",
        y_label="Subject",
        x_label_left="Count",
        x_label_right="Percentage"
    )
    root_fig.suptitle(f"Successful Transition/DC Executions ({model_data['model']['id']}, {_type})", y=1.00)

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)

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

    # if len(previous_results) > 1:
    #     target_success_frequency_data = []
    #     target_success_ratio_data = []
    #     for a, (b, c) in previous_results.items():
    #         b["type"] = a
    #         c["type"] = a
    #         target_success_frequency_data.append(b)
    #         target_success_ratio_data.append(c)
    #     success_frequency_data = pd.concat(target_success_frequency_data)
    #     success_ratio_data = pd.concat(target_success_ratio_data)
    #
    #     root_fig = plt.figure(figsize=(8, 8), dpi=300)
    #     plot_comparison_boxplot(
    #         success_frequency_data, success_ratio_data, root_fig,
    #         title_left="Success Count",
    #         title_right="Success/Total Ratio",
    #         x_label_left="Count",
    #         x_label_right="Percentage",
    #         hue="type",
    #     )
    #     root_fig.suptitle(f"Successful Transition/DC Executions (Comparison)", y=1.00)
    #     plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    #     plt.show()


def plot_concurrency_heatmap(
        plot_data: pd.DataFrame,
        annotations: pd.DataFrame,
        title: str,
        c_bar_label: str,
        color_bar_value_function=None
):
    """Plot the recorded degree of concurrency in the model."""
    # Determine the dimensions.
    n, m = plot_data.shape
    scale_factor = 0.7
    root_fig = plt.figure(figsize=(scale_factor * n + 2, scale_factor * n + 1), dpi=300)

    # Set heatmap options.
    heatmap_options = {
        "annot": annotations,
        "fmt": ".1%",
        "vmin": 0.0,
        "square": True,
    }
    if color_bar_value_function is not None:
        heatmap_options["cbar_kws"] = {"format": FuncFormatter(color_bar_value_function)}

    # Plot the data.
    plot_heatmap(
        plot_data,
        root_fig,
        title=title,
        x_label="Target State Machine",
        y_label="Source State Machine",
        c_bar_label=c_bar_label,
        input_heatmap_options=heatmap_options
    )

    plt.tight_layout(pad=0.5, w_pad=1.0, h_pad=1.0)
    plt.show()


def plot_local_concurrency_heatmap(plot_data: pd.DataFrame, annotations: pd.DataFrame, model_data: Dict):
    """Plot the recorded degree of concurrency in the model."""
    plot_concurrency_heatmap(
        plot_data,
        annotations,
        title=f"State Machine Concurrency Percentage \n({model_data['model']['id']})",
        c_bar_label="Total Frequency"
    )


def plot_global_concurrency_heatmap(plot_data: pd.DataFrame, model_data: Dict):
    """Plot the recorded degree of concurrency in the model."""
    plot_concurrency_heatmap(
        plot_data,
        plot_data,
        title=f"State Machine Global Concurrency Percentage \n({model_data['model']['id']})",
        c_bar_label="Percentage",
        color_bar_value_function=lambda x, pos: f"{x:.1%}"
    )
