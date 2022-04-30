from typing import Dict

from visualization.plot import plot_throughput_reports, plot_message_frequency_similarity_report, \
    plot_thread_workload_balance_report, plot_message_order_similarity_report
from visualization.table import tabulate_message_frequency_similarity_report


def analyze_model_results_log_throughput_consistency(model_data: Dict):
    """Analyze the log throughput of each reported model result run and measure the consistency between runs."""
    plot_throughput_reports(model_data)
    plot_thread_workload_balance_report(model_data)


def analyze_model_results_message_frequency_consistency(model_data: Dict):
    """Analyze the message frequency of each reported model run and report similarity between counting and logging."""
    plot_message_frequency_similarity_report(model_data)
    # tabulate_message_frequency_similarity_report(model_data)


def analyze_model_results_message_order_consistency(model_data: Dict):
    """Analyze the message order of each reported model result run and measure the consistency between runs."""
    plot_message_order_similarity_report(model_data)


def analyze_model_results_consistency(model_data: Dict):
    """Analyze the consistency between the reported model result runs."""
    # analyze_model_results_log_throughput_consistency(model_data)
    analyze_model_results_message_frequency_consistency(model_data)
    analyze_model_results_message_order_consistency(model_data)
