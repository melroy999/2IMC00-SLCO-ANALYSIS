from typing import Dict

from visualization.model import plot_throughput_reports, plot_message_frequency_similarity_report, \
    plot_thread_message_count, plot_thread_message_ratio


def analyze_model_results_log_throughput(model_data: Dict):
    """Analyze the log throughput of each reported model result run and measure the consistency between runs."""
    plot_throughput_reports(model_data)
    plot_thread_message_count(model_data)
    plot_thread_message_ratio(model_data)


def analyze_model_results_message_frequency(model_data: Dict):
    """Analyze the message frequency of each reported model run and report similarity between counting and logging."""
    plot_message_frequency_similarity_report(model_data)


def analyze_model_results_consistency(model_data: Dict):
    """Analyze the consistency between the reported model result runs."""
    analyze_model_results_log_throughput(model_data)
    analyze_model_results_message_frequency(model_data)
