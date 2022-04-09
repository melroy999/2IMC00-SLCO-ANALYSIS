from typing import Dict, List

from analysis.util import create_aggregate_table, create_desc_statistics_table, create_correlation_table, \
    create_difference_sum_table
from plotting import create_global_log_file_throughput_plot, create_succession_heat_map_plot, \
    create_thread_grouped_log_file_throughput_plot, create_concurrency_heat_map_plot


def analyze_data(data: Dict):
    """Analyze the given data."""
    create_global_log_file_throughput_plot(data)
    create_thread_grouped_log_file_throughput_plot(data)
    create_succession_heat_map_plot(data)
    create_succession_heat_map_plot(data, target="transition_succession_graph")
    create_concurrency_heat_map_plot(data)
