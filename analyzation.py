from typing import Dict

from plotting import create_global_log_file_throughput_plot, create_succession_heat_map_plot, \
    create_thread_grouped_log_file_throughput_plot


def analyze_data(data: Dict):
    """Analyze the given data."""
    create_global_log_file_throughput_plot(data)
    create_thread_grouped_log_file_throughput_plot(data)
    # create_succession_heat_map_plot(data)
