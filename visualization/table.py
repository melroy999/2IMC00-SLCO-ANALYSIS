from typing import Dict

import numpy as np

from analysis.util import create_correlation_table, create_difference_sum_table


def tabulate_message_frequency_similarity_report(model_data: Dict):
    """Create a message frequency similarity report table for the given model results."""
    # TODO: Perform the same analysis but per interval and summarized as a table.
    plot_data = model_data["message_frequency"]["intervals"]["table"]
    correlation_table = create_correlation_table(plot_data, method="spearman")
    difference_sum_table = create_difference_sum_table(plot_data)
    correlation_table_min = correlation_table.min()
    difference_sum_table_max = difference_sum_table.max()
    correlation_table_det = np.linalg.det(correlation_table)
    pass
