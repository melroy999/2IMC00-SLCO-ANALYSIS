from typing import Dict

from analysis.consistency import analyze_model_results_consistency
from analysis.results import analyze_model_results


def analyze_target_model(model_data: Dict):
    """Analyze the given model's data."""
    # Analyze the similarity between the results.
    # analyze_model_results_consistency(model_data)

    # Analyze the attained results.
    analyze_model_results(model_data)
