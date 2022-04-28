from typing import Dict

from analysis.consistency import analyze_model_results_consistency


def analyze_model_results(model_data: Dict):
    """Analyze the given model's data."""
    # Analyze the similarity between the results.
    analyze_model_results_consistency(model_data)
    pass
