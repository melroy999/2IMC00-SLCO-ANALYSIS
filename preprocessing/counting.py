from typing import Dict

import pandas as pd

from preprocessing.model import preprocess_model_data


def preprocess_counting_data(data: Dict):
    """Preprocess the json data."""
    # Preprocess the log data such that it can be used more easily in figures and graphs.
    data["event_count"] = pd.DataFrame.from_dict(data["event_count"], orient="index", columns=["count"])
    preprocess_model_data(data["model"])
    return data
