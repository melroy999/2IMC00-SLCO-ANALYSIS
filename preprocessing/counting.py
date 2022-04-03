from typing import Dict

import pandas as pd


def preprocess_counting_data(data: Dict):
    """Preprocess the json data."""
    # Preprocess the log data such that it can be used more easily in figures and graphs.
    data["event_count"] = pd.DataFrame.from_dict(data["event_count"], orient="index", columns=["count"])
    return data
