import networkx as nx
import pandas as pd

root_data = {
    "model": {
        "name": str,
        "id": str,
        "classes": {
            "name": {
                "name": str,
                "state_machines": {
                    "thread_name": {
                        "name": str,
                        "graph": nx.MultiDiGraph
                    }
                }
            }
        }
    },
    "data": {
        "log_frequency": {
            "runs": {
                "run_id": pd.DataFrame
            },
            "threads": {
                "thread_id": pd.DataFrame
            },
            "global": pd.DataFrame
        },
        "message_frequency": {
            "intervals": {
                "interval_id": pd.DataFrame
            },
            "global": pd.DataFrame
        },
        "message_order": {
            "succession": pd.DataFrame
        }
    }
}
