from collections import defaultdict
from functools import reduce
from typing import Dict, List, Union

import pandas as pd

from preprocessing.restructuring import restructure_model_data


def preprocess_model_information(model_results: Dict, target_model: str) -> Dict:
    """Get the model that is tested and verify if all results are of the same model."""
    aggregate_results = [v for v in model_results["logging"] + model_results["counting"]]
    if len(model_results["logging"]) == 0:
        raise Exception("The required logging-based model data is available.")

    # Verify that all of the results have the same model as a source.
    if not all(v["model"] == aggregate_results[0]["model"] for v in aggregate_results[1:]):
        raise Exception("The source model is not the same for all grouped results.")

    # Restructure the model information and return the result.
    return restructure_model_data(model_results["logging"], target_model)


def preprocess_log_frequency_table(
        frequency_data: Dict[str, int], start_timestamp: int, end_timestamp: int = None
) -> Dict[int, int]:
    """Preprocess the timestamps in the given frequency dictionary."""
    preprocessed_frequency_data = dict()
    for message, frequency in frequency_data.items():
        preprocessed_frequency_data[int(message) - start_timestamp] = frequency

    # Fill gaps in the list of timestamps if the end timestamp is given.
    if end_timestamp is not None:
        for i in range(end_timestamp + 1 - start_timestamp):
            if i not in preprocessed_frequency_data:
                preprocessed_frequency_data[i] = 0

    return preprocessed_frequency_data


def create_aggregate_log_frequency_table(target_data_frames: List[pd.DataFrame], fill_na: bool = False) -> pd.DataFrame:
    """Create a data frame that holds aggregate data for the targets in question."""
    # Merge the tables with an outer join.
    aggregate_table = pd.concat(target_data_frames, axis=1)

    # Replace missing values with zeros.
    if fill_na:
        aggregate_table.fillna(.0, inplace=True)

    # Sort the indices.
    return aggregate_table.sort_index()


def preprocess_model_log_frequency_data(model_results: Dict, model_information: Dict) -> Dict:
    """Preprocess all the log frequency related data and convert the found results into aggregate data."""
    # Convert all log frequency global data to data frames.
    log_frequency_target_columns = {
        "run": defaultdict(list), "thread": defaultdict(list)
    }
    log_frequency_data_frames = []
    for i, run_data in enumerate(model_results["logging"]):
        start_timestamp = run_data["log_data"]["global"]["global"]["start"]
        end_timestamp = run_data["log_data"]["global"]["global"]["end"]
        for thread_name, thread_data in run_data["log_data"]["threads"].items():
            state_machine_name, _ = model_information["thread_to_state_machine"][thread_name]
            column_name = f"frequency_{state_machine_name}_{i}"
            frequency_data = thread_data["global"]["count"]
            frequency_data = preprocess_log_frequency_table(frequency_data, start_timestamp, end_timestamp)
            log_frequency_data_frames.append(
                pd.DataFrame.from_dict(frequency_data, orient="index", columns=[column_name])
            )
            log_frequency_data_frames[-1].index.name = "timestamp"
            log_frequency_target_columns["run"][i].append(column_name)
            log_frequency_target_columns["thread"][state_machine_name].append(column_name)

    # Convert all log frequency data grouped per file to data frames.
    file_log_frequency_target_columns = {
        "file": defaultdict(list), "run": defaultdict(list), "thread": defaultdict(list)
    }
    file_log_frequency_data_frames = []
    for i, run_data in enumerate(model_results["logging"]):
        for j, file_data in enumerate(run_data["log_data"]["global"]["files"]):
            start_timestamp = file_data["start"]
            end_timestamp = file_data["end"]
            for thread_name, thread_data in run_data["log_data"]["threads"].items():
                state_machine_name, _ = model_information["thread_to_state_machine"][thread_name]
                column_name = f"frequency_{state_machine_name}_{j}_{i}"
                frequency_data = thread_data["files"][j]["count"]
                frequency_data = preprocess_log_frequency_table(frequency_data, start_timestamp, end_timestamp)
                file_log_frequency_data_frames.append(
                    pd.DataFrame.from_dict(frequency_data, orient="index", columns=[column_name])
                )
                file_log_frequency_data_frames[-1].index.name = "timestamp"
                file_log_frequency_target_columns["file"][j].append(column_name)
                file_log_frequency_target_columns["run"][i].append(column_name)
                file_log_frequency_target_columns["thread"][state_machine_name].append(column_name)

    # Merge all file-grouped log frequency data frames into preprocessed information tables.
    aggregate_file_log_frequency_table = create_aggregate_log_frequency_table(file_log_frequency_data_frames)
    file_log_frequency_sum_data_frames = []
    file_log_frequency_difference_to_min_data_frames = []
    for i, run_data in enumerate(model_results["logging"]):
        # Select the data associated with the target run.
        run_data = aggregate_file_log_frequency_table[file_log_frequency_target_columns["run"][i]]

        # Group all columns of the same file.
        file_grouped_data = run_data.groupby(lambda x: int(x.split("_")[-2]), axis=1)

        # Sum columns within each group. Ensure rows containing just N/A remain N/A.
        file_sum_grouped_data = file_grouped_data.sum(min_count=1)
        file_log_frequency_sum_data_frames.append(file_sum_grouped_data)

        # Create a table that holds the difference of each column to the minimum column value for each row per file.
        # Given that the min is used, no absolute values need to be calculated.
        # Hence, this metric is equivalent to subtracting the minimum times the number of threads from the file sum.
        nr_of_threads = len(file_log_frequency_target_columns["thread"])
        file_min_grouped_data = file_grouped_data.min(min_count=1)
        file_diff_to_min_grouped_data = file_sum_grouped_data.subtract(file_min_grouped_data.multiply(nr_of_threads))
        file_log_frequency_difference_to_min_data_frames.append(file_diff_to_min_grouped_data)

    # Merge all global log frequency data frames.
    aggregate_global_log_frequency_data = {
        "targets": dict(log_frequency_target_columns),
        "table": create_aggregate_log_frequency_table(log_frequency_data_frames, fill_na=True)
    }

    # Merge all file-grouped log frequency data frames.
    aggregate_file_log_frequency_data = {
        "sum": file_log_frequency_sum_data_frames,
        "difference": file_log_frequency_difference_to_min_data_frames
    }

    # Combine the data in a convenient data structure.
    return {
        "global": aggregate_global_log_frequency_data,
        "files": aggregate_file_log_frequency_data
    }


def preprocess_message_frequency_table(frequency_data: Dict[str, int], model_information: Dict) -> Dict[str, int]:
    """Preprocess the message names in the given frequency dictionary."""
    preprocessed_frequency_data = dict()
    for message, frequency in frequency_data.items():
        thread_name, message_id, message_type = message.split(".")
        state_machine_name, _ = model_information["thread_to_state_machine"][thread_name]
        preprocessed_frequency_data[f"{state_machine_name}.{message_id}.{message_type}"] = frequency
    return preprocessed_frequency_data


def create_aggregate_message_frequency_table(target_data_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Create a data frame that holds aggregate data for the targets in question."""
    # Merge the tables with an outer join.
    aggregate_table = pd.concat(target_data_frames, axis=1)

    # Replace missing values with zeros.
    aggregate_table.fillna(.0, inplace=True)

    # Sort the indices.
    aggregate_table.sort_index(inplace=True)

    # Filter out rows that only contain zeroes.
    return aggregate_table[~(aggregate_table == 0).all(axis=1)]


def preprocess_model_message_frequency_data(model_results: Dict, model_information: Dict) -> Dict:
    """Preprocess all the message frequency related data and convert the found results into aggregate data."""
    # Convert all counting-based message frequency data to data frames.
    counting_target_columns = []
    counting_data_frames = []
    for i, run_data in enumerate(model_results["counting"]):
        column_name = f"counting_frequency_{i}"
        frequency_data = preprocess_message_frequency_table(run_data["event_count"], model_information)
        counting_data_frames.append(
            pd.DataFrame.from_dict(frequency_data, orient="index", columns=[column_name])
        )
        counting_data_frames[-1].index.name = "message"
        counting_target_columns.append(column_name)

    # Convert all logging-based message frequency global data to data frames.
    logging_target_columns = []
    logging_data_frames = []
    for i, run_data in enumerate(model_results["logging"]):
        # Convert all global data to data frames.
        column_name = f"logging_frequency_{i}"
        frequency_data = run_data["message_data"]["global_data"]["event_count"]
        frequency_data = preprocess_message_frequency_table(frequency_data, model_information)
        logging_data_frames.append(
            pd.DataFrame.from_dict(frequency_data, orient="index", columns=[column_name])
        )
        logging_target_columns.append(column_name)

    # Convert all logging-based message frequency interval data to data frames.
    logging_interval_target_columns = defaultdict(list)
    logging_interval_data_frames = []
    logging_interval_settings = None
    for i, run_data in enumerate(model_results["logging"]):
        # Check if settings are equal.
        local_logging_interval_settings = {
            "interval": run_data["message_data"]["interval"],
            "start": run_data["message_data"]["start"],
            "end": run_data["message_data"]["end"]
        }
        if logging_interval_settings is None:
            logging_interval_settings = local_logging_interval_settings
        elif logging_interval_settings != local_logging_interval_settings:
            raise Exception("The interval ranges are not equal between runs.")

        # Convert all interval data to data frames.
        for interval_data in run_data["message_data"]["intervals"]:
            column_name = f"{logging_target_columns[i]}_i_{interval_data['start']}_{interval_data['end']}"
            frequency_data = interval_data["data"]["event_count"]
            frequency_data = preprocess_message_frequency_table(frequency_data, model_information)
            logging_interval_data_frames.append(
                pd.DataFrame.from_dict(frequency_data, orient="index", columns=[column_name])
            )
            logging_interval_target_columns[(interval_data["start"], interval_data["end"])].append(column_name)

    # Merge all the global data frames.
    aggregate_global_frequency_data = {
        "targets": {
            "counting": counting_target_columns,
            "logging": logging_target_columns
        },
        "table": create_aggregate_message_frequency_table(counting_data_frames + logging_data_frames)
    }

    # Merge all interval data frames.
    aggregate_interval_frequency_data = {
        "settings": logging_interval_settings,
        "targets": dict(logging_interval_target_columns),
        "table": create_aggregate_message_frequency_table(logging_interval_data_frames)
    }

    # Combine the data in a convenient data structure.
    return {
        "global": aggregate_global_frequency_data,
        "intervals": aggregate_interval_frequency_data
    }


def create_aggregate_message_order_table(target_data_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Create a data frame that holds aggregate data for the targets in question."""
    # Merge the tables with an outer join and replace missing values with zeros.
    return reduce(
        lambda left, right: pd.merge(left, right, on=["source_message", "target_message"], how="outer"),
        target_data_frames
    ).fillna(.0)


def preprocess_message_order_table(
        message_order_data: Dict[str, int], model_information: Dict, column_name: str
) -> Dict[str, List[Union[str, int]]]:
    """Preprocess the message order table such that it can be converted to a data frame."""
    preprocessed_message_order_data = {"source_message": [], "target_message": [], column_name: []}
    for message_pair, frequency in message_order_data.items():
        # Preprocess the message names and re-insert them as a tuple.
        original_source_message, original_target_message = message_pair.split("~")
        source_thread_name, source_message_id, source_message_type = original_source_message.split(".")
        target_thread_name, target_message_id, target_message_type = original_target_message.split(".")
        source_state_machine_name, _ = model_information["thread_to_state_machine"][source_thread_name]
        target_state_machine_name, _ = model_information["thread_to_state_machine"][target_thread_name]
        source_message = f"{source_state_machine_name}.{source_message_id}.{source_message_type}"
        target_message = f"{target_state_machine_name}.{target_message_id}.{target_message_type}"
        preprocessed_message_order_data["source_message"].append(source_message)
        preprocessed_message_order_data["target_message"].append(target_message)
        preprocessed_message_order_data[column_name].append(frequency)
    return preprocessed_message_order_data


def merge_message_order_data_frames(target_data_frames, target_columns):
    """Merge the given data frames containing the to be aggregated message order data."""
    aggregate_data = create_aggregate_message_order_table(target_data_frames)
    aggregate_data["logging_frequency_mean"] = aggregate_data[target_columns].mean(axis=1)
    adjacency_table = aggregate_data.groupby(
        ["source_message", "target_message"]
    )["logging_frequency_mean"].sum().unstack().fillna(0)
    aggregate_data = aggregate_data[["source_message", "target_message"] + target_columns]
    aggregate_data.set_index(["source_message", "target_message"], inplace=True)
    return aggregate_data, adjacency_table


def preprocess_model_message_succession_data(
        model_results: Dict, model_information: Dict, graph_type: str = "transition_succession_graph"
) -> Dict:
    """Preprocess all the message succession data and convert the found results into aggregate data."""
    message_order_target_columns = []
    message_order_data_frames = []
    for i, run_data in enumerate(model_results["logging"]):
        column_name = f"frequency_{i}"
        message_order_data = run_data["message_data"]["global_data"][graph_type]
        message_order_data = preprocess_message_order_table(message_order_data, model_information, column_name)
        message_order_data_frames.append(pd.DataFrame.from_dict(message_order_data))
        message_order_target_columns.append(column_name)

    # Convert all logging-based interval message order data to data frames and graphs.
    message_order_interval_target_columns = defaultdict(list)
    message_order_interval_data_frames = []
    logging_interval_settings = None
    for i, run_data in enumerate(model_results["logging"]):
        # Check if interval settings are equal.
        local_logging_interval_settings = {
            "interval": run_data["message_data"]["interval"],
            "start": run_data["message_data"]["start"],
            "end": run_data["message_data"]["end"]
        }
        if logging_interval_settings is None:
            logging_interval_settings = local_logging_interval_settings
        elif logging_interval_settings != local_logging_interval_settings:
            raise Exception("The interval ranges are not equal between runs.")

        for interval_data in run_data["message_data"]["intervals"]:
            column_name = f"{message_order_target_columns[i]}_i_{interval_data['start']}_{interval_data['end']}"
            message_order_data = interval_data["data"][graph_type]
            message_order_data = preprocess_message_order_table(message_order_data, model_information, column_name)
            message_order_interval_data_frames.append(pd.DataFrame.from_dict(message_order_data))
            message_order_interval_target_columns[(interval_data["start"], interval_data["end"])].append(column_name)

    # Merge all the global data frames.
    aggregate_message_order_data, message_order_adjacency_table = merge_message_order_data_frames(
        message_order_data_frames, message_order_target_columns
    )

    # Create a data structure for the global results.
    aggregate_global_frequency_data = {
        "adjacency_table": message_order_adjacency_table,
        "frequency_table": aggregate_message_order_data
    }

    # Merge all the interval data frames.
    target_columns = [x for v in message_order_interval_target_columns.values() for x in v]
    aggregate_interval_message_order_data, message_interval_order_adjacency_table = merge_message_order_data_frames(
        message_order_interval_data_frames, target_columns
    )

    # Create a data structure for the interval results.
    aggregate_interval_frequency_data = {
        "settings": logging_interval_settings,
        "targets": dict(message_order_interval_target_columns),
        "adjacency_table": message_interval_order_adjacency_table,
        "frequency_table": aggregate_interval_message_order_data
    }

    # Combine the data in a convenient data structure.
    return {
        "global": aggregate_global_frequency_data,
        "intervals": aggregate_interval_frequency_data
    }


def preprocess_model_data(model_results: Dict, model_information: Dict) -> Dict:
    """Preprocess all the data and convert the found results into aggregate data."""
    return {
        "log_frequency": preprocess_model_log_frequency_data(model_results, model_information),
        "message_frequency": preprocess_model_message_frequency_data(model_results, model_information),
        "message_order": preprocess_model_message_succession_data(model_results, model_information)
    }


def preprocess_model_results(model_results: Dict, target_model: str) -> Dict:
    """Preprocess the given data such that it can be used in the analysis and presentation of results."""
    # Preprocess the model information and target data.
    model_information = preprocess_model_information(model_results, target_model)

    # Preprocess the target data.
    model_data = preprocess_model_data(model_results, model_information)

    return {
        "model": model_information,
    } | model_data

