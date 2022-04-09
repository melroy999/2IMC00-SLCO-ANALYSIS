from typing import Dict, List

from analysis.util import create_aggregate_table, create_correlation_table


def select_logging_message_frequency(data):
    """Select the event count within the given dictionary structure."""
    return data["message_data"]["global_data"]["event_count"]


def select_logging_succession_frequency(data):
    """Select the succession table within the given dictionary structure."""
    return data["message_data"]["global_data"]["succession_table"]


def select_logging_transition_succession_frequency(data):
    """Select the transition succession table within the given dictionary structure."""
    return data["message_data"]["global_data"]["transition_succession_table"]


def select_interval_logging_message_frequency(data):
    """Select the event count within the given dictionary structure."""
    return data["data"]["event_count"]


def select_interval_logging_succession_frequency(data):
    """Select the succession table within the given dictionary structure."""
    return data["data"]["succession_table"]


def select_interval_logging_transition_succession_frequency(data):
    """Select the transition succession table within the given dictionary structure."""
    return data["data"]["transition_succession_table"]


def select_interval_aggregate_logging_message_frequency(data):
    """Select the event count within the given dictionary structure."""
    return data["message_frequency"]


def select_interval_aggregate_logging_succession_frequency(data):
    """Select the succession table within the given dictionary structure."""
    return data["succession_frequency"]


def select_interval_aggregate_logging_transition_succession_frequency(data):
    """Select the transition succession table within the given dictionary structure."""
    return data["transition_succession_frequency"]


def create_logging_message_aggregate_data(logging_results: List[Dict]) -> Dict:
    """Create aggregate data for the given collection of message results that are taken from log-based measurements."""
    # Concatenate the event and succession count data of the different runs.
    result_generator = [
        ("message_frequency", select_logging_message_frequency),
        ("succession_frequency", select_logging_succession_frequency),
        ("transition_succession_frequency", select_logging_transition_succession_frequency)
    ]
    aggregate_data = {k: create_aggregate_table(logging_results, f) for k, f in result_generator}

    # Find the intervals to iterate over and validate whether they are of the same domain.
    interval_results_list = [entry["message_data"]["intervals"] for entry in logging_results]
    if len(interval_results_list) == 0:
        raise Exception("Interval comparisons cannot be made, since no intervals are provided.")
    if not all(len(x) == len(interval_results_list[0]) for x in interval_results_list[1:]):
        raise Exception("Interval comparisons cannot be made, since the provided intervals are not of equal length.")

    # Iterate over all intervals and add aggregate tables.
    aggregate_data["intervals"] = {
        "entries": []
    }
    interval_results: List[Dict]
    for interval_results in zip(*interval_results_list):
        # Verify that the data is over the same domain.
        if not all(x["start"] == interval_results[0]["start"] for x in interval_results[1:]) \
                or not all(x["end"] == interval_results[0]["end"] for x in interval_results[1:]):
            raise Exception("Interval comparisons cannot be made, since the provided intervals are different.")

        # Create aggregate data for each interval.
        interval_result = {
            "start": interval_results[0]["start"],
            "end": interval_results[0]["end"]
        }
        result_generator = [
            ("message_frequency", select_interval_logging_message_frequency),
            ("succession_frequency", select_interval_logging_succession_frequency),
            ("transition_succession_frequency", select_interval_logging_transition_succession_frequency)
        ]
        interval_result |= {k: create_aggregate_table(interval_results, f) for k, f in result_generator}
        aggregate_data["intervals"]["entries"].append(interval_result)

    # Create an aggregate table containing all interval entries.
    result_generator = [
        ("message_frequency", select_interval_aggregate_logging_message_frequency),
        ("succession_frequency", select_interval_aggregate_logging_succession_frequency),
        ("transition_succession_frequency", select_interval_aggregate_logging_transition_succession_frequency)
    ]
    aggregate_data["intervals"]["aggregate_data"] = {
        k: create_aggregate_table(aggregate_data["intervals"]["entries"], f) for k, f in result_generator
    }

    return aggregate_data


def select_logging_thread_global_log_frequency(data, thread_name):
    """Select the thread's global frequency table within the given dictionary structure."""
    return data["log_data"]["threads"][thread_name]["global"]["frequency_table"]


def select_logging_thread_files(data, thread_name):
    """Select the files array within the given dictionary structure."""
    return data["log_data"]["threads"][thread_name]["files"]


def select_file_log_frequency(data):
    """Select the file's log frequency within the given dictionary structure."""
    return data["frequency_table"]


def select_run_log_frequency(data):
    """Select the run's log frequency within the given dictionary structure."""
    return data["log_frequency"]


def select_run_global_frequency_table(data):
    """Select the global frequency table within the given dictionary structure."""
    return data["log_data"]["global"]["global"]["frequency_table"]


def create_logging_log_aggregate_data(logging_results: List[Dict]) -> Dict:
    """Create aggregate data for the given collection of logging results that are taken from log-based measurements."""
    aggregate_data = {
        "runs": []
    }

    # Get the thread names and create a constant order.
    thread_names = list(logging_results[0]["log_data"]["threads"])
    thread_names.sort()
    thread_prefixes = {thread_name: f"{thread_name.lower().replace('-', '_')}_" for thread_name in thread_names}

    # Data between runs cannot be compared easily, due to different timestamps and durations.
    for logging_result in logging_results:
        # Select the target global tables and add the appropriate prefixes.
        thread_global_frequencies = [
            select_logging_thread_global_log_frequency(
                logging_result, thread_name
            ).add_prefix(thread_prefixes[thread_name]) for thread_name in thread_names
        ]

        # Create an aggregate table of the global thread counts.
        run_aggregate_data = {
            "log_frequency": create_aggregate_table(thread_global_frequencies, add_index_suffix=False)
        }

        # Find the files to iterate over and validate whether they are of the same domain.
        files_results_list = [select_logging_thread_files(logging_result, thread_name) for thread_name in thread_names]
        if len(files_results_list) == 0:
            raise Exception("File comparisons cannot be made, since no file results are provided.")
        if not all(len(x) == len(files_results_list[0]) for x in files_results_list[1:]):
            raise Exception("File comparisons cannot be made, since the provided file arrays are not of equal length.")

        # Iterate over all files and add aggregate tables.
        run_aggregate_data["files"] = {
            "entries": []
        }
        file_results: List[Dict]
        for file_results in zip(*files_results_list):
            # Select the target file tables and add the appropriate prefixes.
            thread_file_frequencies = [
                select_file_log_frequency(
                    file_results[i]
                ).add_prefix(thread_prefixes[thread_name]) for i, thread_name in enumerate(thread_names)
            ]

            # Create aggregate data for each file.
            file_result = {
                "log_frequency": create_aggregate_table(thread_file_frequencies, add_index_suffix=False)
            }
            run_aggregate_data["files"]["entries"].append(file_result)

        # Add the run data.
        aggregate_data["runs"].append(run_aggregate_data)

    # Combine global results from multiple runs.
    aggregate_data["log_frequencies"] = create_aggregate_table(logging_results, select_run_global_frequency_table)
    aggregate_data["thread_log_frequencies"] = create_aggregate_table(aggregate_data["runs"], select_run_log_frequency)

    return aggregate_data


def create_logging_aggregate_data(logging_results: List[Dict]) -> Dict:
    """Create aggregate data for the given collection of results that are taken from log-based measurements."""
    # Create aggregate data for both message and log data.
    aggregate_data = {
        "message_data": create_logging_message_aggregate_data(logging_results),
        "log_data": create_logging_log_aggregate_data(logging_results)
    }
    return aggregate_data


def select_counting_event_count(data):
    """Select the event count within the given dictionary structure."""
    return data["event_count"]


def create_counting_aggregate_data(results: List[Dict]) -> Dict:
    """Create aggregate data for the given collection of results that are taken from count-based measurements."""
    # A dictionary containing all aggregate data.
    aggregate_data = {
        "message_data": {
            "message_frequency": create_aggregate_table(results, select_counting_event_count)
        }
    }
    return aggregate_data


def combine_aggregate_data(logging_aggregate_data: Dict, counting_aggregate_data: Dict) -> Dict:
    """Merge the aggregate data of the two measurement methods."""
    logging_data = logging_aggregate_data["message_data"]
    counting_data = counting_aggregate_data["message_data"]
    common_measurements = set(logging_data).intersection(set(counting_data))
    return {
        m: create_aggregate_table(
            [logging_data[m].add_prefix(f"logging_"), counting_data[m].add_prefix(f"counting_")], add_index_suffix=False
        ) for m in common_measurements
    }
