import os
import numpy as np
from typing import List, Tuple
import json
from copy import deepcopy
import warnings
import platform

# Get OS, windows has special needs
is_windows = platform.system() == "Windows"


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.normpath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir))


# This structure contains the translation of the taxonomy dataset strings
# into actual HELM dataset results.
# An entry in this dictionary does not need to be exactly one dataset, nor be
# constituted of the same dataset, arbitrary collections can be passed. Only
# keep in mind that the metrics should all be of the same type, since all
# results within a dataset entry will be averaged.
#
# Here is an example entry:
#  "LegalSupport": [
#         {
#             "name": "legal_support,method=multiple_choice_joint",  // Partial string of the result name, all the characters before ",model=..."
#             "metric": "exact_match",  // Type of metric to look for in the "stats.json" file. The first match is the one used.
#             "suffix": "", // Optional field to include for tests that include more partitions after the model name, use "---" to signal strict no suffix and "" for any suffix (without warnings)
#             "field": "mean",  // Field of the metric to keep, usually mean.
#             "split": "test",  // Split of the metric, some dataset have "test" split, other only have "validation"
#         }
#     ],
#
# The data will be read upon importing from the default path or the path indicated in an environment variable
helm_tests_config_path = os.environ.get(
    "HELM_TESTS_CONFIG_PATH", os.path.join(root_dir, "config", "helm_tests.json")
)
with open(helm_tests_config_path) as f:
    dataset_config = json.load(f)

# This config file contains data for each of the models that HELM tested (and potentially more)
models_config_path = os.environ.get(
    "MODELS_CONFIG_PATH", os.path.join(root_dir, "config", "models.json")
)
with open(models_config_path) as f:
    models_config = json.load(f)


def get_model_name_from_test_name(test_name):
    return test_name.split("model=")[-1].split(",")[0]


def read_helm_data(
    helm_data_path: str,
    datasets_list: List[str],
    verbose: bool = False,
    print_prefix: str = "",
    current_dict: dict = {},
    parameters_range: Tuple[float, float] = [0, 0],
) -> dict:
    """
    Given a dataset list from a taxonomy, loads all metrics from the HELM
    results directory.
    Returns a dictionary containing a key for each dataset and within each key
    a dictionary of models and their respective metrics values.
    """

    helm_samples_dict = current_dict

    # Get a list of tasks results from the helm dir
    tasks_dirs = os.listdir(helm_data_path)

    for dataset in datasets_list:
        if dataset in helm_samples_dict.keys():
            if verbose:
                print(
                    print_prefix
                    + "Skipping dataset %s, it is already present in the provided samples dictionary."
                    % dataset
                )
            continue

        # Get list of associated tasks
        task_list = dataset_config.get(dataset, None)
        if task_list is None:
            print(print_prefix + "Dataset not defined in config.t : %s" % dataset)
            continue
        tasks_results_dict = dict()
        ignored = False
        for task in task_list:
            if task["name"] == "IGNORE-ME":
                print(print_prefix + "Ignoring dataset : %s" % dataset)
                ignored = True
                continue
            task_name = task["name"]
            if is_windows:
                # replace the ":" with "-" in the test name
                task_name = task_name.replace(":", "-")
            # Get all results matching this task
            matching_results = list()
            for t_dir in tasks_dirs:
                if task_name + ",model" in t_dir or task_name + ":model" in t_dir:
                    # Check for suffix
                    suffixs = ""
                    for s in t_dir.split("model=")[-1].split(",")[1:]:
                        suffixs += "," + s
                    if len(suffixs) > 0:
                        if task.get("suffix", None) is not None:
                            if task["suffix"] in suffixs:
                                matching_results.append(t_dir)
                        else:
                            if verbose:
                                warnings.warn(
                                    "adding test with non-specified suffix, this can result into errors: %s"
                                    % t_dir
                                )
                                matching_results.append(t_dir)
                    elif task.get("suffix", None) is not None:
                        if task["suffix"] == "---":
                            # We are not requesting one either way, so add
                            matching_results.append(t_dir)
                        else:
                            # This task does not have a suffix but you requested one, ignore
                            continue
                    else:
                        # No suffix, no problem
                        matching_results.append(t_dir)
            if len(matching_results) == 0:
                if verbose:
                    print(print_prefix + "task not found : %s" % task["name"])

            # Get the tested models names
            tested_models = [
                get_model_name_from_test_name(result_name)
                for result_name in matching_results
            ]
            assert len(tested_models) == len(np.unique(tested_models)), (
                "(%s) Got %d unique models in a list of %d models."
                % (task["name"], len(np.unique(tested_models)), len(tested_models))
            )

            # Filter model sizes if requested
            if np.sum(parameters_range) > 0:
                # Sanity check of parameter range
                assert len(parameters_range) == 2
                assert parameters_range[0] < parameters_range[1]
                # Loop over all models found
                tested_models_use = list()
                for model in tested_models:
                    # Get model config
                    this_cfg = models_config.get(model, None)
                    if this_cfg is None:
                        raise ValueError(
                            "Cannot find config data for model %s in config file %s"
                            % (model, models_config)
                        )
                    # Check if within the given parameter and append
                    if (this_cfg["parameters"] > parameters_range[0]) and (
                        this_cfg["parameters"] <= parameters_range[1]
                    ):
                        tested_models_use.append(model)
                if len(tested_models_use) == 0:
                    if verbose:
                        print(
                            print_prefix
                            + "(Task : %s) No models found with the selected parameter range: [%g , %g]"
                            % (task["name"], parameters_range[0], parameters_range[1])
                        )
                # Replace list
                tested_models = deepcopy(tested_models_use)

            # Get the metrics for each model
            for this_result, this_model in zip(matching_results, tested_models):
                metric_val = None
                with open(os.path.join(helm_data_path, this_result, "stats.json")) as f:
                    task_results = json.load(f)
                for r in task_results:
                    try:
                        if (
                            r["name"]["name"] == task["metric"]
                            and r["name"]["split"] == task["split"]
                        ):
                            metric_val = r[task["field"]]

                            if task["metric"] == "bits_per_byte":
                                # just try to set this in the same range as accuracy...
                                metric_val /= 8
                                metric_val = 1.0 - metric_val
                            # The first is the one we want
                            break
                    except Exception:
                        pass
                if metric_val is None:
                    raise ValueError(
                        f"Requested metric {task['metric']}/{task['field']} in split {task['split']} not found in {this_result}.\nTask Results:\n {task_results}"
                    )

                # track all values from the tasks associated to this dataset
                if this_model not in tasks_results_dict.keys():
                    tasks_results_dict[this_model] = list()
                tasks_results_dict[this_model].append(metric_val)

        # Keep only models with all the tasks tested in this dataset and calculate the mean
        tasks_results_dict = {
            k: np.mean(v)
            for k, v in tasks_results_dict.items()
            if len(v) == len(task_list)
        }

        if not ignored:
            if len(tasks_results_dict) > 0:
                helm_samples_dict[dataset] = tasks_results_dict
            else:
                if verbose:
                    print(print_prefix + "No data found for dataset: %s" % dataset)

    return helm_samples_dict


def split_helm_result_folder_name(folder_name):
    split_data = dict()
    split_data["dataset"] = folder_name.split(":")[0]
    split_data["model"] = folder_name.split(",model=")[-1].split(",")[0]
    if "task" in folder_name:
        split_data["task"] = folder_name.split(":task=")[-1].split(",")[0]
    return split_data
