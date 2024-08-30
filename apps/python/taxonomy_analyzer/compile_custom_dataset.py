import numpy as np
import os
import json
import warnings
import tqdm
from copy import deepcopy
import argparse


def main():
    from taxonomy_tools import helm_data as txm_helm_data

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="This script compiles a series of HELM datasets instance results into a series of splits that can be used as new dataset results."
    )

    # Add arguments for the paths
    parser.add_argument(
        "--compilation",
        "-c",
        type=str,
        required=True,
        help="Path to the compilation definition file",
    )
    parser.add_argument(
        "--data",
        "-d",
        type=str,
        help="Path to the HELM data or any other custom dataset.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for the resulting graphs.",
    )
    parser.add_argument(
        "--ignore_wrong_splits",
        "-is",
        type=bool,
        required=False,
        default=True,
        help="Whether to error or continue when a task ID is not found for the requested split.",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=int,
        required=False,
        default=1,
        help="Verbosity level, the higher the more outputs seen.",
    )

    # Parse arguments from the command line
    args = parser.parse_args()
    TASK_COLLECTION_PATH = args.compilation
    HELM_RESULTS_PATH = args.data
    OUTPUT_PATH = args.output
    IGNORE_WRONG_SPLITS = args.ignore_wrong_splits
    verbosity = args.verbosity

    # Read dataset compilation definition file, error if dict entries are duplicate
    def dict_raise_on_duplicates(ordered_pairs):
        """Reject duplicate keys."""
        d = {}
        for k, v in ordered_pairs:
            if k in d:
                raise ValueError("duplicate key: %r" % (k,))
            else:
                d[k] = v
        return d

    with open(TASK_COLLECTION_PATH) as f:
        compilation_definition = json.load(
            f, object_pairs_hook=dict_raise_on_duplicates
        )

    # Recover all the requested data
    compilation_name = compilation_definition["compilation_name"]
    results_structure = dict()
    for dataset_entry in compilation_definition["datasets"]:
        # Get dataset entry data
        dataset_name = list(dataset_entry.keys())[0]
        metric_use = dataset_entry[dataset_name]["metric"]
        split_use = dataset_entry[dataset_name]["split"]
        print("Compiling dataset: %s" % dataset_name)
        dataset_spec = dataset_entry[dataset_name].get("dataset_spec", "")
        if dataset_spec != "":
            print("\tUsing spec: %s" % dataset_spec)

        # Get all datasets here
        data_folders = [
            m
            for m in os.listdir(HELM_RESULTS_PATH)
            if dataset_name in m and dataset_spec in m
        ]
        models_use = [
            txm_helm_data.split_helm_result_folder_name(m)["model"]
            for m in data_folders
        ]
        assert len(models_use) == len(set(models_use)), (
            "The selected dataset (%s) has multiple instances per model, please make sure the dataset spec is filtering sub-tasks in the target dataset"
            % dataset_name
        )

        # Retrieve all selected samples for each target partition
        target_dict = dict()
        samples = dataset_entry[dataset_name]["samples"]
        for sample_id in samples.keys():
            # Get destination partition in compilation
            target_compilation_partition = samples[sample_id]
            # Add to entry
            if target_compilation_partition not in target_dict.keys():
                target_dict[target_compilation_partition] = list()
            target_dict[target_compilation_partition].append(sample_id)

            # print(sample_id, target_compilation_partition)

        # For each folder in the list, we will retrieve all samples
        for folder in tqdm.tqdm(data_folders, total=len(data_folders)):
            # Get model name
            model_name = txm_helm_data.split_helm_result_folder_name(folder)["model"]
            # Load per-instance stats file
            this_stats_file = os.path.join(
                HELM_RESULTS_PATH, folder, "per_instance_stats.json"
            )
            if not os.path.exists(this_stats_file):
                raise ValueError("Cannot find per-instance results for %s" % folder)
            with open(this_stats_file) as f:
                per_instance_data = json.load(f)

            for target in target_dict.keys():
                # Initialize results vector
                metrics_vec = list()
                total_found = 0
                # Retrieve all samples
                for sample in target_dict[target]:
                    found_instance = False
                    found_metric = False
                    found_split = False
                    # Get this instance
                    for instance in per_instance_data:
                        if instance["instance_id"] == sample:
                            found_instance = True
                            for this_stat in instance["stats"]:
                                if this_stat["name"]["name"] == metric_use:
                                    found_metric = True
                                    if this_stat["name"]["split"] == split_use:
                                        found_split = True
                                        assert this_stat["count"] == 1, (
                                            "Selected metric %s in instance %s of entry %s has more than one result"
                                            % (
                                                metric_use,
                                                sample,
                                                folder,
                                            )
                                        )
                                        # Assign
                                        metrics_vec.append(this_stat["sum"])
                                        total_found += 1

                    if not found_instance:
                        raise ValueError(
                            'Instance id "%s" not found in "%s"' % (sample, folder)
                        )
                    if not found_metric:
                        raise ValueError(
                            'Metric "%s" for instance id "%s" not found in "%s"'
                            % (metric_use, sample, folder)
                        )
                    if not IGNORE_WRONG_SPLITS:
                        if not found_split:
                            raise ValueError(
                                'Split "%s" for metric "%s" for instance id "%s" not found in "%s"'
                                % (
                                    split_use,
                                    metric_use,
                                    sample,
                                    folder,
                                )
                            )
                    else:
                        if verbosity >= 2:
                            warnings.warn(
                                'Split "%s" for metric "%s" for instance id "%s" not found in "%s"'
                                % (
                                    split_use,
                                    metric_use,
                                    sample,
                                    folder,
                                )
                            )

                if total_found == 0:
                    raise ValueError(
                        'Target compilation "%s" has no valid entry in "%s"'
                        % (target, folder)
                    )
                if total_found <= 2:
                    if verbosity >= 1:
                        warnings.warn(
                            'Target compilation "%s" has less than 3 entries in "%s"'
                            % (target, folder)
                        )

                # Append results to compilation structure
                if model_name not in results_structure.keys():
                    results_structure[model_name] = dict()
                if target not in results_structure[model_name].keys():
                    results_structure[model_name][target] = dict()
                    results_structure[model_name][target]["metrics_vec"] = deepcopy(
                        metrics_vec
                    )
                    results_structure[model_name][target]["metric"] = dataset_entry[
                        dataset_name
                    ]["metric"]
                    results_structure[model_name][target]["split"] = dataset_entry[
                        dataset_name
                    ]["split"]
                else:
                    assert (
                        results_structure[model_name][target]["metric"] == metric_use
                    ), (
                        'Dataset "%s" selected metric "%s" for compilation "%s" is not compatible accross datasets. Check compilation definition file'
                        % (dataset_name, metric_use, target)
                    )
                    assert (
                        results_structure[model_name][target]["split"] == split_use
                    ), (
                        'Dataset "%s" selected split "%s" for compilation "%s" is not compatible accross datasets. Check compilation definition file'
                        % (dataset_name, split_use, target)
                    )
                    results_structure[model_name][target]["metrics_vec"] += metrics_vec

    # Save the results structure to disk
    for model_name in results_structure.keys():
        for target in results_structure[model_name].keys():
            metrics_vec = np.array(
                results_structure[model_name][target]["metrics_vec"]
            ).astype(float)
            # Calculate stats
            stats_dict = (
                {
                    "name": {
                        "name": results_structure[model_name][target]["metric"],
                        "split": results_structure[model_name][target]["split"],
                    },
                    "count": int(len(metrics_vec)),
                    "sum": metrics_vec.sum(),
                    "sum_squared": (metrics_vec**2).sum(),
                    "min": metrics_vec.min(),
                    "max": metrics_vec.max(),
                    "mean": metrics_vec.mean(),
                    "variance": metrics_vec.var(),
                    "stddev": metrics_vec.std(ddof=1),
                },
            )

            # Save target
            output_folder_name = (
                compilation_name + ":compilation=" + target + ",model=" + model_name
            )
            out_path_this = os.path.join(OUTPUT_PATH, output_folder_name)
            os.makedirs(out_path_this, exist_ok=True)
            with open(os.path.join(out_path_this, "stats.json"), "w") as fp:
                json.dump(stats_dict, fp, indent=4)

    return 0


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
