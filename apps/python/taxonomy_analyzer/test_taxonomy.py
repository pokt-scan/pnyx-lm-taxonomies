import json
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd

import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.normpath(os.path.join(current_dir, os.pardir, os.pardir, os.pardir))
sys.path.insert(0, os.path.join(root_dir, "packages", "python"))


def main():
    from taxonomy_tools import helm_data as txm_helm_data
    from taxonomy_tools import utils as txm_utils

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(
        description="Script for testing taxonomy fitness against a dataset test collection."
    )

    # Add arguments for the paths
    parser.add_argument(
        "--taxonomy", "-t", type=str, required=True, help="Path to the taxonomy file"
    )
    parser.add_argument(
        "--data",
        "-d",
        action="append",
        type=str,
        help="Path (or multiple paths) to the HELM data or any other custom dataset.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for the resulting graphs.",
    )
    parser.add_argument(
        "--metrics",
        "-m",
        type=str,
        required=False,
        default="kendall,mutual_information",
        help="Metric or list of metrics (separated by commas) used to check the taxonomy.",
    )

    # Parse arguments from the command line
    args = parser.parse_args()
    TAXONOMY_PATH = args.taxonomy
    HELM_RESULTS_PATHS = args.data
    OUTPUT_PATH = args.output
    METRICS_USE = args.metrics.split(",")

    # Get taxonomy name
    taxonomy_name = os.path.basename(TAXONOMY_PATH).split(".")[0]
    print("Processing taxonomy: %s" % taxonomy_name)

    # Load taxonomy
    taxonomy_graph, labels_graph, undefined_edges, measurable_edges = (
        txm_utils.load_taxonomy(TAXONOMY_PATH, return_all=True, verbose=True)
    )

    # Get all the required datasets from the taxonomy graph
    datasets_list = txm_utils.get_taxonomy_datasets(taxonomy_graph)

    # Read all the required data from HELM
    helm_samples_dict = dict()
    for data_path in HELM_RESULTS_PATHS:
        helm_samples_dict = txm_helm_data.read_helm_data(
            data_path, datasets_list, current_dict=helm_samples_dict, verbose=True
        )

    # Filter for models that were tested on ALL datasets
    helm_samples_fullytested_dict = txm_utils.filter_for_full_samples(helm_samples_dict)
    if len(helm_samples_fullytested_dict) == 0:
        raise ValueError(
            "No data to process, no node in the taxonomy has available test data."
        )

    # Create taxonomy datasets metrics dataframe
    datasets_data_df = txm_utils.get_taxonomy_datasets_metrics_dataframe(
        helm_samples_fullytested_dict
    )
    # Save
    datasets_data_df.to_csv(
        os.path.join(
            OUTPUT_PATH,
            "%s" % taxonomy_name + "_dataset_metrics.csv",
        )
    )

    # Create taxonomy nodes metrics dataframe
    nodes_data_df = txm_utils.get_taxonomy_datasets_node_dataframe(
        helm_samples_fullytested_dict, taxonomy_graph, verbose=True
    )
    # Save
    nodes_data_df.to_csv(
        os.path.join(
            OUTPUT_PATH,
            "%s" % taxonomy_name + "_nodes_metrics.csv",
        )
    )

    # Get nodes correlations
    corr_dict_list = list()
    names_list = list()
    for metric_use in METRICS_USE:
        correlation_matrix, correlation_matrix_filtered, corr_dict = (
            txm_utils.get_taxonomy_nodes_correlation(
                nodes_data_df,
                taxonomy_graph,
                method=metric_use,
                verbose=True,
            )
        )
        # Save
        pd.DataFrame(correlation_matrix).to_csv(
            os.path.join(
                OUTPUT_PATH,
                "%s" % taxonomy_name + "_full_metric_%s.csv" % metric_use,
            ),
            index=False,
            header=False,
        )
        pd.DataFrame(correlation_matrix_filtered).to_csv(
            os.path.join(
                OUTPUT_PATH,
                "%s" % taxonomy_name + "_filtered_metric_%s.csv" % metric_use,
            ),
            index=False,
            header=False,
        )

        # Get the unbalanced correlation, using all possible models in each edge
        correlation_matrix_imbalanced, corr_dict_imbalanced = (
            txm_utils.get_taxonomy_per_edge_correlation(
                taxonomy_graph, helm_samples_dict, method=metric_use, verbose=True
            )
        )
        # Save
        pd.DataFrame(correlation_matrix_imbalanced).to_csv(
            os.path.join(
                OUTPUT_PATH,
                "%s" % taxonomy_name + "_imbalanced_metric_%s.csv" % metric_use,
            ),
            index=False,
            header=False,
        )

        # Track names and metrics for compilation
        corr_dict_list.append(corr_dict)
        corr_dict_list.append(corr_dict_imbalanced)
        names_list.append(metric_use)
        names_list.append("imabalanced_" + metric_use)

        # Plot compacto de todos los nodos contra todos
        if metric_use == "mutual_information":
            method_use = txm_utils.custom_mi_reg
        else:
            method_use = metric_use
        correlation_matrix = nodes_data_df.loc[:, (nodes_data_df != 0).any()].corr(
            method=method_use
        )
        # Create a heatmap for visualization
        im = plt.matshow(correlation_matrix, cmap="coolwarm")
        im.set_clim([-1.0, 1.0])
        # Add colorbar
        plt.colorbar()
        # Set column labels
        plt.xticks(
            range(len(correlation_matrix.columns)),
            correlation_matrix.columns,
            rotation=90,
        )
        plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
        # Set title
        plt.title("%s" % TAXONOMY_PATH.split("/")[-1])
        plt.draw()
        plt.savefig(
            os.path.join(
                OUTPUT_PATH,
                "%s" % taxonomy_name + "_taxonomy_metric_%s_matrix.png" % metric_use,
            ),
            bbox_inches="tight",
        )

    # Compile all jsons into a single one and save
    def add_metric(other_dicts, others_names):
        dict_out = dict()
        for other_dict, other_name in zip(other_dicts, others_names):
            for key in other_dict.keys():
                if key != "nodes":
                    dict_out[other_name] = other_dict[key]
        if other_dict.get("nodes", None) is None:
            return dict_out
        dict_out["nodes"] = dict()
        for key in other_dict["nodes"].keys():
            dict_out["nodes"][key] = add_metric(
                [other_dict["nodes"][key] for other_dict in other_dicts], others_names
            )
        return dict_out

    corr_dict_comp = dict()
    for key in corr_dict_list[0].keys():
        corr_dict_comp[key] = add_metric(
            [other_dict[key] for other_dict in corr_dict_list], names_list
        )
    with open(
        os.path.join(OUTPUT_PATH, "%s" % taxonomy_name + "_metrics_dict.json"), "w"
    ) as fp:
        json.dump(corr_dict_comp, fp, indent=4)

    ############################################################################
    # ----------- Images
    ############################################################################

    pos = nx.nx_pydot.graphviz_layout(
        taxonomy_graph, prog="dot"
    )  # Choose layout algorithm
    # Draw the graph with desired customizations
    plt.figure(figsize=(15, 6))  # Adjust width and height as desired
    nx.draw_networkx(
        taxonomy_graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="black",
        font_size=8,
    )
    plt.draw()
    plt.savefig(os.path.join(OUTPUT_PATH, "%s" % taxonomy_name + "_taxonomy_graph.png"))

    pos = nx.nx_pydot.graphviz_layout(
        labels_graph, prog="dot"
    )  # Choose layout algorithm
    # Draw the graph with desired customizations
    plt.figure(figsize=(25, 4))  # Adjust width and height as desired
    nx.draw_networkx(
        labels_graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        edge_color="black",
        font_size=8,
    )
    plt.draw()
    plt.savefig(
        os.path.join(
            OUTPUT_PATH, "%s" % taxonomy_name + "_dataset_assignment_graph.png"
        )
    )


# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()
