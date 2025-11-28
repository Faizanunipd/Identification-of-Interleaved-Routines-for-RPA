import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import normalized_mutual_info_score

import networkx as nx
from infomap import Infomap
from itertools import combinations
from collections import defaultdict
import community as community_louvain

from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery


def plot_graph_with_edge_labels(G):
    """
    Properly plots a directed graph G with separate edge labels for each direction.
    """
    plt.figure(figsize=(32, 28))
    pos = nx.spring_layout(G, seed=39)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1600)
    nx.draw_networkx_labels(G, pos)

    # Draw edges with curves
    nx.draw_networkx_edges(
        G,
        pos,
        # connectionstyle='arc3,rad=0.2',
        arrows=True,
        arrowsize=20
    )

    # Prepare edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    # Draw all edge labels at once
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        label_pos=0.2,  # center along the curve
        font_size=10
    )

    plt.axis('off')
    plt.show()


def plot_and_save_adj_matrix(G, plot_matrix, csv_filename="adjacency_matrix.csv"):
    """
    Display and save the adjacency matrix of a directed graph G.
    """

    # Get adjacency matrix (weighted)
    adj_matrix = nx.adjacency_matrix(G, weight='weight').todense()

    # Create a pandas DataFrame with node labels
    adj_matrix_df = pd.DataFrame(adj_matrix, index=G.nodes(), columns=G.nodes())

    # 1. Show the matrix
    # print("Adjacency Matrix:\n")
    # print(adj_matrix_df)

    # 2. Plot the matrix as a heatmap
    if plot_matrix:
        plt.figure(figsize=(10, 8))
        sns.heatmap(adj_matrix_df, annot=True, cmap='YlGnBu', cbar=True, linewidths=0.5, linecolor='black')
        plt.title("Adjacency Matrix Heatmap")
        plt.show()

    # 3. Save the matrix to CSV
    output_dir = "out/csv"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    adj_matrix_df.to_csv(f"{output_dir}/{csv_filename}")
    print(f"Adjacency matrix saved successfully to '{csv_filename}'.")



def plot_graph_with_clusters(G, partition, method="Louvain"):
    if method == "Infomap":
        # Infomap uses integer node IDs, so we need to map nodes in G to Infomap's numeric IDs
        node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
        # Map cluster labels from Infomap partition back to original node labels
        cluster_colors = [partition[node_mapping[node]] for node in G.nodes()]
    elif method == "Louvain":
        # Louvain clustering already has node labels as keys
        cluster_colors = [partition[node] for node in G.nodes()]

    # Get layout for node positions
    pos = nx.spring_layout(G, seed=39)

    # Set up the plot size
    plt.figure(figsize=(32, 28))

    # Draw nodes with colors representing clusters
    nx.draw_networkx_nodes(G, pos, node_size=1600, cmap=plt.cm.jet, node_color=cluster_colors, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=10)

    # Draw edges with curves
    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=20
    )

    # Prepare edge labels
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}

    # Draw all edge labels at once
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        label_pos=0.2,  # center along the curve
        font_size=10
    )

    # Title
    plt.title(f"Graph Clusters - {method} Community Detection")

    # Display the plot
    plt.axis('off')
    plt.show()


def discover_dfg(event_log):
    
    # Drop ground-truth label
    if "routine_type" in event_log.columns:
        event_log = event_log.drop(columns=["routine_type"])
    
    event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
    event_log = log_converter.apply(event_log)
    
    # Discover DFG
    dfg = dfg_discovery.apply(event_log)
    return dfg


def get_Network_Graph(dfg, plot_graph=False, plot_matrix=False, output_filename="Graph_Matrix_Directed.csv"):
    
    # Convert DFG to a NetworkX graph
    G = nx.DiGraph()
    
    for (src, tgt), weight in dfg.items():
        G.add_edge(src, tgt, weight=weight)

    plot_graph_with_edge_labels(G) if plot_graph else None
    plot_and_save_adj_matrix(G, plot_matrix, csv_filename=output_filename)
    return G


def to_Undirected(G, plot_graph=False, plot_matrix=False, output_filename="Graph_Matrix_UnDirected.csv"):
    # Convert to undirected graph and take the maximum weight between edges
    G_undirected = nx.Graph()
    
    # Iterate over directed edges and add them to the undirected graph
    for u, v, data in G.edges(data=True):
        if G_undirected.has_edge(u, v):
            # If edge exists, keep the maximum weight
            G_undirected[u][v]['weight'] = max(G_undirected[u][v]['weight'], data['weight'])
        else:
            # If edge doesn't exist, just add it
            G_undirected.add_edge(u, v, weight=data['weight'])
    
    plot_graph_with_edge_labels(G_undirected) if plot_graph else None
    plot_and_save_adj_matrix(G_undirected, plot_matrix, csv_filename=output_filename)

    return G_undirected


# === Calculate Dependency Score for each edge ===
def calculate_dependency_score1(G_directed):
    dependency_scores = {}

    for (A, B) in G_directed.edges():
        count_A_to_B = G_directed[A][B]["weight"]
        count_B_to_A = G_directed[B][A]["weight"] if G_directed.has_edge(B, A) else 0

        numerator = abs(count_A_to_B - count_B_to_A)
        denominator = count_A_to_B + count_B_to_A + 1  # Add 1 only here

        score = numerator / denominator
        dependency_scores[(A, B)] = score

    return dependency_scores


# === Calculate Dependency Score for each edge ===
def calculate_dependency_score2(G_directed):
    # Initialize a dictionary to store dependency scores
    dependency_scores = {}

    # Initialize a dictionary to store the total outgoing counts for each activity
    total_outgoing_count = {node: 0 for node in G_directed.nodes()}
    
    # Calculate total outgoing counts for each activity (node)
    for u, v, data in G_directed.edges(data=True):
        total_outgoing_count[u] += data['weight']  # Add the weight of the outgoing edge from u

    # Calculate the dependency score for each directed edge (A -> B)
    for (A, B) in G_directed.edges():
        # Count of A -> B (the weight of the edge from A to B)
        count_A_to_B = G_directed[A][B]["weight"]
        
        # Total outgoing count from A (sum of all outgoing edges from A)
        total_A_outgoing = total_outgoing_count[A]

        # Calculate the dependency score as the ratio of count_A_to_B to total_A_outgoing
        if total_A_outgoing == 0:
            # If total outgoing count from A is 0, avoid division by zero
            score = 0
        else:
            score = count_A_to_B / total_A_outgoing

        # Store the dependency score for the edge (A -> B)
        dependency_scores[(A, B)] = score

    return dependency_scores
    

# === Remove parallel activities (dependency score < 0.6) ===
def remove_parallel_activities(G_directed, dependency_scores, threshold):
    to_remove = [edge for edge, score in dependency_scores.items() if score < threshold]
    # print(f"\nRemoved Edges: {to_remove}")
    G_directed.remove_edges_from(to_remove)
    return G_directed


# === Replace the edge weights with the dependency score ===
def update_graph_with_dependency_score(G_directed, dependency_scores):
    # Update graph with dependency score as the new edge weight
    for (A, B), score in dependency_scores.items():
        if G_directed.has_edge(A, B):
            G_directed[A][B]["weight"] = score  # Update edge weight to dependency score
    return G_directed


def get_scored1_grpah_directed(G, threshold=0.6, plot_graph=False, plot_matrix=False, output_filename="Graph_Matrix_Directed_Score1.csv"):
    # creating the copy of the original object
    G_copy = G.copy()

    # Calculate dependency scores

    dependency_scores = calculate_dependency_score1(G_copy)
    # print(dependency_scores)
    G_directed_truncated = remove_parallel_activities(G_copy, dependency_scores, threshold)
    
    # Replace the edge weights with the dependency score in the truncated graph
    G_directed_updated = update_graph_with_dependency_score(G_directed_truncated, dependency_scores)
    
    plot_graph_with_edge_labels(G_directed_updated) if plot_graph else None
    plot_and_save_adj_matrix(G_directed_updated, plot_matrix, csv_filename=output_filename)

    return G_directed_updated


def get_scored2_grpah_directed(G, threshold=0.1, plot_graph=False, plot_matrix=False, output_filename="Graph_Matrix_Directed_Score2.csv"):
    # creating the copy of the original object
    G_copy = G.copy()

    # Calculate dependency scores
    dependency_scores = calculate_dependency_score2(G_copy)
    # print(dependency_scores)
    G_directed_truncated = remove_parallel_activities(G_copy, dependency_scores, threshold)
    
    # Replace the edge weights with the dependency score in the truncated graph
    G_directed_updated = update_graph_with_dependency_score(G_directed_truncated, dependency_scores)
    
    plot_graph_with_edge_labels(G_directed_updated) if plot_graph else None
    plot_and_save_adj_matrix(G_directed_updated, plot_matrix, csv_filename=output_filename)

    return G_directed_updated


def get_oprimal_modularity(G_undirected, gamma_min=0.1, gamma_max=3.0, values=20, plot_graph=False):
    gamma_values = np.linspace(gamma_min, gamma_max, values)
    num_runs = 5
    num_random_graphs = 10
    
    stability_scores = []
    modularity_real = []
    modularity_random = []
    community_counts = []
    
    for gamma in gamma_values:
        partitions = []
        modularities = []
    
        # print(f"=== Louvain Clustering for γ = {gamma:.2f} ===")
    
        for run in range(num_runs):
            partition = community_louvain.best_partition(G_undirected, resolution=gamma, weight='weight')
            partitions.append(partition)
            mod = community_louvain.modularity(partition, G_undirected, weight='weight')
            modularities.append(mod)
    
            # Group by cluster
            clustered_actions = defaultdict(list)
            for activity_name, cluster in partition.items():
                clustered_actions[cluster].append(activity_name)
    
            # # Print nicely
            # print(f"\nRun {run+1}:")
            # for cluster_num in sorted(clustered_actions):
            #     for activity in sorted(clustered_actions[cluster_num]):
            #         print(f"Gamma: {gamma:.2f}, Cluster: {cluster_num}, Activity: {activity}")
    
        # Stability (approximated using NMI)
        all_nmi = []
        for p1, p2 in combinations(partitions, 2):
            labels1 = list(p1.values())
            labels2 = list(p2.values())
            nmi = normalized_mutual_info_score(labels1, labels2)
            all_nmi.append(nmi)
        stability_scores.append(np.mean(all_nmi))
    
        modularity_real.append(np.mean(modularities))
        community_counts.append(np.mean([len(set(p.values())) for p in partitions]))
    
        # Random modularity
        random_modularities = []
        for _ in range(num_random_graphs):
            rand_G = nx.configuration_model([d for _, d in G_undirected.degree()])
            rand_G = nx.Graph(rand_G)
            rand_G.remove_edges_from(nx.selfloop_edges(rand_G))
            rand_partition = community_louvain.best_partition(rand_G, resolution=gamma)
            rand_mod = community_louvain.modularity(rand_partition, rand_G)
            random_modularities.append(rand_mod)
        modularity_random.append(np.mean(random_modularities))

    if plot_graph:
        # Plotting
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(gamma_values, stability_scores, marker='o')
        plt.title("A: Stability (1 - VI via NMI)")
        plt.xlabel("Gamma")
        plt.ylabel("Stability (NMI)")
        
        plt.subplot(1, 3, 2)
        plt.plot(gamma_values, modularity_real, marker='o', label="Real")
        plt.plot(gamma_values, modularity_random, marker='x', label="Random")
        plt.title("B: Modularity Comparison")
        plt.xlabel("Gamma")
        plt.ylabel("Modularity")
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(gamma_values, community_counts, marker='o')
        plt.title("C: Number of Communities")
        plt.xlabel("Gamma")
        plt.ylabel("Avg Communities")
        
        plt.tight_layout()
        plt.show()

    # Get the optimal gamma based on modularity (real modularity)
    optimal_gamma_modularity = gamma_values[np.argmax(modularity_real)]
    print(f"Optimal Gamma based on Modularity: {optimal_gamma_modularity:.2f}")
    
    # Get the optimal gamma based on stability (NMI)
    optimal_gamma_stability = gamma_values[np.argmax(stability_scores)]
    print(f"Optimal Gamma based on Stability: {optimal_gamma_stability:.2f}")
    
    return optimal_gamma_modularity, optimal_gamma_stability


def louvin_Clustering(G_undirected, optimal_gamma, document, plot_graph=False):
    # Apply Louvain using the selected optimal gamma (e.g., 1.5)
    final_partition = community_louvain.best_partition(G_undirected, resolution=optimal_gamma, weight='weight')
    
    plot_graph_with_clusters(G_undirected, final_partition) if plot_graph else None   

    final_clusters = defaultdict(list)

    for node, cluster in final_partition.items():
        final_clusters[cluster].append(node)
    
    # Print final community assignments
    print(f"\n=== Final Louvain Clustering for γ = {optimal_gamma} ===")
    for cluster_id in sorted(final_clusters):
        print(f"\nCluster {cluster_id}:")
        document.add_heading(f"\nCluster {cluster_id}:", level=6)
        for activity in sorted(final_clusters[cluster_id]):
            print(f" - {activity}")
            document.add_paragraph(f" - {activity}")
            
    return final_clusters, document


def display_cluster(infomap_partition, node_mapping, document):
    # Print clusters
    print("\n Infomap Clustering (Directed):")
    
    cluster_activity_pairs = []
    for node, cluster in infomap_partition.items():
        activity_name = list(node_mapping.keys())[list(node_mapping.values()).index(node)]
        cluster_activity_pairs.append((cluster, activity_name))
    
    # Sort the list by cluster number
    cluster_activity_pairs.sort()
    
    # Print sorted results
    for cluster, activity in cluster_activity_pairs:
        print(f"Cluster {cluster}: {activity}")
        document.add_paragraph(f"Cluster {cluster}: {activity}")
    return document

    
def infomap_clustering(G, document, plot_graph=False, MRT=None):
    # Always use a markov-time value, default to 1.0 if MRT is None
    current_MRT = MRT if MRT is not None else 1.0

    params = f"--directed --two-level --num-trials 20 --markov-time {current_MRT}"
    im = Infomap(params)
    node_to_id = {node: idx for idx, node in enumerate(G.nodes())}
    id_to_node = {idx: node for node, idx in node_to_id.items()}
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1.0)
        im.add_link(node_to_id[u], node_to_id[v], weight)
    im.run()
    cluster_assignments = {id_to_node[node.node_id]: node.module_id for node in im.nodes}
    final_clusters = defaultdict(list)
    for node, cluster_id in cluster_assignments.items():
        final_clusters[cluster_id].append(node)
        
    # Optional: plot the clustered graph
    if plot_graph:
        plot_graph_with_clusters(G, cluster_assignments, method="Infomap")

    return final_clusters, document


def min_max_scale_graph(G):
    # Get all edge weights in a list
    edge_weights = np.array([data['weight'] for u, v, data in G.edges(data=True)])

    # Apply Min-Max Scaling
    min_weight = edge_weights.min()
    max_weight = edge_weights.max()

    # Scale edge weights between 0 and 1
    for u, v, data in G.edges(data=True):
        data['weight'] = (data['weight'] - min_weight) / (max_weight - min_weight)

    return G