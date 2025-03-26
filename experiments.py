import ete3
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
import numpy as np
import pandas as pd

# Enhanced layout function with spacing parameters
def get_tree_layout(tree, horizontal_spacing=2, vertical_scale=1):
    node_positions = {}
    def traverse(node, x_offset=[0], y_position=0):
        if node.is_leaf():
            x = x_offset[0] * horizontal_spacing  # Increased horizontal spacing
            y = y_position * vertical_scale         # Apply vertical scaling
            node_positions[node.name] = (x, y)
            x_offset[0] += 1
        else:
            children = node.get_children()
            for child in children:
                traverse(child, x_offset, y_position - child.dist)
            child_x_positions = [node_positions[child.name][0] for child in children]
            node_positions[node.name] = (sum(child_x_positions) / len(child_x_positions), y_position * vertical_scale)
    traverse(tree, y_position=0)
    return node_positions

# Updated function for a nicer node layout
def plot_tree_network(G, tree_positions, edge_lengths):
    plt.figure(figsize=(12, 10))
    # Removed the gradient color; using a fixed color instead
    nx.draw(G, pos=tree_positions, with_labels=True, node_size=500,
            node_color="lightblue", edge_color="black", width=1.5, font_size=10)
    edge_labels = {(u, v): f"{w:.2f}" for (u, v), w in edge_lengths.items()}
    nx.draw_networkx_edge_labels(G, tree_positions, edge_labels=edge_labels, font_size=10)
    plt.title("Enhanced Phylogenetic Tree Layout", fontsize=14)
    plt.axis('off')
    plt.show()

def detect_communities(G, edge_lengths, num_clusters=3):
    # Convert to undirected graph and build a similarity matrix
    UG = G.to_undirected()
    nodes = list(UG.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    N = len(nodes)
    sim_matrix = np.zeros((N, N))
    for u, v, data in UG.edges(data=True):
        # Use similarity = 1/(1 + branch length)
        dist = edge_lengths.get((u, v), edge_lengths.get((v, u), 1))
        similarity = 1 / (1 + dist)
        i, j = node_index[u], node_index[v]
        sim_matrix[i, j] = similarity
        sim_matrix[j, i] = similarity
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(sim_matrix)
    return {nodes[i]: labels[i] for i in range(N)}

def plot_tree_network_with_communities(G, tree_positions, edge_lengths, communities):
    plt.figure(figsize=(12, 10))
    cmap = plt.cm.Set1
    colors = [communities[node] for node in G.nodes()]
    nx.draw(G, pos=tree_positions, with_labels=True, node_size=500,
            node_color=colors, cmap=cmap, edge_color="black", width=1.5, font_size=10)
    edge_labels = {(u, v): f"{w:.2f}" for (u, v), w in edge_lengths.items()}
    nx.draw_networkx_edge_labels(G, tree_positions, edge_labels=edge_labels, font_size=10)
    plt.title("Enhanced Phylogenetic Tree Layout with Communities", fontsize=14)
    
    # Create legend for communities
    unique_comms = sorted(set(communities.values()))
    from matplotlib.patches import Patch
    legend_patches = [Patch(color=cmap(float(comm)/max(unique_comms) if max(unique_comms) > 0 else 0, alpha=1.0),
                            label=f"Community {comm}") for comm in unique_comms]
    plt.legend(handles=legend_patches, loc="best")
    
    plt.axis('off')
    plt.show()

def compute_community_transfers(G, communities):
    transfers = {}
    for u, v, data in G.edges(data=True):
        source_comm = communities[u]
        target_comm = communities[v]
        if source_comm != target_comm:
            key = (source_comm, target_comm)
            transfers[key] = transfers.get(key, 0) + data.get('weight', 1)
    return transfers

# Add new functions for clustering based on transfers
def load_transfer_network():
    # Load the transfers CSV and build a graph based on transfers
    file_path = "/home/enzo/Documents/git/WP1/DeepGhosts/experiments/biological_data/aggregated_transfers.csv"
    df = pd.read_csv(file_path)
    G_transfer = nx.DiGraph()
    for _, row in df.iterrows():
        G_transfer.add_edge(row['from'], row['to'], weight=float(row['freq']))
    return G_transfer

def perform_transfer_clustering(G_transfer, num_clusters=5):
    nodes = list(G_transfer.nodes())
    node_index = {node: i for i, node in enumerate(nodes)}
    adj_matrix = np.zeros((len(nodes), len(nodes)))
    for u, v, data in G_transfer.edges(data=True):
        i, j = node_index[u], node_index[v]
        adj_matrix[i, j] = data['weight']
    spectral = SpectralClustering(n_clusters=num_clusters, affinity='precomputed', random_state=42)
    labels = spectral.fit_predict(adj_matrix)
    return {nodes[i]: labels[i] for i in range(len(nodes))}

def compute_exchange_matrix(transfers, num_communities):
    import numpy as np
    matrix = np.zeros((num_communities, num_communities))
    for (src, tgt), weight in transfers.items():
        matrix[src, tgt] = weight
    return matrix

# Main execution
if __name__ == '__main__':
    # Load and cluster the transfer network
    transfer_graph = load_transfer_network()
    transfer_communities = perform_transfer_clustering(transfer_graph, num_clusters=5)
    print("Transfer Communities:")
    for node, comm in transfer_communities.items():
        print(f"{node}: Community {comm}")
    
    # Load tree for visualization only
    tree_path = "/home/enzo/Documents/git/WP1/DeepGhosts/experiments/biological_data/data_cyano/gene_tree/aletree"
    tree = ete3.Tree(tree_path, format=1)
    tree_positions = get_tree_layout(tree, horizontal_spacing=2, vertical_scale=1)
    G = nx.DiGraph()
    for node in tree_positions:
        G.add_node(node)
    edge_lengths = {}
    for node in tree.traverse():
        for child in node.children:
            G.add_edge(node.name, child.name)
            edge_lengths[(node.name, child.name)] = child.dist
    
    # Visualize the tree; use the transfer communities for node coloring if available,
    # or default to lightblue if a node is not in the transfer network.
    communities_for_vis = {node: transfer_communities.get(node, -1) for node in G.nodes()}
    plot_tree_network_with_communities(G, tree_positions, edge_lengths, communities_for_vis)

    # Compute and display transfers between communities
    transfers_between = compute_community_transfers(G, transfer_communities)
    print("Transfers between communities:")
    for (src, tgt), weight in transfers_between.items():
        print(f"Community {src} -> Community {tgt}: {weight:.2f}")

    # Compute and display exchange matrix
    num_comms = max(transfer_communities.values()) + 1
    exchange_matrix = compute_exchange_matrix(transfers_between, num_comms)
    import pandas as pd
    comm_labels = [f"Comm {i}" for i in range(num_comms)]
    exchange_df = pd.DataFrame(exchange_matrix, index=comm_labels, columns=comm_labels)
    print("\nMatrix of exchanges between communities:")
    print(exchange_df)
