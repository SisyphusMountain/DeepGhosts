import pandas as pd
import os
import re
import subprocess
import concurrent.futures
import shutil
from ete3 import Tree
import numpy as np
import torch
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.utils import coalesce
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pickle
import argparse

def stats_df_fn(dir_path):
    """
        dir_path: str
            The path to the directory containing the reconciliation files    
        
    """
    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=['Gene_Index', 'Duplications', 'Transfers', 'Losses', 
                            'Total_Duplications', 'Total_Transfers', 'Total_Losses', 'Total_Speciations'])

    # Compile the regex pattern for "..._uml" extraction
    pattern = re.compile(r"reconciliation_(\d+)_uml")

    # Loop over each file in the directory
    for filename in os.listdir(dir_path):
        # Only open files that end with _uml
        if filename.endswith("_uml"):
            # Extract the gene index from the filename
            gene_index = re.findall(pattern, filename)[0]

            with open(os.path.join(dir_path, filename), 'r') as file:
                lines = file.readlines()

                for i, line in enumerate(lines):
                    # Check if this line starts with 'rate of'
                    if line.startswith('rate of'):
                        # The numbers are on the next line, split by tabs
                        numbers = lines[i + 1].split()
                        dup, trans, loss = map(float, numbers[1:4])

                    # Check if this line starts with '# of' and ends with 'Speciations'
                    elif line.startswith('# of') and line.strip().endswith('Speciations'):
                        # The numbers are on the next line, split by tabs
                        total_numbers = lines[i + 1].split()
                        total_dup, total_trans, total_loss, total_spec = map(float, total_numbers[1:5])

                # Add these numbers to the DataFrame
                new_data = pd.DataFrame({
                    'Gene_Index': [gene_index],
                    'Duplications': [dup],
                    'Transfers': [trans],
                    'Losses': [loss],
                    'Total_Duplications': [total_dup],
                    'Total_Transfers': [total_trans],
                    'Total_Losses': [total_loss],
                    'Total_Speciations': [total_spec]
                })
                df = pd.concat([df, new_data], ignore_index=True)
    return df

def transfers_df_fn(base_folder_path):
    """
    base_folder_path: str
        The folder containing all the .uTs reconciliation files

    """
    temp_dfs = []  # Initialize a list to hold dataframes temporarily
    pattern = re.compile(r"reconciliation_(\d+)_uTs")
    
    for ind, filename in enumerate(os.listdir(base_folder_path)):
        if filename.endswith("_uTs"):
            gene_index_match = re.findall(pattern, filename)
            if gene_index_match:
                gene_index = gene_index_match[0]
                file_path = os.path.join(base_folder_path, filename)
                corrected_file_path = f"{file_path}.corrected"
                
                # Use sed to create a corrected copy of the file
                sed_command = ['sed', 's/^[ \t]*//', file_path]
                with open(corrected_file_path, 'w') as corrected_file:
                    subprocess.run(sed_command, stdout=corrected_file)
                
                try:
                    # Read the corrected file into a DataFrame
                    temp_df = pd.read_csv(corrected_file_path, sep='\t', comment='#', header=None, names=['Donor', 'Receiver', 'Frequency'], skipinitialspace=True)
                    
                    # Add the gene index to the dataframe
                    temp_df['Gene_Index'] = gene_index
                    
                    # Collect the dataframe
                    temp_dfs.append(temp_df)
                except pd.errors.ParserError as e:
                    print(f"Error reading {corrected_file_path}: {e}")
                finally:
                    # Optionally, remove the corrected file to clean up
                    os.remove(corrected_file_path)
    
    # Concatenate all collected dataframes once
    df = pd.concat(temp_dfs, ignore_index=True) if temp_dfs else pd.DataFrame()
    return df

def rename_internal_nodes(newick_path, output_path):
    tree = Tree(newick_path, format=1)
    count = 0
    for node in tree.traverse():
        if not node.is_leaf():
            node.name = str(count)
            count += 1
    tree.write(outfile=output_path, format=1, format_root_node=True)

def process_tree_directory(args):
    """
    tree_dir: str
        The directory containing all the reconciliation files
    """
    tree_dir, translation_bin_path = args
    # Initialize the output directory
    output_dir = os.path.join(tree_dir, "processed_data_for_pytorch")
    
    # Remove the output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the stats dataframe
    stats_df = stats_df_fn(tree_dir + "/reconciliations")
    stats_df.to_csv(os.path.join(output_dir, "stats.csv"), index=False)
    
    # Get the transfers dataframe
    transfers_df = transfers_df_fn(tree_dir + "/reconciliations")
    transfers_df.to_csv(os.path.join(output_dir, "transfers.csv"), index=False)

    complete_tree_path = os.path.join(tree_dir, "complete_species_tree.nwk")
    shutil.copy(complete_tree_path, os.path.join(output_dir, "complete_species_tree.nwk"))

    rename_internal_nodes(complete_tree_path, os.path.join(output_dir, "complete_species_tree_internal_nodes_renamed.nwk"))

    sampled_tree_path = os.path.join(tree_dir, "sampled/sampled_trees/sampled_species_tree.nwk")
    shutil.copy(sampled_tree_path, os.path.join(output_dir, "sampled_species_tree.nwk"))
    
    newick_pattern = re.compile(r"S:\s*(\([^;]+;)", re.DOTALL)

    sampled_ale_tree_path = os.path.join(tree_dir, "reconciliations/reconciliation_0_uml")
    sampled_ale_tree_target_path = os.path.join(output_dir, "ale_sampled_species_tree.nwk")
    if os.path.exists(sampled_ale_tree_path):
        with open(sampled_ale_tree_path, 'r') as file:
            content = file.read()
        newick_match = re.search(newick_pattern, content)
        if newick_match:
            newick = newick_match.group(1)
            with open(sampled_ale_tree_target_path, 'w') as file:
                file.write(newick)
        else:
            print(f"Newick pattern not found in {sampled_ale_tree_path}")
    # TODO: remove the hardcoded path
    translation_command = [translation_bin_path,
                           tree_dir + "/processed_data_for_pytorch/sampled_species_tree.nwk",
                           tree_dir + "/processed_data_for_pytorch/ale_sampled_species_tree.nwk",
                           tree_dir + "/processed_data_for_pytorch"]
    
    result = subprocess.run(translation_command, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("Error executing command:")

def is_part_of_sampled_tree(complete_tree, sampled_tree):
    """This function takes a complete tree and a sampled tree and marks nodes that are either present in the sampled tree,
    or which have a descendant that is present in the sampled tree.
    
    It needs to take the trees sampled_species_tree.nwk and complete_species_tree.nwk because they should have coherent names.
    """
    sampled_tree_nodes = set([n.name for n in sampled_tree.traverse()])

    for node in complete_tree.traverse():
        descendants = node.get_descendants()
        if node.name in sampled_tree_nodes:
            node.add_features(in_sampled_tree = 1)
        elif any(descendant.name in sampled_tree_nodes for descendant in descendants):
            # Not part of the sampled tree, but has a descendant that is.
            node.add_features(in_sampled_tree = 2)
        else:
            node.add_features(in_sampled_tree = 0)
    return complete_tree, sampled_tree

def lists_to_tensors(sampled_tree):
    """Makes the times_list attribute of each node in the sampled tree a tensor.
    Returns the sampled tree."""
    for node in sampled_tree.traverse():
        node.add_features(times_list=torch.tensor(node.times_list, dtype=torch.float32))
    return sampled_tree

def process_trees(complete_tree, sampled_tree):
    """Attributes the ghost nodes' times to the corresponding nodes in the sampled tree."""
    # Initialize an empty times_list attribute for each node in the sampled tree
    for node in sampled_tree.traverse():
        node.add_features(times_list=[])
        node.add_features(ghost_descendants=[])
        node.add_features(LTT_variation = []) # List of +1 or -1, depending on whether the ghost node is terminal (lowering the LTT) or internal (increasing the LTT by 1)

    for node in complete_tree.traverse():
        # If the node is in the sampled tree, we do nothing. 
        # If the node has a descendant in the sampled tree, but is not in the sampled tree, we find the node it attaches to in the sampled tree.
        # If the node is not in the sampled tree and has no descendant in the sampled tree, we need to find which node in the sampled tree it attaches to.
        # If the node is a leaf, then either it is part of the sampled tree, in which case it is not added as a ghost tree, or 
        # it is not part of the sampled tree, and it should be counted as a death. When a ghost branch G stemming from a branch B
        # dies at time t, the "ghost LTT" of the branch B decreases by 1 at time t.
        if node.in_sampled_tree != 1:#In this case, the node is not in the sampled tree, so it counts as a ghost branch
            if node.in_sampled_tree == 0:#In this case either the node's sister has a descendant in the sampled tree it will attach on, or we must move up to find the parent's sibling
                sibling = node.get_sisters()[0] if node.get_sisters() else None
            elif node.in_sampled_tree == 2:
                sibling = node.get_children()[0] if node.get_children()[0].in_sampled_tree!=0 else node.get_children()[1]

            # if the sibling is in the sampled tree, then we just append to it (and we don't get in the loop)
            # else, if the sibling has a descendant in the sampled tree, then we append to the descendant (and we don't get in the loop)
            # While sibling's in_sampled_tree is not 1
            while sibling and sibling.in_sampled_tree != 1:
                # If sibling's in_sampled_tree is 0, move up to parent's sibling. In this case, the sibling does not have an extant descendent either
                if sibling.in_sampled_tree == 0:
                    sibling = sibling.up.get_sisters()[0] if sibling.up and sibling.up.get_sisters() else None

                # If sibling's in_sampled_tree is 2, move down to children. In this case, it should only have one child with an extant descendent
                elif sibling.in_sampled_tree == 2:
                    siblings = sibling.get_children()
                    sibling = siblings[0] if siblings[0].in_sampled_tree!=0 else siblings[1]

            # If we found a sibling in the sampled tree
            if sibling and sibling.in_sampled_tree == 1:
                # Get the node in the sampled tree corresponding to the sibling
                sampled_tree_node = sampled_tree&sibling.name
                # Append the node's distance to the root to the sibling's times_list attribute
                sampled_tree_node.times_list.append(node.get_distance(complete_tree))
                sampled_tree_node.ghost_descendants.append((node.name, node.get_distance(complete_tree)))
                sampled_tree_node.LTT_variation.append(-1 if node.is_leaf() else 1)
                node.add_features(sampled_tree_parent = sibling.name)
    # Return the processed sampled tree
    return lists_to_tensors(sampled_tree)

def compute_ghost_length(times_list, LTT_variation, time_endpoint):
    """
    times_list is the list of times at which a branch either splits or goes extinct.
    LTT_variation is a vector with values in {1, -1}, indicating whether the given time
    corresponds to an extinction or a split.
    
    We can compute the ghost length in the following way: when a branch splits, we count it as 
    +1*(time_endpoint - time_split), and when a branch dies, we count it as -1*(time_endpoint - time_split) 
    """
    # First compute the list of the intervals.
    np_events = time_endpoint - np.array(times_list)
    np_LTT_variation = np.array(LTT_variation, dtype = float)
    return np.dot(np_events, np_LTT_variation)

def assign_ghost_lengths(sampled_tree):
    """Every node needs to have attributes event_times, and LTT_variation"""
    time_endpoint = sampled_tree.get_farthest_leaf()[1]
    for node in sampled_tree.traverse():
        node.add_features(ghost_length = compute_ghost_length(node.times_list,
                                                              node.LTT_variation,
                                                              time_endpoint))
        
    return sampled_tree

"""def create_edge_index_and_attr(sampled_tree, transfers_df):
    # Define a function that gets the index from sampled_tree given a node name
    def get_index(node_name):
        return (sampled_tree & str(node_name)).index

    # Apply this function to every element in the 'donor' and 'receiver' columns
    donors = transfers_df['donor'].apply(get_index).values
    receivers = transfers_df['receiver'].apply(get_index).values

    # Convert the 'donor' and 'receiver' columns to PyTorch tensors
    donors_tensor = torch.tensor(donors, dtype=torch.long)
    receivers_tensor = torch.tensor(receivers, dtype=torch.long)

    # Stack the tensors into a 2xN tensor
    edge_index = torch.stack((donors_tensor, receivers_tensor), dim=0).contiguous()

    # Convert 'frequency' column to a PyTorch tensor
    edge_attr = torch.tensor(transfers_df['frequency'].values, dtype=torch.float).view(-1, 1).contiguous()

    return edge_index, edge_attr"""

def remove_parentheses(s):
    return re.sub(r"\([^)]*\)", "", s)

def open_transfers_df(transfers_df_path):
    # The donor and receiver columns are opened as strings,
    # so that we can remove the parentheses from the names
    # using strip_df
    transfers_df = pd.read_csv(
        transfers_df_path,
        sep=",",
        skiprows=1,
        header=None,
        names=["Donor", "Receiver", "Frequency", "Gene_Index"],
        dtype={"Donor": str, "Receiver": str, "Frequency": float, "Gene_Index": int}
    )
    
    return transfers_df

def strip_df(df):
    """ALE outputs a donor and a receiver for each event.
    The donor and the receiver can have the format 'new_name(old_name)'
    where new_name is the name given by ALE and old_name was the previous name of the node."""
    df["Donor"] = df["Donor"].apply(remove_parentheses)
    df["Receiver"] = df["Receiver"].apply(remove_parentheses)
    return df

def open_translation_df(translation_df_path):
    translation_df = pd.read_csv(translation_df_path, sep = ",", skiprows=1, header = None, names = ["ale_name", "original_name"])
    return translation_df

def translate_tree(sampled_tree, translation_df):
    # Convert DataFrame to a dictionary for faster lookup
    name_dict = translation_df.set_index('original_name')['ale_name'].to_dict()

    # Traverse through each node in the tree
    for node in sampled_tree.traverse():
        try:
            # Directly use the dictionary key which raises KeyError if the key is not found
            node.add_features(ale_name=name_dict[int(node.name)])
        except KeyError:
            print(translation_df)
            raise KeyError(f"No ale_name found for original_name '{node.name}' in translation_df")
    return sampled_tree

def give_dist_to_root(sampled_tree):
    for node in sampled_tree.traverse():
        if node.is_root():
            node.add_features(dist_to_root = 0)
        else:
            node.add_features(dist_to_root = node.up.dist_to_root + node.dist)
    return sampled_tree

def filter_transfers_df(transfers_df, threshold):
    """Filter the dataframe, keeping only transfers with a frequency higher than a threshold value."""
    return transfers_df[transfers_df["Frequency"] >= threshold]

def add_life_dates_to_df(tree, df):
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Dictionary to memoize distances
    life_data = {}

    # Helper function to get distances
    def get_life_data(ale_node_name):
        if ale_node_name not in life_data:
            node = tree.search_nodes(ale_name=ale_node_name)[0]  # Assume each name uniquely identifies a node

            # Memoize data
            life_data[ale_node_name] = {
                'begin_life': node.up.dist_to_root if node.up else 0,
                'end_life': node.dist_to_root
            }
        return life_data[ale_node_name]

    # Initialize empty lists for new data
    donor_begin = []
    donor_end = []
    receiver_begin = []
    receiver_end = []


    # Iterate through each row in the dataframe
    for idx, row in df_copy.iterrows():
        # For Donor

        donor_data = get_life_data(row['Donor'])
        donor_begin.append(donor_data['begin_life'])
        donor_end.append(donor_data['end_life'])

        # For Receiver
        receiver_data = get_life_data(row['Receiver'])
        receiver_begin.append(receiver_data['begin_life'])
        receiver_end.append(receiver_data['end_life'])

    # Add the collected data as new columns in the dataframe copy
    df_copy['Donor_Begin_Life'] = donor_begin
    df_copy['Donor_End_Life'] = donor_end
    df_copy['Receiver_Begin_Life'] = receiver_begin
    df_copy['Receiver_End_Life'] = receiver_end

    return df_copy

def add_comparison_column(df):
    # Define the conditions for assigning values to the new column
    condition1 = df['Donor_End_Life'] <= df['Receiver_Begin_Life']
    condition2 = df['Donor_Begin_Life'] >= df['Receiver_End_Life']

    # Use numpy.select to vectorize the condition checking
    df['Comparison_Result'] = np.select(
        [condition1, condition2],
        [1, -1],
        default=0
    )
    
    return df

def add_br_length(sampled_tree):
    for node in sampled_tree.traverse():
        node.add_features(br_length = node.dist)
    return sampled_tree

def determine_descendant_relations(tree, df):
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()


    # Prepare to store the descendant sets for quick lookup
    descendants_cache = {}

    # Function to fetch and cache descendants
    def get_descendants(node_name):
        if node_name not in descendants_cache:
            node = tree.search_nodes(ale_name=int(node_name))[0]  # Assume each name uniquely identifies a node
            descendants = set(n.ale_name for n in node.get_descendants())
            descendants_cache[node_name] = descendants
        return descendants_cache[node_name]

    # Vector to store the results
    results = []

    # Iterate through the prefiltered DataFrame
    for idx, row in df.iterrows():
        if row["Comparison_Result"] == 0:
            results.append(0) # The nodes are contemporaneous, they cannot have descendant relationships to each other.
            continue
        donor_descendants = get_descendants(row['Donor'])
        receiver_descendants = get_descendants(row['Receiver'])

        # Check descendant relationships
        if row['Receiver'] in donor_descendants:
            results.append(1)
        elif row['Donor'] in receiver_descendants:
            results.append(-1)
        else:
            results.append(0)

    # Assign the results to the DataFrame
    df['Descendant_Relation'] = results

    return df

import pandas as pd

def create_stats_df(sampled_tree, transfers_df):
    # Dictionary to cache node attributes
    node_cache = {}

    # Function to fetch and cache node attributes
    def get_node_attributes(ale_node_name):
        if ale_node_name not in node_cache:
            nodes = sampled_tree.search_nodes(ale_name=ale_node_name)
            if nodes:
                node = nodes[0]
                node_cache[ale_node_name] = {
                    'br_length': getattr(node, 'br_length', None),
                    'dist_to_root': getattr(node, 'dist_to_root', None)
                }
            else:
                node_cache[ale_node_name] = {'br_length': None, 'dist_to_root': None}
        return node_cache[ale_node_name]

    # Ensure the DataFrame has the expected structure
    expected_columns = [
        'Donor', 'Receiver', 'Frequency', 'Gene_Index',
        'Donor_Begin_Life', 'Donor_End_Life',
        'Receiver_Begin_Life', 'Receiver_End_Life',
        'Comparison_Result', 'Descendant_Relation'
    ]
    assert list(transfers_df.columns) == expected_columns, "DataFrame does not have the expected columns"

    # Prepare the DataFrame
    df = transfers_df.copy()

    # Extract all unique ale_names from the tree
    all_nodes = pd.DataFrame([node.ale_name for node in sampled_tree.traverse() if hasattr(node, 'ale_name')], columns=['name'])

    # Group and sum frequencies for donors and receivers
    donors_sums = df.groupby("Donor")["Frequency"].sum().reset_index()
    donors_sums.rename(columns={"Donor": "name", "Frequency": "N_transfers_donor"}, inplace=True)

    recips_sums = df.groupby("Receiver")["Frequency"].sum().reset_index()
    recips_sums.rename(columns={"Receiver": "name", "Frequency": "N_transfers_recip"}, inplace=True)

    # Merge donor and receiver sums using a common 'name' column
    stats_df = pd.merge(donors_sums, recips_sums, on="name", how="outer").fillna(0)
    # Convert both 'name' columns to integers (if possible)
    all_nodes['name'] = all_nodes['name'].astype(int)
    stats_df['name'] = stats_df['name'].astype(int)


    # Ensure all nodes are in stats_df by merging with all_nodes

    stats_df = pd.merge(all_nodes, stats_df, on="name", how="outer").fillna(0)
    
    

    # Calculate transfer sums for direct and general descendants
    df['to_descendant'] = df.apply(lambda row: row['Frequency'] if row['Comparison_Result'] == 1 else 0, axis=1)
    df['to_direct_descendant'] = df.apply(lambda row: row['Frequency'] if row['Descendant_Relation'] == 1 else 0, axis=1)

    direct_descendant_sums = df.groupby("Donor")["to_direct_descendant"].sum().reset_index()
    direct_descendant_sums.rename(columns={"Donor": "name"}, inplace=True)
    descendant_sums = df.groupby("Donor")["to_descendant"].sum().reset_index()
    descendant_sums.rename(columns={"Donor": "name"}, inplace=True)

    direct_descendant_sums['name'] = direct_descendant_sums['name'].astype(int)
    descendant_sums['name'] = descendant_sums['name'].astype(int)
    
    # Merge these sums into the main stats DataFrame
    stats_df = pd.merge(stats_df, direct_descendant_sums, on="name", how="left")
    stats_df = pd.merge(stats_df, descendant_sums, on="name", how="left")

    # Add tree node attributes
    stats_df['br_length'] = stats_df['name'].apply(lambda x: get_node_attributes(x)['br_length'])
    stats_df['dist_to_root'] = stats_df['name'].apply(lambda x: get_node_attributes(x)['dist_to_root'])

    # Fill missing values
    stats_df.fillna(0, inplace=True)

    return stats_df

def give_attributes(sampled_tree, stats_df):
    """
    Gives the attributes dist_to_root, br_length, N_transfers_donor, N_transfers_recip,
    to_direct_descendant, and to_descendant to each node in sampled_tree based on the stats_df dataframe.
    
    Args:
        sampled_tree: The tree containing nodes with an 'ale_name' attribute.
        stats_df: A DataFrame with columns including 'name', 'dist_to_root', 'br_length', 
                  'N_transfers_donor', 'N_transfers_recip', 'to_direct_descendant', 'to_descendant'.
    """
    # Index the DataFrame by 'name' for quick lookup
    stats_indexed = stats_df.set_index('name')

    # Traverse all nodes in the tree
    for node in sampled_tree.traverse():
        # Match each node's ale_name with the DataFrame index
        node_data = stats_indexed.loc[node.ale_name]

        # Assign attributes to each node from the DataFrame
        node.add_features(
            dist_to_root=node_data['dist_to_root'],
            br_length=node_data['br_length'],
            N_transfers_donor=node_data['N_transfers_donor'],
            N_transfers_recip=node_data['N_transfers_recip'],
            to_direct_descendant=node_data['to_direct_descendant'],
            to_descendant=node_data['to_descendant']
        )
    return sampled_tree

def give_index(sampled_tree):
    counter = 0
    for node in sampled_tree.traverse():
        node.add_features(index = counter)
        counter += 1
    return sampled_tree

def rescale_stats_df(stats_df, scaling_factor, use_transfers_rescaling = False):
    """We rescale the tree branches, so we must rescale other lengths in the problem. However,
    we can choose not to change the number of transfers, which would be equivalent to multiplying
    the transfer rate by the scaling factor."""
    stats_df['br_length'] = stats_df['br_length'] / scaling_factor
    stats_df['dist_to_root'] = stats_df['dist_to_root'] / scaling_factor
    stats_df['L_ghost_branch'] = stats_df['L_ghost_branch'] / scaling_factor
    if use_transfers_rescaling:
        stats_df['N_transfers_donor'] = stats_df['N_transfers_donor'] / scaling_factor
        stats_df['N_transfers_recip'] = stats_df['N_transfers_recip'] / scaling_factor
        stats_df['to_direct_descendant'] = stats_df['to_direct_descendant'] / scaling_factor
        stats_df['to_descendant'] = stats_df['to_descendant'] / scaling_factor
    return stats_df

def rescale_tree(sampled_tree, rescale_transfers = False):
    """Rescales features of the species tree that are proportional to the tree height."""
    s = sampled_tree.scaling_factor
    for node in sampled_tree.traverse():
        node.br_length /= s
        node.dist /= s
        node.dist_to_root /= s
        node.ghost_length /= s
        if rescale_transfers:
            node.N_transfers_donor /= s
            node.N_transfers_recip /= s
            node.to_descendant /= s
            node.to_direct_descendant /= s
    return sampled_tree

def create_edge_index_and_attr(sampled_tree, transfers_df):
    # Define a function that gets the index from sampled_tree given a node name
    index_translation = {node.ale_name:node.index for node in sampled_tree.traverse()}


    # Apply this function to every element in the 'donor' and 'receiver' columns
    donors = transfers_df['Donor'].map(index_translation).values
    receivers = transfers_df['Receiver'].map(index_translation).values

    # Convert the 'donor' and 'receiver' columns to PyTorch tensors
    donors_tensor = torch.tensor(donors, dtype=torch.long)
    receivers_tensor = torch.tensor(receivers, dtype=torch.long)

    # Stack the tensors into a 2xN tensor
    edge_index = torch.stack((donors_tensor, receivers_tensor), dim=0).contiguous()

    # Convert 'frequency' column to a PyTorch tensor
    edge_attr = torch.tensor(transfers_df['Frequency'].values, dtype=torch.float).view(-1, 1).contiguous()

    return edge_index, edge_attr 

def make_parenthood_graph(sampled_tree):
    """Adds an index to each node that corresponds to the index for the pyg tree, constructs the parenthood graph, and returns
    the tensor of node features and the parenthood graph."""
    parenthood_edges = []
    x = []
    node_counter = 0


    for node in sampled_tree.traverse():
        if not node.is_root():
            parenthood_edges.append([node.up.index, node.index])
            node.add_features(index = node_counter)

        x.append([node.dist_to_root,
            node.br_length,
            node.N_transfers_donor,
            node.N_transfers_recip,
            node.to_direct_descendant,
            node.to_descendant,]
            )

        node_counter += 1

    x = torch.tensor(x, dtype=torch.float32).contiguous()
    parenthood_edges = torch.tensor(parenthood_edges, dtype=torch.long).t().contiguous()
    
    if __debug__:
        assert all([hasattr(node, "index") for node in sampled_tree.traverse()]), "Not all nodes have an index attribute."
        assert x.shape == (len(sampled_tree)*2-1, 6), f"x has the wrong shape. {x.shape=}"
    return x, parenthood_edges


def create_y_L_ghost(sampled_tree):
    """Creates the target for the deep learning model. In this case,
    we only predict the ghost branch length of each node and not a density matrix."""
    side = 2*len(sampled_tree) - 1
    y_matrix = torch.zeros(side, dtype=torch.float32)
    for node_1 in sampled_tree.traverse():
        if not node_1.is_root():
            y_matrix[node_1.index] = node_1.ghost_length
    return y_matrix


def tree_to_pyg(sampled_tree, transfers_df):
    """This function can construct a single pyg_tree.
    We do not use it for bootstrapping, because open_and_bootstrap_batch
    is more efficient.
    This function does not rescale anything. It only builds the tree from previously processed information.
    The input of this function should have gone through tree_preprocessing"""
    pyg_tree = HeteroData()
    x, parenthood_edges = make_parenthood_graph(sampled_tree)
    HGT_edges, HGT_attr = create_edge_index_and_attr(transfers_df = transfers_df, sampled_tree=sampled_tree)
    HGT_edges, HGT_attr = coalesce(edge_index=HGT_edges,
                                    edge_attr=HGT_attr,
                                    reduce="add",)

    y = create_y_L_ghost(sampled_tree)
    pyg_tree["node"].x = x
    pyg_tree["node", "is_parent_of", "node"].edge_index = parenthood_edges
    pyg_tree["node", "sends_gene_to", "node"].edge_index = HGT_edges
    pyg_tree["node", "sends_gene_to", "node"].edge_attr = HGT_attr
    pyg_tree["node"].y = y
    return pyg_tree


def files_to_pyg_tree(complete_tree_path, sampled_tree_path, transfers_df_path, translation_df_path, frequency_threshold):
    complete_tree = Tree(complete_tree_path, format=1)
    sampled_tree = Tree(sampled_tree_path, format=1)
    complete_tree, sampled_tree = is_part_of_sampled_tree(complete_tree, sampled_tree)
    sampled_tree = process_trees(complete_tree, sampled_tree)
    assign_ghost_lengths(sampled_tree)
    transfers_df = open_transfers_df(transfers_df_path)
    transfers_df = strip_df(transfers_df)
    transfers_df["Donor"] = transfers_df["Donor"].astype(int)
    transfers_df["Receiver"] = transfers_df["Receiver"].astype(int)
    translation_df = open_translation_df(translation_df_path)
    sampled_tree = translate_tree(sampled_tree, translation_df)
    sampled_tree = give_dist_to_root(sampled_tree)
    transfers_df = filter_transfers_df(transfers_df, threshold=frequency_threshold)
    transfers_df = add_life_dates_to_df(sampled_tree, transfers_df)
    sampled_tree = add_br_length(sampled_tree)
    transfers_df = add_comparison_column(transfers_df)
    transfers_df = determine_descendant_relations(sampled_tree, transfers_df)
    stats_df = create_stats_df(sampled_tree, transfers_df)
    sampled_tree = give_attributes(sampled_tree, stats_df)
    sampled_tree = give_index(sampled_tree)
    sampled_tree.add_features(scaling_factor = sampled_tree.get_farthest_leaf()[1])
    sampled_tree = rescale_tree(sampled_tree)
    
    return tree_to_pyg(sampled_tree, transfers_df)

# Worker function that creates the necessary paths and returns pyg_tree
def process_tree(i, frequency_threshold, dataset_path):
    base_path = dataset_path + f"/species_tree_{i}/processed_data_for_pytorch/"
    complete_tree_path = base_path + "complete_species_tree.nwk"
    sampled_tree_path = base_path + "sampled_species_tree.nwk"
    transfers_df_path = base_path + "transfers.csv"
    translation_df_path = base_path + "translations.csv"
    frequency_threshold = frequency_threshold

    pyg_tree = files_to_pyg_tree(
        complete_tree_path,
        sampled_tree_path,
        transfers_df_path,
        translation_df_path,
        frequency_threshold,
    )

    return pyg_tree

def fuse_stats_csv(dataset_path, output_path):
    """
    Fuse all stats.csv files across species_tree folders into a global stats.csv.
    
    Args:
        dataset_path (str): Path to the dataset containing species_tree_* directories.
        output_path (str): Path to save the global stats.csv file.
    """
    all_stats = []
    species_tree_pattern = re.compile(r"species_tree_(\d+)$")

    for folder in os.listdir(dataset_path):
        if species_tree_pattern.match(folder):
            stats_file = os.path.join(dataset_path, folder, "processed_data_for_pytorch", "stats.csv")
            if os.path.exists(stats_file):
                df = pd.read_csv(stats_file)
                df['species_tree'] = folder  # Add a column to track the source folder
                all_stats.append(df)

    if all_stats:
        global_stats = pd.concat(all_stats, ignore_index=True)
        global_stats.to_csv(output_path, index=False)
        print(f"Global stats saved to {output_path}")
    else:
        print("No stats.csv files found.")

def detect_species_tree_folders(dataset_path):
    """Detects all folders matching the pattern 'species_tree_{i}' and returns sorted list of indices."""
    pattern = re.compile(r"species_tree_(\d+)$")
    indices = [
        int(match.group(1)) for folder in os.listdir(dataset_path)
        if (match := pattern.match(folder)) and os.path.isdir(os.path.join(dataset_path, folder))
    ]
    return sorted(indices)

def process_single_tree(args):
    """Wrapper function to unpack arguments and call process_tree."""
    i, frequency_threshold, dataset_path = args
    return process_tree(i, frequency_threshold, dataset_path)

def process_all_species_trees(dataset_path, num_processes=None, frequency_threshold=0.1, pickle_path=None, translation_bin_path=None):
    """
    Processes all species trees found in the specified dataset path in parallel and saves the results as a pickled file.

    Args:
        dataset_path (str): Base path containing species tree folders.
        num_processes (int, optional): Number of parallel processes to use. Defaults to os.cpu_count().
        frequency_threshold (float, optional): Frequency threshold for filtering transfers. Defaults to 0.1.
        pickle_path (str, optional): Path to save the pickled results. If None, results are not saved.

    Returns:
        list: List of processed PyG trees.
    """
    num_processes = num_processes or 1
    if not translation_bin_path:
        raise ValueError("translation_bin_path is required.")
    # Detect species_tree folders
    tree_indices = detect_species_tree_folders(dataset_path)
    print(f"Detected {len(tree_indices)} species trees.")
    tree_directories = [os.path.join(dataset_path, f"species_tree_{i}") for i in range(len(tree_indices))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(process_tree_directory, (tree_dir, translation_bin_path)) for tree_dir in tree_directories]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f"An error occurred: {exc}")
    # Run parallel processing
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Prepare arguments for each process
        args_list = [(i, frequency_threshold, dataset_path) for i in tree_indices]

        # Run parallel processing
        # results = list(tqdm(executor.map(process_single_tree, args_list), total=len(tree_indices)))
    # debug : run sequentially: 
    results = []
    for arg in args_list:
        results.append(process_single_tree(arg))
    # Save results as a pickle file if pickle_path is provided
    if pickle_path:
        with open(pickle_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Results saved to {pickle_path}")
    # Fuse stats.csv files into a global stats.csv
    global_stats_path = os.path.join(dataset_path, "global_stats.csv")
    fuse_stats_csv(dataset_path, global_stats_path)
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process species trees in parallel and generate PyG trees.")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Base path to the dataset.")
    parser.add_argument("--frequency_threshold", type=float, default=0.1, help="Frequency threshold for filtering transfers.")
    parser.add_argument("--pickle_path", type=str, required=True, help="Path to save the pickled results.")
    parser.add_argument("--translation_bin_path", type=str, required=True, help="Path to the translation binary.")
    args = parser.parse_args()

    # Call the main function with command-line arguments
    process_all_species_trees(
        dataset_path=args.dataset_path,
        num_processes=args.num_processes,
        frequency_threshold=args.frequency_threshold,
        pickle_path=args.pickle_path,
        translation_bin_path=args.translation_bin_path
    )