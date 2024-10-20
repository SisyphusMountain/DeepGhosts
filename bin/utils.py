########################################################################
# Python functions for readability.
import hashlib
import subprocess
import os
import numpy as np


def generate_seed_from_parameters(*args):
    # Convert all arguments to strings and concatenate them
    concatenated = "_".join(map(str, args))

    # Use hashlib to create a hash of the concatenated string
    return int(hashlib.sha256(concatenated.encode()).hexdigest(), 16) % (2**30)


def generate_species_tree(extant_species,
                          birth_rate,
                          death_rate,
                          output_complete,
                          output_extant,
                          seed,
                          r_script,
                          rust_script,
                          total_branch_length_txt,
                          n_extant_nodes):
    # Arguments for the R script
    args = [str(extant_species),
            str(birth_rate),
            str(death_rate),
            output_complete,
            str(seed),]
    # Combine script path and arguments
    cmd = ['Rscript', r_script] + args

    # Execute the script
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    # Extract the sum of branch lengths

    sum_branch_lengths = result.stdout.strip().split('\n')[-1]

    with open(total_branch_length_txt, "w+") as f:
        f.write(sum_branch_lengths)
    # Now obtain the sampled tree
    cmd = [rust_script, output_complete, str(n_extant_nodes), output_extant]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    


def generate_transfers_file(n_genes, average_transfers=0, distribution="constant", seed=123, path="./transfers_file.txt"):
    # Seed the random number generator
    np.random.seed(seed)

    # Generate random variables based on the specified distribution
    if distribution == "Poisson":
        # Generate Poisson-distributed random variables
        vars = np.random.poisson(lam=average_transfers, size=n_genes)
    else:
        # Generate a constant array
        vars = np.ones(shape=(n_genes), dtype=int) * int(average_transfers)

    # Convert to a comma-separated string
    vars_str = ",".join(map(str, vars))

    folder = os.path.dirname(path)
    # Create directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)


    # Write to the file at the specified path
    with open(path, "w") as file:
        file.write(vars_str)

def generate_gene_trees(gene_transfer_script, path_to_tree, transfers_file, output_dir, seed):
   # Combine script path and arguments
   cmd = [gene_transfer_script, path_to_tree, output_dir, transfers_file, str(seed)]

   # Execute the script
   result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

   # Output results
   print("STDOUT:", result.stdout)
   print("STDERR:", result.stderr)

def sample_trees(sample_script, species_tree, extant_species_tree, gene_trees, n_sampled_nodes, start_index, end_index, output_dir, seed):
   cmd = [sample_script, species_tree, extant_species_tree, gene_trees, str(n_sampled_nodes), str(start_index), str(end_index), output_dir, str(seed)]
   os.makedirs(output_dir, exist_ok=True)
   original_dir = os.getcwd()
   os.chdir(output_dir)
   # Execute the script
   result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
   os.chdir(original_dir)
   # Output results
   print("STDOUT:", result.stdout)
   print("STDERR:", result.stderr)

def prepare_trees(sampled, start_index, end_index, output_dir):
    sampled_species_tree = os.path.join(sampled, "sampled_species_tree.nwk")
    old_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    for gene_index in range(start_index, end_index):
        gene_tree = os.path.join(sampled, f"sampled_gene_{gene_index}.nwk")
        observe_sp_cmd = f"ALEobserve {sampled_species_tree}"
        observe_gene_cmd = f"ALEobserve {gene_tree}"
        ml_undated_sp_cmd = f"ALEml_undated {sampled_species_tree} {sampled_species_tree}.ale output_species_tree=y sample=0 delta=0 tau=0 lambda=0"
        subprocess.run(observe_sp_cmd, shell=True, check=True)

        # Run ALEml_undated on species tree
        subprocess.run(ml_undated_sp_cmd, shell=True, check=True)

        # Run ALEobserve on gene tree
        subprocess.run(observe_gene_cmd, shell=True, check=True)
    os.chdir(old_dir)

def reconcile_trees(sampled, start_index, end_index, aletrees, output_dir):
    old_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    sampled_species_tree_ale = os.path.join(aletrees,"sampled_species_tree.nwk_sampled_species_tree.nwk.ale.spTree")
    for gene_index in range(start_index, end_index):
        gene_trees_ale = os.path.join(sampled, f"sampled_gene_{gene_index}.nwk.ale")
        command = f"ALEml_undated {sampled_species_tree_ale} {gene_trees_ale}"
        subprocess.run(command, shell=True, check=True)
    os.chdir(old_dir)

def compute_averages(reconciliations_folder, start_index, end_index):
    duplications_total = 0
    transfers_total = 0
    losses_total = 0
    for gene_index in range(start_index, end_index):
        gene = f"sampled_species_tree.nwk_sampled_species_tree.nwk.ale.spTree_sampled_gene_{gene_index}.nwk.ale.uml_rec"
        file = os.path.join(reconciliations_folder, gene)
        with open(file, 'r') as file:
            lines = file.readlines()

        # Initialize variables
        duplications, transfers, losses = None, None, None

        for i, line in enumerate(lines):
            if line.startswith("rate of"):
                # Assuming the next line contains the values
                values = lines[i + 1].split()
                duplications = float(values[1])
                transfers = float(values[2])
                losses = float(values[3])
                break

        duplications_total += duplications
        transfers_total += transfers
        losses_total += losses

    duplications_rate = duplications_total/(end_index-start_index)
    transfers_rate = transfers_total/(end_index-start_index)
    losses_rate = losses_total/(end_index-start_index)
    return duplications_rate, transfers_rate, losses_rate

def reconcile_trees_estimation(sampled, start_index, end_index, aletrees, output_dir, duplications, transfers, losses):
    old_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    os.chdir(output_dir)
    sampled_species_tree_ale = os.path.join(aletrees,"sampled_species_tree.nwk_sampled_species_tree.nwk.ale.spTree")
    for gene_index in range(start_index, end_index):
        gene_trees_ale = os.path.join(sampled, f"sampled_gene_{gene_index}.nwk.ale")

        command = f"ALEml_undated {sampled_species_tree_ale} {gene_trees_ale} delta={duplications} tau={transfers} lambda={losses}"
        subprocess.run(command, shell=True, check=True)
    os.chdir(old_dir)

###############################################################################