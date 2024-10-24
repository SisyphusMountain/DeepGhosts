import subprocess
from scripts.rename import rename_internal_nodes
def generate_species_tree(
    extant_species,
    birth_rate,
    death_rate,
    sp_tree_index,
    output_folder,
    output_complete_tree,
    seed,
    r_script,
    extract_extant_script,
    total_branch_length_txt,
    n_extant_nodes,
):
    output_extant_tree = output_folder / f"species_tree_{sp_tree_index}/" # Leave this as is, it is how the rust script is coded.
    # Arguments for the R script
    args = [
        str(extant_species),
        str(birth_rate),
        str(death_rate),
        output_complete_tree,
        str(seed),
    ]
    # Combine script path and arguments
    cmd = ['Rscript', r_script] + args
    # import IPython; IPython.embed()
    # Execute the R script
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    rename_internal_nodes(output_complete_tree, output_complete_tree)
    # Extract the sum of branch lengths
    sum_branch_lengths = result.stdout.strip().split('\n')[-1]

    with open(total_branch_length_txt, "w+") as f:
        f.write(sum_branch_lengths)

    # Now obtain the extant species tree
    cmd = [
        extract_extant_script,
        output_complete_tree,
        str(n_extant_nodes),
        output_extant_tree
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
