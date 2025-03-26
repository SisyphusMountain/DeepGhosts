import subprocess
from ete3 import Tree


def rename_internal_nodes(newick_path, output_path):
    # Load the Newick tree
    tree = Tree(newick_path, format=1)
    # Dirty hack to rename internal nodes, found no other solution. Giving non-integer node names has been problematic in ALE reconciliations, so we need to do this.
    # After pruning leaves and using ALEObserve, it could be that internal node names are the same as some leave names.
    # Therefore, we add 10000 to all node names. This makes sure that internal node names created by ALE will never be the same as leave names, since
    # the size of pruned trees is always much smaller than 10000 in our experiments.
    count = 10000
    for node in tree.traverse(strategy="postorder"):
            node.name = str(count)
            count += 1
    # Save the tree with renamed internal nodes
    tree.write(outfile=output_path, format=1, format_root_node=True)

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
    # print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    rename_internal_nodes(output_complete_tree, output_complete_tree)
    # Extract the sum of branch lengths
    sum_branch_lengths = result.stdout.strip().split('\n')[-1]

    with open(total_branch_length_txt, "w+") as f:
        f.write(sum_branch_lengths)

    # Now obtain the extant species tree.
    # We generate the complete species tree conditional on a certain number of extant species.
    # Therefore, in order to obtain the extant tree, it suffices to extract
    # the n_extant nodes that are furthest away from the root.
    cmd = [
        extract_extant_script,
        output_complete_tree,
        str(n_extant_nodes),
        output_extant_tree
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
