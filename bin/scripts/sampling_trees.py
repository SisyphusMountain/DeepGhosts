# scripts/sampling_trees.py
from IPython import embed
import os
from pathlib import Path
import os
import subprocess


def sample_trees(
    complete_species_tree,
    extant_species_tree,
    generated_gene_trees_folder,
    n_sampled_nodes,
    start_index,
    end_index,
    output_dir,
    seed,
    sample_script,
):
   """This function takes both the complete tree and the extant tree.
   Using these two trees, it samples leaves from the complete tree which are also present
   in the extant tree. It then generates gene trees for the sampled leaves.
   start_index and end_index specify the range of gene tree indices to generate."""
   cmd = [sample_script, complete_species_tree, extant_species_tree, generated_gene_trees_folder, str(n_sampled_nodes), str(start_index), str(end_index), output_dir, str(seed)]
   os.makedirs(output_dir, exist_ok=True)
   original_dir = os.getcwd()
   os.chdir(output_dir)
   result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
   os.chdir(original_dir)
   # Output results
   # print("STDOUT:", result.stdout)
   print("STDERR:", result.stderr)