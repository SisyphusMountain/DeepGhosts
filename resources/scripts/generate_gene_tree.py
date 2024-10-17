# scripts/generate_gene_tree.py

import os
import subprocess
def generate_gene_tree(
    complete_species_tree,
    transfers_file,
    output_complete,
    seed,
    gene_transfer_script,
    generated_gene_trees_proof,
):
    os.makedirs(output_complete, exist_ok=True)

    with open(generated_gene_trees_proof, "w+") as f:
        f.write("done!")
    cmd = [gene_transfer_script, complete_species_tree, output_complete, transfers_file, str(seed)]

    # Execute the script
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Output results
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)