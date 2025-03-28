from pathlib import Path
import subprocess
import shutil
import os

def prepare_gene_tree_for_reconciliation(
    sampled_gene_trees_dir,
    prepared_gene_trees_dir,
    start_index,
    end_index,
):
    # Convert the paths to Path objects
    sampled_gene_trees_dir = Path(sampled_gene_trees_dir).resolve()
    prepared_gene_trees_dir = Path(prepared_gene_trees_dir).resolve()
    # Create the output directory if it doesn't exist
    prepared_gene_trees_dir.mkdir(parents=True, exist_ok=True)

    
    # Change working directory
    original_dir = os.getcwd()
    os.chdir(sampled_gene_trees_dir)

    for gene_index in range(start_index, end_index):
        # Construct the paths using pathlib
        input_sampled_gene_tree = sampled_gene_trees_dir / f"{gene_index+1}_sampledtree.nwk"
        output_prepared_gene_tree = prepared_gene_trees_dir / f"prepared_gene_{gene_index+1}.nwk"

        # Run ALEobserve on the sampled gene tree
        cmd = f"ALEobserve {input_sampled_gene_tree}"
        subprocess.run(cmd,
                       shell=True,
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE,)

        # Move the resulting .ale file to the prepared_gene_trees_dir
        ale_file = f"{gene_index+1}_sampledtree.nwk.ale"
        shutil.move(ale_file, output_prepared_gene_tree)
    os.chdir(original_dir)
