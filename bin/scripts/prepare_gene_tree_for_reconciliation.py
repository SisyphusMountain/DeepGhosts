from pathlib import Path
import subprocess
import shutil

def prepare_gene_tree_for_reconciliation(
    sampled_gene_trees_dir,
    prepared_gene_trees_dir,
    start_index,
    end_index,
    prepared_gene_trees_proof,
):
    # Convert the paths to Path objects
    sampled_gene_trees_dir = Path(sampled_gene_trees_dir).resolve()
    prepared_gene_trees_dir = Path(prepared_gene_trees_dir).resolve()
    prepared_gene_trees_proof = Path(prepared_gene_trees_proof).resolve()

    # Create the output directory if it doesn't exist
    prepared_gene_trees_dir.mkdir(parents=True, exist_ok=True)

    # Change working directory
    Path(prepared_gene_trees_dir).cwd()

    for gene_index in range(start_index, end_index):
        # Construct the paths using pathlib
        input_sampled_gene_tree = sampled_gene_trees_dir / f"sampled_gene_{gene_index}.nwk"
        output_prepared_gene_tree = prepared_gene_trees_dir / f"prepared_gene_{gene_index}.nwk"

        # Run ALEobserve on the sampled gene tree
        cmd = f"ALEobserve {input_sampled_gene_tree}"
        subprocess.run(cmd,
                       shell=True,
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE,)

        # Move the resulting .ale file to the prepared_gene_trees_dir
        ale_file = input_sampled_gene_tree.with_suffix('.nwk.ale')
        shutil.move(ale_file, output_prepared_gene_tree)

    # Write proof that the preparation is complete
    with open(prepared_gene_trees_proof, "w+") as f:
        f.write("done!")
