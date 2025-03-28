from pathlib import Path
import subprocess
import shutil
import os
from IPython import embed

def reconcile_gene_tree(
    prepared_species_tree,
    prepared_gene_trees_dir,
    start_index,
    end_index_estimation,
    reconciliations_estimation_dir,
    ):
    # Convert string paths to Path objects
    prepared_species_tree = Path(prepared_species_tree).resolve()
    prepared_gene_trees_dir = Path(prepared_gene_trees_dir).resolve()
    reconciliations_estimation_dir = Path(reconciliations_estimation_dir).resolve()

    # Ensure the reconciliations directory exists
    reconciliations_estimation_dir.mkdir(parents=True, exist_ok=True)
    # embed()
    # Change to the reconciliations estimation directory
    # Using pathlib doesn't have a `cwd()` method, so os.chdir is kept here
    os.chdir(reconciliations_estimation_dir)

    for gene_index in range(start_index, end_index_estimation):
        # Construct the path to the prepared gene tree
        prepared_gene_tree = prepared_gene_trees_dir / f"prepared_gene_{gene_index}.nwk"

        # Run ALEml_undated on the species tree and gene tree
        cmd = f"ALEml_undated {prepared_species_tree} {prepared_gene_tree} seed=42"
        subprocess.run(cmd,
                       shell=True,
                       check=True,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE,)

        # Construct the source and target file paths
        source_file_uml = f"{prepared_species_tree.stem}.nwk_{prepared_gene_tree.stem}.nwk.uml_rec"
        source_file_uTs = f"{prepared_species_tree.stem}.nwk_{prepared_gene_tree.stem}.nwk.uTs"
        target_file_uml = reconciliations_estimation_dir / f"reconciliation_{gene_index}_uml"
        target_file_uTs = reconciliations_estimation_dir / f"reconciliation_{gene_index}_uTs"
        # Move the generated files to their target locations
        shutil.move(source_file_uml, target_file_uml)
        shutil.move(source_file_uTs, target_file_uTs)
        