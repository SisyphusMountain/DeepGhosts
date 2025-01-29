from pathlib import Path
import subprocess
import shutil
import os
import logging

def reconcile_all_gene_trees(
    prepared_species_tree,
    prepared_gene_trees_dir,
    rates_file,
    start_index,
    end_index,
    reconciliations_dir,
    proof_reconciliation,
):
    # Convert paths to Path objects
    prepared_species_tree = Path(prepared_species_tree).resolve()
    prepared_gene_trees_dir = Path(prepared_gene_trees_dir).resolve()
    rates_file = Path(rates_file).resolve()
    reconciliations_dir = Path(reconciliations_dir).resolve()
    proof_reconciliation = Path(proof_reconciliation).resolve()

    # Create the reconciliations directory if it doesn't exist
    reconciliations_dir.mkdir(parents=True, exist_ok=True)

    # Change to the reconciliations directory
    original_dir = os.getcwd()
    os.chdir(reconciliations_dir)


    # Read the rates file
    with rates_file.open('r') as rates_file_handle:
        rates = rates_file_handle.read().strip()
    duplications, transfers, losses = rates.split()

    for gene_index in range(start_index, end_index):
        # Construct paths using pathlib
        prepared_gene_tree = prepared_gene_trees_dir / f"prepared_gene_{gene_index}.nwk"
        cmd = f"ALEml_undated {prepared_species_tree} {prepared_gene_tree} delta={duplications} tau={transfers} lambda={losses} seed=42"
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,  # Ensure output is returned as a string
            )
            # Log the output and error (if any)
            logging.info(f"Output:\n{result.stdout}")
            if result.stderr:
                logging.error(f"Errors:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Command failed for Gene Index {gene_index} with error:\n{e.stderr}")
            # Handle failure or decide to stop further processing
            break
        # Move the resulting .uml_rec and .uTs files
        source_file_uml = f"{prepared_species_tree.name}_{prepared_gene_tree.name}.uml_rec"
        source_file_uTs = f"{prepared_species_tree.name}_{prepared_gene_tree.name}.uTs"
        target_file_uml = f"reconciliation_{gene_index}_uml"
        target_file_uTs = f"reconciliation_{gene_index}_uTs"

        shutil.move(source_file_uml, target_file_uml)
        shutil.move(source_file_uTs, target_file_uTs)
    
    os.chdir(original_dir)
    # Write proof that the reconciliation is complete
    with proof_reconciliation.open("w+") as f:
        f.write("done!")
