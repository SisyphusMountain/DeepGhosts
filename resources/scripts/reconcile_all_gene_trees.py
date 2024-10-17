from pathlib import Path
import subprocess
import shutil

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
    reconciliations_dir.cwd()

    # Read the rates file
    with rates_file.open('r') as rates_file_handle:
        rates = rates_file_handle.read().strip()
    duplications, transfers, losses = rates.split()

    for gene_index in range(start_index, end_index):
        # Construct paths using pathlib
        prepared_gene_tree = prepared_gene_trees_dir / f"prepared_gene_{gene_index}.nwk"
        cmd = f"ALEml_undated {prepared_species_tree} {prepared_gene_tree} delta={duplications} tau={transfers} lambda={losses}"
        subprocess.run(cmd, shell=True)

        # Move the resulting .uml_rec and .uTs files
        source_file_uml = f"{prepared_species_tree.name}_{prepared_gene_tree.name}.uml_rec"
        source_file_uTs = f"{prepared_species_tree.name}_{prepared_gene_tree.name}.uTs"
        target_file_uml = reconciliations_dir / f"reconciliation_{gene_index}_uml"
        target_file_uTs = reconciliations_dir / f"reconciliation_{gene_index}_uTs"

        shutil.move(source_file_uml, target_file_uml)
        shutil.move(source_file_uTs, target_file_uTs)

    # Write proof that the reconciliation is complete
    with proof_reconciliation.open("w+") as f:
        f.write("done!")
