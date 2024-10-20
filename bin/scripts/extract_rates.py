from pathlib import Path

def extract_rates(
    reconciliations_estimation_dir,
    start_index,
    end_index_estimation,
    rates_file,
):
    # Convert string paths to Path objects
    reconciliations_estimation_dir = Path(reconciliations_estimation_dir).resolve()
    rates_file = Path(rates_file).resolve()

    duplications_total = 0
    transfers_total = 0
    losses_total = 0

    for gene_index in range(start_index, end_index_estimation):
        # Construct the path to the reconciliation file
        file_path = reconciliations_estimation_dir / f"reconciliation_{gene_index}_uml"
        
        # Read the reconciliation file
        with file_path.open('r') as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if line.startswith('rate of'):
                    values = lines[i + 1].split()
                    duplications_total += float(values[1])
                    transfers_total += float(values[2])
                    losses_total += float(values[3])
                    break

    # Calculate averages
    n_genes = end_index_estimation - start_index
    duplications = duplications_total / n_genes
    transfers = transfers_total / n_genes
    losses = losses_total / n_genes

    # Write the rates to the rates file
    with rates_file.open('w') as file:
        file.write(f"{duplications} {transfers} {losses}")
