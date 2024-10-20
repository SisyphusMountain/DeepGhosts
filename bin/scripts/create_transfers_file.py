# scripts/create_transfers_file.py

from utils import generate_transfers_file
import os
import numpy as np
def create_transfers_file(
    branch_length_file,
    transfers_file,
    n_genes,
    distribution,
    seed,
    transfer_rate,
):
    with open(branch_length_file, "r") as f:
        br_length = float(f.readline().strip())

    average_transfers = br_length * transfer_rate

    # Seed the random number generator
    np.random.seed(seed)

    # Generate random variables based on the specified distribution
    if distribution == "Poisson":
        # Generate Poisson-distributed random variables
        vars = np.random.poisson(lam=average_transfers, size=n_genes)
    else:
        # Generate a constant array
        vars = np.ones(shape=(n_genes), dtype=int) * int(average_transfers)

    # Convert to a comma-separated string
    vars_str = ",".join(map(str, vars))

    folder = os.path.dirname(transfers_file)
    # Create directory if it doesn't exist
    os.makedirs(folder, exist_ok=True)


    # Write to the file at the specified path
    with open(transfers_file, "w") as file:
        file.write(vars_str)
