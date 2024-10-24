from pathlib import Path
import subprocess
import shutil
import os
def prepare_species_tree_for_reconciliation(
    sampled_species_tree,
    sampled_species_tree_ale,
):
    # Convert to Path objects
    sampled_species_tree = Path(sampled_species_tree).resolve()
    sampled_species_tree_ale = Path(sampled_species_tree_ale).resolve()

    prepared_folder = sampled_species_tree_ale.parent
    prepared_folder.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    original_dir = os.getcwd()
    # Change to the prepared folder directory
    os.chdir(prepared_folder)
    
    # Run ALEobserve on the species tree
    subprocess.run(f"ALEobserve {sampled_species_tree}",
                   shell=True,
                   check=True,
                   stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE,)
    
    # Run ALEml_undated on the species tree
    subprocess.run(
        f"ALEml_undated {sampled_species_tree} {sampled_species_tree}.ale "
        "output_species_tree=y sample=0 delta=0 tau=0 lambda=0 seed=42",
        shell=True,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Move the ale.spTree file to the desired location
    ale_sp_tree_file = f"{sampled_species_tree.name}_{sampled_species_tree.name}.ale.spTree"
    shutil.move(ale_sp_tree_file, sampled_species_tree_ale)
    os.chdir(original_dir)