#! /usr/bin/env python
import os
from pathlib import Path
from scripts.utils import generate_seed_from_parameters
from scripts.params import (
    generate_species_tree_params,
    get_params_extract_rates,
)


# PARENT_PATH = Path(config['parent_path']).resolve()
# BIN_PATH = Path(config['bin_path']).resolve()
# R_SCRIPT = BIN_PATH / "species_tree_generator.R"
# GENE_TRANSFER_SCRIPT = BIN_PATH / "gene_transfer_script"
# SAMPLE_SCRIPT = BIN_PATH / "sample_script"
# EXTRACT_EXTANT_SPECIES_SCRIPT = BIN_PATH / "extract_extant_script"
# TRANSLATION_BIN_PATH = BIN_PATH / "Translate_trees"
# ## OUTPUT FOLDERS PATHS
# OUTPUT_FOLDER = Path(config['output_folder']).resolve()
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# ## OUTPUT FILES PATHS
# DATASET_PICKLE_PATH = OUTPUT_FOLDER / "dataset.pkl"
# OUTPUT_NAME_COMPLETE = "complete_species_tree.nwk"
# OUTPUT_NAME_EXTANT = "extant_species_tree.nwk"
# N_SP_TREES = config['n_sp_trees']
# N_GENES = config['n_genes']
# START_INDEX = config['start_index']
# SEED = config['seed']
# DISTRIBUTION = config['distribution']
# PARAMETERS_ESTIMATION_GENE_PROPORTION = config['parameters_estimation_gene_proportion']
# END_INDEX_ESTIMATION = int(N_GENES * PARAMETERS_ESTIMATION_GENE_PROPORTION)
# END_INDEX = N_GENES
# FREQUENCY_THRESHOLD = float(config['frequency_threshold'])
# # Generate species tree parameters
# species_tree_params = generate_species_tree_params(N_SP_TREES, SEED, config, OUTPUT_FOLDER)
# # create modified transfer rate dictionary
# modified_transfer_rates = dict()
# NORMALIZE_TREES = config['normalize_trees']

def modify_parameters(input_file_path: str,
                             output_file_path: str,
                             replacements: dict):
    try:
        # Read the file content
        with open(input_file_path, 'r') as file:
            content = file.read()

        # Replace all parameters ending with '_SNAKEMAKE'
        for key, value in replacements.items():
            pattern = re.compile(rf"{key}_SNAKEMAKE")
            content = pattern.sub(value, content)

        # Write the modified content to the output file
        with open(output_file_path, 'w+') as file:
            file.write(content)
        print(f"Successfully modified the file and saved to: {output_file_path}")
    except FileNotFoundError:
        print(f"Error: The file {input_file_path} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


rule all:
    input:
        ct=rules.generate_species_tree.output.complete_tree,
        et=rules.generate_species_tree.output.extant_tree,
        bl=rules.compute_transfer_rates.output.branch_length,
        nf=rules.compute_transfer_rates.output.normalization_factor,
    run:
        print(f"Complete tree: {input.ct}, extant tree: {input.et}, branch length: {input.bl}, normalization factor: {input.nf}")

rule generate_species_tree:
    output:
        complete_tree=str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / OUTPUT_NAME_COMPLETE),
        extant_tree=str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / OUTPUT_NAME_EXTANT),
        branch_length=str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "branch_length.txt"),
        normalization_factor=str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "normalization_factor.txt"),
    run:
        from scripts.generate_species_tree import generate_species_tree
        from scripts.sampling_trees import sample_trees
        from ete3 import Tree
        import subprocess
        sp_tree_index = int(wildcards.sp_tree_index)
        params = species_tree_params[sp_tree_index]
        replacements_species = {"SPECIATION": str(params["speciation"]),
                                "EXTINCTION": str(params["extinction"]),
                                "MAX_LINEAGES": str(params["max_lineages"]),
                                "SEED": str(params["seed"])}
        input_file_path = str(BIN_PATH / "SpeciesTreeParametersTemplate.tsv")
        output_file_path = str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "SpeciesTreeParameters.tsv")
        modify_parameters(input_file_path, output_file_path, replacements_species)
        # Now just use Zombi with the constructed parameters file to generate the species tree
        command = f"python3 /home/enzo/Documents/git/Zombi_ale_friendly/Zombi.py T /home/enzo/Documents/git/WP1/DeepGhosts/bin/zombi_configs/SpeciesTreeParameters_2.tsv {tree_output_folder}"
        subprocess.run(command, shell=True)
        # Now compute the names of the species that will be sampled.
        
        
        
        if NORMALIZE_TREES:
            sampled_species_tree = Path(OUTPUT_FOLDER / f"species_tree_{sp_tree_index}/sampled/sampled_trees/sampled_species_tree.nwk")
            # now use the sampling script to obtain the sampled tree before normalization
            from scripts.sampling_trees import sample_trees
            generated_gene_trees_folder = sampled_species_tree.parent.parent
            generated_gene_trees_folder = str(generated_gene_trees_folder)
            sampled_species_tree = str(sampled_species_tree)


            # sample_trees(
            #     complete_species_tree=output.complete_tree,
            #     extant_species_tree=output.extant_tree,
            #     generated_gene_trees_folder=generated_gene_trees_folder,
            #     n_sampled_nodes=params['sampled_species'],
            #     start_index=0,
            #     end_index=0,
            #     output_dir=os.path.dirname(sampled_species_tree),
            #     seed=params['seed'],
            #     sample_script=SAMPLE_SCRIPT,
            # )
            # Now we have generated the sampled tree. We need to find its sum of branch lengths.
            sampled_tree = Tree(sampled_species_tree, format=1)
            total_branch_length = sum([node.dist for node in sampled_tree.traverse()])
            # The normalization factor is given by the ratio of the expected tree length under a pure-birth process over the actual tree length
            expected_tree_length = params["sampled_species"]/params["birth_rate"]
            normalization_factor = expected_tree_length/total_branch_length
            # save the factor in the output folder
            with open(output.normalization_factor, "w") as f:
                f.write(str(normalization_factor))
        else:
            normalization_factor = 1
            with open(output.normalization_factor, "w") as f:
                f.write(str(normalization_factor))

rule generate_gene_trees:
    output:
        gene_tree=".",
    input:
        complete_tree=rules.generate_species_tree.output.complete_tree,
    run:
        import subprocess
 
        sp_tree_index = int(wildcards.sp_tree_index)
        params = species_tree_params[sp_tree_index]
        replacements_genome = {"DUPLICATION_RATE": str(params["duplication_rate"]),
                "LOSS_RATE": str(params["loss_rate"]),
                "TRANSFER_RATE": str(params["transfer_rate"]),
                "ASSORTATIVE_TRANSFER": str(params["assortative_transfer"]),
                "ALPHA": str(params["alpha"]),
                "N_GENES": str(params["n_genes"]),
                "SEED": str(params["seed"])}
        input_file_path = str(BIN_PATH / "GenomeParametersTemplate.tsv")
        output_file_path = str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "GenomeParameters.tsv")
        modify_parameters(input_file_path, output_file_path, replacements_genome)
        command = f"python3 /home/enzo/Documents/git/Zombi_ale_friendly/Zombi.py G /home/enzo/Documents/git/WP1/DeepGhosts/bin/zombi_configs/GenomeParameters_2.tsv {tree_output_folder}"
        subprocess.run(command, shell=True)


















































rule give_sampled_species_names:
    # On donne d'abord les noms des espèces samplées

rule compute_transfer_rates:
    # Grâce aux espèces samplées, on peut calculer la longueur de l'arbre, et donc normaliser les taux de transfert par la longueur de l'arbre.
    # Dans cette règle, on calcule les taux de transfert pairwise entre toutes les espèces de l'arbre, en normalisant par la longueur de l'arbre.
    # Pour cela, on doit :
    # 1. Calculer la matrice de coexistence des espèces 