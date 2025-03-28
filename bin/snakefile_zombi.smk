#! /usr/bin/env python
import os
from pathlib import Path
from scripts.utils import generate_seed_from_parameters
from scripts.params import (
    generate_species_tree_params,
    get_params_extract_rates,
)

# snakemake --snakefile snakefile_zombi.smk --configfile config_test.yaml --cores 1
# PARENT_PATH = Path(config['parent_path']).resolve()
BIN_PATH = Path(config['bin_path']).resolve()
# ## OUTPUT FOLDERS PATHS
OUTPUT_FOLDER = Path(config['output_folder']).resolve()
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# ## OUTPUT FILES PATHS
DATASET_PICKLE_PATH = OUTPUT_FOLDER / "dataset.pkl"
OUTPUT_NAME_COMPLETE = "CompleteTree.nwk"
OUTPUT_NAME_EXTANT = "ExtantTree.nwk"
N_SP_TREES = config['n_sp_trees']
SAMPLING_MODE = config['sampling_mode']
N_GENES = config['n_genes']
START_INDEX = config['start_index'] + 1
SEED = config['seed']
# DISTRIBUTION = config['distribution']
PARAMETERS_ESTIMATION_GENE_PROPORTION = config['parameters_estimation_gene_proportion']
END_INDEX_ESTIMATION = int(N_GENES * PARAMETERS_ESTIMATION_GENE_PROPORTION)+1
END_INDEX = N_GENES + 1
FREQUENCY_THRESHOLD = float(config['frequency_threshold'])
# # Generate species tree parameters
species_tree_params = generate_species_tree_params(N_SP_TREES, SEED, config, OUTPUT_FOLDER)
# # create modified transfer rate dictionary
# modified_transfer_rates = dict()
NORMALIZE_TREES = config['normalize_trees']
ASSORTATIVE_TRANSFER = config['assortative_transfer']
ALPHA = config['alpha']


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
        gene_trees=expand(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "G" / "GenomeParameters.tsv", sp_tree_index=range(N_SP_TREES)),
        same = expand(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "SAMPLE_1" / "SampledSpeciesTree.nwk", sp_tree_index=range(N_SP_TREES)),
        prepared_species_tree=expand(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "ale_trees" / "SampledSpeciesTree_ale.nwk", sp_tree_index=range(N_SP_TREES)),
        prepared_gene_tree_ale=expand(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "ale_trees" / "prepared_gene_1.nwk", sp_tree_index=range(N_SP_TREES)),
        reconciled_partial=expand(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "reconciliations_estimation" / "reconciliation_1_uml", sp_tree_index=range(N_SP_TREES)),
        rates=expand(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "rates.txt", sp_tree_index=range(N_SP_TREES)),
        reconciled_all=expand(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "reconciliations_all" / "reconciliation_1_uml", sp_tree_index=range(N_SP_TREES)),
        dataset=str(OUTPUT_FOLDER / "dataset.pkl"),
    run:
        print("All done!")

rule generate_species_tree:
    output:
        complete_tree=str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "T" / OUTPUT_NAME_COMPLETE),
        extant_tree=str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "T" / OUTPUT_NAME_EXTANT),
        normalization_factor=str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "normalization_factor.txt"),
        sampled_species=str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "sampled_species.txt"),
    run:
        from scripts.sampling_algorithms import sampling
        from ete3 import Tree
        import subprocess
        sp_tree_index = int(wildcards.sp_tree_index)
        params = species_tree_params[sp_tree_index]
        replacements_species = {"SPECIATION": str(params["birth_rate"]),
                                "EXTINCTION": str(params["death_rate"]),
                                "MAX_LINEAGES": str(params["extant_species"]),
                                "SEED": str(params["seed"])}
        input_file_path = str(BIN_PATH / "zombi_configs/SpeciesTreeParametersTemplate.tsv")
        output_file_path = str(OUTPUT_FOLDER / f"species_tree_{sp_tree_index}" / "SpeciesTreeParameters.tsv")
        modify_parameters(input_file_path, output_file_path, replacements_species)
        # Now just use Zombi with the constructed parameters file to generate the species tree
        command = f"python3 /home/enzo/Documents/git/Zombi_ale_friendly/Zombi.py T {output_file_path} {str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}')}"
        subprocess.run(command, shell=True)
        # Now compute the names of the species that will be sampled.
        complete_tree = Tree(output.complete_tree, format=1)
        sampled_species_names = sampling(tree=complete_tree,
                                            k=params["sampled_species"],
                                            sampling_mode=SAMPLING_MODE,
                                            seed=params["seed"])
        with open(output.sampled_species, "w") as f:
            f.write("\n".join(sampled_species_names))
        
        if NORMALIZE_TREES:
            # Compute the length of the sampled species tree by pruning it with ete3
            
            complete_tree.prune(sampled_species_names, preserve_branch_length=True)
            sampled_branch_length = sum([node.dist for node in complete_tree.traverse()])
            # Compute the expected length of this tree if the sampled tree was not sampled,
            # but generated with the birth and death rates of the original tree
            expected_sampled_length = params["sampled_species"]/(params["birth_rate"]-params["death_rate"])
            normalization_factor = expected_sampled_length/sampled_branch_length
        else:
            normalization_factor = 1
        with open(output.normalization_factor, "w") as f:
            f.write(str(normalization_factor))




rule generate_gene_trees:
    output:
        gene_trees=str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "G" / "GenomeParameters.tsv"),
    input:
        complete_tree=rules.generate_species_tree.output.complete_tree,
    run:
        import subprocess
        sp_tree_index = int(wildcards.sp_tree_index)
        params = species_tree_params[sp_tree_index]
        transfer_rate = params["transfer_rate"]
        # normalize the transfer rate
        with open(OUTPUT_FOLDER / f"species_tree_{sp_tree_index}" / "normalization_factor.txt", "r") as f:
            normalization_factor = float(f.read())
        # the normalization factor will be 1 if we decided not to normalize the trees
        normalized_transfer_rate = transfer_rate * normalization_factor
        replacements_genome = {"DUPLICATION_RATE": "0",
                "LOSS_RATE": "0",
                "TRANSFER_RATE": str(normalized_transfer_rate),
                "ASSORTATIVE_TRANSFER": str(ASSORTATIVE_TRANSFER),
                "ALPHA": str(ALPHA),
                "N_GENES": str(N_GENES),
                "SEED": str(params["seed"])}
        input_file_path = str(BIN_PATH / "zombi_configs" / "GenomeParametersTemplate.tsv")
        output_file_path = str(OUTPUT_FOLDER / f"species_tree_{sp_tree_index}" / "GenomeParameters.tsv")
        modify_parameters(input_file_path, output_file_path, replacements_genome)
        # test whether the file GenomeParameters.tsv was correctly generated
        command = f"python3 /home/enzo/Documents/git/Zombi_ale_friendly/Zombi.py G {output_file_path} {str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}')}"
        subprocess.run(command, shell=True)

rule sampling_trees:
    input:
        sampled_species_path = str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "sampled_species.txt"),
        gene_trees=str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "G" / "GenomeParameters.tsv"),
    output:
        sampled_tree=str(OUTPUT_FOLDER / "species_tree_{sp_tree_index}" / "SAMPLE_1" / "SampledSpeciesTree.nwk"),
    run:
        import subprocess
        import os
        sp_tree_index = int(wildcards.sp_tree_index)
        sampled_species = str(OUTPUT_FOLDER / f"species_tree_{sp_tree_index}" / "sampled_species.txt")
        tree_path = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}')
        os.rmdir(str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "SAMPLE_1"))
        command = f"python3 /home/enzo/Documents/git/Zombi_ale_friendly/SpeciesSampler.py i {sampled_species} {tree_path}"
        print(f"running command {command}")
        subprocess.run(command, shell=True)

# Il reste à :
# écrire le reste des règles (il faut juste changer les noms de fichiers normalement)
# Intégrer les changements à Zombi en un fork qu'on peut ensuite utiliser.
# Choisir l'option de ne pas normaliser avant de faire le softmax
# Ecrire un ensemble d'expériences à lancer, dont on peut parler demain.
# Pour cela, il faudra choisir des alpha, ainsi que le nombre d'expériences à lancer.
# Il faut aussi choisir le type de normalisation que l'on applique.

rule prepare_species_tree_for_reconciliation:
    input:
        sampled_species_tree=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "SAMPLE_1" / "SampledSpeciesTree.nwk"),
    output:
        sampled_species_tree_ale=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "ale_trees" / "SampledSpeciesTree_ale.nwk"),
    run:
        from scripts.prepare_species_tree_for_reconciliation import prepare_species_tree_for_reconciliation
        sp_tree_index = int(wildcards.sp_tree_index)
        input_tree = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "SAMPLE_1" / "SampledSpeciesTree.nwk")
        output_tree = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "ale_trees" / "SampledSpeciesTree_ale.nwk")
        prepare_species_tree_for_reconciliation(
            sampled_species_tree=input_tree,
            sampled_species_tree_ale=output_tree,
        )

rule prepare_gene_tree_for_reconciliation:
    input:
        sampled_gene_trees=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "SAMPLE_1" / "SampledSpeciesTree.nwk"),
    output:
        sampled_gene_trees_ale=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "ale_trees" / "prepared_gene_1.nwk"),
    run:
        from scripts.prepare_gene_tree_for_reconciliation import prepare_gene_tree_for_reconciliation
        sp_tree_index = int(wildcards.sp_tree_index)
        input_dir = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "SAMPLE_1")
        output_dir = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "ale_trees")
        prepare_gene_tree_for_reconciliation(
            sampled_gene_trees_dir=input_dir,
            prepared_gene_trees_dir=output_dir,
            start_index=0,
            end_index=N_GENES,
        )

rule reconcile_gene_tree:
    """Reconcile 10% of the prepared sampled gene trees to estimate rates."""
    input:
        prepared_gene_trees=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "ale_trees" / "prepared_gene_1.nwk"),
        prepared_species_tree=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "ale_trees" / "SampledSpeciesTree_ale.nwk"),
    output:
        reconciliations_estimation_dir=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "reconciliations_estimation" / "reconciliation_1_uml"),
    run:
        from scripts.reconcile_gene_tree import reconcile_gene_tree
        sp_tree_index = int(wildcards.sp_tree_index)
        prepared_species_tree = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "ale_trees" / "SampledSpeciesTree_ale.nwk")
        prepared_gene_trees_dir = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "ale_trees")
        reconciliations_estimation_dir = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "reconciliations_estimation")
        reconcile_gene_tree(
            prepared_species_tree=prepared_species_tree,
            prepared_gene_trees_dir=prepared_gene_trees_dir,
            start_index=START_INDEX,
            end_index_estimation=END_INDEX_ESTIMATION,
            reconciliations_estimation_dir=reconciliations_estimation_dir,
        )


rule extract_rates:
    """Extract rates of duplications, transfers, and losses."""
    input:
        reconciliations_estimation_dir=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "reconciliations_estimation" / "reconciliation_1_uml"),
    output:
        rates_estimation=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "rates.txt"),
    run:
        from scripts.extract_rates import extract_rates
        sp_tree_index = int(wildcards.sp_tree_index)
        reconciliations_estimation_dir = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "reconciliations_estimation")
        rates_estimation = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "rates.txt")
        extract_rates(
            reconciliations_estimation_dir=reconciliations_estimation_dir,
            start_index=START_INDEX,
            end_index_estimation=END_INDEX_ESTIMATION,
            rates_file=rates_estimation,
        )

rule reconcile_all_gene_trees:
    """Reconcile all prepared sampled gene trees using estimated rates."""
    input:
        prepared_species_tree=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "ale_trees" / "SampledSpeciesTree_ale.nwk"),
        prepared_gene_trees=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "ale_trees" / "prepared_gene_1.nwk"),
        rates_estimation=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "rates.txt"),
    output:
        reconciliations_all_dir=str(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "reconciliations_all" / "reconciliation_1_uml"),
    run:
        from scripts.reconcile_all_gene_trees import reconcile_all_gene_trees
        sp_tree_index = int(wildcards.sp_tree_index)
        prepared_species_tree = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "ale_trees" / "SampledSpeciesTree_ale.nwk")
        prepared_gene_trees_dir = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "ale_trees")
        rates_file = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "rates.txt")
        reconciliations_dir = str(OUTPUT_FOLDER / f'species_tree_{sp_tree_index}' / "reconciliations_all")

        reconcile_all_gene_trees(
            prepared_species_tree=prepared_species_tree,
            prepared_gene_trees_dir=prepared_gene_trees_dir,
            rates_file=rates_file,
            start_index=START_INDEX,
            end_index=END_INDEX,
            reconciliations_dir=reconciliations_dir,
        )

rule prepare_pytorch_dataset:
    """
    Synthesize the generated reconciliation data into a pickled dataset for PyTorch,
    and generate a csv file with the inferred rates and other information.

    The dataset is a list of pytorch geometric objects, saved in the output folder.
    The csv file is saved in the output folder as well. 
    """
    input:
        reconciliations_all_dir=expand(OUTPUT_FOLDER / 'species_tree_{sp_tree_index}' / "reconciliations_all" / "reconciliation_1_uml", sp_tree_index=range(N_SP_TREES))
    output:
        dataset=str(OUTPUT_FOLDER / "dataset.pkl"),
        csv_file=str(OUTPUT_FOLDER / "global_stats.csv"),
    threads:
        min(32, workflow.cores)
    run:
        from scripts.create_pytorch_dataset import process_all_species_trees
        dataset = str(OUTPUT_FOLDER / "dataset.pkl")
        csv_file = str(OUTPUT_FOLDER / "global_stats.csv")
        process_all_species_trees(dataset_path=str(OUTPUT_FOLDER),
                                    num_processes=threads,
                                    frequency_threshold=FREQUENCY_THRESHOLD,
                                    pickle_path=str(DATASET_PICKLE_PATH),)