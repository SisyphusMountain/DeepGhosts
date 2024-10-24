# params.py

import numpy as np
import numpy.random as npr
from scripts.utils import generate_seed_from_parameters
import os

def generate_species_tree_params(N_SP_TREES, SEED, config):
    species_tree_params = {}
    for sp_tree_index in range(N_SP_TREES):
        npr.seed(SEED + sp_tree_index)
        death_rate = npr.uniform(
            low=config['death_rate_low'], high=config['death_rate_high']
        )
        transfer_rate = npr.uniform(
            low=config['transfer_rate_low'], high=config['transfer_rate_high']
        )
        sampled_species = npr.randint(
            low=config['sampled_species_low'], high=config['sampled_species_high']
        )
        extant_species = npr.randint(
            low=config['extant_species_low'], high=config['extant_species_high']
        )
        birth_rate = npr.uniform(
            low=config['birth_rate_low'], high=config['birth_rate_high']
        )
        species_tree_params[sp_tree_index] = {
            'death_rate': death_rate,
            'transfer_rate': transfer_rate,
            'sampled_species': sampled_species,
            'extant_species': extant_species,
            'birth_rate': birth_rate,
        }
    return species_tree_params

def get_params_generate_species_tree(
    wildcards,
    species_tree_params,
    SEED,
    N_GENES
):
    sp_tree_index = int(wildcards.sp_tree_index)
    # Ensure that all required parameters are included in the params dictionary
    seed = generate_seed_from_parameters(
        species_tree_params[sp_tree_index]['extant_species'],
        N_GENES,
        species_tree_params[sp_tree_index]['transfer_rate'],
        species_tree_params[sp_tree_index]['death_rate'],
        species_tree_params[sp_tree_index]['birth_rate'],
        SEED,
    )
    return {
        'seed': seed,
        'extant_species': species_tree_params[sp_tree_index]['extant_species'],
        'birth_rate': species_tree_params[sp_tree_index]['birth_rate'],
        'death_rate': species_tree_params[sp_tree_index]['death_rate'],
        'n_extant_nodes': species_tree_params[sp_tree_index]['extant_species'],
    }

def get_params_create_transfers_file(
    wildcards, species_tree_params, DISTRIBUTION, N_GENES, SEED
):
    sp_tree_index = int(wildcards.sp_tree_index)
    seed = generate_seed_from_parameters(
        species_tree_params[sp_tree_index]['extant_species'],
        N_GENES,
        species_tree_params[sp_tree_index]['transfer_rate'],
        species_tree_params[sp_tree_index]['death_rate'],
        species_tree_params[sp_tree_index]['birth_rate'],
        SEED,
    )
    return {
        'n_genes': N_GENES,
        'distribution': DISTRIBUTION,
        'seed': seed,
        'transfer_rate': species_tree_params[sp_tree_index]['transfer_rate'],
    }

def get_params_generate_gene_tree(
    wildcards, species_tree_params, OUTPUT_FOLDER, SEED, N_GENES,
):
    sp_tree_index = int(wildcards.sp_tree_index)
    seed = generate_seed_from_parameters(
        species_tree_params[sp_tree_index]['extant_species'],
        N_GENES,  # Ensure N_GENES is passed correctly
        species_tree_params[sp_tree_index]['transfer_rate'],
        species_tree_params[sp_tree_index]['death_rate'],
        species_tree_params[sp_tree_index]['birth_rate'],
        SEED,
    )
    return {
        'seed': seed,
        'output_complete': os.path.join(
            OUTPUT_FOLDER, f"species_tree_{sp_tree_index}/complete/"
        ),
    }

def get_params_sampling_trees(
    wildcards, species_tree_params, START_INDEX, END_INDEX, SEED, N_GENES
):
    sp_tree_index = int(wildcards.sp_tree_index)
    seed = generate_seed_from_parameters(
        species_tree_params[sp_tree_index]['extant_species'],
        N_GENES,
        species_tree_params[sp_tree_index]['transfer_rate'],
        species_tree_params[sp_tree_index]['death_rate'],
        species_tree_params[sp_tree_index]['birth_rate'],
        SEED,
    )
    return {
        'seed': seed,
        'n_sampled_nodes': species_tree_params[sp_tree_index]['sampled_species'],
        'start_index': START_INDEX,
        'end_index': END_INDEX,
    }

def get_params_reconcile_gene_tree(
        START_INDEX, END_INDEX_ESTIMATION
):
    return {
        'start_index': START_INDEX,
        'end_index_estimation': END_INDEX_ESTIMATION,
    }

def get_params_extract_rates(
    START_INDEX, END_INDEX_ESTIMATION
):
    return {
        'start_index': START_INDEX,
        'end_index_estimation': END_INDEX_ESTIMATION,
    }

def get_params_reconcile_all_gene_trees(
    START_INDEX, END_INDEX
):
    return {
        'start_index': START_INDEX,
        'end_index': END_INDEX,
    }
