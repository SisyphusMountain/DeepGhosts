#!/bin/bash

CONFIG_DIR="/home/enzo/Documents/git/WP1/DeepGhosts/bin/pure_birth_renormalized_configs"
SNAKEFILE="snakefile_norm"
CORES=32

for config_file in "$CONFIG_DIR"/*.yaml; do
  echo "Running Snakemake with $config_file"
  snakemake --snakefile "$SNAKEFILE" --cores "$CORES" --configfile "$config_file"
done
