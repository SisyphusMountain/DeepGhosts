#!/bin/bash

for i in {1..6}; do
  snakemake --snakefile snakefile --cores 32 --configfile "config_s${i}.yaml"
done
