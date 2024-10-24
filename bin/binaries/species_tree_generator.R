#!/usr/bin/env Rscript

# Load necessary library
library(ape)

# Function to generate and save a phylogenetic tree
generate_phylo_tree <- function(seed,
                                n,
                                b,
                                d,
                                file_path_complete) {
  set.seed(seed)
  tree <- rphylo(n, b, d, fossils = TRUE)
  # Set the tree root length to 0
  write.tree(tree, file = file_path_complete, append = FALSE,
             digits = 5, tree.names = FALSE)
  tree
}

# Parsing command line arguments
args <- commandArgs(trailingOnly = TRUE)
# Assuming the order of arguments is: seed, n, b, d, file_path

n <- as.integer(args[1])
b <- as.numeric(args[2])
d <- as.numeric(args[3])
file_path_complete <- args[4]
seed <- as.integer(args[5])

# Call the function with parsed arguments
tree <- generate_phylo_tree(seed, n, b, d, file_path_complete)
sum_branch_lengths <- sum(tree$edge.length)
cat(sum_branch_lengths, "\n")
