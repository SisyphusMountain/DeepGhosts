#!/usr/bin/env Rscript

# Load required packages
library(ape)
library(ggtree)
library(ggplot2)
library(dplyr)
library(stringr)
library(tidyr)

# Set the file paths
tree_path <- "/home/enzo/Documents/git/WP1/DeepGhosts/Output/test_run/species_tree_0/T/CompleteTree.nwk"
events_path <- "/home/enzo/Documents/git/WP1/DeepGhosts/Output/test_run/species_tree_0/G/Gene_families/1_events.tsv"

# Print file paths to verify
cat("Reading tree from:", tree_path, "\n")
cat("Reading events from:", events_path, "\n")

# Check if files exist
if (!file.exists(tree_path)) {
  stop("Tree file not found: ", tree_path)
}
if (!file.exists(events_path)) {
  stop("Events file not found: ", events_path)
}

# Read the phylogenetic tree
tree <- read.tree(tree_path)
cat("Tree loaded with", length(tree$tip.label), "tips and", tree$Nnode, "internal nodes\n")

# Print a few tip labels to verify
cat("Sample tip labels:", paste(head(tree$tip.label, 3), collapse=", "), "...\n")

# Read the events data
events_data <- read.delim(events_path, header = TRUE)
cat("Events data loaded with", nrow(events_data), "rows\n")

# Filter to only get transfer (T) events
transfer_events <- events_data %>% 
  filter(EVENT == "T")
cat("Found", nrow(transfer_events), "transfer events\n")

# Function to parse the NODES column and extract donor and recipient IDs
get_transfer_info <- function(nodes_str) {
  parts <- strsplit(nodes_str, ";")[[1]]
  # Based on the format 100006;7;100006;8;100004;9
  # Donor is the first ID (100006) and recipient is the fifth ID (100004)
  donor_id <- as.numeric(parts[1])
  recipient_id <- as.numeric(parts[5])
  return(data.frame(donor = donor_id, recipient = recipient_id))
}

# Apply the function to each row of transfer events
transfers <- data.frame()
for (i in 1:nrow(transfer_events)) {
  nodes_str <- transfer_events$NODES[i]
  transfer_info <- get_transfer_info(nodes_str)
  transfer_info$time <- transfer_events$TIME[i]
  transfers <- rbind(transfers, transfer_info)
}

cat("Parsed transfer events:\n")
print(head(transfers, 10))

# Create a tree plot - use layout_dendrogram for better branch visualization
p <- ggtree(tree, layout="dendrogram") + 
     geom_tiplab() +
     geom_nodelab() +  # Add internal node labels
     theme_tree2()

# Get the node positions from the plot data
node_data <- p$data
cat("First few rows of tree node data:\n")
print(head(node_data, 10))

# Save node mapping information
write.csv(node_data, "tree_node_mapping.csv", row.names = FALSE)
cat("Saved tree node mapping to tree_node_mapping.csv\n")

# Find the scaling factor between tree x-coordinates and time
# Calculate the min and max x values in the tree
min_x <- min(node_data$x)
max_x <- max(node_data$x)

# Find the maximum time in the events data
max_time <- max(events_data$TIME)

# Calculate the scaling factor
time_scaling_factor <- (max_x - min_x) / max_time
cat("Tree scaling factor (x-coordinate per time unit):", time_scaling_factor, "\n")

# Function to find the branch position at a specific time
find_branch_at_time <- function(node_id, transfer_time, node_data, time_scaling_factor) {
  # Convert time to x-coordinate
  transfer_x <- min_x + (transfer_time * time_scaling_factor)
  
  # Get node data
  node_info <- node_data[node_data$node == node_id, ]
  if (nrow(node_info) == 0) {
    return(list(found = FALSE))
  }
  
  # Get parent information
  parent_id <- node_info$parent
  if (is.na(parent_id)) {
    # Root node - just return its position
    return(list(found = TRUE, x = transfer_x, y = node_info$y))
  }
  
  parent_info <- node_data[node_data$node == parent_id, ]
  if (nrow(parent_info) == 0) {
    return(list(found = FALSE))
  }
  
  # Check if transfer time is within this branch
  node_x <- node_info$x
  parent_x <- parent_info$x
  
  # For debugging
  # cat("Node:", node_id, "x:", node_x, "Parent:", parent_id, "x:", parent_x, "Transfer x:", transfer_x, "\n")
  
  if ((parent_x <= transfer_x && transfer_x <= node_x) || 
      (node_x <= transfer_x && transfer_x <= parent_x)) {
    # Transfer is on this branch
    # Return the y-coordinate of the node (branches are horizontal in the dendrogram layout)
    return(list(found = TRUE, x = transfer_x, y = node_info$y))
  } else {
    # Transfer is not on this branch
    # If transfer time is more recent than the node, use the node's position
    if ((transfer_x > node_x && node_x > parent_x) || 
        (transfer_x > parent_x && parent_x > node_x)) {
      return(list(found = TRUE, x = transfer_x, y = node_info$y))
    } else {
      # Otherwise, we can't reliably place the transfer on this lineage
      return(list(found = FALSE))
    }
  }
}

# Create a list to store transfer arrow coordinates
transfer_arrows <- data.frame(
  x = numeric(0),
  y = numeric(0),
  xend = numeric(0),
  yend = numeric(0),
  arrow_color = character(0),
  transfer_id = integer(0)
)

# Try to find nodes in the tree that match our transfer node IDs
cat("Processing transfer events...\n")
valid_transfers <- 0

for (i in 1:nrow(transfers)) {
  # Get donor and recipient IDs
  donor_id <- transfers$donor[i]
  recipient_id <- transfers$recipient[i]
  transfer_time <- transfers$time[i]
  
  # Try to find donor and recipient nodes in the tree
  donor_node <- NULL
  recipient_node <- NULL
  
  # First try to find in tip labels
  donor_tip_match <- which(tree$tip.label == as.character(donor_id))
  recipient_tip_match <- which(tree$tip.label == as.character(recipient_id))
  
  if (length(donor_tip_match) > 0) {
    donor_node <- donor_tip_match[1]
  }
  
  if (length(recipient_tip_match) > 0) {
    recipient_node <- recipient_tip_match[1]
  }
  
  # Then try to find in internal node labels
  if (is.null(donor_node) && !is.null(tree$node.label)) {
    donor_internal_match <- which(tree$node.label == as.character(donor_id))
    if (length(donor_internal_match) > 0) {
      donor_node <- donor_internal_match[1] + length(tree$tip.label)
    }
  }
  
  if (is.null(recipient_node) && !is.null(tree$node.label)) {
    recipient_internal_match <- which(tree$node.label == as.character(recipient_id))
    if (length(recipient_internal_match) > 0) {
      recipient_node <- recipient_internal_match[1] + length(tree$tip.label)
    }
  }
  
  # If we found both donor and recipient, calculate the arrow coordinates
  if (!is.null(donor_node) && !is.null(recipient_node)) {
    # Find branch positions at the transfer time
    donor_pos <- find_branch_at_time(donor_node, transfer_time, node_data, time_scaling_factor)
    recipient_pos <- find_branch_at_time(recipient_node, transfer_time, node_data, time_scaling_factor)
    
    if (donor_pos$found && recipient_pos$found) {
      valid_transfers <- valid_transfers + 1
      
      # Add to transfer arrows dataframe
      transfer_arrows <- rbind(transfer_arrows, data.frame(
        x = donor_pos$x,
        y = donor_pos$y,
        xend = recipient_pos$x,
        yend = recipient_pos$y,
        arrow_color = "red",
        transfer_id = i
      ))
      
      cat("Transfer", i, "from", donor_id, "to", recipient_id, "at time", transfer_time, 
          "- coordinates: (", donor_pos$x, ",", donor_pos$y, ") to (", 
          recipient_pos$x, ",", recipient_pos$y, ")\n")
    }
  }
}

cat("Successfully processed", valid_transfers, "transfers with valid coordinates\n")

# Plot the tree with the transfer arrows
if (nrow(transfer_arrows) > 0) {
  # Since we're using the dendrogram layout, the transfers should be horizontal segments
  # For the dendrogram layout:
  # - x coordinate represents time
  # - y coordinate represents the vertical position of the branch
  p_with_transfers <- p +
    geom_segment(
      data = transfer_arrows,
      aes(x = x, y = y, xend = x, yend = yend, group = transfer_id),
      arrow = arrow(length = unit(0.2, "cm"), type = "closed"),
      color = "red",
      alpha = 0.7,
      size = 0.8,
      inherit.aes = FALSE
    )
  
  # Add a title
  p_with_transfers <- p_with_transfers + 
    ggtitle("Phylogenetic Tree with Horizontal Gene Transfers") +
    theme(plot.title = element_text(hjust = 0.5))
} else {
  p_with_transfers <- p + 
    ggtitle("Phylogenetic Tree (No Valid Transfers Found)") +
    theme(plot.title = element_text(hjust = 0.5))
}

# Save the plots
ggsave("tree_basic.pdf", p, width =.0, height = 10)
cat("Saved basic tree plot to tree_basic.pdf\n")

ggsave("tree_with_transfers.pdf", p_with_transfers, width = 15, height = 15)
cat("Saved tree with transfers to tree_with_transfers.pdf\n")

# Save the transfer arrow data for debugging
write.csv(transfer_arrows, "transfer_arrows.csv", row.names = FALSE)
cat("Saved transfer arrow coordinates to transfer_arrows.csv\n")

# Display the plot
print(p_with_transfers)
cat("Script completed!\n")