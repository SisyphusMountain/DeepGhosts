#!/bin/bash

# Updated submodules path
SUBMODULES=("extract_extant_script" "gene_transfer_script" "sample_script")
SUBMODULES_DIR="bin/scripts/rust_scripts"

# Use absolute path for the destination directory
current_dir=$(pwd)
DEST_DIR="$current_dir/bin/binaries"

# Create necessary directories
mkdir -p "$DEST_DIR"
mkdir -p "$current_dir/Output"

for SUBMODULE in "${SUBMODULES[@]}"; do
  echo "Building $SUBMODULE..."
  
  # Change directory to the submodule
  cd "$current_dir/$SUBMODULES_DIR/$SUBMODULE" || exit 1  # Exit if cd fails
  echo "Changing directory to $current_dir/$SUBMODULES_DIR/$SUBMODULE"
  
  # Build the submodule using Cargo
  cargo build --release
  
  # Move the built binary to the destination directory using absolute paths
  mv "target/release/$SUBMODULE" "$DEST_DIR/" || exit 1
  echo "$SUBMODULE built and moved successfully."

  # Return to the original directory before moving to the next submodule
  cd "$current_dir" || exit 1
  echo "Returned to $current_dir"
done

echo "All binaries are built and moved to $DEST_DIR."
