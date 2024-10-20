#!/bin/bash

# Updated submodules path
SUBMODULES=("extract_extant_script" "gene_transfer_script" "sample_script")
SUBMODULES_DIR="rust_projects"

DEST_DIR="bin"

mkdir -p $DEST_DIR
mkdir -p Output

for SUBMODULE in "${SUBMODULES[@]}"; do
  echo "Building $SUBMODULE..."
  cd $SUBMODULES_DIR/$SUBMODULE

  cargo build --release

  mv target/release/$SUBMODULE ../../$DEST_DIR/
  cd ../..
  echo "$SUBMODULE built and moved successfully."
done

echo "All binaries are built and moved to $DEST_DIR."
