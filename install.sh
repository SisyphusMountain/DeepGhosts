#!/bin/bash


SUBMODULES=("extract_extant_script" "gene_transfer_script" "sample_script")

DEST_DIR="resources"


mkdir -p $DEST_DIR
mkdir -p Output

for SUBMODULE in "${SUBMODULES[@]}"; do
  echo "Building $SUBMODULE..."
  cd $SUBMODULE

  cargo build --release

  mv target/release/$SUBMODULE ../$DEST_DIR/
  cd ..
  echo "$SUBMODULE built and moved successfully."
done

echo "All binaries are built and moved to $DEST_DIR."
