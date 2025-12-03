#!/bin/bash

echo "Downloading processed_type3_parquet_shuffled.tar.gz into data/processed_type3_parquet_shuffled/"

DEST_DIR="data/processed_type3_parquet_shuffled"
mkdir -p "$DEST_DIR"

TAR_PATH="$DEST_DIR/processed_type3_parquet_shuffled.tar.gz"
DROPBOX_URL="https://www.dropbox.com/scl/fi/bbs6sxlkpbx11rhyvdfto/processed_type3_parquet_shuffled.tar.gz?rlkey=s73utmumq6xldmv3y6kh9bz75&e=3&st=8pslwy2a&dl=1"

# Download tar.gz
wget -O "$TAR_PATH" "$DROPBOX_URL"

echo "Extracting tar.gz..."

tar -xzf "$TAR_PATH" -C "$DEST_DIR"
rm "$TAR_PATH"

echo "Done. Files are in $DEST_DIR"
