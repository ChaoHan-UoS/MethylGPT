#!/bin/bash

echo "Downloading processed_type3_parquet_shuffled.tar.gz into data/processed_type3_parquet_shuffled/"

DEST_DIR="data/processed_type3_parquet_shuffled"
mkdir -p "$DEST_DIR"

TAR_PATH="$DEST_DIR/processed_type3_parquet_shuffled.tar.gz"
DROPBOX_URL="https://uc3f5fa96fd1b2c01a916c66dba2.dl.dropboxusercontent.com/cd/0/get/C2WOKQ7F9xhwLe0xfL6Plt53ZRbuIe9FWyp8XEyrLNfUCqR9xq7zNY0mxs8iqul9jq1nRAVA2L_9PyyQwJsnkkVsf6J4g97FjHbomNUKrDvwN0SauuWoAkJb9ZqFlZYLZKh_dQkOPhxKQTHaZVp-q-qJ8VYJkW5gB5k_hFSkinOjEA/file?_download_id=32764388564660274110998624726781978777689027179389912730641765422&_log_download_success=1&_notify_domain=www.dropbox.com&dl=1"

# Download tar.gz
wget -O "$TAR_PATH" "$DROPBOX_URL"

echo "Extracting tar.gz..."

tar -xzf "$TAR_PATH" -C "$DEST_DIR"
rm "$TAR_PATH"

echo "Done. Files are in $DEST_DIR"
