#!/bin/bash

echo "Downloading processed_type3_parquet_shuffled.tar.gz into data/processed_type3_parquet_shuffled/"

DEST_DIR="data/processed_type3_parquet_shuffled"
mkdir -p "$DEST_DIR"

TAR_PATH="$DEST_DIR/processed_type3_parquet_shuffled.tar.gz"
DROPBOX_URL="https://uced7957724472baf6f80735c741.dl.dropboxusercontent.com/cd/0/get/C2Xkb-2CdF_mPDn5bDinGHzRrxBZthc3WhbmlLcL-mStjUgzDYuTOyzWPCm3MsDxujmDzubj_Q5OCaBOeRUQ9k3irIC5DY9vNYMna474J-PeIu5YSajGwnQkwunTrDiEY1ot9wQ0WYrR690rhLe9vsFWxWt7o50ZQz7hxOxW3INe9A/file?_download_id=16090692631357484951216844092507454752866486066417629603414689776&_log_download_success=1&_notify_domain=www.dropbox.com&dl=1"

# Download tar.gz
wget -O "$TAR_PATH" "$DROPBOX_URL"

echo "Extracting tar.gz..."

tar -xzf "$TAR_PATH" -C "$DEST_DIR"
rm "$TAR_PATH"

echo "Done. Files are in $DEST_DIR"
