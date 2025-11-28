#!/bin/bash

set -e

echo "Downloading MethylGPT data into data_examples/..."

mkdir -p data_examples

ZIP_PATH="data_examples/methylgpt_data.zip"
DROPBOX_URL="https://www.dropbox.com/scl/fo/9113fa2h5jbrud2zcb7s9/ACXKuYZA5goHdHcHIKz-LOo?rlkey=tni0bjs7y722znr98hkmir5q8&st=mq3we384&dl=1"

# Download zip
wget -O "$ZIP_PATH" "$DROPBOX_URL"

echo "Extracting zip..."
if unzip -q "$ZIP_PATH" -d data_examples/; then
    echo "Extraction successful. Removing zip..."
    rm "$ZIP_PATH"
else
    echo "⚠️  unzip failed. Keeping zip file for debugging."
    echo "Try extracting manually with: unzip $ZIP_PATH -d data_examples/"
    exit 1
fi

echo "Done. Files are in data_examples/"