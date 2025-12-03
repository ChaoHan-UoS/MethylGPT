#!/bin/bash

echo "Downloading MethylGPT raw data into data_examples/raw_data/"

mkdir -p "data_examples/raw_data/"

ZIP_PATH="data_examples/raw_data/methylgpt_data.zip"
DROPBOX_URL="https://www.dropbox.com/scl/fo/9113fa2h5jbrud2zcb7s9/ACXKuYZA5goHdHcHIKz-LOo?rlkey=tni0bjs7y722znr98hkmir5q8&st=mq3we384&dl=1"

# Download zip
wget -O "$ZIP_PATH" "$DROPBOX_URL"

echo "Extracting zip..."

unzip -qo "$ZIP_PATH" -d "data_examples/raw_data/"
rm "$ZIP_PATH"

echo "Done. Files are in data_examples/raw_data/"