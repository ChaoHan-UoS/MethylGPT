#!/bin/bash

# Data
RUN_PREPROCESSING=true
RAW_DATA_DIR="data_examples/raw_data"                  # Dir with raw CSV/CSV.gz files
PROBE_ID_REF="data_examples/probe_ids_type3.csv"            #  Probe ID ref for preprocessing & vocab
PREPROCESSED_PARQUET_DIR="data_examples/parquet_files"            # Parquet files
PREPROCESSED_METADATA_PATH="data_examples/QCed_samples_type3.csv"  # Metadata file after QC

# Config and script files
CONFIG_FILE="config_ex.json"
PREPROCESS_SCRIPT="preprocess_data.py"
TRAIN_SCRIPT="pretraining.py"

echo "--- Configuration ---"
echo "Execution Directory: $(pwd)"
echo "Run Preprocessing: ${RUN_PREPROCESSING}"
echo "Raw Data Dir: ${RAW_DATA_DIR}"
echo "Probe ID Ref: ${PROBE_ID_REF}"
echo "Output Metadata File: ./${PREPROCESSED_METADATA_PATH}"
echo "Output Parquet Dir: ./${PREPROCESSED_PARQUET_DIR}"
echo "Config File: ${CONFIG_FILE}"
echo "---------------------"

if [ "${RUN_PREPROCESSING}" = true ]; then
  echo ""
  echo "--- Starting Preprocessing Step ---"

  python "${PREPROCESS_SCRIPT}" \
      --input_raw_csv_dir "${RAW_DATA_DIR}" \
      --probe_id_ref_path "${PROBE_ID_REF}" \
      --output_parquet_dir "${PREPROCESSED_PARQUET_DIR}" \
      --output_metadata_path "${PREPROCESSED_METADATA_PATH}"


  if [ $? -ne 0 ]; then
      echo "Preprocessing failed. Exiting."
      exit 1
  fi
  echo "--- Preprocessing Step Completed ---"
else
  echo ""
  echo "--- Skipping Preprocessing Step (RUN_PREPROCESSING is false) ---"
  echo "To enable, edit this script and set ƒ=true."
fi

echo ""
echo "--- Starting Training Step ---"

python "${TRAIN_SCRIPT}" \
    --config_file "${CONFIG_FILE}" \
    --probe_id_path "${PROBE_ID_REF}" \
    --parquet_data_dir "${PREPROCESSED_PARQUET_DIR}" \
    --qced_metadata_path "${PREPROCESSED_METADATA_PATH}" \
    "$@"           # additional args of the TRAIN_SCRIPT passed when invoking this bash script

if [ $? -ne 0 ]; then
    echo "Training script exited with an error."
    exit 1
fi
echo "--- Training Step Script Invoked ---"
