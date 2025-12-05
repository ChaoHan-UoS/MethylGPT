#!/bin/bash

BASE_DIR="/scratch/wsspaces/chao.han-"
PROJ="MethylGPT/MethylGPT"
TASK_DIR="$BASE_DIR$PROJ"

mkdir -p "${TASK_DIR}/out"

sbatch --output=${TASK_DIR}/out/output_%j.txt \
       --export=TASK_DIR=TASK_DIR \
       ./bash_griffin/run_pretrain.sh