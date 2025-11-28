#!/bin/bash

SRC_DIR="/data/cep/chao.han/"
DST_DIR="/scratch/wsspaces/chao.han-"
PROJ="MethylGPT"

# Dir where code is stored
CODE_DIR="$SRC_DIR$PROJ"
DST_PROJ="$DST_DIR$PROJ"

# Create a timestamped dir for the snapshot (at submission time)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_DIR="$DST_PROJ/Snapshot_$TIMESTAMP"

# Mirror the current state of the code to the snapshot dir
mkdir -p $SNAPSHOT_DIR
rsync -a --delete $CODE_DIR/ $SNAPSHOT_DIR/

# Submit the SLURM job, passing the snapshot dir as an argument
# Run 1 task
JOBID=$(sbatch --hold --parsable \
               --output=${DST_PROJ}/%A/out/output_%A_%a.txt \
               --export=DST_PROJ=$DST_PROJ,SNAPSHOT_DIR=$SNAPSHOT_DIR \
               --array=0 \
               ./bash_griffin/run_pretrain.sh)

# Create parent dirs (job-level) for SLURM output
mkdir -p "${DST_PROJ}/${JOBID}/out"
scontrol release "$JOBID"
echo "Submitted $JOBID"