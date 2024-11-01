#!/bin/bash

DATASET=biochem_star_2
MODEL=g2s_series_rel
TASK=retrosynthesis
REPR_START=smiles
REPR_END=smiles
N_WORKERS=32

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

cd /path/to/Graph2SMILES
echo $pwd

python preprocess.py \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --task="$TASK" \
  --representation_start=$REPR_START \
  --representation_end=$REPR_END \
  --train_src="./data/revision/$DATASET/src-train.txt" \
  --train_tgt="./data/revision/$DATASET/tgt-train.txt" \
  --val_src="./data/revision/$DATASET/src-val.txt" \
  --val_tgt="./data/revision/$DATASET/tgt-val.txt" \
  --test_src="./data/revision/$DATASET/src-test.txt" \
  --test_tgt="./data/revision/$DATASET/tgt-test.txt" \
  --log_file="$PREFIX.preprocess.log" \
  --preprocess_output_path="./preprocessed/revision/$PREFIX/" \
  --seed=42 \
  --max_src_len=1024 \
  --max_tgt_len=1024 \
  --num_workers="$N_WORKERS"
