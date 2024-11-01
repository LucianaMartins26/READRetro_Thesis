#!/bin/bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate graph2smiles

root=/home/taein/READRetro/scripts/singlestep_eval/g2s/clean
# root=/path/to/READRetro/scripts/singlestep_eval/g2s/clean
mkdir $root/preprocessed

DATASET=clean
MODEL=g2s_series_rel
TASK=retrosynthesis
REPR_START=smiles
REPR_END=smiles
N_WORKERS=32

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

python preprocess.py \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --task="$TASK" \
  --representation_start=$REPR_START \
  --representation_end=$REPR_END \
  --train_src="$root/src-train.txt" \
  --train_tgt="$root/tgt-train.txt" \
  --val_src="$root/src-val.txt" \
  --val_tgt="$root/tgt-val.txt" \
  --test_src="$root/src-test.txt" \
  --test_tgt="$root/tgt-test.txt" \
  --log_file="$PREFIX.preprocess.log" \
  --preprocess_output_path="$root/preprocessed/$PREFIX/" \
  --seed=42 \
  --max_src_len=1024 \
  --max_tgt_len=1024 \
  --num_workers="$N_WORKERS"
