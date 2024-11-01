#!/bin/bash

MODEL=g2s_series_rel

EXP_NO=2
DATASET=biochem_star_1
CHECKPOINT=./checkpoints/biochem_star_1_g2s_series_rel_smiles_smiles.2
FIRST_STEP=50000
LAST_STEP=100000

BS=30
T=1.0
NBEST=10
MPN_TYPE=dgcn
REL_POS=emb_only

REPR_START=smiles
REPR_END=smiles

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

cd /path/to/Graph2SMILES
CUDA_VISIBLE_DEVICES=1 python validate.py \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --valid_bin="./preprocessed/revision/$PREFIX/val_0.npz" \
  --val_tgt="./data/revision/$DATASET/tgt-test.txt" \
  --val_src="./data/revision/$DATASET/src-test.txt" \
  --result_file="./results/$DATASET/$PREFIX.$EXP_NO.result.txt" \
  --log_file="$PREFIX.validate.$EXP_NO.log" \
  --load_from="$CHECKPOINT" \
  --checkpoint_step_start="$FIRST_STEP" \
  --checkpoint_step_end="$LAST_STEP" \
  --mpn_type="$MPN_TYPE" \
  --rel_pos="$REL_POS" \
  --seed=42 \
  --batch_type=tokens \
  --predict_batch_size=4096 \
  --beam_size="$BS" \
  --n_best="$NBEST" \
  --temperature="$T" \
  --predict_min_len=1 \
  --predict_max_len=512 \
  --eval_iter 2000
  
