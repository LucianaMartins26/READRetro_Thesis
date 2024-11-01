#!/bin/bash

source /home/taein/oddments/anaconda3/etc/profile.d/conda.sh
conda activate graph2smiles
root=/home/taein/READRetro/scripts/singlestep_eval/g2s/clean

cd /home/taein/Retrosynthesis/Graph2SMILES

MODEL=g2s_series_rel

EXP_NO=1
DATASET=clean
FIRST_STEP=72000
LAST_STEP=74000

BS=200
T=1.0
NBEST=10
MPN_TYPE=dgcn


RESULT=${DATASET}_${MODEL}_${MPN_TYPE}
CHECKPOINT=$root/checkpoints/$RESULT.$EXP_NO

REPR_START=smiles
REPR_END=smiles

PREFIX=${DATASET}_${MODEL}_${REPR_START}_${REPR_END}

CUDA_VISIBLE_DEVICES=6 python validate.py \
  --model="$MODEL" \
  --data_name="$DATASET" \
  --valid_bin="$root/preprocessed/$PREFIX/val_0.npz" \
  --val_tgt="$root/tgt-test.txt" \
  --val_src="$root/src-test.txt" \
  --result_file="$root/result.txt" \
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
  --log_iter=100
