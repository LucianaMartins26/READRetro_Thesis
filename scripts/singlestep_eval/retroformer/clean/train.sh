#!/bin/bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate readretro
path=/path/to/READRetro/scripts/singlestep_eval/retroformer/clean
mkdir $path/intermediates

python retroformer/train.py \
  --encoder_num_layers 8 \
  --decoder_num_layers 8 \
  --heads 8 \
  --max_step 1600000 \
  --batch_size_trn 4 \
  --batch_size_val 4 \
  --batch_size_token 4096 \
  --save_per_step 50000 \
  --val_per_step 10000 \
  --report_per_step 200 \
  --device cuda \
  --known_class False \
  --shared_vocab True \
  --shared_encoder False \
  --data_dir $path/data \
  --intermediate_dir $path/intermediates \
  --checkpoint_dir $path/ckpt
