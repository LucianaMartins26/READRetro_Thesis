#!/bin/bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate readretro

python eval_single.py -m g2s -s 200 -p $path -v $vocab_path
