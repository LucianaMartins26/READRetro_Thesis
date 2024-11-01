#!/bin/bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate readretro

python eval_single.py -m retroformer -p $path -v $vocab_path
