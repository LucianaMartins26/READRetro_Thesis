#!/bin/bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate readretro

python eval.py $save_file -c $product_class