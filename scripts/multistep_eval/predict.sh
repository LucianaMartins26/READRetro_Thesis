#!/bin/bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate readretro

python run_mp.py -m $model_type -mp $model_path
