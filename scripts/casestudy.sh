#! /usr/bin/bash
source /path/to/anaconda3/etc/profile.d/conda.sh
conda activate readretro

# Figure 3
# Fig.3a Catharanghine and Tarbersonine
CUDA_VISIBLE_DEVICES=1 python run.py -pr false 'CCC1=CC2CC3(C1N(C2)CCC4=C3NC5=CC=CC=C45)C(=O)OC'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false 'CCC12CC(=C3C4(C1N(CC4)CC=C2)C5=CC=CC=C5N3)C(=O)OC'
# Fig.3b Menisdaurilide
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'O=C1C=C2C=C[C@@H](O)C[C@H]2O1'
# Fig.3c Cannabichromenic acid
CUDA_VISIBLE_DEVICES=0 python run.py 'CCCCCc1cc2c(c(O)c1C(=O)O)C=CC(C)(CCC=C(C)C)O2'
# Fig.3d Glucotropaeolin
CUDA_VISIBLE_DEVICES=0 python run.py 'C1=CC=C(C=C1)CC(=NOS(=O)(=O)O)S[C@H]2[C@@H]([C@H]([C@@H]([C@H](O2)CO)O)O)O'



# Supplementary Fig. 4
# 4a Quercetin 
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'C1=CC(=C(C=C1C2=C(C(=O)C3=C(C=C(C=C3O2)O)O)O)O)O'
# 4b Harmin
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC1=NC=CC2=C1NC3=C2C=CC(=C3)OC'
# 4c α-pinene, β-thujene ,α-terpineol,camphene 
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC1=CCC2CC1C2(C)C'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC1C=CC2(C1C2)C(C)C'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC1=CCC(CC1)C(C)(C)O'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC1(C2CCC(C2)C1=C)C'
# 4d α-gurjunene, aristolochene, β-bourbonene
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC1CCC2C(C2(C)C)C3=C(CCC13)C'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC1CCCC2=CCC(CC12C)C(=C)C'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC(C)C1CCC2(C1C3C2CCC3=C)C'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC1CCC(C2C1C=CC(C2)C)C(C)C'
# 4e cadinene, Cyanthane, Gersolanoid, and Ingenane
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC(C1)CCCC(C)(C)CC[C@H](C)CC2C1C[C@H](C)C2'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC12CCC3(C)CCC(C)CCC3C1C(C(C)C)CC2'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC12CCCCCC(C(C)C)CCC(C)CCC1C2'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false -r false 'CC1CC2CC(CC23CC(C1)C4C(C4(C)C)CC3C)C'



# Supplementary Fig. 7 Cannabichromenic acid, and Glucotropaeolin
CUDA_VISIBLE_DEVICES=0 python run.py -pr false 'CCCCCc1cc2c(c(O)c1C(=O)O)C=CC(C)(CCC=C(C)C)O2'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false 'C1=CC=C(C=C1)CC(=NOS(=O)(=O)O)S[C@H]2[C@@H]([C@H]([C@@H]([C@H](O2)CO)O)O)O'



# Supplementary Fig. 9 ent-Phyllocladane, Cattleyene, Betaerane, Allokutznerene, Aphidicolane, Dolabellane
CUDA_VISIBLE_DEVICES=0 python run.py -pr false 'CC1CCC23CCC4C(C)(C)CCCC4(C)C2CCC1C3'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false 'CC1CCC2=C3CC4(C)CCC(C)(C)C4C3CCC21C'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false 'CC1CC2CCC3C(C)(C)CCCC3(C)C23CCC1C3'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false 'C[C@@H]1CC[C@H]2[C@@]13CCC(=C3C[C@]4([C@@H]2C(CC4)(C)C)C)C'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false 'CC1(C)CCCC2(C)C1CCC1CC3CCCC12C3'
CUDA_VISIBLE_DEVICES=0 python run.py -pr false 'CC(CC1)CCC[C@H](C)CC[C@]2(C)C1[C@H](C(C)C)CC2'