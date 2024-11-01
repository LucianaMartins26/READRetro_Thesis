import pandas as pd
from rxnmapper import BatchedMapper

def add_atom_mapping(smiles_list):
    bm = BatchedMapper(batch_size=5)
    mapped_smiles = bm.map_reactions(list(smiles_list))
    return mapped_smiles

compounds = ['avenacinA1', 'momilactoneB', 'afrormosin', 'lycosantalonol', 'vincamine', 'brucine', 'diaboline', 'falcarindiol']

for compound in compounds:
    tgt_am = []
    src_am = []

    input_filename = f'./{compound}_reactionSMILES.txt'
    output_filename = f'./{compound}_reactionSMILES_AM.txt'

    with open(input_filename, 'r') as f:
        reactions = f.readlines()
        reactions = [reaction.strip() for reaction in reactions]

    mapped_reactions = add_atom_mapping(reactions)

    with open(output_filename, 'w') as f:
        for reaction in mapped_reactions:
            f.write(reaction + '\n')