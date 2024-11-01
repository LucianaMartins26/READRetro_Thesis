#!/usr/bin/env python3

from retro_star.api import RSPlanner
from time import time
import argparse
import torch
import pandas as pd
from retro_star.common import prepare_starting_molecules, prepare_molstar_planner, smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger
from retroformer.translate import prepare_retroformer, run_retroformer
from megan.translate import prepare_megan, run_megan
from utils.ensemble import prepare_ensemble, run_ensemble, prepare_g2s, run_g2s
from retro_star.retriever import Retriever, run_retriever, run_retriever_only, neutralize_atoms, kegg_search, pathRetriever, run_path_retriever, run_both_retriever
from rdkit import Chem
import os
import sys

root = os.path.abspath('.')
if root not in sys.path:
    sys.path.insert(0, root)

def value_fn(mol, retrieved):
    if retrieved:
        return 0.
    fp = smiles_to_fp(mol, fp_dim=2048).reshape(1, -1)
    fp = torch.FloatTensor(fp).to(device)
    v = model(fp).item()
    return v

def __keggpath_find(routes, token, mol_db, path_db, top_k):
    modi_list = []
    for route in routes[:top_k]:
        r = route.split(">")
        token_position = [i for i, j in enumerate(r) if token in j]
        for pos in token_position:
            cid, _ = kegg_search(neutralize_atoms(r[pos-2].split("|")[-1]), mol_db)
            target_maps = path_db["Map"][path_db['Pathways'].apply(lambda x: any(cid in sublist for sublist in x))].to_list()
            if target_maps:
                map = target_maps[0]  # Take the first map if available
                r[pos] = r[pos].replace(token, f'{token}=kegg.jp/pathway/{map}+{cid}')
            else:  # No maps found
                modi_list.append(route)
        modi_route = '>'.join(r)
        modi_list.append(modi_route)
    return modi_list

products_dict = {
    'esculeosideA' : 'C[C@H]1[C@H]2[C@H](C[C@@H]3[C@@]2(CC[C@H]4[C@H]3CC[C@@H]5[C@@]4(CC[C@@H](C5)O[C@H]6[C@@H]([C@H]([C@H]([C@H](O6)CO)O[C@H]7[C@@H]([C@H]([C@@H]([C@H](O7)CO)O)O[C@H]8[C@@H]([C@H]([C@@H](CO8)O)O)O)O[C@H]9[C@@H]([C@H]([C@@H]([C@H](O9)CO)O)O)O)O)O)C)C)O[C@]11[C@H](C[C@@H](CN1)CO[C@H]1[C@@H]([C@H]([C@@H]([C@H](O1)CO)O)O)O)OC(=O)C',
    'albine' : 'C=CC[C@@H]1[C@H]2C[C@H](CN1)CN3[C@@H]2CC(=O)C=C3',
    'arecoline' : 'CN1CCC=C(C1)C(=O)OC',
    'calystegineB1' : 'C1[C@H]2[C@@H](C[C@](N2)([C@H]([C@@H]1O)O)O)O',
    'convicine' : 'C([C@@H]1[C@H]([C@@H]([C@H]([C@@H](O1)OC2=C(NC(=O)NC2=O)N)O)O)O)O',
    'guvacine' : 'C1CNCC(=C1)C(=O)O',
    'isoesculeogeninA' : 'C[C@H]1[C@H]2[C@H](C[C@@H]3[C@@]2(CC[C@H]4[C@H]3CC[C@@H]5[C@@]4(CC[C@@H](C5)O)C)C)O[C@@]16[C@@H](C[C@@H](CN6)CO)O',
    'lycoperosideF' : 'C[C@H]1[C@H]2[C@H](C[C@@H]3[C@@]2(CC[C@H]4[C@H]3CC[C@@H]5[C@@]4(CC[C@@H](C5)O[C@H]6[C@@H]([C@H]([C@H]([C@H](O6)CO)O[C@H]7[C@@H]([C@H]([C@@H]([C@H](O7)CO)O)O[C@H]8[C@@H]([C@H]([C@@H](CO8)O)O)O)O[C@H]9[C@@H]([C@H]([C@@H]([C@H](O9)CO)O)O)O)O)O)C)C)O[C@@]11[C@@H](C[C@@H](CN1)CO[C@H]1[C@@H]([C@H]([C@@H]([C@H](O1)CO)O)O)O)OC(=O)C',
    'esculeogeninB' : 'C[C@H]1[C@H]2[C@H](C[C@@H]3[C@@]2(CC[C@H]4[C@H]3CC[C@@H]5[C@@]4(CC[C@@H](C5)O)C)C)O[C@]6([C@H]1NC[C@H](C6)CO)O'
}

for n, p in products_dict.items():
    product = p
    name = n

    blocks = 'data/building_block_unknown.csv'
    iterations = 100
    exp_topk = 10
    route_topk = 10
    beam_size = 10
    model_type = 'retroformer'
    model_path = None
    retrieval = 'true'
    path_retrieval = 'true'
    retrieval_db = 'data/train_canonicalized.txt'
    path_retrieval_db = 'data/pathways.pickle'
    device = 'cuda'
    starting_molecules = 'data/building_block_unknown.csv'
    value_model = 'retro_star/saved_model/best_epoch_final_4.pt'
    
    retroformer_path = 'retroformer/saved_models/transfer_learning_1000_1000times_lowerLR.pt'

    model_retroformer, args = prepare_retroformer(device, beam_size, exp_topk, path=retroformer_path)
    expansion_handler = lambda x: run_retroformer(model_retroformer, args, x)

    model = ValueMLP(n_layers=1, fp_dim=2048, latent_dim=128, dropout_rate=0.1, device=device).to(device)
    model.load_state_dict(torch.load(value_model, map_location=device))
    model.eval()

    starting_mols = prepare_starting_molecules(starting_molecules)
    starting_mols.add('keggpath')

    plan_handle = prepare_molstar_planner(
        expansion_handler=expansion_handler,
        value_fn=value_fn,
        starting_mols=starting_mols,
        iterations=iterations
    )

    target_mol = Chem.MolToSmiles(Chem.MolFromSmiles(product))
    succ, msg = plan_handle(target_mol)

    kegg_mol_db = "/home/lmartins/READRetro/data/kegg_neutral_iso_smi.csv"
    kegg_mol_db = pd.read_csv(kegg_mol_db)
    routes_list = []
    modi_routes_list = []

    for route in msg:
        routes_list.append(route.serialize())
        modi_routes_list = __keggpath_find(routes=routes_list, token='keggpath', mol_db=kegg_mol_db, path_db=path_retrieval_db, top_k=10)

    if modi_routes_list:
        with open(f'./data/multi_step_result/unknown_compounds/{name}_TL1000_1000x.txt', 'w') as f:
            for route in modi_routes_list:
                f.write(route)
                f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('standard', type=str, help="Standard model to use ('True' or 'False')")
    parser.add_argument('--epochs', type=str, help="Epochs for transfer learning (e.g. '1000' or '700')", default='1000')
    args = parser.parse_args()

    standard = args.standard
    epochs = args.epochs

    if standard == 'True':
        retroformer_path = '/home/lmartins/READRetro/retroformer/saved_models/biochem.pt'
        
        if modi_routes_list:
            with open(f'./data/multi_step_result/unknown_compounds/{name}_biochem.txt', 'w') as f:
                for route in modi_routes_list:
                    f.write(route)
                    f.write('\n')
    else:
        if epochs == '1000':
            retroformer_path = 'retroformer/saved_models/transfer_learning_1000_1000times_lowerLR.pt'
            if modi_routes_list:
                with open(f'./data/multi_step_result/unknown_compounds/{name}_TL1000_1000.txt', 'w') as f:
                    for route in modi_routes_list:
                        f.write(route)
                        f.write('\n')
        elif epochs == '700':
            retroformer_path = 'retroformer/saved_models/transfer_learning_700_1000times_lowerLR.pt'
            if modi_routes_list:
                with open(f'./data/multi_step_result/unknown_compounds/{name}_TL700_1000.txt', 'w') as f:
                    for route in modi_routes_list:
                        f.write(route)
                        f.write('\n')