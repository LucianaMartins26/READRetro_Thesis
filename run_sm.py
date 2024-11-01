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
    'avenacinA1' : '[CH3:1][C@:2]12[CH2:3][CH2:4][C@H:5]([O:46][C@H:47]3[C@H:48]([O:66][C@H:67]4[C@H:68]([OH:77])[C@@H:69]([OH:76])[C@H:70]([OH:75])[C@@H:71]([CH2:73][OH:74])[O:72]4)[C@@H:49]([OH:65])[C@@H:50]([O:53][C@H:54]4[C@H:55]([OH:64])[C@@H:56]([OH:63])[C@H:57]([OH:62])[C@@H:58]([CH2:60][OH:61])[O:59]4)[CH2:51][O:52]3)[C@@:6]([CH3:43])([CH2:44][OH:45])[C@@H:7]1[CH2:8][CH2:9][C@:10]1([CH3:42])[C@@H:11]2[CH2:12][C@@H:13]2[C@:14]3([C@@:15]1([CH3:40])[CH2:16][C@H:17]([OH:39])[C@:18]1([CH3:38])[C@H:19]3[CH2:20][C@@:21]([CH3:35])([CH:36]=[O:37])[C@@H:22]([O:24][C:25](=[O:26])[C:27]3=[CH:28][CH:29]=[CH:30][CH:31]=[C:32]3[NH:33][CH3:34])[CH2:23]1)[O:41]2',
    'momilactoneB' : '[CH3:1][C@@:2]1([CH:23]=[CH2:24])[CH2:3][CH2:4][C@@H:5]2[C:6](=[CH:7][CH:8]3[C@@H:9]4[C@:10]25[CH2:11][CH2:12][C:13]([OH:21])([C@:14]4([CH3:18])[C:15](=[O:16])[O:17]3)[O:19][CH2:20]5)[CH2:22]1',
    'afrormosin' : '[CH3:1][O:2][C:3]1=[CH:4][CH:5]=[C:6]([C:9]2=[CH:10][O:11][C:12]3=[CH:13][C:14]([OH:22])=[C:15]([O:20][CH3:21])[CH:16]=[C:17]3[C:18]2=[O:19])[CH:7]=[CH:8]1',
    'lycosantalonol' : '[CH3:1][C:2](=[CH:3][CH2:4][CH2:5][C:6]([CH3:7])([C:8](=[O:9])[CH2:10][CH2:11][C:12]1([CH3:20])[CH:13]2[CH2:14][CH:15]3[C:16]1([CH3:19])[CH:17]3[CH2:18]2)[OH:21])[CH3:22]' ,
    'vincamine' : '[CH3:1][CH2:2][C@@:3]12[CH2:4][CH2:5][CH2:6][N:7]3[C@@H:8]1[C:9]1=[C:10]([CH2:11][CH2:12]3)[C:13]3=[CH:14][CH:15]=[CH:16][CH:17]=[C:18]3[N:19]1[C@:20]([C:22](=[O:23])[O:24][CH3:25])([OH:26])[CH2:21]2',
    'brucine' : '[CH3:1][O:2][C:3]1=[C:4]([O:28][CH3:29])[CH:5]=[C:6]2[C:7](=[CH:8]1)[C@:9]13[CH2:10][CH2:11][N:12]4[C@H:13]1[CH2:14][C@@H:15]1[C@@H:16]5[C@@H:17]3[N:18]2[C:19](=[O:20])[CH2:21][C@@H:22]5[O:23][CH2:24][CH:25]=[C:26]1[CH2:27]4',
    'diaboline' : '[CH3:1][C:2](=[O:3])[N:4]1[C@H:5]2[C@H:6]3[C@H:7]4[CH2:8][C@H:9]5[C@@:10]2([CH2:11][CH2:12][N:13]5[CH2:14][C:15]4=[CH:16][CH2:17][O:18][C@H:19]3[OH:20])[C:21]2=[CH:22][CH:23]=[CH:24][CH:25]=[C:26]12',
    'falcarindiol' : '[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH2:6][CH2:7]/[CH:8]=[CH:9]\[C@@H:10]([C:11]#[C:12][C:13]#[C:14][C@@H:15]([CH:16]=[CH2:17])[OH:18])[OH:19]'
}

for n, p in products_dict.items():
    product = p
    name = n

    blocks = '/home/lmartins/READRetro/data/building_block.csv'
    iterations = 100
    exp_topk = 10
    route_topk = 10
    beam_size = 10
    retrieval = 'true'
    path_retrieval = 'true'
    retrieval_db = '/home/lmartins/READRetro/data/train_canonicalized.txt'
    path_retrieval_db = '/home/lmartins/READRetro/data/pathways.pickle'
    device = 'cuda'
    starting_molecules = '/home/lmartins/READRetro/data/building_block.csv'
    value_model = '/home/lmartins/READRetro/retro_star/saved_model/best_epoch_final_4.pt'
    
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
        with open(f'./data/multi_step_result/{name}_TL1000_1000x.txt', 'w') as f:
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
            with open(f'./data/multi_step_result/{name}_biochem.txt', 'w') as f:
                for route in modi_routes_list:
                    f.write(route)
                    f.write('\n')
    else:
        if epochs == '1000':
            retroformer_path = 'retroformer/saved_models/transfer_learning_1000_1000times_lowerLR.pt'
            if modi_routes_list:
                with open(f'./data/multi_step_result/{name}_TL1000_1000.txt', 'w') as f:
                    for route in modi_routes_list:
                        f.write(route)
                        f.write('\n')
        elif epochs == '700':
            retroformer_path = 'retroformer/saved_models/transfer_learning_700_1000times_lowerLR.pt'
            if modi_routes_list:
                with open(f'./data/multi_step_result/{name}_TL700_1000.txt', 'w') as f:
                    for route in modi_routes_list:
                        f.write(route)
                        f.write('\n')