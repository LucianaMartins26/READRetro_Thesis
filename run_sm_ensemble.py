from retro_star.api import RSPlanner
from time import time
import argparse
import torch
import pandas as pd
from retro_star.common import prepare_starting_molecules, \
     prepare_molstar_planner, smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger
from retroformer.translate import prepare_retroformer, run_retroformer
from megan.translate import prepare_megan, run_megan
from utils.ensemble import prepare_ensemble, run_ensemble, prepare_g2s, run_g2s
from retro_star.retriever import Retriever, run_retriever, run_retriever_only, \
    neutralize_atoms, kegg_search, \
    pathRetriever, run_path_retriever, run_both_retriever
from rdkit import Chem
import os

import sys
root = os.path.abspath('.')
if root not in sys.path:
    sys.path.insert(0, root)

def value_fn(mol, retrieved):
    if retrieved: return 0.
    # import pdb; pdb.set_trace()
    fp = smiles_to_fp(mol, fp_dim=2048).reshape(1, -1)
    fp = torch.FloatTensor(fp).to(device)
    v = model(fp).item()
    return v

def __keggpath_find(routes,token,mol_db,path_db,top_k):
    modi_list = []
    for route in routes[:top_k]:
        r = route.split(">")
        token_position = [i for i,j in enumerate(r) if token in j]
        for pos in token_position:
            cid, _ = kegg_search(neutralize_atoms(r[pos-2].split("|")[-1]),mol_db)
            target_maps = path_db["Map"][path_db['Pathways'].apply(lambda x: any(cid in sublist for sublist in x))].to_list()
            map = target_maps[0]  # check a representation method
            r[pos] = r[pos].replace(token,f'{token}=kegg.jp/pathway/{map}+{cid}')
            if target_maps == []:  # not the case
                modi_list.append(route)

        modi_route = '>'.join(r)
        modi_list.append(modi_route)
    return modi_list

products_dict = {
    'avenacinA1' : 'CC12CCC(C(C1CCC3(C2CC4C5(C3(CC(C6(C5CC(C(C6)OC(=O)C7=CC=CC=C7NC)(C)C=O)C)O)C)O4)C)(C)CO)OC8C(C(C(CO8)OC9C(C(C(C(O9)CO)O)O)O)O)OC1C(C(C(C(O1)CO)O)O)O',
    'momilactoneB' : 'C[C@]1(CC[C@@H]2C(=CC3[C@@H]4[C@]25CCC([C@@]4(C(=O)O3)C)(OC5)O)C1)C=C',
    'afrormosin' : 'COC1=CC=C(C=C1)C2=COC3=CC(=C(C=C3C2=O)OC)O',
    'lycosantalonol' : 'CC(=CCCC(C)(C(=O)CCC1(C2CC3C1(C3C2)C)C)O)C' ,
    'vincamine' : 'CC[C@@]12CCCN3[C@@H]1C4=C(CC3)C5=CC=CC=C5N4[C@](C2)(C(=O)OC)O',
    'brucine' : 'COC1=C(C=C2C(=C1)[C@]34CCN5[C@H]3C[C@@H]6[C@@H]7[C@@H]4N2C(=O)C[C@@H]7OCC=C6C5)OC',
    'diaboline' : 'CC(=O)N1[C@H]2[C@H]3[C@H]4C[C@H]5[C@@]2(CCN5CC4=CCO[C@H]3O)C6=CC=CC=C61',
    'falcarindiol' : 'CCCCCCC/C=C\[C@@H](C#CC#C[C@@H](C=C)O)O'
}


blocks = '/home/lmartins/READRetro/data/building_block.csv'
iterations = 100
exp_topk = 10
route_topk = 10
beam_size = 10
model_type = 'ensemble'
model_path = None
retrieval = 'true'
path_retrieval = 'true'
retrieval_db = '/home/lmartins/READRetro/data/train_canonicalized.txt'
path_retrieval_db = '/home/lmartins/READRetro/data/pathways.pickle'
device = 'cuda'
starting_molecules = '/home/lmartins/READRetro/data/building_block.csv'
value_model = '/home/lmartins/READRetro/retro_star/saved_model/best_epoch_final_4.pt'
kegg_mol_db = "/home/lmartins/READRetro/data/kegg_neutral_iso_smi.csv"
expansion_topk = 10

path_retrieve_token = 'keggpath'
starting_mols = prepare_starting_molecules(starting_molecules)
starting_mols.add(path_retrieve_token)


for n, p in products_dict.items():
    product = p
    name = n
    
    model_retroformer, model_g2s, args_retroformer, args_g2s, vocab, vocab_tokens, _ = \
        prepare_ensemble(device, beam_size, expansion_topk)

    path_retriever = pathRetriever(kegg_mol_db, path_retrieval_db, path_retrieve_token)
    retriever = Retriever(retrieval_db)
    expansion_handler = lambda x: run_both_retriever(x, path_retriever, retriever, model_type, model_retroformer, model_g2s, args_retroformer, args_g2s,
                                                vocab, vocab_tokens, device, expansion_topk)

    model = ValueMLP(
        n_layers=1,
        fp_dim=2048,
        latent_dim=128,
        dropout_rate=0.1,
        device=device
    ).to(device)
    model.load_state_dict(torch.load(value_model, map_location=device))
    model.eval()


    plan_handle = prepare_molstar_planner(
        expansion_handler=expansion_handler,
        value_fn=value_fn,
        starting_mols=starting_mols,
        iterations=iterations
    )

    target_mol = Chem.MolToSmiles(Chem.MolFromSmiles(product))
    succ, msg = plan_handle(target_mol)

    routes_list = []
    modi_routes_list = []

    kegg_mol_db_readable = pd.read_csv(kegg_mol_db)
    path_retrieval_db_readable = pd.read_pickle(path_retrieval_db)

    for route in msg:
        routes_list.append(route.serialize())
        modi_routes_list = __keggpath_find(routes=routes_list,token='keggpath',mol_db=kegg_mol_db_readable,path_db=path_retrieval_db_readable,top_k=10)

    if modi_routes_list:
        with open('./data/multi_step_result/{}_ensemble.txt'.format(name), 'w') as f:
            for route in modi_routes_list:
                f.write(route)
                f.write('\n')