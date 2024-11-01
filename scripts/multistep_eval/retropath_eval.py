from tqdm import tqdm
from rdkit import Chem
import pandas as pd
import os, sys

def gen_pathways(path_path, compound_path):
    
    with open(compound_path) as f:
        compounds =  f.readlines()
        compounds = {i.rstrip().split()[0]:i.rstrip().split()[1] for i in compounds[1:]}
    with open(path_path) as f:
        path_lines = f.readlines()
        path_lines = [i.rstrip().split(',') for i in path_lines[1:]]
    # print(compounds)
    starting = Chem.MolToSmiles(Chem.MolFromSmiles(compounds[path_lines[0][4].split('.')[-1].replace('"','')]))
    pathways = []
    pathway, id = [], '0'
    for line in path_lines:
        if id != line[0]:
            pathways.append(pathway)
            id = line[0]
            pathway = [starting]
        try:
            pathway.append(Chem.MolToSmiles(Chem.MolFromSmiles(compounds[line[-2].split('.')[-1].replace('"','')])))
        except:
            print(line[3])
    pathways = pathways[1:]
        
    return pathways

def dict_update(tdict, tkey, tval):
    if tkey in tdict:
        pos = len(tdict[tkey])
        tdict[tkey].update({pos: tval})
    else:
        tdict[tkey] = {0: tval}

def get_lst_from_pred_dict(s):  
    '''
    :param s: 'smiles1>score1>smiles2|smiles2>score2>smiles3|smiles3>score3>smiles4...'
    :return: [smiles1, smiles2, smiles3, ...]
    '''
    lst = []
    rea = s.split('|')
    for i in range(len(rea)):
        line = rea[i].split('>')
        if i == 0:
            lst.append(line[0])
        lst.append(line[-1])
    return lst

def get_lst_from_gt_dict(s):
    '''
    :param s: 'smiles1|smiles2|smiles3...'
    :return: [smiles1, smiles2, smiles3, ...]
    '''
    l = s.split('|')
    mols = []
    for i in l:
        try:
            mols.append(i)
        except:
            print(f"wrong reaction: {i}")
    return mols

def get_ground_truth_dict(pth):
    '''
    :param pth: the path of ground truth file(.txt)
    :return: a dict, format: {
                    smiles1: {0: ground truth path1; 1: ground truth path2; ...}
                    smiles2: {0: ground truth path1; 1: ground truth path2; ...}
                }
    '''
    ground_truth_dict = {}
    with open(pth, 'r') as f:
        l = [i.strip('\n') for i in f.readlines()]
    
    for line in l:
        ele = line.split('\t')
        ele = [i for i in ele if len(i) != 0]
        target = ele[3]
        dict_update(ground_truth_dict, target, '|'.join(ele[3:]))
    return ground_truth_dict

def blockHit(ori_path, pred_path):
    ori_bb = ori_path[-1]
    pred_bb = pred_path[-1]
    return ori_bb == pred_bb


def pathHit(ori_path, pred_path):
    ori_set = set(ori_path)
    pred_set = set(pred_path)
    intersect = len(ori_set & pred_set)
    num = 0
    if len(ori_path) == len(pred_path) and intersect == len(ori_path):
        num = 1
    return num

class MultiPredRes:
    def __init__(self, testSetLength):
        self.successPath = 0
        self.blockHit = 0
        self.pathHit = 0
        self.testSetLen = testSetLength
        self.result_dict = {}

    def resultShow(self):
        s =  f"Success rate:\t\t\t{self.successPath / self.testSetLen*100:.4f}%\n"
        s += f"Hit rate of building blocks:\t{self.blockHit / self.testSetLen*100:.4f}%\n"
        s += f"Hit rate of pathways:\t\t{self.pathHit / self.testSetLen*100:.4f}%"
        print(s)

if __name__ == '__main__':
    
    root = "/path/to/rp2paths/examples/paths"
    targets = os.listdir(root)
    ground_truth_dict = get_ground_truth_dict('/path/to/READRetro/data/test_gt.txt')

    mRes = MultiPredRes(len(targets))
    
    for target in tqdm(targets):
        tmp_results = os.listdir(f"{root}/{target}")
        if 'out_paths.csv' not in tmp_results:
            continue
        mRes.successPath += 1
        pathways = gen_pathways(f'{root}/{target}/out_paths.csv',f'{root}/{target}/compounds.txt')
        smiles = pathways[0][0]
        try:
            gt_dict = ground_truth_dict[smiles]
        except:
            import pdb; pdb.set_trace()
        building_block_same = False
        path_same = False
        
        for path in pathways:
            for item in gt_dict.values():
                gt_lst = get_lst_from_gt_dict(item)

                bb_same = blockHit(gt_lst, path)
                if bb_same:
                    mRes.blockHit += 1

                num = pathHit(gt_lst, path)
                if num:
                    mRes.pathHit += 1
                    
    mRes.resultShow()