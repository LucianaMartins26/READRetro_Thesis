import argparse
from rdkit import Chem

def rm_stereo(smi):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi),isomericSmiles=False)
    except:
        # import pdb; pdb.set_trace()
        # print(smi)
        return smi

def dict_update(tdict, tkey, tval):
    if tkey in tdict:
        pos = len(tdict[tkey])
        tdict[tkey].update({pos: tval})
    else:
        tdict[tkey] = {0: tval}


def get_pred_dict(pth, mol2class, product_class='all'):
    mol_dict= {}
    with open(pth, 'r') as f:
        lines = [i.strip('\n') for i in f.readlines()]
    for line in lines:
        path = line.split(' ')[-1]
        if path in {'None', 'Error'}:
            continue
        product = path.split('>')[0]
        if product_class == 'all' or mol2class[product] == product_class:
            dict_update(mol_dict, product, path)
    return mol_dict


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
            lst.append(rm_stereo(line[0]))
        lst.append(rm_stereo(line[-1]))
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


def blockHit(ori_path, pred_path):
    ori_bb = ori_path[-1]
    pred_bb = pred_path[-1]
    return ori_bb == pred_bb


def pathHit(ori_path, pred_path):
    ori_set = set(ori_path)
    pred_set = set(pred_path)
    intersect = len(ori_set & pred_set)
    num = 0
    # if len(ori_path) == len(pred_path) and intersect == len(ori_path):
    if intersect == len(ori_path):

        num = 1
    return num


class MultiPredRes:
    def __init__(self, pred_p, testSetLength):
        self.successPath = len(pred_p)
        self.total_path = 0

    def resultShow(self):
        s =  f"Average Path Number:\t\t\t{self.total_path / self.successPath:.4f}\n"
        print(s)


def multiVal(pred_path, gt_file, product_class='all'):
    with open(pred_path) as f:
        mol2class = f.readlines()
        
    # mol2class = [l.strip().split('\t') for l in mol2class]
    mol2class = [l.strip().split() for l in mol2class]
    # mol2class = {l[2]: l[1] for l in mol2class}
    mol2class = {l[-1]: l[1] for l in mol2class}

    products = set()
    for product, _class in mol2class.items():
        if product_class == 'all' or _class == product_class:
            products.add(product)
    # testSetNum = len(products)
    testSetNum = 368
    print(f'Number of test molecules:\t{testSetNum}')

    pred_dict = get_pred_dict(pred_path, mol2class, product_class)
    ground_truth_dict = get_ground_truth_dict(gt_file)


    mRes = MultiPredRes(pred_dict, testSetNum)

    for smiles, v_dict in pred_dict.items():
        if v_dict is None:
            continue
        try:
            gt_dict = ground_truth_dict[smiles]
        except:
            continue
        if len(v_dict) > 5: 
            mRes.total_path += 5
        else:
            mRes.total_path += len(v_dict)
    mRes.resultShow()
    # import pdb; pdb.set_trace()

parser = argparse.ArgumentParser()
parser.add_argument('save_file', type=str)
parser.add_argument('gt_file', type=str)
args = parser.parse_args()

multiVal(args.save_file,args.gt_file)
