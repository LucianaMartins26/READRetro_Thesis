from rdkit import Chem

def read_txt(path):
    data = []
    with open(path, 'r') as f:
        for each in f.readlines():
            data.append(each.strip('\n'))
        return data

def remove_chiral(smi,atomMap=False):
    if '>>' not in smi:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi
        if atomMap:
            [a.SetAtomMapNum(0) for a in mol.GetAtoms()]
        ret = Chem.MolToSmiles(mol, isomericSmiles=False)
    else:
        reactants, product = smi.split('>>')[0], smi.split('>>')[-1]
        mol_react, mol_prod = Chem.MolFromSmiles(reactants), Chem.MolFromSmiles(product)
        ret_react, ret_prod = Chem.MolToSmiles(mol_react, isomericSmiles=False), Chem.MolToSmiles(mol_prod, isomericSmiles=False)
        ret = '>>'.join([ret_react, ret_prod])
    return ret


def cal_acc(preds, target, n_best):
    # top 1 3 5 10 acc
    correct_cnt = {'top1': 0, 'top3': 0, 'top5': 0, 'top10': 0}
    for i, tgt_smi in enumerate(target):
        pred_list = preds[i*n_best: i*n_best + n_best]
        if tgt_smi in pred_list[:1]:
            correct_cnt['top1'] += 1
        if tgt_smi in pred_list[:3]:
            correct_cnt['top3'] += 1
        if tgt_smi in pred_list[:5]:
            correct_cnt['top5'] += 1
        if tgt_smi in pred_list[:10]:
            correct_cnt['top10'] += 1
    acc_dict = {key: value / len(target) for key, value in correct_cnt.items()}
    return acc_dict

def evaluate(target, pred_path, AM=False):
    preds = read_txt(pred_path)
    preds = [remove_chiral(each.replace(' ', ''),atomMap = AM) for each in preds]
    acc_dict = cal_acc(preds, target, 10)
    return acc_dict
    
