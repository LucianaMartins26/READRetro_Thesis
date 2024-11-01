import math
import argparse
from tqdm import tqdm
from rdkit import Chem
from retroformer.translate import prepare_retroformer, run_retroformer
from g2s.translate import prepare_g2s, run_g2s
from utils.ensemble import prepare_ensemble, run_ensemble_singlestep

def remove_chiral(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return smi
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def cal_acc(preds, target, n_best):
    correct_cnt = {'top1': 0, 'top3': 0, 'top5': 0, 'top10': 0}
    for i, tgt_smi in enumerate(target):
        pred_list = preds[i][:n_best]
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

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_type',       type=str, default='retroformer', choices=['ensemble', 'retroformer', 'g2s'])
parser.add_argument('-b', '--batch_size',       type=int, default=32)
parser.add_argument('-s', '--beam_size',        type=int, default=10)
parser.add_argument('-d', '--device',           type=str, default='cuda', choices=['cuda', 'cpu'])
parser.add_argument('-a', '--all',              type=str, default='False')

args = parser.parse_args()

model_paths = [
    'retroformer/saved_models/biochem.pt',
    'retroformer/saved_models/clean.pt',
    'retroformer/saved_models/transfer_learning_700_10times_lowerLR.pt',
    'retroformer/saved_models/transfer_learning_1000_10times_lowerLR.pt',
    'retroformer/saved_models/transfer_learning_700_100times_lowerLR.pt',
    'retroformer/saved_models/transfer_learning_700_1000times_lowerLR.pt',
    'retroformer/saved_models/transfer_learning_1000_100times_lowerLR.pt',
    'retroformer/saved_models/transfer_learning_1000_1000times_lowerLR.pt',
    'retroformer/saved_models/transfer_learning_700_1000x_clean.pt'
]

g2s_path = 'g2s/saved_models/biochem.pt'
retroformer_vocab_path = 'retroformer/saved_models/vocab_share.pk'
g2s_vocab_path = 'g2s/saved_models/vocab_smiles.txt'

for retroformer_path in model_paths:
    print(f"Running model: {retroformer_path}")

    if args.model_type == 'ensemble':
        model_retroformer, model_g2s, args_retroformer, args_g2s, vocab, vocab_tokens, device = \
            prepare_ensemble(args.device == 'cuda', args.beam_size,
                             retroformer_path=retroformer_path,
                             retroformer_vocab_path=retroformer_vocab_path,
                             g2s_path=g2s_path,
                             g2s_vocab_path=g2s_vocab_path)
                             
    elif args.model_type == 'retroformer':
        model_retroformer, args_retroformer = prepare_retroformer(args.device == 'cuda', args.beam_size,
                                                                  path=retroformer_path,
                                                                  vocab_path=retroformer_vocab_path)
    else:
        model_g2s, args_g2s, vocab, vocab_tokens, device = prepare_g2s(args.device == 'cuda', args.beam_size,
                                                                       path=g2s_path,
                                                                       vocab_path=g2s_vocab_path)

    if args.all == 'True':
        with open('data/thesis_dataset_compounds/all_reactionSMILES_AM.txt', 'r') as f:
            reactions = f.readlines()
        tgt_AM = [r.strip().split('>>')[0] for r in reactions]
        src_AM = [r.strip().split('>>')[1] for r in reactions]
        tgt_AM = [remove_chiral(t) for t in tgt_AM]

        preds = []
        if args.model_type == 'ensemble':
            for i in tqdm(range(math.ceil(len(src_AM) / args.batch_size))):
                smi_list = src_AM[i * args.batch_size : (i + 1) * args.batch_size]
                preds.extend(run_ensemble_singlestep(smi_list, model_retroformer, model_g2s,
                                                     args_retroformer, args_g2s,
                                                     vocab, vocab_tokens, device))
        elif args.model_type == 'retroformer':
            for i in tqdm(range(math.ceil(len(src_AM) / args.batch_size))):
                smi_list = src_AM[i * args.batch_size : (i + 1) * args.batch_size]
                preds.extend(run_retroformer(model_retroformer, args_retroformer, smi_list)[0])
        else:
            for smi in tqdm(src_AM):
                preds.append(run_g2s(model_g2s, args_g2s, smi, vocab, vocab_tokens, device)['reactants'])

        for pred in preds:
            for i in range(len(pred)):
                pred[i] = remove_chiral(pred[i])

        with open('data/thesis_dataset_compounds/all_reactionSMILES.txt', 'r') as f:
            reactions = f.readlines()
        tgt = [r.strip().split('>>')[0] for r in reactions]
        src = [r.strip().split('>>')[1] for r in reactions]
        tgt = [remove_chiral(t) for t in tgt]

        result = cal_acc(preds, tgt, n_best=10)
        print(result)

        model = retroformer_path.split('/')[-1].split('.')[0]

        with open('data/retroformer_{}_results/all_reactionSMILES_pred.txt'.format(model), 'w') as f:
            for key, value in result.items():
                f.write(f'{key}: {value}\n')

    else:
        compounds = ['avenacinA1', 'momilactoneB', 'afrormosin', 'lycosantalonol', 'vincamine', 'brucine', 'diaboline', 'falcarindiol']

        for compound in compounds:

            with open('data/thesis_dataset_compounds/{}_reactionSMILES_AM.txt'.format(compound), 'r') as f:
                reactions = f.readlines()
            tgt_AM = [r.strip().split('>>')[0] for r in reactions]
            src_AM = [r.strip().split('>>')[1] for r in reactions]
            tgt_AM = [remove_chiral(t) for t in tgt_AM]
            
            preds = []
            if args.model_type == 'ensemble':
                for i in tqdm(range(math.ceil(len(src_AM) / args.batch_size))):
                    smi_list = src_AM[i * args.batch_size : (i + 1) * args.batch_size]
                    preds.extend(run_ensemble_singlestep(smi_list, model_retroformer, model_g2s,
                                                         args_retroformer, args_g2s,
                                                         vocab, vocab_tokens, device))
            elif args.model_type == 'retroformer':
                for i in tqdm(range(math.ceil(len(src_AM) / args.batch_size))):
                    smi_list = src_AM[i * args.batch_size : (i + 1) * args.batch_size]
                    preds.extend(run_retroformer(model_retroformer, args_retroformer, smi_list)[0])
            else:
                for smi in tqdm(src_AM):
                    preds.append(run_g2s(model_g2s, args_g2s, smi, vocab, vocab_tokens, device)['reactants'])

            for pred in preds:
                for i in range(len(pred)):
                    pred[i] = remove_chiral(pred[i])

            with open('data/thesis_dataset_compounds/{}_reactionSMILES.txt'.format(compound), 'r') as f:
                reactions = f.readlines()
            tgt = [r.strip().split('>>')[0] for r in reactions]
            src = [r.strip().split('>>')[1] for r in reactions]
            tgt = [remove_chiral(t) for t in tgt]

            result = cal_acc(preds, tgt, n_best=10)
            print(result)

            model = retroformer_path.split('/')[-1].split('.')[0]

            with open('data/retroformer_{}_results/{}_reactionSMILES_pred.txt'.format(model, compound), 'w') as f:
                for key, value in result.items():
                    f.write(f'{key}: {value}\n')