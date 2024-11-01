import sys
wd = '/path/to/READRetro/scripts/singlestep_eval/mhnreact/'
mhn_react = '/path/to/mhn-react'
sys.path.append(mhn_react)

from mhnreact.inspect import load_clf, list_models
from mhnreact.data import load_dataset_from_csv
from mhnreact.utils import top_k_accuracy


clf = load_clf(list_models()[2],model_path=f"{wd}", model_type='mhn', device='cpu',json_file="test5_1676271921_config.json")
X,y,_,_ = load_dataset_from_csv(f"{wd}/biochem_prepro_recre.csv")
preds = clf.forward_smiles(X['test'])

ks = [1,3,5,10]
a = top_k_accuracy(y_true=y['test'], y_pred=preds, k=ks)
for idx,i in enumerate(ks):
    print(f"top-{i}: {round(float(a[idx]),3)}")
