# built-in libraries
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import random
import json
from copy import deepcopy
# third-party libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
# custom libraries
from .model import EndTaskClassifier


DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PICKLE_PATH    = "autodl-tmp/data/noFE_windowed_segraw_allEMG.pkl"
GENERIC_CKPT   = "emg_finetune/checkpoints/best_finetuned_model.pth"        # 你的 generic 模型
K_SHOT         = 1
MAX_QUERY      = 64
FULL_LR        = 1e-5
SUP_EPOCHS     = 3                 # 每次重采样只训 3 epoch
BATCH_SIZE     = 32
N_REPEAT       = 5                 # 5 次重采样 + 平均 logits
ENSEMBLE_TEMP  = 1.0               # softmax 温度，可微调
SEED           = 42
FULL_FINETUNE  = False             # True: 追加 encoder 小 LR 微调
USER_FT_CKPT_DIR = "user_ft_ckpts"
TRAINING_PARAMS_PATH = "emg_finetune/best_hpo_params.json"
TEST_USERS     = ['P004','P006','P107','P108','P111','P112','P114','P115']

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

df_all = (pd.read_pickle(PICKLE_PATH)
            .rename(columns={'windowed_ts_data':'feature'}))
df_all['Gesture_ID'] = pd.factorize(df_all['Gesture_ID'])[0]
N_CLASSES = df_all['Gesture_ID'].nunique()


class FewShotDataset(Dataset):
    def __init__(self, df_user: pd.DataFrame, lbl_col: str, mode: str):
        self.x, self.y = [], []
        for g_id, g_df in df_user.groupby(lbl_col):
            idx = g_df.sample(frac=1, random_state=np.random.randint(10**6)).index
            sup_idx = idx[:K_SHOT]
            qry_idx = idx[K_SHOT:][:MAX_QUERY] if MAX_QUERY else idx[K_SHOT:]
            chosen  = sup_idx if mode=='support' else qry_idx
            self.x.extend(df_user.loc[chosen, 'feature'])
            self.y.extend([g_id]*len(chosen))
        self.x = np.stack(self.x).astype(np.float32)
        self.y = np.asarray(self.y, dtype=np.int64)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i):
        xi = torch.tensor(self.x[i])
        xi = (xi - xi.mean(0,keepdim=True))/(xi.std(0,keepdim=True)+1e-6)
        return xi, torch.tensor(self.y[i])

with open(TRAINING_PARAMS_PATH, 'r') as f:
    config = json.load(f)


# Initialize the model
# Set in_ch to 64 to match the data, and ckpt_path to None to disable internal loading.
BASE_MODEL = EndTaskClassifier(in_ch=16, num_classes=N_CLASSES, config=config, ckpt_path=GENERIC_CKPT)
model = deepcopy(BASE_MODEL)

#one-shot
chosen_users = random.sample(TEST_USERS, 8)
print("One-shot fine-tune users:", chosen_users)

user_acc = {}
for uid in chosen_users:
    df_u  = df_all[df_all.Participant==uid].copy()
    labels_present = sorted(df_u.Gesture_ID.unique())
    lbl_map = {g:i for i,g in enumerate(labels_present)}
    df_u['cid'] = df_u.Gesture_ID.map(lbl_map)

    n_cls   = len(labels_present)
    ds_qry  = FewShotDataset(df_u,'cid','query')
    dl_qry  = DataLoader(ds_qry, batch_size=BATCH_SIZE, shuffle=False)

    # -------- logits ensemble ----------
    all_logits = []

    for rep in range(N_REPEAT):
        #重采样 support-set
        ds_sup = FewShotDataset(df_u,'cid','support')
        dl_sup = DataLoader(ds_sup, batch_size=BATCH_SIZE, shuffle=True)
        model = deepcopy(BASE_MODEL)

        # 3) head
        # Optimizer
        optimizer_str = config['optimizer']
        
        if optimizer_str == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=config['lr'],
                momentum=config.get('momentum', 0.9),
                weight_decay=config['weight_decay'],
                nesterov=config.get('nesterov', False)
            )
        elif optimizer_str == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay'])
        else:  # Adam
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config['lr'],
                weight_decay=config['weight_decay']
            )    
        
        # Scheduler
        scheduler_str = config['scheduler']
        if scheduler_str == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=config['step_size'],
                gamma=config['gamma']
            )
        elif scheduler_str == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=10,  # T_max is epochs for HPO
                eta_min=config['min_lr']
            )
        else:
            scheduler = None
            
        # Criterion
        criterion = torch.nn.CrossEntropyLoss()
        model.train()
        for _ in range(SUP_EPOCHS):
            for xb,yb in dl_sup:
                xb,yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                
            if scheduler:
                scheduler.step()

        #query logits
        model.eval()
        with torch.no_grad():
            logits_rep = []
            for xb, yb in dl_qry:
                xb = xb.to(DEVICE)
                logits_rep.append(model(xb))
            all_logits.append(torch.cat(logits_rep, dim=0))

    ensembled_logits = torch.stack(all_logits).mean(dim=0)
    pred_y = ensembled_logits.argmax(dim=1).cpu().numpy()
    true_y = ds_qry.y
    acc = (pred_y == true_y).mean()
    user_acc[uid] = acc
    print(f"  -> User {uid} | Acc: {acc:.3f}")


    save_path = os.path.join(USER_FT_CKPT_DIR, f"{uid}_k{K_SHOT}_finetuned.pth")
    torch.save(model.state_dict(), save_path)

print(f"\nAverage acc over {len(chosen_users)} users: {np.mean(list(user_acc.values())):.3f}")
