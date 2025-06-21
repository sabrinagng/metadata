import os, json, pickle, random, argparse, time, numpy as np
import pandas as pd
import torch, optuna
from torch.utils.data import DataLoader, Dataset, get_worker_info
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from masked_sequence import MaskedEMGMAE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4  # Set to 0 if you encounter subprocess issues
WINDOW_LEN = 400
STEP = 100

PICKLE_PATH = "/Users/km82/Documents/metadata/data/noFE_windowed_segraw_allEMG.pkl"


class EMGWindowDataset(Dataset):
    """返回 (C,T) float32 tensor；每个样本已按通道 z-score 归一化"""
    def __init__(self, df_part):
        feats = np.stack(df_part["feature"].values)   # (N, ?, ?)
        # 转成 (N, C, T) 且 C=16
        if feats.shape[-1] != feats.shape[-2]:
            if feats.shape[1] == 16:
                feats = feats.transpose(0, 1, 2)
            else:
                feats = feats.transpose(0, 2, 1)
        self.x = torch.tensor(feats, dtype=torch.float32)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        x = self.x[idx]
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        return x
    

def make_loaders(batch_size, df, VAL_USERS, val_split=0.2):
    """把 16 位验证用户内部再按 windows 8:2 划分 train/val"""
    df_val_all = df[df.Participant.isin(VAL_USERS)].reset_index(drop=True)

    # 打乱索引再切分
    idx = np.random.permutation(len(df_val_all))
    np.random.shuffle(idx)
    cut = max(1, int(len(idx) * (1 - val_split)))
    idx_train, idx_val = idx[:cut], idx[cut:]

    ds_train = EMGWindowDataset(df_val_all.iloc[idx[:cut]])
    ds_val   = EMGWindowDataset(df_val_all.iloc[idx[cut:]])

    dl_train = DataLoader(ds_train, batch_size=batch_size,
                          shuffle=True,  drop_last=True,  num_workers=NUM_WORKERS)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size,
                          shuffle=False, drop_last=False, num_workers=NUM_WORKERS)
    in_ch = ds_train[0].shape[0]      # 16
    return dl_train, dl_val, in_ch


def objective(trial, df, VAL_USERS):
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    mask_ratio = trial.suggest_float("mask_ratio", 0.1, 0.5, step=0.1)
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
    epochs = trial.suggest_int("epochs", 5, 30)
    opt_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])

    dl_train, dl_val, in_ch = make_loaders(batch_size, df, VAL_USERS)
    model = MaskedEMGMAE(in_ch, hidden_dim=hidden_dim, mask_ratio=mask_ratio).to(DEVICE)
    if opt_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_val, patience, wait = np.inf, 3, 0

    for epoch in range(epochs):
        model.train()
        for x in dl_train:
            x = x.to(DEVICE, non_blocking=True)
            rec, xo, mask = model(x)
            loss = model.compute_loss(rec, xo, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in dl_val:
                x = x.to(DEVICE, non_blocking=True)
                rec, xo, mask = model(x)
                val_loss += model.compute_loss(rec, xo, mask).item()

        val_loss /= len(dl_val)
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
        if trial.should_prune():
            raise optuna.TrialPruned(f"Validation loss {val_loss:.4f} exceeded pruning threshold")
        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered")
                break
    return best_val


def main(n_trials):
    df = pickle.load(open(PICKLE_PATH, 'rb'))

    if "windowed_ts_data" in df.columns:
        # Adjust channels from 64 to 12 by taking the first 12
        print("Original data shape (first sample):", df['windowed_ts_data'].iloc[0].shape)
        df['windowed_ts_data'] = df['windowed_ts_data'].apply(lambda x: x[:12, :])
        print("Adjusted data shape (first sample):", df['windowed_ts_data'].iloc[0].shape)
        df = df.rename(columns={"windowed_ts_data": "feature"})

    all_users = df["Participant"].unique().tolist()
    assert len(all_users) == 32, "期望 32 位用户，请检查数据"

    random.seed(42)                        # 固定种子，可复现
    VAL_USERS  = set(random.sample(all_users, 16))

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_warmup_steps=5)
    )
    t0 = time.time()
    study.optimize(lambda trial: objective(trial, df, VAL_USERS), n_trials=n_trials) 
    print(f"\n Total time: {(time.time()-t0)/60:.1f} min")
    print("Best val-loss:", study.best_value)
    print("Best hyperparameters:", study.best_params)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Optuna hyperparameter optimization for MaskedEMGMAE")
    ap.add_argument("--n_trials", type=int, default=30,
                    help="Number of trials for hyperparameter optimization")
    args = ap.parse_args()
    main(args.n_trials)