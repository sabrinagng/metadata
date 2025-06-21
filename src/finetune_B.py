import os
import random
import pickle
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, get_worker_info
from sklearn.preprocessing import LabelEncoder

# ---------- 全局配置 ----------
PICKLE_PATH   = "/Users/km82/Documents/metadata/noFE_windowed_segraw_allEMG.pkl"   # $B 数据
PRETRAIN_PATH = "/Users/km82/Documents/metadata/models/sequence_masked_full_model.pth"               # Ninapro 预训练 + Optuna 最优
SAVE_FT_PATH  = "/Users/km82/Documents/metadata/models/ft_best.pth"                       # 微调后最优
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MASK_RATIO  = 0.2
BATCH_SIZE  = 256
MAX_EPOCHS  = 13
LR          = 1.2157220601699218e-3
OPTIMIZER   = "adam"          
WEIGHT_DECAY = 1e-4           # 若想完全照 Optuna，可设为 0
PATIENCE    = 5             
DROPOUT_P   = 0.3
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 4         
VAL_SPLIT     = 0.1      

WINDOW_LEN = 400          
IN_CH      = 12
N_CLASSES  = 10           
HIDDEN_DIM = 128         

print("Loading $B pickle ...")
df = pickle.load(open(PICKLE_PATH, "rb"))
if "windowed_ts_data" in df.columns:
    df = df.rename(columns={"windowed_ts_data": "feature"})

if "Gesture_Encoded" not in df.columns:
    df["Gesture_Encoded"] = LabelEncoder().fit_transform(df["Gesture_ID"])
all_users = df["Participant"].unique().tolist()
assert len(all_users) == 32, "不符合 32 位受试者，检查数据！"

random.seed(42)
VAL_USERS  = set(random.sample(all_users, 16))  
TEST_USERS = set(all_users) - VAL_USERS   

print("VAL_USERS :", sorted(VAL_USERS))
print("TEST_USERS:", sorted(TEST_USERS), "\n")


class BGestureDataset(Dataset):
    def __init__(self, df_sub):
        self.x = np.stack(df_sub["feature"].to_numpy())
        self.y = df_sub["Gesture_Encoded"].to_numpy()

    def __len__(self): return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
        return x, self.y[idx]


class FineTuneModel(nn.Module):
    """
    Uses the pre-trained Encoder and adds a new classification head for fine-tuning.
    The encoder architecture must be identical to the one used during pre-training.
    """
    def __init__(self, in_ch, hidden_dim, n_classes, dropout_p=0.3):
        super().__init__()
        # Encoder: Structure must be identical to pre-training
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),  nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_dim), nn.ReLU(True),
        )
        # Classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.head(feat)
        return logits


def make_ft_loaders(batch_size=BATCH_SIZE, val_split=VAL_SPLIT):
    """
    Creates data loaders for fine-tuning with a given batch size and validation split.

    Returns:
        dl_train, dl_val, dl_test: Data loaders for training, validation, and testing.
    """
    df_val_all = df[df.Participant.isin(VAL_USERS)].reset_index(drop=True)
    perm = np.random.permutation(len(df_val_all))
    cut  = int(len(perm)*(1-val_split))
    ds_train = BGestureDataset(df_val_all.iloc[perm[:cut]])
    ds_val   = BGestureDataset(df_val_all.iloc[perm[cut:]])
    dl_train = DataLoader(ds_train, batch_size, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

    df_test  = df[df.Participant.isin(TEST_USERS)]
    dl_test  = DataLoader(BGestureDataset(df_test), batch_size,
                          shuffle=False, num_workers=NUM_WORKERS,
                          pin_memory=True)
    return dl_train, dl_val, dl_test


def main(args):
    dl_train, dl_val, dl_test = make_ft_loaders()

    # 1) Model
    model = FineTuneModel(IN_CH, HIDDEN_DIM, N_CLASSES, DROPOUT_P).to(DEVICE)

    # Load pre-trained Encoder weights
    print(f"Loading pre-trained model from {PRETRAIN_PATH}")
    full_ckpt = torch.load(PRETRAIN_PATH, map_location=DEVICE)
    
    # Extract the state_dict for the encoder
    encoder_ckpt = {k.replace('encoder.', ''): v for k, v in full_ckpt.items() if k.startswith('encoder.')}
    model.encoder.load_state_dict(encoder_ckpt)
    print("Successfully loaded pre-trained encoder weights.")

    # Freeze Encoder
    if not args.unfreeze:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen.")
        optimizer_params = model.head.parameters()
    else:
        print("Encoder parameters are trainable (unfrozen).")
        optimizer_params = model.parameters()

    # 2) Optimizer
    if OPTIMIZER == "adam":
        optimizer = torch.optim.Adam(optimizer_params, lr=LR, weight_decay=WEIGHT_DECAY)
    elif OPTIMIZER == "adamw":
        optimizer = torch.optim.AdamW(optimizer_params, lr=LR, weight_decay=WEIGHT_DECAY)
    else:
        optimizer = torch.optim.SGD(optimizer_params, lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY)
    
    criterion = nn.CrossEntropyLoss()
    best_acc, patience_cnt = 0, 0

    # 3) Training loop
    for epoch in range(MAX_EPOCHS):
        model.train(); train_loss = 0
        for x,y in dl_train:
            x,y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*y.size(0)
        train_loss /= len(dl_train.dataset)

        # ---- val ----
        model.eval(); val_loss, n_corr, n = 0,0,0
        with torch.no_grad():
            for x,y in dl_val:
                x,y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                val_loss += criterion(logits, y).item()*y.size(0)
                n_corr  += (logits.argmax(1)==y).sum().item()
                n += y.size(0)
        val_loss /= n; val_acc = n_corr/n

        print(f"[{epoch+1:02d}] train {train_loss:.4f} | "
              f"val {val_loss:.4f} acc {val_acc:.2%}")

        # ---- early-stop ----
        if val_acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_FT_PATH)
            patience_cnt = 0
        else:
            patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 4) Test on the best model
    model.load_state_dict(torch.load(SAVE_FT_PATH))
    model.eval(); n_corr, n = 0,0
    with torch.no_grad():
        for x,y in dl_test:
            x,y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            n_corr += (logits.argmax(1)==y).sum().item()
            n += y.size(0)
    print(f"\n★ FINAL TEST ACCURACY on 16 TEST_USERS: {n_corr/n:.2%}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fine-tuning script for EMG classification")
    ap.add_argument("--unfreeze", action="store_true", help="Unfreeze encoder layers for full fine-tuning")
    args = ap.parse_args()
    main(args)
