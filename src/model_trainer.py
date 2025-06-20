#!/usr/bin/env python3
# coding: utf-8

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob
from tqdm import tqdm # Add this import


import torch, inspect
def debug_cuda(msg):
    print("[DEBUG]", msg, "| cuda?", torch.cuda.is_available(),
          "| pid", os.getpid())
debug_cuda("script start")

# ---------------- Config ----------------
DATA_FOLDERS  = ['data/Ninapro/DB2_emg_only/split set']
TEST_FOLDERS  = ['data/Ninapro/DB3_emg_only/s1_0_emg.npy']
WINDOW_LEN    = 400
STEP          = 100
MASK_RATIO    = 0.2
BATCH_SIZE    = 2048
EPOCHS        = 10
LR            = 1e-3
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS   = 4  # 如果遇到子进程问题，可改为 0

# ---------------- Dataset ----------------
class NinaproMaskedDataset(Dataset):
    def __init__(self, folders, window_len, step):
        self.windows = []
        self.window_len = window_len
        for folder in folders:
            files = glob(os.path.join(folder, '*.npy'))
            print(f"Found {len(files)} .npy files in {folder}", flush=True)
            for path in files:
                emg = np.load(path, mmap_mode='r')
                T, _ = emg.shape
                cnt = 0
                for st in range(0, T - window_len + 1, step):
                    self.windows.append((path, st))
                    cnt += 1
                print(f"  {os.path.basename(path)} → {cnt} windows", flush=True)
        print(f"Total windows: {len(self.windows)}\n", flush=True)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        path, st = self.windows[idx]
        emg = np.load(path)                           # (T, C)
        seg = emg[st:st + self.window_len]             # (window_len, C)
        # Normalize per-channel
        seg = (seg - seg.mean(0)) / (seg.std(0) + 1e-6)
        return torch.from_numpy(seg.T.astype(np.float32))  # (C, T)

# ---------------- Model ----------------
class MaskedEMGMAE(nn.Module):
    def __init__(self, in_ch, hidden_dim=128, mask_ratio=0.2):
        super().__init__()
        self.mask_ratio = mask_ratio
        # Encoder: 3-layer CNN
        self.encoder = nn.Sequential(
            nn.Conv1d(in_ch, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),  nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_dim), nn.ReLU(True),
        )
        # Decoder: mirror structure
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(True),
            nn.Conv1d(128, 64, kernel_size=5, padding=2),  nn.BatchNorm1d(64),  nn.ReLU(True),
            nn.Conv1d(64, in_ch, kernel_size=5, padding=2),
        )

    def random_mask(self, x):
        B, C, T = x.shape
        num = int(T * self.mask_ratio)
        mask = torch.zeros((B, 1, T), dtype=torch.bool, device=x.device)
    
        # 一次性生成随机索引（CUDA 上）
        idx = torch.randint(0, T, (B, num), device=x.device)   # [B, num]
        mask.scatter_(2, idx.unsqueeze(1), True)               # 写 True 到掩码
        xm = x.masked_fill(mask.expand_as(x), 0.)
        return xm, mask

    def forward(self, x):
        xm, mask = self.random_mask(x)
        feat = self.encoder(xm)
        rec  = self.decoder(feat)
        return rec, x, mask

    def compute_loss(self, rec, x, mask):
        loss_mask   = F.mse_loss(rec[mask.expand_as(rec)],    x[mask.expand_as(x)])
        inv_mask    = ~mask.expand_as(rec)
        loss_unmask = F.mse_loss(rec[inv_mask], x[inv_mask])
        return loss_mask + 0.1 * loss_unmask

def train():
    ds = NinaproMaskedDataset(DATA_FOLDERS, WINDOW_LEN, STEP)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                    drop_last=True, num_workers=NUM_WORKERS, pin_memory=True)

    #模型 & 优化器
    in_ch  = ds[0].shape[0]
    model  = MaskedEMGMAE(in_ch, hidden_dim=128,
                          mask_ratio=MASK_RATIO).to(DEVICE)
    optim  = torch.optim.Adam(model.parameters(), lr=LR)

    total_steps = EPOCHS * len(dl)          # 告诉 tqdm 总迭代数
    step = 0
    with tqdm(total=total_steps,
              desc="Training",
              unit="step") as pbar:         # 这一根条就会显示剩余时间

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for x in dl:
                x = x.to(DEVICE, non_blocking=True)

                rec, x_orig, mask = model(x)
                loss = model.compute_loss(rec, x_orig, mask)

                optim.zero_grad()
                loss.backward()
                optim.step()

                step += 1
                pbar.set_postfix(epoch=epoch,       
                                 loss=f"{loss.item():.4f}",
                                 lr=f"{optim.param_groups[0]['lr']:.1e}")
                pbar.update(1)

                # 只在第 1 个 mini-batch 打印显存，占用可选
                if step == 1:
                    mb = torch.cuda.memory_allocated() / 1024**2
                    tqdm.write(f"[DEBUG] GPU alloc = {mb:.1f} MB")

    torch.save(model.state_dict(), 'models/full_model.pth')
    print("Saved full_model.pth", flush=True)



def test():
    # 1) 数据
    ds_test = NinaproMaskedDataset(TEST_FOLDERS, WINDOW_LEN, STEP)
    dl_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False, 
                         drop_last=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 2) 模型
    in_ch = ds_test[0].shape[0]
    model = MaskedEMGMAE(in_ch, hidden_dim=128, mask_ratio=MASK_RATIO).to(DEVICE)
    
    # 加载 encoder 权重
    ckpt = torch.load('data/Ninapro/full_model.pth', map_location=DEVICE)
    model.load_state_dict(ckpt)
    print("Loaded full encoder+decoder weights", flush=True)

    model.eval()
    total_loss = 0.0
    print("\nStarting testing...", flush=True)
    with torch.no_grad():
        # Wrap DataLoader with tqdm for a progress bar
        loop = tqdm(enumerate(dl_test, 1), total=len(dl_test), leave=True)
        for i, x in loop:
            x = x.to(DEVICE, non_blocking=True)
            rec, x_orig, mask = model(x) # The model's forward pass uses the encoder
            loss = model.compute_loss(rec, x_orig, mask) # Using the same loss for consistency
            total_loss += loss.item()
            # Update tqdm description
            loop.set_description("Testing")
            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dl_test)
    # Use tqdm.write to print messages without interfering with the progress bar
    tqdm.write(f"Testing Complete — Avg Test Loss: {avg_loss:.6f}\n", end='')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train or Test MaskedEMGMAE model.")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'],
                        help='Set to "train" for training, "test" for testing.')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    else:
        print("Invalid mode. Choose 'train' or 'test'.")