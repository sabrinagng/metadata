#!/usr/bin/env python3
# coding: utf-8

print(">>> test_cnn_ae.py start", flush=True)

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# import exactly the classes you used in training
from train import NinaproMaskedDataset, MaskedEMGMAE

# ---------- Config ----------
ENCODER_PATH  = 'encoder.pth'
DATA_FOLDERS  = ['Ninapro/DB3_emg_only']
WINDOW_LEN    = 200
STEP          = 100
BATCH_SIZE    = 1024
NUM_PLOTS     = 3
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS   = 0    # avoid multiprocessing issues
HIDDEN_DIM    = 64   # same as training
MASK_RATIO    = 0.2  # same as training
SAVE_ORIG     = 'all_orig.npy'
SAVE_REC      = 'all_rec.npy'

def test_and_visualize():
    print(">>> Loading dataset...", flush=True)
    ds = NinaproMaskedDataset(DATA_FOLDERS, WINDOW_LEN, STEP)
    dl = DataLoader(ds,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True)

    # infer number of channels from first sample
    sample = ds[0]            # Tensor of shape (C, T)
    in_ch = sample.shape[0]
    print(f">>> in_ch = {in_ch}", flush=True)

    # build model and load encoder weights
    model = MaskedEMGMAE(in_ch, hidden_dim=HIDDEN_DIM, mask_ratio=MASK_RATIO).to(DEVICE)
    encoder_ckpt = torch.load(ENCODER_PATH, map_location=DEVICE)
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}")
        print(f"Parameter shape: {param.shape}")
    # sanity check the encode_ckpt keys
    print(">>> Encoder checkpoint keys:", encoder_ckpt.keys(), flush=True)
    model.encoder.load_state_dict(encoder_ckpt, strict=False)
    model.eval()
    print(">>> Encoder loaded", flush=True)

    # compute masked-position MSE
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x in dl:
            x = x.to(DEVICE, non_blocking=True)
            rec, orig, mask = model(x)
            loss = model.compute_loss(rec, orig, mask)
            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    print(f"Test avg masked-MSE: {total_loss/n:.6f}", flush=True)

    # reconstruct every window and save
    all_orig = []
    all_rec  = []
    with torch.no_grad():
        for x in DataLoader(ds,
                             batch_size=1,
                             shuffle=False,
                             num_workers=NUM_WORKERS):
            x = x.to(DEVICE).unsqueeze(0)  # shape (1, C, T)
            rec, orig, mask = model(x)
            all_orig.append(orig.cpu().squeeze(0).numpy())
            all_rec.append(rec.cpu().squeeze(0).numpy())

    all_orig = np.stack(all_orig, axis=0)  # (N, C, T)
    all_rec  = np.stack(all_rec, axis=0)
    np.save(SAVE_ORIG, all_orig)
    np.save(SAVE_REC,  all_rec)
    print(f"Saved {SAVE_ORIG}, {SAVE_REC}, shape {all_orig.shape}", flush=True)

    # visualize some random windows
    # for idx in random.sample(range(len(all_orig)), NUM_PLOTS):
    #     orig = all_orig[idx]
    #     rec  = all_rec[idx]
    #     C, T = orig.shape

    #     fig, axes = plt.subplots(3, 4, figsize=(12, 8),
    #                              sharex=True, sharey=True)
    #     fig.suptitle(f"Sample #{idx}", fontsize=16)
    #     vmin = min(orig.min(), rec.min())
    #     vmax = max(orig.max(), rec.max())
    #     margin = 0.02 * (vmax - vmin)

    #     for ch in range(C):
    #         ax = axes[ch//4, ch%4]
    #         ax.plot(orig[ch], linewidth=0.7, label='orig')
    #         ax.plot(rec[ch],  linewidth=0.7, label='rec')
    #         ax.set_title(f"Ch {ch}", fontsize=8)
    #         ax.set_ylim(vmin - margin, vmax + margin)
    #         ax.axis('off')

    #     plt.tight_layout(rect=[0,0.03,1,0.95])
    #     plt.show()

if __name__ == '__main__':
    print(">>> Entered main guard", flush=True)
    test_and_visualize()
