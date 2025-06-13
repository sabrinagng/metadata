import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from glob import glob


DATA_FOLDERS = [
    '/Users/km82/Documents/train/Ninapro/DB3_emg_only'
]

WINDOW_LEN = 200
STEP = 100
MASK_RATIO = 0.4
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset
class NinaproMaskedDataset(Dataset):
    def __init__(self, folder_list, window_len, step):
        self.windows = []
        self.window_len = window_len

        print("Indexing all windows...")
        for folder in folder_list:
            for file_path in glob(os.path.join(folder, '*.npy')):
                emg = np.load(file_path, mmap_mode='r')
                T, C = emg.shape
                for start in range(0, T - window_len + 1, step):
                    self.windows.append((file_path, start))
                print(f"  {os.path.basename(file_path)}: {T} samples, {len(range(0, T - window_len + 1, step))} windows")

        print(f"Total windows: {len(self.windows)}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        file_path, start = self.windows[idx]
        emg = np.load(file_path, mmap_mode='r')
        seg = emg[start:start + self.window_len]  # shape (T, C)

        # Normalize
        seg = (seg - seg.mean(axis=0)) / (seg.std(axis=0) + 1e-6)
        return torch.from_numpy(seg.T.astype(np.float32))  # (C, T)

# model
class MaskedEMGMAE(nn.Module):
    def __init__(self, in_channels=12, hidden_dim=128, mask_ratio=0.4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Conv1d(64, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(True),
            nn.Conv1d(64, in_channels, kernel_size=5, padding=2)
        )

    def random_mask(self, x):
        B, C, T = x.shape
        mask = torch.zeros((B, 1, T), dtype=torch.bool, device=x.device)
        num_mask = int(T * self.mask_ratio)

        for i in range(B):
            # mask `num_mask` random indices
            indices = torch.randperm(T)[:num_mask]
            mask[i, 0, indices] = True

        x_masked = x.clone()
        x_masked[mask.expand_as(x)] = 0
        return x_masked, mask

    def forward(self, x):
        x_masked, mask = self.random_mask(x)
        feat = self.encoder(x_masked)
        recon = self.decoder(feat)
        return recon, x, mask

    def compute_loss(self, recon, x, mask):
        masked_elements = mask.sum().item()
        if masked_elements == 0:
            return torch.tensor(0.0, device=x.device)
        return F.mse_loss(recon[mask.expand_as(recon)], x[mask.expand_as(x)])

# train
def train():
    dataset = NinaproMaskedDataset(DATA_FOLDERS, WINDOW_LEN, STEP)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)

    model = MaskedEMGMAE(in_channels=12, hidden_dim=128, mask_ratio=MASK_RATIO).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for i, x in enumerate(dataloader):
            x = x.to(DEVICE)
            recon, x_orig, mask = model(x)
            loss = model.compute_loss(recon, x_orig, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 50 == 0:
                print(f"    [Batch {i}] Loss: {loss.item():.6f} | Recon mean: {recon.mean().item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch}/{EPOCHS} | Avg Loss: {avg_loss:.6f}")

    torch.save(model.encoder.state_dict(), 'masked_emg_encoder.pth')
    print("Saved encoder weights to masked_emg_encoder.pth")


if __name__ == '__main__':
    train()

