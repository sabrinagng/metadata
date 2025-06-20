import os
import glob
import zipfile
import scipy.io
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#feature extraction
def extract_windows(emg: np.ndarray, labels: np.ndarray, win_len=200, step=100):
    T, C = emg.shape
    feats, labs = [], []
    for start in range(0, T - win_len + 1, step):
        w = emg[start:start+win_len, :]
        rms = np.sqrt((w**2).mean(axis=0))
        mav = np.abs(w).mean(axis=0)
        feats.append(np.concatenate([rms, mav]))
        labs.append(int(labels[start + win_len//2]))
    if not feats:
        return None, None
    return np.stack(feats), np.array(labs, dtype=int)

def build_dataset_from_zips(zip_dir, win_len=200, step=100):
    X_list, y_list, g_list = [], [], []
    zip_paths = sorted(glob.glob(os.path.join(zip_dir, "DB2_s*.zip")))
    for group_id, zippath in enumerate(zip_paths):
        with zipfile.ZipFile(zippath) as z:
            for matf in [f for f in z.namelist() if f.lower().endswith(".mat")]:
                m = scipy.io.loadmat(z.open(matf))
                emg = m['emg']
                #emg is (T, 12)
                if emg.ndim == 2 and emg.shape[1] > emg.shape[0]:
                    emg = emg.T
                labels = m['stimulus'].ravel()
                Xw, yw = extract_windows(emg, labels, win_len, step)
                if Xw is None:
                    print(f"[WARN] {os.path.basename(zippath)}->{matf}: "
                          f"T={emg.shape[0]} < win_len, skipped")
                    continue
                n = Xw.shape[0]
                X_list.append(Xw)
                y_list.append(yw)
                g_list.append(np.full(n, group_id, dtype=int))
                print(f"Loaded {n} windows from {matf} (group {group_id})")
    X      = np.vstack(X_list)
    y      = np.concatenate(y_list)
    groups = np.concatenate(g_list)
    return X, y, groups

#model
class EMGDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = correct = total = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
        pred = out.argmax(dim=1)
        correct   += (pred == yb).sum().item()
        total     += Xb.size(0)
    return total_loss/total, correct/total

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            out = model(Xb)
            loss = criterion(out, yb)
            total_loss += loss.item() * Xb.size(0)
            pred = out.argmax(dim=1)
            correct   += (pred == yb).sum().item()
            total     += Xb.size(0)
    return total_loss/total, correct/total

# —— 主流程：GroupKFold + PyTorch MLP + k-NN ——
def main():
    ZIP_ROOT = "/Users/km82/Documents/train/Ninapro"
    WIN_LEN  = 200
    STEP     = 100
    BATCH    = 128
    EPOCHS   = 10
    LR       = 1e-3


    X, y, groups = build_dataset_from_zips(ZIP_ROOT, win_len=WIN_LEN, step=STEP)
    print(f"\nTotal windows: {X.shape[0]}, feature dim: {X.shape[1]}, "
          f"subjects: {len(np.unique(groups))}")
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    num_classes = len(np.unique(y))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gkf = GroupKFold(n_splits=len(np.unique(groups)))
    fold_mlp_accs = []
    fold_knn_accs = []

    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        subj_te = groups[te_idx][0]

        #MLP
        train_ds = EMGDataset(X_tr, y_tr)
        test_ds  = EMGDataset(X_te, y_te)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, shuffle=True)
        test_loader  = torch.utils.data.DataLoader(test_ds, batch_size=BATCH)

        model     = SimpleMLP(X.shape[1], num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)


        for epoch in range(EPOCHS):
            train_epoch(model, train_loader, criterion, optimizer, device)
        _, mlp_acc = eval_epoch(model, test_loader, criterion, device)
        fold_mlp_accs.append(mlp_acc)

        #KNN
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_tr, y_tr)
        knn_acc = knn.score(X_te, y_te)
        fold_knn_accs.append(knn_acc)

        print(f"\nFold {fold}: test subject S{subj_te}")
        print(f"  MLP Test acc: {mlp_acc:.4f}")
        print(f"  kNN Test acc: {knn_acc:.4f}")


    mlp_mean, mlp_std = np.mean(fold_mlp_accs), np.std(fold_mlp_accs)
    knn_mean, knn_std = np.mean(fold_knn_accs), np.std(fold_knn_accs)
    print(f"\nAverage MLP accuracy: {mlp_mean:.4f} ± {mlp_std:.4f}")
    print(f"Average kNN accuracy: {knn_mean:.4f} ± {knn_std:.4f}")


    results = {
        "mlp_accs": fold_mlp_accs,
        "knn_accs": fold_knn_accs
    }
    with open("results.json", "w") as f:
        json.dump(results, f)
    print("Saved fold accuracies to results.json")

if __name__ == "__main__":
    main()
