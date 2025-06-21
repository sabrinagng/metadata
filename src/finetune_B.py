import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import os
import argparse
import optuna

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PICKLE_PATH = os.path.join("..", "data", "noFE_windowed_segraw_allEMG.pkl")
SAVE_FT_PATH = os.path.join("..", "models", "best_hpo_model.pth")

# --- User Split (Fixed) ---
VAL_USERS = ['P005', 'P008', 'P010', 'P011', 'P102', 'P103', 'P104', 'P105', 'P106', 'P109', 'P110', 'P116', 'P122', 'P123', 'P126', 'P128']
TEST_USERS = ['P004', 'P006', 'P107', 'P108', 'P111', 'P112', 'P114', 'P115', 'P118', 'P119', 'P121', 'P124', 'P125', 'P127', 'P131', 'P132']

# --- Dataset Definition ---
class BGestureDataset(Dataset):
    def __init__(self, df_sub):
        self.feats = np.stack(df_sub["feature"].values)
        self.labels = df_sub["Gesture_Encoded"].to_numpy()

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        x = torch.tensor(self.feats[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        # Instance-wise normalization
        x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
        return x, y

# --- Dynamic Model Definition ---
class DynamicConvModel(nn.Module):
    def __init__(self, n_classes, config):
        super().__init__()
        layers = []
        in_channels = 16  # Input channels are fixed

        for i in range(config["num_layers"]):
            out_channels = config["hidden_dim"]
            kernel = config["kernel_size"]
            stride = config["stride"]
            padding = (kernel - 1) // 2  # 'same' padding

            layers.append(nn.Conv1d(in_channels, out_channels, kernel, stride, padding))
            if config["use_batchnorm"]:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(True))
            
            in_channels = out_channels # For the next layer

        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(config["head_dropout"]),
            nn.Linear(in_channels, n_classes)
        )

    def forward(self, x):
        if x.shape[1] != 16:
             x = x.transpose(1, 2)
        feat = self.encoder(x)
        logits = self.head(feat)
        return logits

# --- Helper to build optimizer and scheduler ---
def build_optimizer_scheduler(model, config):
    params = model.parameters()
    optimizer_name = config["optimizer"]
    
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=config["lr"], momentum=config["momentum"], 
            weight_decay=config["weight_decay"], nesterov=config["nesterov"]
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=config["lr"], weight_decay=config["weight_decay"])
    else: # Adam
        optimizer = torch.optim.Adam(params, lr=config["lr"], weight_decay=config["weight_decay"])

    scheduler_type = config["scheduler"]
    scheduler = None
    if scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
    elif scheduler_type == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=config["min_lr"]) # T_max is epochs for HPO

    return optimizer, scheduler

# --- Optuna Objective Function ---
def objective(trial, df_val_users, n_classes):
    # Define search space and store in a config dict
    config = {
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "scheduler": trial.suggest_categorical("scheduler", ["None", "StepLR", "CosineAnnealingLR"]),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "num_layers": trial.suggest_int("num_layers", 2, 4),
        "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
        "stride": trial.suggest_categorical("stride", [1, 2]),
        "use_batchnorm": trial.suggest_categorical("use_batchnorm", [True, False]),
        "head_dropout": trial.suggest_float("head_dropout", 0.1, 0.6)
    }
    # Conditional Hyperparameters
    if config["optimizer"] == "SGD":
        config["momentum"] = trial.suggest_float("momentum", 0.8, 0.99)
        config["nesterov"] = trial.suggest_categorical("nesterov", [True, False])
    if config["scheduler"] == "StepLR":
        config["step_size"] = trial.suggest_int("step_size", 2, 5)
        config["gamma"] = trial.suggest_float("gamma", 0.1, 0.9)
    if config["scheduler"] == "CosineAnnealingLR":
        config["min_lr"] = trial.suggest_float("min_lr", 1e-6, 1e-4, log=True)

    # DataLoaders
    df_train, df_val = train_test_split(df_val_users, test_size=0.2, random_state=42, stratify=df_val_users['Gesture_Encoded'])
    ds_train = BGestureDataset(df_train)
    ds_val = BGestureDataset(df_val)
    dl_train = DataLoader(ds_train, batch_size=config["batch_size"], shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=config["batch_size"], shuffle=False, num_workers=2)

    # Model, Optimizer, Scheduler
    model = DynamicConvModel(n_classes=n_classes, config=config).to(DEVICE)
    optimizer, scheduler = build_optimizer_scheduler(model, config)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop for HPO
    best_acc = 0
    for epoch in range(10): # Fixed 10 epochs for HPO
        model.train()
        for x, y in dl_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        if scheduler:
            scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        acc = correct / total
        if acc > best_acc:
            best_acc = acc
        
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_acc

# --- Final Training and Evaluation ---
def final_train_and_eval(best_params, df_val_users, df_test_users, n_classes):
    print("\n--- Starting Final Training with Best Hyperparameters ---")
    print(f"Best Parameters: {best_params}")

    # DataLoaders
    ds_train = BGestureDataset(df_val_users)
    ds_test = BGestureDataset(df_test_users)
    dl_train = DataLoader(ds_train, batch_size=best_params['batch_size'], shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=best_params['batch_size'], shuffle=False)

    # Model, Optimizer, Scheduler
    model = DynamicConvModel(n_classes=n_classes, config=best_params).to(DEVICE)
    optimizer, scheduler = build_optimizer_scheduler(model, best_params)
    criterion = nn.CrossEntropyLoss()

    # Final training loop
    for epoch in range(25): # Train for more epochs
        model.train()
        for x, y in dl_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()
        print(f"Final Training Epoch {epoch+1}/25, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), SAVE_FT_PATH)
    print(f"Final model saved to {SAVE_FT_PATH}")

    # Final evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dl_test:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    final_acc = 100 * correct / total
    print(f"\n--- Final Test Accuracy: {final_acc:.2f}% ---")

# --- Main Execution ---
def main(n_trials):
    print("Loading and preparing data...")
    df = pd.read_pickle(PICKLE_PATH)
    df = df.rename(columns={"windowed_ts_data": "feature"})
    le = LabelEncoder()
    df["Gesture_Encoded"] = le.fit_transform(df["Gesture"])
    n_classes = len(le.classes_)
    
    df_val_users = df[df.Participant.isin(VAL_USERS)].reset_index(drop=True)
    df_test_users = df[df.Participant.isin(TEST_USERS)].reset_index(drop=True)
    print("Data loaded.")

    print("\n--- Starting Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, df_val_users, n_classes), n_trials=n_trials)

    print("\n--- Hyperparameter Optimization Finished ---")
    print(f"Best trial accuracy: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_trial.params}")

    final_train_and_eval(study.best_trial.params, df_val_users, df_test_users, n_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Fine-tuning HPO script")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    args = parser.parse_args()
    main(args.n_trials)

