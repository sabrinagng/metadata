import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import random
import os
import argparse
import optuna

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PRETRAIN_PATH = os.path.join("..", "models", "sequence_masked_full_model.pth")
PICKLE_PATH = os.path.join("..", "data", "noFE_windowed_segraw_allEMG.pkl")
SAVE_FT_PATH = os.path.join("..", "models", "best_finetuned_model.pth")

# --- User Split (Fixed) ---
# Using the same user split as before for consistency
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

# --- Model Definition ---
class HybridFineTuneModel(nn.Module):
    def __init__(self, n_classes, hidden_dim=128, dropout_p=0.5):
        super().__init__()
        # Encoder with a new first layer and pre-trainable 2nd/3rd layers
        self.encoder = nn.Sequential(
            # Layer 1: New, randomly initialized for 16 channels
            nn.Conv1d(16, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(True),
            # Layer 2: Structure matches pre-trained model to load weights
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.ReLU(True),
            # Layer 3: Structure matches pre-trained model to load weights
            nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1), nn.BatchNorm1d(hidden_dim), nn.ReLU(True),
        )
        # New classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, n_classes)
        )

    def forward(self, x):
        # The input x is expected to be (B, C, T), e.g., (batch, 16, 400)
        # We need to transpose it to (B, T, C) for Conv1d if it's not already
        if x.shape[1] != 16:
             x = x.transpose(1, 2)
        feat = self.encoder(x)
        logits = self.head(feat)
        return logits

def load_partial_weights(new_model, pretrained_path):
    """Loads weights from layers 2 and 3 of a pretrained encoder."""
    try:
        pretrained_dict = torch.load(pretrained_path, map_location=DEVICE)
        new_model_dict = new_model.state_dict()

        weights_to_load = {}
        # We want to load weights for layers starting from index 3 of the encoder
        # e.g., 'encoder.3.weight', 'encoder.4.bias', etc.
        for key, value in pretrained_dict.items():
            if key.startswith('encoder.3.') or key.startswith('encoder.4.') or \
               key.startswith('encoder.6.') or key.startswith('encoder.7.'):
                if key in new_model_dict and new_model_dict[key].shape == value.shape:
                    weights_to_load[key] = value
        
        new_model_dict.update(weights_to_load)
        new_model.load_state_dict(new_model_dict)
        print(f"Successfully loaded {len(weights_to_load)} tensors from pretrained model.")
        return new_model
    except FileNotFoundError:
        print(f"Pretrained weights not found at {pretrained_path}. Training from scratch.")
        return new_model
    except Exception as e:
        print(f"Error loading partial weights: {e}. Training from scratch.")
        return new_model

# --- Optuna Objective Function ---
def objective(trial, df_val_users, n_classes):
    # Hyperparameters to tune
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    dropout_p = trial.suggest_float("dropout", 0.1, 0.6)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])

    # Create train/validation split within the validation user set for HPO
    df_train, df_val = train_test_split(df_val_users, test_size=0.2, random_state=42, stratify=df_val_users['Gesture_Encoded'])
    ds_train = BGestureDataset(df_train)
    ds_val = BGestureDataset(df_val)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2)

    # Model setup
    model = HybridFineTuneModel(n_classes=n_classes, dropout_p=dropout_p).to(DEVICE)
    model = load_partial_weights(model, PRETRAIN_PATH)

    # Freeze the loaded layers, train the new layer and the head
    for name, param in model.encoder.named_parameters():
        if not name.startswith('0.'): # Layer 0 is the new Conv1d
            param.requires_grad = False

    optimizer = getattr(torch.optim, optimizer_name)(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop for HPO
    best_acc = 0
    for epoch in range(10): # Fixed 10 epochs for HPO for speed
        model.train()
        for x, y in dl_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

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

    # DataLoaders: Train on the full validation set, test on the test set
    ds_train = BGestureDataset(df_val_users)
    ds_test = BGestureDataset(df_test_users)
    dl_train = DataLoader(ds_train, batch_size=best_params['batch_size'], shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=best_params['batch_size'], shuffle=False)

    # Model setup
    model = HybridFineTuneModel(n_classes=n_classes, dropout_p=best_params['dropout']).to(DEVICE)
    model = load_partial_weights(model, PRETRAIN_PATH)
    
    # Unfreeze all layers for final fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    optimizer = getattr(torch.optim, best_params['optimizer'])(model.parameters(), lr=best_params['lr'])
    criterion = nn.CrossEntropyLoss()

    # Final training loop
    for epoch in range(20): # Train for more epochs in the final run
        model.train()
        for x, y in dl_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        print(f"Final Training Epoch {epoch+1}/20, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), SAVE_FT_PATH)
    print(f"Final model saved to {SAVE_FT_PATH}")

    # Final evaluation on the test set
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
    # Load and prepare data
    print("Loading and preparing data...")
    df = pd.read_pickle(PICKLE_PATH)
    df = df.rename(columns={"windowed_ts_data": "feature"})
    le = LabelEncoder()
    df["Gesture_Encoded"] = le.fit_transform(df["Gesture"])
    n_classes = len(le.classes_)
    
    df_val_users = df[df.Participant.isin(VAL_USERS)].reset_index(drop=True)
    df_test_users = df[df.Participant.isin(TEST_USERS)].reset_index(drop=True)
    print("Data loaded.")

    # --- HPO Phase ---
    print("\n--- Starting Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, df_val_users, n_classes), n_trials=n_trials)

    print("\n--- Hyperparameter Optimization Finished ---")
    print(f"Best trial accuracy: {study.best_value:.4f}")
    print(f"Best parameters: {study.best_trial.params}")

    # --- Final Training Phase ---
    final_train_and_eval(study.best_trial.params, df_val_users, df_test_users, n_classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tuning HPO script")
    parser.add_argument("--n_trials", type=int, default=25, help="Number of Optuna trials")
    args = parser.parse_args()
    main(args.n_trials)

