import argparse
import json

import torch
from torch.utils.data import random_split, DataLoader
import optuna
import pandas as pd

from .model import EndTaskClassifier
from .b_gesture_dataset import BGestureDataset
from .final_train_and_eval import final_train_and_eval
from .hpo_const import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def objective(trial, df_val_users, n_classes):
    # Required Hyperparameters
    config = {
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'optimizer': trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'SGD']),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True),
        'scheduler': trial.suggest_categorical('scheduler', ['None', 'StepLR', 'CosineAnnealingLR']),
        'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512]),
        'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 2, 4),
        'kernel_size': trial.suggest_categorical('kernel_size', [3, 5, 7]),
        'stride': trial.suggest_categorical('stride', [1, 2]),
        'use_batchnorm': trial.suggest_categorical('use_batchnorm', [True, False]),
        'head_dropout': trial.suggest_float('head_dropout', 0.1, 0.6),
        # Add backbone selection
        'backbone': trial.suggest_categorical('backbone', ['mae'])
    }
    
    # Conditional Hyperparameters
    if config['optimizer'] == 'SGD':
        config['momentum'] = trial.suggest_float('momentum', 0.8, 0.99)
        config['nesterov'] = trial.suggest_categorical('nesterov', [True, False])
    if config['scheduler'] == 'StepLR':
        config['step_size'] = trial.suggest_int('step_size', 2, 5)
        config['gamma'] = trial.suggest_float('gamma', 0.1, 0.9)
    if config['scheduler'] == 'CosineAnnealingLR':
        config['min_lr'] = trial.suggest_float('min_lr', 1e-6, 1e-4, log=True)
    
    # Dataloaders
    train_size = int(0.8 * len(df_val_users))
    val_size = len(df_val_users) - train_size
    ds = BGestureDataset(df_val_users)
    train_dataset, val_dataset = random_split(ds, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    
    # Model
    N, T, C = ds.features.shape
    print(f'Input shape: {N} samples, {C} channels, {T} time steps')
    if config['backbone'] == 'mae':
        model = EndTaskClassifier(C, num_classes=n_classes, config=config, ckpt_path=PRETRAINED_CKPT).to(DEVICE)
    else:
        # not implement cnn yet
        raise NotImplementedError('Only \'mae\' backbone is implemented for now.')
    
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
    
    # HT Loop
    best_acc = 0.0
    for epoch in range(10):
        # Training Loop
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            
        if scheduler:
            scheduler.step()
            
        # Validation Loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
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

def main(args):
    df = pd.read_pickle(PICKLE_PATH)
    df_val_users = df[df.Participant.isin(VAL_USERS)].reset_index(drop=True)
    df_test_users = df[df.Participant.isin(TEST_USERS)].reset_index(drop=True)

    ds = BGestureDataset(df_val_users)
    n_classes = len(ds.label2idx)
    
    print("\n--- Starting Hyperparameter Optimization ---")
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, df_val_users, n_classes), n_trials=args.n_trials)
    
    print("\n--- Hyperparameter Optimization Finished ---")
    print(f'Best trial accuracy: {study.best_value:.4f}')
    print(f'Best parameters: {study.best_trial.params}')
    
    # Save the best parameters
    with open('emg_finetune/best_hpo_params.json', 'w') as f:
        json.dump(study.best_trial.params, f, indent=4)
        
    if args.final_train:
        print("\n--- Starting Final Training with Best Hyperparameters ---")
        print(f'Best Parameters: {study.best_trial.params}')
        
        # Final training and evaluation
        final_train_and_eval(study.best_trial.params, df_val_users, df_test_users, n_classes)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Advanced Fine-tuning HPO script")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument('--final-train', action='store_true', help="Run final training and evaluation with best hyperparameters")
    args = parser.parse_args()
    main(args)
