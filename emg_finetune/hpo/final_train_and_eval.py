import torch
import torch.nn as nn
import pandas as pd
import json
from torch.utils.data import DataLoader

from .model import EndTaskClassifier
from .b_gesture_dataset import BGestureDataset
from .hpo_const import *

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def final_train_and_eval(best_params, df_train_users, df_test_users, n_classes):
    print('\n--- Starting Final Training with Best Hyperparameters ---')
    print(f'Best Parameters: {best_params}')

    # DataLoaders
    ds_train = BGestureDataset(df_train_users)
    ds_test = BGestureDataset(df_test_users)
    dl_train = DataLoader(ds_train, batch_size=best_params['batch_size'], shuffle=True)
    dl_test = DataLoader(ds_test, batch_size=best_params['batch_size'], shuffle=False)

    # Model, Optimizer, Scheduler
    N, T, C = ds_train.features.shape
    model = EndTaskClassifier(C, num_classes=n_classes, config=best_params, ckpt_path=PRETRAINED_CKPT).to(DEVICE)
    
    optimizer_name = best_params['optimizer']
    if optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=best_params['lr'], momentum=best_params['momentum'], 
            weight_decay=best_params['weight_decay'], nesterov=best_params['nesterov']
        )
    elif optimizer_name == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
    else: # Adam
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    scheduler_type = best_params['scheduler']
    scheduler = None
    if scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=best_params['step_size'], gamma=best_params['gamma'])
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=best_params['min_lr'])

    criterion = nn.CrossEntropyLoss()

    # Final training loop
    for epoch in range(15): # Train for more epochs
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for x, y in dl_train:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()

        if scheduler:
            scheduler.step()

        train_acc = 100 * train_correct / train_total
        
        # Evaluation for test set
        model.eval()
        test_loss, test_correct, test_total = 0, 0, 0
        with torch.no_grad():
            for x, y in dl_test:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                test_loss += loss.item()
                _, predicted = torch.max(logits.data, 1)
                test_total += y.size(0)
                test_correct += (predicted == y).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        print(f'Epoch {epoch+1}/15 | Train Loss: {train_loss/len(dl_train):.4f}, Train Acc: {train_acc:.2f}% | Test Loss: {test_loss/len(dl_test):.4f}, Test Acc: {test_acc:.2f}%')


    print(f'\n--- Final Test Accuracy: {test_acc:.2f}% ---')

    # Save the model and config
    torch.save(model.state_dict(), 'emg_finetune/checkpoints/best_finetuned_model.pth')
    print('Saved final model and config.')
