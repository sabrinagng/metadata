#!/usr/bin/env python3
# coding: utf-8

import os
import datetime
import argparse
import json
from prettytable import PrettyTable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EMGPrediction

from ninapro_masked_dataset import NinaproMaskedDataset 

# ---------------- Config ----------------
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataloader parameters
DATA_FOLDERS  = ['autodl-tmp/data/DB2_all']
WINDOW_LEN    = 64
STEP          = 16
NUM_WORKERS   = 4

# Training parameters
BATCH_SIZE    = 4096
EPOCHS        = 50
LR            = 1e-3

# Model parameters for the prediction task
PRED_RATIO    = 0.5 # The ratio of the signal to predict
hidden_size   = 256
num_layers    = 2

def train(report=False, test=False):
    print(f'Using device: {DEVICE}', flush=True)
    if DEVICE.type == 'cuda':
        print(f'Found {torch.cuda.device_count()} CUDA device(s).', flush=True)
        print(f'Current CUDA device: {torch.cuda.current_device()}', flush=True)
        print(f'Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}', flush=True)

    # Use a dataset that provides full, unmasked windows
    ds = NinaproMaskedDataset(DATA_FOLDERS, WINDOW_LEN, STEP)
    train_dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)

    # --- Test Data Loading (Optional) ---
    test_dl = None
    if test:
        test_folders = ['autodl-tmp/data/DB3_emg_only']
        try:
            test_ds = NinaproMaskedDataset(test_folders, WINDOW_LEN, STEP)
            if len(test_ds) > 0:
                test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                     num_workers=NUM_WORKERS, pin_memory=True)
                print(f'Loaded {len(test_ds)} test windows for per-epoch evaluation.', flush=True)
            else:
                print("Warning: Test data folder provided, but no data was found.", flush=True)
        except Exception as e:
            print(f"Warning: Could not load test data: {e}", flush=True)


    # ---------------- Model ----------------
    in_ch = ds[0].shape[0]
    pred_len = int(WINDOW_LEN * PRED_RATIO)
    
    # Instantiate the EMGPrediction model
    model = EMGPrediction(
        in_ch=in_ch,
        pred_len=pred_len,
        hidden_size=hidden_size,
        num_layers=num_layers).to(DEVICE)
    
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    # --- Model Summary ---
    table = PrettyTable(["Layer Name", "Shape", "N. of Parameters"])
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            table.add_row([name, list(param.shape), param.numel()])
            total_params += param.numel()
    print(table)
    print(f"Total Trainable Params: {total_params}")

    training_start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    # --- Initialize training report ---
    if report:
        report_data = {
            'model_parameters': {
                'in_ch': model.in_ch,
                'pred_len': model.pred_len,
                'hidden_size': model.hidden_size,
                'num_layers': model.num_layers,
            },
            'training_parameters': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LR,
                'pred_ratio': PRED_RATIO,
            },
            'loss_history': [],
            'test_loss_history': []
        }
        report_path = f'/root/pretrain/training_report/predictor/training_report_{training_start_time}.json'
        print(f'Training report will be saved to {report_path}', flush=True)

    total_steps = EPOCHS * len(train_dl)
    step = 0

    with tqdm(total=total_steps,
              desc='Training Predictor',
              unit='step') as pbar:

        for epoch in range(1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            for x in train_dl:
                x = x.to(DEVICE, non_blocking=True)

                # Updated model forward pass and loss calculation
                pred, target = model(x)
                loss = model.compute_loss(pred, target)

                epoch_loss += loss.item()

                optim.zero_grad()
                loss.backward()
                optim.step()

                step += 1
                pbar.set_postfix(epoch=epoch,
                                 loss=f'{loss.item():.6f}',
                                 lr=f"{optim.param_groups[0]['lr']:.1e}")
                pbar.update(1)

            if report:
                avg_epoch_loss = epoch_loss / len(train_dl)
                report_data['loss_history'].append(avg_epoch_loss)

            # --- Testing ---
            if test_dl:
                model.eval()
                total_test_loss = 0.0
                with torch.no_grad():
                    test_pbar = tqdm(test_dl, desc=f'Testing epoch {epoch}', leave=False)
                    for x_test in test_pbar:
                        x_test = x_test.to(DEVICE, non_blocking=True)
                        
                        # Updated test logic
                        pred, target = model(x_test)
                        loss = model.compute_loss(pred, target)
                        
                        total_test_loss += loss.item()
                        current_avg_loss = total_test_loss / (test_pbar.n + 1)
                        test_pbar.set_postfix(loss=f'{current_avg_loss:.6f}')

                avg_test_loss = total_test_loss / len(test_dl)
                if report:
                    report_data['test_loss_history'].append(avg_test_loss)
                tqdm.write(f'Epoch {epoch} | Test Loss: {avg_test_loss:.6f}')

            # --- Save Model at End of Each Epoch ---
            if epoch % 5 == 0 or epoch == EPOCHS:
                # Updated checkpoint path for the predictor model
                ckpt_dir = f'/root/pretrain/checkpoints/predictor/{training_start_time}'
                os.makedirs(ckpt_dir, exist_ok=True)

                ckpt_path = os.path.join(ckpt_dir, f'ckpt_predictor_epoch_{epoch}.pth')
                torch.save(model.state_dict(), ckpt_path)

                tqdm.write(f'  -> Saved model at epoch {epoch} to {ckpt_path}')

    if report:
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        print('Saved training_report.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EMG Future Signal Predictor.')
    parser.add_argument('--report', action='store_true', help='Enable saving a training report.')
    parser.add_argument('--test', action='store_true', help='Enable per-epoch test evaluation.')
    args = parser.parse_args()

    train(report=args.report, test=args.test)
