 #!/usr/bin/env python3
# coding: utf-8

import os
import datetime
import argparse
import json
from prettytable import PrettyTable

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import EMGMaskedAE
from ninapro_dataset import NinaproDataset

# ---------------- Config ----------------
def setup_ddp():
    dist.init_process_group(backend="nccl")        # fastest on NVIDIA
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)              # one GPU ↔ one process
    return torch.device(f"cuda:{local_rank}")

def cleanup_ddp():
    dist.destroy_process_group()

DEVICE = setup_ddp()

# Dataloader parameters
TRAIN_DATA_PATH = 'data/Ninapro/DB2_emg_only_all_subjects.json'
TEST_DATA_PATH  = 'data/Ninapro/DB3_emg_only_all_subjects.json'
NUM_WORKERS     = 4

# Training parameters
BATCH_SIZE      = 32
EPOCHS          = 50
LR              = 1e-3

# Model parameters
MASK_TYPE       = 'block'
MASK_RATIO      = 0.4
BLOCK_LEN       = 512

def train(report=False, test=False):
    print(f'Using device: {DEVICE}', flush=True)
    if DEVICE.type == 'cuda':
        print(f'Found {torch.cuda.device_count()} CUDA device(s).', flush=True)
        print(f'Current CUDA device: {torch.cuda.current_device()}', flush=True)
        print(f'Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}', flush=True)

    train_ds = NinaproDataset(TRAIN_DATA_PATH)
    print(f'Finished loading dataset with length {len(train_ds)}.')

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    train_dl = DataLoader(
        train_ds,
        batch_size = BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # --- Test Data Loading (Optional) ---
    test_dl = None
    if test:
        try:
            test_ds = NinaproDataset(TEST_DATA_PATH)
            test_sampler = DistributedSampler(test_ds, shuffle=False)
            if len(test_ds) > 0:
                test_dl = DataLoader(
                    test_ds,
                    batch_size=BATCH_SIZE,
                    sampler=test_sampler,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                )
                print(f'Loaded {len(test_ds)} test windows for per-epoch evaluation.', flush=True)
            else:
                print("Warning: Test data folder provided, but no data was found.", flush=True)
        except Exception as e:
            print(f"Warning: Could not load test data: {e}", flush=True)


    # ---------------- Model ----------------
    in_ch = train_ds[0][0].shape[0]
    model = EMGMaskedAE(
        in_ch=in_ch,
        mask_type=MASK_TYPE,
        mask_ratio=MASK_RATIO,
        block_len=BLOCK_LEN,
        different_mask_per_channel=True,
        device=DEVICE
    ).to(DEVICE)
    
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

    # --- Initialize training report ---
    if report:
        report_data = {
            'model_parameters': {
                'in_ch': in_ch,
                'mask_type': model.mask_type,
                'mask_ratio': model.mask_ratio,
                'block_len': model.block_len,
                'freq_band': model.freq_band if model.mask_type == "freq" else None,
                'different_mask_per_channel': model.different_mask_per_channel,
                'device': str(model.device),
            },
            'training_parameters': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'learning_rate': LR,
            },
            'loss_history': [],
            'test_loss_history': []
        }
        training_start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        report_path = f'pretrain/training_report/{model.mask_type}/training_report_{training_start_time}.json'
        print(f'Training report will be saved to {report_path}', flush=True)

    model = DistributedDataParallel(model, device_ids=[DEVICE.index], output_device=DEVICE.index)
    model = torch.compile(model)

    total_steps = EPOCHS * len(train_dl)
    step = 0

    with tqdm(total=total_steps,
              desc='Training',
              unit='step') as pbar:

        for epoch in range(1, EPOCHS + 1):
            train_sampler.set_epoch(epoch)
            model.train()
            epoch_loss = 0.0
            for x, _ in train_dl:
                x = x.to(DEVICE, non_blocking=True)

                reconstructed = model(x)
                loss = model.compute_loss(reconstructed, x)

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
                        reconstructed = model(x_test)
                        # Use the new loss function for testing as well
                        loss = model.compute_loss(reconstructed, x_test)
                        total_test_loss += loss.item()
                        current_avg_loss = total_test_loss / (test_pbar.n + 1)
                        test_pbar.set_postfix(loss=f'{current_avg_loss:.6f}')

                avg_test_loss = total_test_loss / len(test_dl)
                if dist.get_rank() == 0 and report:
                    report_data['test_loss_history'].append(avg_test_loss)
                tqdm.write(f'Epoch {epoch} | Test Loss: {avg_test_loss:.6f}')

            # --- Save Model at End of Each Epoch ---
            if dist.get_rank() == 0 and (epoch % 5 == 0 or epoch == EPOCHS):
                ckpt_dir = f'pretrain/checkpoints/{MASK_TYPE}/{training_start_time}'
                os.makedirs(ckpt_dir, exist_ok=True)

                ckpt_path = os.path.join(ckpt_dir, f'ckpt_{MASK_TYPE}_epoch_{epoch}.pth')
                torch.save(model.state_dict(), ckpt_path)

                tqdm.write(f'  -> Saved model at epoch {epoch} to {ckpt_path}')
    
    cleanup_ddp()
    
    if report:
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=4)
        print('Saved training_report.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EMG Masked Autoencoder with per-channel frequency masking.')
    parser.add_argument('--report', action='store_true', help='Enable saving a training report.')
    parser.add_argument('--test', action='store_true', help='Enable per-epoch test evaluation.')
    args = parser.parse_args()

    train(report=args.report, test=args.test)
