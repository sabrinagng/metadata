import torch

# Load the checkpoint
checkpoint_path = "encoder.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Inspect the keys in the checkpoint
print("Checkpoint keys:", checkpoint.keys())

# If 'MaskedEMGMAE.encoder' is a key, inspect its contents
if 'MaskedEMGMAE.encoder' in checkpoint:
    encoder = checkpoint['MaskedEMGMAE.encoder']
    print("MaskedEMGMAE.encoder keys:", encoder.keys())
    # Optionally, print shapes of parameters
    for k, v in encoder.items():
        print(f"{k}: {v.shape}")
else:
    print("'MaskedEMGMAE.encoder' not found in checkpoint.")