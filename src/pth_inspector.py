import argparse
import torch

def inspect_pth(pth_path):
    checkpoint = torch.load(pth_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    for key, value in state_dict.items():
        if hasattr(value, 'shape'):
            print(f"{key}: {tuple(value.shape)}")
        else:
            print(f"{key}: {type(value)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect .pth file parameters")
    parser.add_argument("pth_path", type=str, help="Path to the .pth file")
    args = parser.parse_args()
    inspect_pth(args.pth_path)