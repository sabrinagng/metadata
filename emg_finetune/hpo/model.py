import torch
import torch.nn as nn
from pretrain.model import MAEEncoderBlock

class EndTaskChannelProjector(nn.Module):
    """Encoder for changing any number of input channels to 12 channels."""
    def __init__(self, in_ch, pretrain_in_ch=12):
        super().__init__()
        self.pretrain_in_ch = pretrain_in_ch
        self.channel_project = nn.Linear(in_ch, pretrain_in_ch)

    def forward(self, x):
        if x.shape[1] != self.pretrain_in_ch:
            x = self.channel_project(x)  # (B, T, C)
        return x

class EndTaskEncoder(nn.Module):
    """
    Encoder for the end task. It uses a pretrained MAEEncoderBlock and freezes its weights.
    It also includes a channel projector to adapt the input channels to the pretrained model's requirements.
    """
    def __init__(self, in_ch, pretrain_in_ch=12, ckpt_path=None):
        super().__init__()
        self.channel_projector = EndTaskChannelProjector(in_ch, pretrain_in_ch)
        self.encoder = MAEEncoderBlock(pretrain_in_ch)

        if ckpt_path:
            self._load_ckpt(ckpt_path)
        else:
            print(f"Warning: No checkpoint path provided. The encoder will be trained from scratch.")

        # Freeze the encoder parameters
        for param in self.encoder.parameters():
            param.requires_grad = False

    def _load_ckpt(self, ckpt_path):
        """
        Loads the weights from the encoder of a pretrained EMGMaskedAE model.
        """
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # The state dict from the checkpoint is for the entire EMGMaskedAE model.
            # We need to extract the state dict for the encoder only.
            encoder_state_dict = {}
            for k, v in ckpt.items():
                if k.startswith('encoder.'):
                    # Remove the 'encoder.' prefix to match the keys in MAEEncoderBlock
                    encoder_state_dict[k[len('encoder.'):]] = v
            
            if not encoder_state_dict:
                raise KeyError("No encoder weights found in the checkpoint.")

            self.encoder.load_state_dict(encoder_state_dict)
            print(f"Successfully loaded pretrained encoder from {ckpt_path}")
        except FileNotFoundError:
            print(f"Warning: Checkpoint file not found at {ckpt_path}. The encoder will be trained from scratch.")
        except Exception as e:
            print(f"An error occurred while loading the checkpoint: {e}")

    def forward(self, x):
        # Project the channels first
        x = self.channel_projector(x)
        x = x.permute(0, 2, 1)  # (B, C, T)
        return self.encoder(x)

class EndTaskClassifierConv(nn.Module):
    def __init__(self, config, in_channels=512):
        super().__init__()
        layers = []
        # The number of input channels is determined by the output of the EndTaskEncoder
        current_channels = in_channels

        for _ in range(config["num_layers"]):
            out_channels = config["hidden_dim"]
            kernel = config["kernel_size"]
            stride = config["stride"]
            # To maintain the dimension with stride 1, or halve it with stride 2
            padding = (kernel - 1) // 2

            layers.append(nn.Conv1d(current_channels, out_channels, kernel, stride, padding))
            if config.get("use_batchnorm", False):
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(True))
            
            current_channels = out_channels # Update for the next layer

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)

class EndTaskClassifierHead(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(config['hidden_dim'], num_classes)
        )
    
    def forward(self, x):
        return self.head(x)

class EndTaskClassifier(nn.Module):
    def __init__(self, in_ch, num_classes, config, ckpt_path=None):
        super().__init__()
        self.encoder = EndTaskEncoder(in_ch, ckpt_path=ckpt_path)
        self.conv = EndTaskClassifierConv(config)
        self.head = EndTaskClassifierHead(config, num_classes)
    
    def forward(self, x):
        latent = self.encoder(x)  # (B, C, T)
        hidden = self.conv(latent)
        logits = self.head(hidden)
        return logits