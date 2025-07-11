import torch
import torch.nn as nn
import torch.nn.functional as F

## Masked Autoencoder for EMG signals
# This model implements a masked autoencoder for EMG signals with different masking strategies.

class MAEEncoderBlock(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.in_ch = in_ch

        self.encoder = nn.Sequential(
            nn.Conv1d( in_channels=in_ch,  out_channels=32,  kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(32),  nn.GELU(),  # 8192→4096
            nn.Conv1d( in_channels=32,     out_channels=32,  kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(32),  nn.GELU(),  # 4096→2048
            nn.Conv1d( in_channels=32,     out_channels=64,  kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(64),  nn.GELU(),  # 2048→1024
            nn.Conv1d( in_channels=64,     out_channels=64,  kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(64),  nn.GELU(),  # 1024→512
            nn.Conv1d( in_channels=64,     out_channels=128, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(128), nn.GELU(),  # 512 →256
            nn.Conv1d( in_channels=128,    out_channels=128, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(128), nn.GELU(),  # 256 →128
            nn.Conv1d( in_channels=128,    out_channels=256, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(256), nn.GELU(),  # 128 →64
            nn.Conv1d( in_channels=256,    out_channels=512, kernel_size=3, stride=2, padding=1), nn.BatchNorm1d(512), nn.GELU(),  # 64  →32
            nn.Conv1d( in_channels=512,    out_channels=1024,kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(1024),nn.GELU(),  # keep 32
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class MAEDecoderBlock(nn.Module):
    """
    Upsamples (B, 1024, 32) back to (B, C, 8192) in three stages:
        32 → 128 → 512 → 8192
    """
    def __init__(self, out_ch: int):
        super().__init__()

        mid_dim   = 256          # fixed internal width
        latent_in = 1024         # must match encoder’s last channel count

        self.decoder = nn.Sequential(
            nn.Conv1d( in_channels=latent_in, out_channels=mid_dim, kernel_size=1), nn.GELU(),

            # 32 → 128
            nn.ConvTranspose1d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=4, stride=4, padding=0), nn.GELU(),
            nn.Conv1d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1), nn.GELU(),

            # 128 → 512
            nn.ConvTranspose1d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=4, stride=4, padding=0), nn.GELU(),
            nn.Conv1d(in_channels=mid_dim, out_channels=mid_dim, kernel_size=3, stride=1, padding=1), nn.GELU(),

            # 512 → 8192
            nn.ConvTranspose1d(in_channels=mid_dim, out_channels=out_ch, kernel_size=16, stride=16, padding=0),
        )

        # sanity check
        L = 32
        for k, s in [(4,4), (4,4), (16,16)]:
            L = (L - 1) * s + k
        assert L == 8192, "decoder length mismatch (got %d)" % L
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)

class EMGMaskedAE(nn.Module):
    def __init__(
        self,
        in_ch: int,
        mask_type: str = 'block',
        mask_ratio: float = 0.5,
        block_len: int = 512,
        freq_band: str = 'high',
        different_mask_per_channel = False,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Masked Autoencoder for EMG signals.
        Args:
            in_ch (int): Number of input channels (EMG sensors).
            mask_type (str): Type of masking ('random', 'block', or 'freq').
            mask_ratio (float): Ratio of the input to be masked.
            block_len (int): Length of each block for block masking.
            freq_band (str): Frequency band to mask ('high' or 'low').
            different_mask_per_channel (bool): If True, apply different masks per channel.
            device (str): Device to run the model on ('cuda' or 'cpu').

        Note:
            - `mask_type` can be 'random', 'block', or 'freq'.
            - `freq_band` is only relevant if `mask_type` is 'freq'.
            - `block_len` and `different_mask_per_channel` is only relevant if `mask_type` is 'block'.
            - The model uses a 5-layer CNN encoder and decoder.
            - The input is expected to be of shape (B, C, T) where B
              is batch size, C is number of channels, and T is the time dimension.
        """
        super().__init__()
        self.in_ch = in_ch
        self.mask_ratio = mask_ratio
        self.block_len = block_len
        self.freq_band = freq_band
        self.mask_type = mask_type
        self.different_mask_per_channel = different_mask_per_channel
        self.device = device

        self.encoder = MAEEncoderBlock(in_ch=self.in_ch)
        self.decoder = MAEDecoderBlock(out_ch=self.in_ch)

        self.mask = None # To store the mask used during forward pass

    def random_mask(self, x):
        B, C, T = x.shape
        num = int(T * self.mask_ratio)
        mask = torch.zeros((B, 1, T), dtype=torch.bool, device=x.device)
        idx = torch.randint(0, T, (B, num), device=x.device)
        mask.scatter_(2, idx.unsqueeze(1), True)
        
        mask_full = mask.expand(B, C, T)
        x_masked = x.masked_fill(mask_full, 0.)
        self.mask = mask_full
        
        return x_masked

    def block_mask(self, x):
        B, C, T = x.shape
        
        num_blocks_total = T // self.block_len
        num_blocks_to_mask = int(num_blocks_total * self.mask_ratio)

        if self.different_mask_per_channel:
            # Create different masks for each channel.
            # block_indices shape: (C, num_blocks_to_mask)
            block_indices = torch.stack([
                torch.randperm(num_blocks_total, device=x.device)[:num_blocks_to_mask]
                for _ in range(C)
            ])
            Cm = C
        else:
            # Create one mask and expand it to all channels.
            # block_indices shape: (1, num_blocks_to_mask)
            block_indices = torch.randperm(num_blocks_total, device=x.device)[:num_blocks_to_mask].unsqueeze(0)
            Cm = 1

        block_starts = block_indices * self.block_len
        block_ranges = torch.arange(self.block_len, device=x.device)
        
        block_positions = block_starts.unsqueeze(-1) + block_ranges     # (Cm, Nm, Lb)
        block_positions = block_positions.clamp(max=T - 1)
        
        flat_indices = block_positions.reshape(Cm, -1)                  # (Cm, Nm * Lb)
        
        # Channel indices for advanced indexing.
        channel_indices = torch.arange(Cm, device=x.device).unsqueeze(1)
        
        mask = torch.zeros((B, Cm, T), dtype=torch.bool, device=x.device)
        mask[:, channel_indices, flat_indices] = True
        
        # If the same mask was created for all channels, expand it.
        if not self.different_mask_per_channel:
            mask = mask.expand(B, C, T)
        
        x_masked = x.masked_fill(mask, 0.)
        self.mask = mask

        return x_masked

    def freq_mask(self, x):
        """
        Not implemented yet.
        This method applies frequency masking in the frequency domain.
        """
        raise NotImplementedError("freq masking not implemented yet")
    

    def forward(self, x):
        if self.mask_type == 'random':
            x_masked = self.random_mask(x)
        elif self.mask_type == 'block':
            x_masked = self.block_mask(x)
        elif self.mask_type == 'freq':
            x_masked = self.freq_mask(x)
        else:
            raise ValueError(f'Unknown mask_type: {self.mask_type}')
        
        latent = self.encoder(x_masked)
        reconstructed = self.decoder(latent)
        return reconstructed

    def compute_loss(self, reconstructed: torch.Tensor, original: torch.Tensor, spec_weight: float = 0.0):
        """Composite loss = time-domain Smooth-L1  +  spectral L1.

        Args
        ----
        reconstructed : (B, C, T)
        original      : (B, C, T)
        spec_weight   : weight for the spectral term. 0 ⇒ time-domain only.
        """
        if self.mask is None:
            raise RuntimeError("Mask has not been generated before loss computation.")

        # ── time-domain loss on masked tokens ──
        l_time = F.smooth_l1_loss(
            reconstructed[self.mask],
            original[self.mask],
        )

        if spec_weight == 0.0:
            return l_time

        # ── simple magnitude‑spectrum loss (FFT) ──
        # Only compute on the **masked** portions to keep behaviour consistent.
        # Gather the masked samples, FFT over last axis, compare magnitude.
        rec_masked = reconstructed[self.mask].view(-1, n)  # flatten
        orig_masked = original[self.mask].view(-1, n)

        # zero‑pad to next power‑of‑two for speed / identical length
        n = 1 << (rec_masked.numel() - 1).bit_length()
        rec_fft  = torch.fft.rfft(rec_masked, n=n)
        orig_fft = torch.fft.rfft(orig_masked, n=n)
        l_spec = torch.mean(torch.abs(torch.abs(rec_fft) - torch.abs(orig_fft)))

        return l_time + spec_weight * l_spec


## CNN-LSTM Predictor for EMG signals
# This model implements a CNN-LSTM architecture for predicting future EMG signals.
# TODO: THIS PART NEEDS REWRITE

class EMGConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class CNNLSTM(nn.Module):
    def __init__(self, in_ch: int, pred_len: int,
                 hidden_size: int = 128, num_layers: int = 3):
        super().__init__()
        self.pred_len = pred_len
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, 64,  kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(128, hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.pred_block = nn.Conv1d(hidden_size, in_ch, kernel_size=1)

    def forward(self, x):           # x: (B, C, T_in)
        z = self.cnn(x)             # (B, 128, T_in)
        z = z.permute(0, 2, 1)      # (B, T_in, 128)
        z, _ = self.lstm(z)         # (B, T_in, H)

        z = z[:, -self.pred_len:, :]          # (B, T_pred, H)
        z = z.permute(0, 2, 1)                # (B, H, T_pred)
        pred = self.time_linear(z)            # (B, C, T_pred)
        return pred
    
class EMGConvLSTMPredictor(nn.Module):
    def __init__(self, in_ch, pred_len, hidden_size, num_layers=2):
        super().__init__()
        self.in_ch = in_ch
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, in_ch * pred_len)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, T, C]

        # divide into observed and target sequences
        obs_len = x.shape[1] - self.pred_len
        obs = x[:, :obs_len, :]
        target = x[:, obs_len:, :]

        lstm_out, _ = self.lstm(obs)  # [B, T, H]
        
        # We only need the output of the last time step
        last_hidden_state = lstm_out[:, -1, :]
        
        # Pass through the fully connected layer
        prediction = self.fc(last_hidden_state)                     # [B, pred_len * in_ch]
        prediction = prediction.view(-1, self.pred_len, self.in_ch) # [B, pred_len, in_ch]
        
        return prediction, target

    def compute_loss(self, pred, target):
        return self.loss_fn(pred, target)

    def predict_future(self, x, pred_len=None):
        if pred_len is None:
            pred_len = self.pred_len

        # x shape: (batch_size, in_ch, sequence_length)
        # Permute to (batch_size, sequence_length, in_ch) for LSTM
        x = x.permute(0, 2, 1)

        # Get the initial hidden state from the input sequence
        _, (h, c) = self.lstm(x)
        
        # Last observation from the input sequence
        last_obs = x[:, -1:, :]

        predictions = []
        for _ in range(pred_len):
            lstm_out, (h, c) = self.lstm(last_obs, (h, c))
            # lstm_out is (B, 1, H)
            
            # Use the hidden state from the LSTM to predict the next step
            prediction = self.fc(lstm_out.squeeze(1)) # (B, H) -> (B, C*P)
            
            # Reshape to (B, P, C) and take the first step
            prediction_one_step = prediction.view(-1, self.pred_len, self.in_ch)[:, 0, :] # (B, C)
            
            # Reshape to (B, 1, C) to feed back into LSTM
            last_obs = prediction_one_step.unsqueeze(1)
            
            predictions.append(last_obs)

        # Concatenate predictions along the sequence dimension
        return torch.cat(predictions, dim=1).permute(0, 2, 1)
