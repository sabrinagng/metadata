import os
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
from glob import glob
import tqdm

# ---------- 参数配置 ----------
DATA_FOLDER = '/Users/km82/Documents/train/Ninapro/DB2_emg_only'  # 输入原始文件
OUT_FOLDER = '/Users/km82/Documents/train/Ninapro'  # 输出文件
os.makedirs(OUT_FOLDER, exist_ok=True)

SAMPLING_RATE = 2000
WINDOW_LEN = 200
STEP = 100

# ---------- 滤波器 ----------
def bandpass_filter(signal, low=20, high=450, fs=2000, order=4):
    b, a = butter(order, [low / fs * 2, high / fs * 2], btype='band')
    return filtfilt(b, a, signal, axis=0)

def notch_filter(signal, notch_freq=50, fs=2000, Q=30):
    b, a = iirnotch(notch_freq, Q, fs)
    return filtfilt(b, a, signal, axis=0)

# ---------- 信号切窗 ----------
def window_signal(emg, window_len, step):
    windows = []
    for start in range(0, emg.shape[0] - window_len + 1, step):
        windows.append(emg[start:start + window_len])
    return np.stack(windows)

# ---------- 完整预处理流程 ----------
def preprocess_emg_signal(emg_raw, use_notch=True):
    emg = bandpass_filter(emg_raw)
    if use_notch:
        emg = notch_filter(emg)
    
    # 切窗（不在整流前做，否则截断失真）
    emg_windows = window_signal(emg, window_len=WINDOW_LEN, step=STEP)
    
    # 每个 window 做整流 + 标准化
    processed = []
    for win in emg_windows:
        win = np.abs(win)  # 整流
        # z-score 标准化
        win = (win - np.mean(win, axis=0)) / (np.std(win, axis=0) + 1e-6)
        # 3σ 截断（防止极端值干扰）
        win = np.clip(win, -3, 3)
        processed.append(win)
    
    return np.stack(processed)  # [N, 200, C]

# ---------- 主函数 ----------
all_files = glob(os.path.join(DATA_FOLDER, '*.npy'))

for file_path in tqdm.tqdm(all_files, desc="⚙️ Processing EMG"):
    filename = os.path.basename(file_path).replace('.npy', '')
    emg_raw = np.load(file_path)  # [T, C]
    
    emg_cleaned = preprocess_emg_signal(emg_raw, use_notch=True)
    
    out_path = os.path.join(OUT_FOLDER, f"{filename}_clean.npy")
    np.save(out_path, emg_cleaned)

    print(f"✅ Saved: {emg_cleaned.shape} → {out_path}")
