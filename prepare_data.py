import os
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy import stats
import pickle

# Configuration
WINDOW_SIZE = 60
STEP_SIZE = 30
CHANNELS = 8

ACTIVITY_MERGE = {
    # WISDM (18 -> 8)
    "A": "Walking", "B": "Jogging", "C": "Stairs",
    "D": "Still", "E": "Still",
    "F": "Hand Activity", "G": "Hand Activity", "Q": "Hand Activity",
    "H": "Eating", "I": "Eating", "J": "Eating", "K": "Eating", "L": "Eating",
    "M": "Sports", "O": "Sports", "P": "Sports",
    "R": "Active Hands", "S": "Active Hands",
    
    # Heterogeneity (6 -> 4 mapped to the 8)
    "walk": "Walking", "sit": "Still", "stand": "Still", 
    "stairsup": "Stairs", "stairsdown": "Stairs", "bike": "Sports",
    
    # UCI HAR (6 -> 3 mapped to the 8)
    "1": "Walking", "2": "Stairs", "3": "Stairs", 
    "4": "Still", "5": "Still", "6": "Still"
}

from scipy.signal import butter, sosfiltfilt

# 1. Filter Design
FS = 20.0 # Sampling rate in Hz
NYQ = 0.5 * FS

# Noise filter (5Hz cutoff)
sos_noise = butter(3, 5.0 / NYQ, btype='low', output='sos')

# Gravity filter (0.3Hz cutoff)
sos_gravity = butter(3, 0.3 / NYQ, btype='low', output='sos')

def preprocess_window(window_6d):
    """
    Input: (N, 6) array [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    Output: (N, 8) array [body_acc_xyz, gyro_xyz, acc_mag, gyro_mag]
    """
    # 2. Noise Filtering
    filtered = np.zeros_like(window_6d)
    for i in range(6):
        filtered[:, i] = sosfiltfilt(sos_noise, window_6d[:, i])
    
    accel = filtered[:, 0:3]
    gyro = filtered[:, 3:6]
    
    # 3. Gravity Separation
    gravity = np.zeros_like(accel)
    for i in range(3):
        gravity[:, i] = sosfiltfilt(sos_gravity, accel[:, i])
        
    body_accel = accel - gravity
    
    # 4. Feature Extraction (Magnitude)
    accel_mag = np.linalg.norm(body_accel, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)
    
    # 5. Concatenate to 8 channels
    enhanced_window = np.hstack((body_accel, gyro, accel_mag, gyro_mag))
    
    return enhanced_window

def clean_wisdm(file_path):
    try:
        records = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().rstrip(';').split(',')
                if len(parts) >= 6:
                    try:
                        records.append([parts[1], float(parts[3]), float(parts[4]), float(parts[5])])
                    except ValueError:
                        pass
        return pd.DataFrame(records, columns=["ActivityCode", "X", "Y", "Z"])
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return pd.DataFrame()

def extract_wisdm():
    print("Processing WISDM...")
    accel_dir = "wisdm_data/extracted/wisdm-dataset/raw/phone/accel"
    gyro_dir = "wisdm_data/extracted/wisdm-dataset/raw/phone/gyro"
    
    all_frames = []
    all_labels = []

    for sid in range(1600, 1651):
        a_path = f"{accel_dir}/data_{sid}_accel_phone.txt"
        g_path = f"{gyro_dir}/data_{sid}_gyro_phone.txt"
        
        if not os.path.exists(a_path) or not os.path.exists(g_path):
            continue
            
        df_a = clean_wisdm(a_path)
        df_g = clean_wisdm(g_path)
        
        if df_a.empty or df_g.empty:
            continue
            
        for act in df_a["ActivityCode"].unique():
            a_act = df_a[df_a["ActivityCode"] == act]
            g_act = df_g[df_g["ActivityCode"] == act]
            
            min_len = min(len(a_act), len(g_act))
            if min_len < WINDOW_SIZE: continue
            
            merged = np.column_stack((
                a_act["X"].values[:min_len],
                a_act["Y"].values[:min_len],
                a_act["Z"].values[:min_len],
                g_act["X"].values[:min_len],
                g_act["Y"].values[:min_len],
                g_act["Z"].values[:min_len]
            ))
            
            label = ACTIVITY_MERGE.get(act, "Unknown")
            if label == "Unknown": continue
            
            for i in range(0, min_len - WINDOW_SIZE, STEP_SIZE):
                window = merged[i:i+WINDOW_SIZE]
                window = preprocess_window(window)
                all_frames.append(window)
                all_labels.append(label)
                
    return all_frames, all_labels

def extract_heterogeneity():
    print("Processing Heterogeneity...")
    import zipfile
    all_frames, all_labels = [], []
    
    zip_path = "heterogeneity+activity+recognition/Activity recognition exp.zip"
    if not os.path.exists(zip_path):
        return [], []
        
    with zipfile.ZipFile(zip_path, 'r') as z:
        # Load Phones
        with z.open('Activity recognition exp/Phones_accelerometer.csv') as f:
            df_a = pd.read_csv(f)
        with z.open('Activity recognition exp/Phones_gyroscope.csv') as f:
            df_g = pd.read_csv(f)
            
    # Group by User and split into windows. 
    # The Heterogeneity data is roughly 100Hz-200Hz. Let's just resample/subsample by taking every 5th row to simulate closer to 20-40Hz.
    df_a = df_a.iloc[::5].reset_index(drop=True)
    df_g = df_g.iloc[::5].reset_index(drop=True)
    
    # We will just greedily align by minimum length for each user+device+gt since they started roughly together
    users = df_a["User"].unique()
    for u in users:
        u_a = df_a[df_a["User"] == u]
        u_g = df_g[df_g["User"] == u]
        for gt in u_a["gt"].unique():
            act_a = u_a[u_a["gt"] == gt]
            act_g = u_g[u_g["gt"] == gt]
            
            min_len = min(len(act_a), len(act_g))
            if min_len < WINDOW_SIZE: continue
            
            merged = np.column_stack((
                act_a["x"].values[:min_len], act_a["y"].values[:min_len], act_a["z"].values[:min_len],
                act_g["x"].values[:min_len], act_g["y"].values[:min_len], act_g["z"].values[:min_len]
            ))
            
            label = ACTIVITY_MERGE.get(gt, "Unknown")
            if label == "Unknown": continue
            
            for i in range(0, min_len - WINDOW_SIZE, STEP_SIZE):
                window = merged[i:i+WINDOW_SIZE]
                window = preprocess_window(window)
                all_frames.append(window)
                all_labels.append(label)
                
    return all_frames, all_labels

def extract_uci_har():
    print("Processing UCI HAR...")
    # UCI HAR is pre-windowed. 128 samples per window. 
    # We will resize using scipy.ndimage.zoom from 128 to 200.
    
    base = "UCI HAR Dataset (1)/UCI HAR Dataset"
    all_frames, all_labels = [], []
    
    for split in ["train", "test"]:
        labels_path = f"{base}/{split}/y_{split}.txt"
        if not os.path.exists(labels_path): continue
            
        with open(labels_path, "r") as f:
            y = [line.strip() for line in f.readlines()]
            
        channels = []
        c_names = ["body_acc_x", "body_acc_y", "body_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]
        for c in c_names:
            c_path = f"{base}/{split}/Inertial Signals/{c}_{split}.txt"
            with open(c_path, "r") as f:
                # Each line is 128 floats separated by space
                data = [[float(v) for v in line.split()] for line in f.readlines()]
                channels.append(data)
                
        # channels is list of 6, each (N, 128)
        N = len(y)
        for i in range(N):
            label = ACTIVITY_MERGE.get(y[i], "Unknown")
            if label == "Unknown": continue
            
            window_128 = np.column_stack([channels[c][i] for c in range(6)]) # (128, 6)
            
            # zoom to (60, 6)
            window_60 = zoom(window_128, (60/128, 1))
            
            window_processed = preprocess_window(window_60)
            
            all_frames.append(window_processed)
            all_labels.append(label)
            
    return all_frames, all_labels


# ── Custom Mobile CSV Data ────────────────────────────────────────────────────
def extract_custom_csv(csv_path, activity_label):
    """
    Parse custom mobile sensor CSV recorded from a phone app.
    Format: Timestamp, Sensor Type, Value1, Value2, Value3
    
    Sensor Types (Android):
      1 = Accelerometer (m/s²)
      4 = Gyroscope (rad/s)
    
    Extracts accel + gyro, aligns by nearest timestamp, creates windows.
    """
    print(f"  Processing custom CSV: {csv_path} -> {activity_label}")
    
    if not os.path.exists(csv_path):
        print(f"    ⚠ File not found: {csv_path}")
        return [], []
    
    df = pd.read_csv(csv_path)
    
    # Extract accelerometer (type 1) and gyroscope (type 4)
    accel = df[df['Sensor Type'] == 1][['Timestamp', 'Value1', 'Value2', 'Value3']].copy()
    gyro = df[df['Sensor Type'] == 4][['Timestamp', 'Value1', 'Value2', 'Value3']].copy()
    
    # Average duplicate timestamps (batch sensor events)
    accel = accel.groupby('Timestamp').mean().reset_index().sort_values('Timestamp')
    gyro = gyro.groupby('Timestamp').mean().reset_index().sort_values('Timestamp')
    
    print(f"    Accel readings: {len(accel)}, Gyro readings: {len(gyro)}")
    
    if len(accel) < WINDOW_SIZE or len(gyro) < WINDOW_SIZE:
        print(f"    ⚠ Not enough data for even one window")
        return [], []
    
    # Merge by nearest timestamp using merge_asof
    accel = accel.rename(columns={'Value1': 'ax', 'Value2': 'ay', 'Value3': 'az'})
    gyro = gyro.rename(columns={'Value1': 'gx', 'Value2': 'gy', 'Value3': 'gz'})
    
    merged = pd.merge_asof(
        accel.sort_values('Timestamp'),
        gyro.sort_values('Timestamp'),
        on='Timestamp',
        direction='nearest',
        tolerance=500  # max 500ms gap
    ).dropna()
    
    print(f"    Merged paired readings: {len(merged)}")
    
    if len(merged) < WINDOW_SIZE:
        print(f"    ⚠ Not enough paired data")
        return [], []
    
    # Build 6-channel array: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    data_6ch = merged[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
    
    # Create windows
    all_frames = []
    all_labels = []
    
    for i in range(0, len(data_6ch) - WINDOW_SIZE, STEP_SIZE):
        window = data_6ch[i:i + WINDOW_SIZE]
        window_processed = preprocess_window(window)
        all_frames.append(window_processed)
        all_labels.append(activity_label)
    
    print(f"    Created {len(all_frames)} windows")
    return all_frames, all_labels


def augment_windows(frames, labels, num_augments=15):
    """
    Create augmented copies of windows using noise injection, scaling,
    and time warping to boost representation of real recorded data.
    """
    print(f"  Augmenting {len(frames)} windows x{num_augments}...")
    aug_frames = []
    aug_labels = []
    
    for frame, label in zip(frames, labels):
        frame = np.array(frame)
        for _ in range(num_augments):
            aug = frame.copy()
            
            # 1. Add Gaussian noise (small)
            noise = np.random.normal(0, 0.02, aug.shape)
            aug = aug + noise
            
            # 2. Random scaling (95%-105%)
            scale = np.random.uniform(0.95, 1.05)
            aug = aug * scale
            
            # 3. Random time shift (shift by 0-3 samples)
            shift = np.random.randint(0, 4)
            if shift > 0:
                aug = np.roll(aug, shift, axis=0)
            
            aug_frames.append(aug)
            aug_labels.append(label)
    
    print(f"    Created {len(aug_frames)} augmented windows")
    return aug_frames, aug_labels


def extract_all_custom():
    """Extract and augment all custom recorded CSV data."""
    print("\nProcessing Custom Mobile Data...")
    
    custom_files = {
        "still data.csv": "Still",
        "jogging.csv": "Jogging",
        "eating.csv": "Eating",
    }
    
    all_frames = []
    all_labels = []
    
    for csv_file, activity in custom_files.items():
        csv_path = csv_file  # Files are in the project root
        f, l = extract_custom_csv(csv_path, activity)
        
        if f:
            # Add original windows
            all_frames.extend(f)
            all_labels.extend(l)
            
            # Add augmented windows (15x to boost real data representation)
            af, al = augment_windows(f, l, num_augments=15)
            all_frames.extend(af)
            all_labels.extend(al)
    
    return all_frames, all_labels


if __name__ == "__main__":
    frames, labels = [], []
    
    # 1. Existing datasets
    f1, l1 = extract_uci_har()
    frames.extend(f1); labels.extend(l1)
    
    f2, l2 = extract_heterogeneity()
    frames.extend(f2); labels.extend(l2)
    
    f3, l3 = extract_wisdm()
    frames.extend(f3); labels.extend(l3)
    
    # 2. Custom real recorded data (with augmentation)
    f4, l4 = extract_all_custom()
    frames.extend(f4); labels.extend(l4)
    
    X = np.array(frames)
    y = np.array(labels)
    
    print(f"\nTotal Frames: {X.shape}")
    print(f"Total Labels: {y.shape}")
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u:15s}: {c:6d} ({c/len(y)*100:.1f}%)")
    
    np.save("X_all.npy", X)
    np.save("y_all.npy", y)
    print("\nSaved to X_all.npy and y_all.npy")
