import os
import sys
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
    # WISDM (18 -> 7)
    "A": "Walking", "B": "Jogging", "C": "Stairs",
    "D": "Still", "E": "Still",
    "F": "Hand Activity", "G": "Hand Activity", "Q": "Hand Activity",
    "H": "Eating", "I": "Eating", "J": "Eating", "K": "Eating", "L": "Eating",
    "M": "Sports", "O": "Sports", "P": "Sports",
    "R": "Hand Activity", "S": "Hand Activity",  # Was "Active Hands" — merged with Hand Activity
    
    # Heterogeneity (6 -> 4 mapped to the 7)
    "walk": "Walking", "sit": "Still", "stand": "Still", 
    "stairsup": "Stairs", "stairsdown": "Stairs", "bike": "Sports",
    
    # UCI HAR (6 -> 3 mapped to the 7)
    "1": "Walking", "2": "Stairs", "3": "Stairs", 
    "4": "Still", "5": "Still", "6": "Still"
}

from scipy.signal import butter, sosfiltfilt

# Filter Design
FS = 20.0
NYQ = 0.5 * FS
sos_noise = butter(3, 5.0 / NYQ, btype='low', output='sos')
sos_gravity = butter(3, 0.3 / NYQ, btype='low', output='sos')

def preprocess_window(window_6d):
    """
    Input: (N, 6) array [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    Output: (N, 8) array [body_acc_xyz, gyro_xyz, acc_mag, gyro_mag]
    """
    filtered = np.zeros_like(window_6d)
    for i in range(6):
        filtered[:, i] = sosfiltfilt(sos_noise, window_6d[:, i])
    
    accel = filtered[:, 0:3]
    gyro = filtered[:, 3:6]
    
    gravity = np.zeros_like(accel)
    for i in range(3):
        gravity[:, i] = sosfiltfilt(sos_gravity, accel[:, i])
        
    body_accel = accel - gravity
    accel_mag = np.linalg.norm(body_accel, axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(gyro, axis=1, keepdims=True)
    
    return np.hstack((body_accel, gyro, accel_mag, gyro_mag))

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
    print("Processing WISDM...", flush=True)
    accel_dir = "wisdm_data/extracted/wisdm-dataset/raw/phone/accel"
    gyro_dir = "wisdm_data/extracted/wisdm-dataset/raw/phone/gyro"
    
    all_frames, all_labels = [], []

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
                a_act["X"].values[:min_len], a_act["Y"].values[:min_len], a_act["Z"].values[:min_len],
                g_act["X"].values[:min_len], g_act["Y"].values[:min_len], g_act["Z"].values[:min_len]
            ))
            
            label = ACTIVITY_MERGE.get(act, "Unknown")
            if label == "Unknown": continue
            
            for i in range(0, min_len - WINDOW_SIZE, STEP_SIZE):
                window = merged[i:i+WINDOW_SIZE]
                window = preprocess_window(window)
                all_frames.append(window)
                all_labels.append(label)
    
    print(f"  WISDM: {len(all_frames)} windows", flush=True)
    return all_frames, all_labels

def extract_heterogeneity():
    """Process Heterogeneity dataset - optimized to avoid memory issues."""
    print("Processing Heterogeneity...", flush=True)
    import zipfile
    all_frames, all_labels = [], []
    
    zip_path = "heterogeneity+activity+recognition/Activity recognition exp.zip"
    if not os.path.exists(zip_path):
        print("  Heterogeneity dataset not found, skipping.", flush=True)
        return [], []
        
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open('Activity recognition exp/Phones_accelerometer.csv') as f:
            df_a = pd.read_csv(f)
        with z.open('Activity recognition exp/Phones_gyroscope.csv') as f:
            df_g = pd.read_csv(f)
    
    print(f"  Loaded accel: {len(df_a)} rows, gyro: {len(df_g)} rows", flush=True)
    
    # Subsample heavily (every 10th row) to speed up processing
    df_a = df_a.iloc[::10].reset_index(drop=True)
    df_g = df_g.iloc[::10].reset_index(drop=True)
    
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
    
    print(f"  Heterogeneity: {len(all_frames)} windows", flush=True)
    return all_frames, all_labels

def extract_uci_har():
    print("Processing UCI HAR...", flush=True)
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
                data = [[float(v) for v in line.split()] for line in f.readlines()]
                channels.append(data)
                
        N = len(y)
        for i in range(N):
            label = ACTIVITY_MERGE.get(y[i], "Unknown")
            if label == "Unknown": continue
            
            window_128 = np.column_stack([channels[c][i] for c in range(6)])
            window_60 = zoom(window_128, (60/128, 1))
            window_processed = preprocess_window(window_60)
            
            all_frames.append(window_processed)
            all_labels.append(label)
    
    print(f"  UCI HAR: {len(all_frames)} windows", flush=True)
    return all_frames, all_labels


# ── Custom Mobile CSV Data ────────────────────────────────────────────────────

def extract_custom_csv(csv_path, activity_label):
    """
    Parse custom mobile sensor CSV recorded from a phone app.
    Sensor Types (Android): 1 = Accelerometer (m/s²), 4 = Gyroscope (rad/s)
    """
    print(f"  Processing: {csv_path} -> {activity_label}", flush=True)
    
    if not os.path.exists(csv_path):
        print(f"    File not found: {csv_path}", flush=True)
        return [], []
    
    df = pd.read_csv(csv_path)
    
    # Extract accelerometer (type 1) and gyroscope (type 4)
    accel = df[df['Sensor Type'] == 1][['Timestamp', 'Value1', 'Value2', 'Value3']].copy()
    gyro = df[df['Sensor Type'] == 4][['Timestamp', 'Value1', 'Value2', 'Value3']].copy()
    
    # Average duplicate timestamps (batch sensor events)
    accel = accel.groupby('Timestamp').mean().reset_index().sort_values('Timestamp')
    gyro = gyro.groupby('Timestamp').mean().reset_index().sort_values('Timestamp')
    
    print(f"    Accel: {len(accel)}, Gyro: {len(gyro)} readings", flush=True)
    
    if len(accel) < WINDOW_SIZE or len(gyro) < WINDOW_SIZE:
        return [], []
    
    # Merge by nearest timestamp
    accel = accel.rename(columns={'Value1': 'ax', 'Value2': 'ay', 'Value3': 'az'})
    gyro = gyro.rename(columns={'Value1': 'gx', 'Value2': 'gy', 'Value3': 'gz'})
    
    merged = pd.merge_asof(
        accel.sort_values('Timestamp'),
        gyro.sort_values('Timestamp'),
        on='Timestamp',
        direction='nearest',
        tolerance=500
    ).dropna()
    
    print(f"    Paired: {len(merged)} readings", flush=True)
    
    if len(merged) < WINDOW_SIZE:
        return [], []
    
    data_6ch = merged[['ax', 'ay', 'az', 'gx', 'gy', 'gz']].values
    
    all_frames, all_labels = [], []
    for i in range(0, len(data_6ch) - WINDOW_SIZE, STEP_SIZE):
        window = data_6ch[i:i + WINDOW_SIZE]
        window_processed = preprocess_window(window)
        all_frames.append(window_processed)
        all_labels.append(activity_label)
    
    print(f"    Windows: {len(all_frames)}", flush=True)
    return all_frames, all_labels


def augment_windows(frames, labels, num_augments=30):
    """Create augmented copies via diverse, advanced techniques."""
    aug_frames, aug_labels = [], []
    
    for frame, label in zip(frames, labels):
        frame = np.array(frame)
        for i in range(num_augments):
            aug = frame.copy()
            
            # Technique 1: Gaussian jitter (varied noise levels)
            noise_level = np.random.uniform(0.01, 0.05)
            aug += np.random.normal(0, noise_level, aug.shape)
            
            # Technique 2: Random scaling / magnitude warping
            scale = np.random.uniform(0.90, 1.10)
            aug *= scale
            
            # Technique 3: Random time shift
            shift = np.random.randint(0, 8)
            if shift > 0:
                aug = np.roll(aug, shift, axis=0)
            
            # Technique 4: Time warping (randomly stretch/compress segments)
            if i % 3 == 0:
                mid = aug.shape[0] // 2
                warp_factor = np.random.uniform(0.8, 1.2)
                first_half = zoom(aug[:mid], (warp_factor, 1))
                second_half = zoom(aug[mid:], ((2.0 - warp_factor), 1))
                combined = np.vstack((first_half, second_half))
                # Resize back to original window size
                if combined.shape[0] != aug.shape[0]:
                    combined = zoom(combined, (aug.shape[0] / combined.shape[0], 1))
                aug = combined
            
            # Technique 5: Channel-wise permutation (every 5th)
            if i % 5 == 0:
                perm = np.random.permutation(3)
                aug[:, :3] = aug[:, perm]
                perm2 = np.random.permutation(3)
                aug[:, 3:6] = aug[:, 3 + perm2]
            
            # Technique 6: Random signal inversion on a channel
            if i % 4 == 0:
                ch = np.random.randint(0, aug.shape[1])
                aug[:, ch] *= -1
            
            aug_frames.append(aug)
            aug_labels.append(label)
    
    return aug_frames, aug_labels


def cap_dataset_per_class(frames, labels, max_per_class=500):
    """Cap the number of samples per class from a public dataset."""
    from collections import defaultdict
    class_buckets = defaultdict(list)
    for f, l in zip(frames, labels):
        class_buckets[l].append(f)
    
    capped_frames, capped_labels = [], []
    for label, bucket in class_buckets.items():
        np.random.shuffle(bucket)
        selected = bucket[:max_per_class]
        capped_frames.extend(selected)
        capped_labels.extend([label] * len(selected))
    
    return capped_frames, capped_labels


def extract_all_custom():
    """Extract and heavily augment all custom recorded CSV data."""
    print("\nProcessing Custom Mobile Data...", flush=True)
    
    custom_files = {
        "still data.csv": "Still",
        "jogging.csv": "Jogging",
        "eating.csv": "Eating",
        "sensor_recording_20260424_131838.csv": "Stairs",
        "sensor_recording_20260424_184254.csv": "Stairs",
        "sensor_recording_20260424_184845.csv": "Stairs",
        "sensor_recording_20260426_145530.csv": "Stairs",
        "sensor_recording_20260426_145547.csv": "Stairs",
        "sensor_recording_20260426_162439.csv": "Walking",
    }
    
    all_frames, all_labels = [], []
    
    for csv_file, activity in custom_files.items():
        f, l = extract_custom_csv(csv_file, activity)
        
        if f:
            all_frames.extend(f)
            all_labels.extend(l)
            
            # 30x augmentation — real data gets maximum representation
            af, al = augment_windows(f, l, num_augments=30)
            all_frames.extend(af)
            all_labels.extend(al)
            print(f"    + {len(af)} augmented windows", flush=True)
    
    print(f"  Total custom: {len(all_frames)} windows", flush=True)
    return all_frames, all_labels


MAX_PUBLIC_PER_CLASS = 500  # Cap public dataset samples per class


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("  PREPARING HAR TRAINING DATA (CUSTOM-PRIORITY)", flush=True)
    print("=" * 60, flush=True)
    
    frames, labels = [], []
    
    # 1. Custom real recorded data FIRST (highest priority, uncapped)
    f4, l4 = extract_all_custom()
    frames.extend(f4); labels.extend(l4)
    print(f"\n  Custom data: {len(f4)} windows (UNCAPPED)", flush=True)
    
    # 2. UCI HAR (capped to prevent domination)
    f1, l1 = extract_uci_har()
    f1, l1 = cap_dataset_per_class(f1, l1, max_per_class=MAX_PUBLIC_PER_CLASS)
    frames.extend(f1); labels.extend(l1)
    print(f"  UCI HAR (capped): {len(f1)} windows", flush=True)
    
    # 3. Heterogeneity (capped)
    f2, l2 = extract_heterogeneity()
    f2, l2 = cap_dataset_per_class(f2, l2, max_per_class=MAX_PUBLIC_PER_CLASS)
    frames.extend(f2); labels.extend(l2)
    print(f"  Heterogeneity (capped): {len(f2)} windows", flush=True)
    
    # 4. WISDM (capped)
    f3, l3 = extract_wisdm()
    f3, l3 = cap_dataset_per_class(f3, l3, max_per_class=MAX_PUBLIC_PER_CLASS)
    frames.extend(f3); labels.extend(l3)
    print(f"  WISDM (capped): {len(f3)} windows", flush=True)
    
    X = np.array(frames)
    y = np.array(labels)
    
    print(f"\n{'=' * 60}", flush=True)
    print(f"Total Frames: {X.shape}", flush=True)
    print(f"Total Labels: {y.shape}", flush=True)
    
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:", flush=True)
    for u, c in zip(unique, counts):
        print(f"  {u:15s}: {c:6d} ({c/len(y)*100:.1f}%)", flush=True)
    
    # Show custom data dominance ratio
    total = len(y)
    custom_count = len(f4)
    print(f"\n  Custom data ratio: {custom_count/total*100:.1f}% of total", flush=True)
    
    np.save("X_all.npy", X)
    np.save("y_all.npy", y)
    print("\nSaved to X_all.npy and y_all.npy", flush=True)
    print("=" * 60, flush=True)
