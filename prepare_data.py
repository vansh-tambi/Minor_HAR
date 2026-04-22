import os
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from scipy import stats
import pickle

# Configuration
WINDOW_SIZE = 200
STEP_SIZE = 100
CHANNELS = 6

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

def clean_wisdm(file_path):
    try:
        # Some rows have extra columns or semicolons, reading manually is safer
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
            
            # zoom to (200, 6)
            window_200 = zoom(window_128, (200/128, 1))
            
            all_frames.append(window_200)
            all_labels.append(label)
            
    return all_frames, all_labels

if __name__ == "__main__":
    frames, labels = [], []
    
    f1, l1 = extract_uci_har()
    frames.extend(f1); labels.extend(l1)
    
    f2, l2 = extract_heterogeneity()
    frames.extend(f2); labels.extend(l2)
    
    f3, l3 = extract_wisdm()
    frames.extend(f3); labels.extend(l3)
    
    X = np.array(frames)
    y = np.array(labels)
    
    print(f"Total Frames: {X.shape}")
    print(f"Total Labels: {y.shape}")
    
    np.save("X_all.npy", X)
    np.save("y_all.npy", y)
    print("Saved to X_all.npy and y_all.npy")
