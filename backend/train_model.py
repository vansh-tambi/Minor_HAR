"""
Train CNN on phone accelerometer + gyroscope data from WISDM dataset.
Run from Minor_HAR folder: python backend/train_model.py
Saves: backend/har_model.keras, scaler.pkl, label_encoder.pkl,
       activity_names.pkl, confusion_matrix.png, report.txt

Key improvements over v1:
  - 6-channel input (accel XYZ + gyro XYZ) instead of 3
  - 8 merged activity classes instead of 18
  - 50% overlapping sliding windows for more training data
  - Deeper CNN with BatchNormalization
  - Learning rate scheduling + early stopping
"""

import os
import numpy as np
import pandas as pd
import pickle
import scipy.stats as stats
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from keras.models import Sequential
from keras.layers import (
    Conv1D, MaxPooling1D, Dense, Dropout,
    BatchNormalization, LSTM
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ── Config ────────────────────────────────────────────────────────────────────
ACCEL_PATH      = "code/data/raw_data/phone/accel/"
GYRO_PATH       = "code/data/raw_data/phone/gyro/"
OUTPUT_DIR      = "backend"
FREQUENCY       = 20        # Hz
TIME_PERIOD     = 10        # seconds
FRAME_SIZE      = FREQUENCY * TIME_PERIOD   # 200 samples per window
STEP_SIZE       = FRAME_SIZE // 2           # 50% overlap -> ~2x more training data
NUM_CHANNELS    = 6                         # accel(X,Y,Z) + gyro(X,Y,Z)
EPOCHS          = 30
BATCH_SIZE      = 64
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# ── Merge original 18 codes -> 8 simplified classes ───────────────────────────
# Original codes: A=Walking, B=Jogging, C=Stairs, D=Sitting, E=Standing,
#   F=Typing, G=Brushing Teeth, H=Eating Soup, I=Eating Chips, J=Eating Pasta,
#   K=Drinking, L=Eating Sandwich, M=Kicking, O=Playing Catch, P=Dribbling,
#   Q=Writing, R=Clapping, S=Folding Clothes
ACTIVITY_MERGE = {
    "A": "Walking",
    "B": "Jogging",
    "C": "Stairs",
    "D": "Still",            # Sitting -> Still
    "E": "Still",            # Standing -> Still
    "F": "Hand Activity",    # Typing
    "G": "Hand Activity",    # Brushing Teeth
    "H": "Eating",           # Eating Soup
    "I": "Eating",           # Eating Chips
    "J": "Eating",           # Eating Pasta
    "K": "Eating",           # Drinking
    "L": "Eating",           # Eating Sandwich
    "M": "Sports",           # Kicking
    "O": "Sports",           # Playing Catch
    "P": "Sports",           # Dribbling
    "Q": "Hand Activity",    # Writing
    "R": "Active Hands",     # Clapping
    "S": "Active Hands",     # Folding Clothes
}

ACTIVITY_NAMES_SORTED = sorted(set(ACTIVITY_MERGE.values()))
NUM_CLASSES = len(ACTIVITY_NAMES_SORTED)  # 8


# ── Modular Functions ─────────────────────────────────────────────────────────

def load_and_merge_data(accel_path, gyro_path):
    """
    Load accelerometer and gyroscope CSV files, merge them per subject into
    6-channel DataFrames (accel_X, accel_Y, accel_Z, gyro_X, gyro_Y, gyro_Z).
    Returns a list of merged DataFrames.
    """
    print("=" * 60)
    print("  LOADING & MERGING ACCEL + GYRO DATA")
    print("=" * 60)

    accel_files = sorted([f for f in os.listdir(accel_path) if f.endswith(".csv")])
    gyro_files  = sorted([f for f in os.listdir(gyro_path)  if f.endswith(".csv")])
    print(f"  Found {len(accel_files)} accel files, {len(gyro_files)} gyro files.")

    # Build a lookup: subject_id -> (accel_file, gyro_file)
    accel_by_subject = {}
    for f in accel_files:
        # filename like: data_1600_accel_phone.csv
        parts = f.split("_")
        subject_id = parts[1]  # e.g. "1600"
        accel_by_subject[subject_id] = os.path.join(accel_path, f)

    gyro_by_subject = {}
    for f in gyro_files:
        parts = f.split("_")
        subject_id = parts[1]
        gyro_by_subject[subject_id] = os.path.join(gyro_path, f)

    # Find subjects that have BOTH accel and gyro
    common_subjects = sorted(set(accel_by_subject.keys()) & set(gyro_by_subject.keys()))
    print(f"  Subjects with both sensors: {len(common_subjects)}")

    merged_dfs = []
    total_rows = 0

    for sid in common_subjects:
        try:
            df_a = pd.read_csv(accel_by_subject[sid])
            df_g = pd.read_csv(gyro_by_subject[sid])

            # Merge strategy: for each (subject, activity) pair, align by position
            # since timestamps don't perfectly match between sensors
            merged_parts = []
            for activity_code in df_a["ActivityCode"].unique():
                a_act = df_a[df_a["ActivityCode"] == activity_code].reset_index(drop=True)
                g_act = df_g[df_g["ActivityCode"] == activity_code].reset_index(drop=True)

                # Use the shorter length (both sensors should have similar counts)
                min_len = min(len(a_act), len(g_act))
                if min_len < FRAME_SIZE:
                    continue  # Skip if not enough data for even one window

                merged_part = pd.DataFrame({
                    "ActivityCode": a_act["ActivityCode"].iloc[:min_len],
                    "accel_X": a_act["X"].iloc[:min_len].values,
                    "accel_Y": a_act["Y"].iloc[:min_len].values,
                    "accel_Z": a_act["Z"].iloc[:min_len].values,
                    "gyro_X":  g_act["X"].iloc[:min_len].values,
                    "gyro_Y":  g_act["Y"].iloc[:min_len].values,
                    "gyro_Z":  g_act["Z"].iloc[:min_len].values,
                })
                merged_parts.append(merged_part)

            if merged_parts:
                df_merged = pd.concat(merged_parts, ignore_index=True)
                merged_dfs.append(df_merged)
                total_rows += len(df_merged)

        except Exception as e:
            print(f"  [SKIP] Subject {sid}: {e}")

    print(f"  Total merged rows: {total_rows}")
    print(f"  Subjects successfully loaded: {len(merged_dfs)}")
    return merged_dfs


def preprocess(dataframes, frame_size, step_size):
    """
    Clean, merge activity codes, encode, scale, and window all subject DataFrames.
    Returns X (frames), y (labels), fitted LabelEncoder, fitted StandardScaler,
    and a list of activity names.
    """
    print("\n" + "=" * 60)
    print("  PREPROCESSING DATA")
    print("=" * 60)

    feature_cols = ["accel_X", "accel_Y", "accel_Z", "gyro_X", "gyro_Y", "gyro_Z"]

    # Step 1: Apply activity merging
    print("  Merging activity codes -> 8 classes...")
    for df in dataframes:
        df["Activity"] = df["ActivityCode"].map(ACTIVITY_MERGE)

    # Step 2: Fit label encoder on the merged activity names
    label_encoder = LabelEncoder()
    label_encoder.fit(ACTIVITY_NAMES_SORTED)
    print(f"  Classes: {list(label_encoder.classes_)}")

    # Step 3: Fit a global scaler on ALL data for consistency
    print("  Fitting global StandardScaler on 6 channels...")
    all_features = pd.concat([df[feature_cols] for df in dataframes], ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(all_features)
    print(f"  Scaler fitted on {len(all_features)} total samples across 6 channels.")

    # Step 4: Window each subject
    all_X, all_y = [], []
    for i, df in enumerate(dataframes):
        try:
            df_copy = df.copy()
            df_copy["ActivityLabel"] = label_encoder.transform(df_copy["Activity"])
            df_copy[feature_cols] = scaler.transform(df_copy[feature_cols])

            X, y = _get_frames(df_copy, feature_cols, frame_size, step_size)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            print(f"  [ERROR] Subject {i}: {e}")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    print(f"  Total frames: {X_all.shape[0]}  |  Shape per frame: {X_all.shape[1:]}")

    # Print class distribution
    print("\n  Class distribution:")
    unique, counts = np.unique(y_all, return_counts=True)
    for u, c in zip(unique, counts):
        name = label_encoder.inverse_transform([u])[0]
        print(f"    {name:15s}: {c:5d} frames ({c/len(y_all)*100:.1f}%)")

    return X_all, y_all, label_encoder, scaler


def _get_frames(df, feature_cols, frame_size, step_size):
    """Segment a single DataFrame into fixed-size windows of 6 channels."""
    frames, labels = [], []

    for i in range(0, len(df) - frame_size, step_size):
        window = df[feature_cols].iloc[i: i + frame_size].values  # (200, 6)
        label = stats.mode(
            df["ActivityLabel"].values[i: i + frame_size], keepdims=True
        )[0][0]
        frames.append(window)
        labels.append(label)

    frames = np.asarray(frames)   # (N, 200, 6)
    labels = np.asarray(labels)
    return frames, labels


def create_model(input_shape, num_classes):
    """
    Build a hybrid Conv1D + LSTM model for better time-series accuracy.
    Input shape: (FRAME_SIZE, NUM_CHANNELS) = (200, 6)
    """
    model = Sequential([
        # Feature Extraction: 1D Convolutions
        Conv1D(64, kernel_size=5, activation="relu", padding="same", input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        Conv1D(128, kernel_size=3, activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),

        # Temporal Learning: LSTM
        LSTM(64, return_sequences=False),
        Dropout(0.5),

        # Classifier
        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(model, X_test, y_test, label_encoder, output_dir):
    """
    Evaluate the model on the test set.
    Prints accuracy, classification report, and saves artifacts.
    """
    print("\n" + "=" * 60)
    print("  EVALUATING MODEL")
    print("=" * 60)

    # Predict
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {acc * 100:.2f}%\n")

    # Class names
    target_names = list(label_encoder.classes_)

    # Classification Report
    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0,
    )
    print("  Classification Report:")
    print("  " + "-" * 56)
    for line in report.split("\n"):
        print(f"  {line}")

    # Save report to file
    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Classification Report\n")
        f.write("=" * 56 + "\n")
        f.write(report)
    print(f"\n  Report saved -> {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(f"Confusion Matrix — Hybrid Conv1D+LSTM (Accuracy: {acc*100:.1f}%)", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved -> {cm_path}")

    return acc


def save_artifacts(model, scaler, label_encoder, output_dir):
    """Save model, scaler, encoder, and activity name map."""
    print("\n" + "=" * 60)
    print("  SAVING ARTIFACTS")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Model
    model_path = os.path.join(output_dir, "har_model.keras")
    model.save(model_path)
    print(f"  Model saved          -> {model_path}")

    # Scaler (for 6 channels)
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved         -> {scaler_path}")

    # Label encoder
    encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"  Label encoder saved  -> {encoder_path}")

    # Activity name map: { index: "ClassName" }
    activity_names = {}
    for i, name in enumerate(label_encoder.classes_):
        activity_names[i] = name

    names_path = os.path.join(output_dir, "activity_names.pkl")
    with open(names_path, "wb") as f:
        pickle.dump(activity_names, f)
    print(f"  Activity names saved -> {names_path}")
    print(f"  Classes: {activity_names}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load & merge accel + gyro
    dataframes = load_and_merge_data(ACCEL_PATH, GYRO_PATH)

    # 2. Preprocess (merge classes, scale, window)
    X_all, y_all, label_encoder, scaler = preprocess(
        dataframes, FRAME_SIZE, STEP_SIZE
    )

    # No reshape needed for Conv1D+LSTM, already (samples, 200, 6)
    # X_all = X_all.reshape(X_all.shape[0], FRAME_SIZE, NUM_CHANNELS)

    # 3. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )
    print(f"\n  Train set: {X_train.shape}  |  Test set: {X_test.shape}")

    # 4. Build & train
    print("\n" + "=" * 60)
    print("  TRAINING MODEL")
    print("=" * 60)
    model = create_model(
        input_shape=(FRAME_SIZE, NUM_CHANNELS),
        num_classes=NUM_CLASSES
    )
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    print(f"\n  Training for up to {EPOCHS} epochs (early stopping enabled)...\n")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )

    # 5. Evaluate
    evaluate_model(model, X_test, y_test, label_encoder, OUTPUT_DIR)

    # 6. Save everything
    save_artifacts(model, scaler, label_encoder, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print(f"  Model: {NUM_CLASSES} classes, {NUM_CHANNELS}-channel input")
    print("  Run the app:  python backend/app.py")
    print("=" * 60 + "\n")