"""
Train CNN on phone accelerometer data from WISDM dataset.
Run from Minor_HAR folder: python backend/train_model.py
Saves: backend/har_model.keras, scaler.pkl, label_encoder.pkl,
       activity_names.pkl, confusion_matrix.png, report.txt
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
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam

# ── Config ────────────────────────────────────────────────────────────────────
BASE_PATH       = "code/data/raw_data/phone/accel/"
OUTPUT_DIR      = "backend"
FREQUENCY       = 20        # Hz
TIME_PERIOD     = 10        # seconds
FRAME_SIZE      = FREQUENCY * TIME_PERIOD   # 200 samples per window
STEP_SIZE       = FRAME_SIZE                # no overlap
NUM_CLASSES     = 18
EPOCHS          = 15
BATCH_SIZE      = 128
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

ACTIVITY_MAP = {
    "A": "Walking",        "B": "Jogging",       "C": "Stairs",
    "D": "Sitting",        "E": "Standing",      "F": "Typing",
    "G": "Brushing Teeth", "H": "Eating Soup",   "I": "Eating Chips",
    "J": "Eating Pasta",   "K": "Drinking",      "L": "Eating Sandwich",
    "M": "Kicking",        "O": "Playing Catch",  "P": "Dribbling",
    "Q": "Writing",        "R": "Clapping",      "S": "Folding Clothes"
}


# ── Modular Functions ─────────────────────────────────────────────────────────

def load_data(base_path):
    """
    Load all CSV files from the given folder.
    Returns a list of DataFrames and a sorted list of unique activity codes.
    """
    print("=" * 60)
    print("  LOADING DATA")
    print("=" * 60)

    csv_files = [
        os.path.join(base_path, f)
        for f in os.listdir(base_path)
        if f.endswith(".csv")
    ]
    print(f"  Found {len(csv_files)} subject CSV files.")

    dataframes = []
    all_codes = []

    for fpath in csv_files:
        try:
            df = pd.read_csv(fpath)
            all_codes.extend(df["ActivityCode"].unique().tolist())
            dataframes.append(df)
        except Exception as e:
            print(f"  [SKIP] {os.path.basename(fpath)}: {e}")

    unique_codes = sorted(set(all_codes))
    print(f"  Unique activity codes: {unique_codes}")
    print(f"  Total subjects loaded: {len(dataframes)}")
    return dataframes, unique_codes


def preprocess(dataframes, unique_codes, frame_size, step_size):
    """
    Clean, encode, scale, and window all subject DataFrames.
    Returns X (frames), y (labels), fitted LabelEncoder, and fitted StandardScaler.
    """
    print("\n" + "=" * 60)
    print("  PREPROCESSING DATA")
    print("=" * 60)

    # Fit encoder on all unique codes
    label_encoder = LabelEncoder()
    label_encoder.fit(unique_codes)

    # Fit a global scaler on ALL data first for consistency
    print("  Fitting global StandardScaler...")
    all_features = []
    for df in dataframes:
        cleaned = df.drop(["SubjectID", "Timestamp"], axis=1, errors="ignore").copy()
        all_features.append(cleaned[["X", "Y", "Z"]])

    global_features = pd.concat(all_features, ignore_index=True)
    scaler = StandardScaler()
    scaler.fit(global_features)
    print(f"  Scaler fitted on {len(global_features)} total samples.")

    # Window each subject
    all_X, all_y = [], []
    for i, df in enumerate(dataframes):
        try:
            cleaned = df.drop(["SubjectID", "Timestamp"], axis=1, errors="ignore").copy()
            cleaned["ActivityCode"] = label_encoder.transform(cleaned["ActivityCode"])
            cleaned[["X", "Y", "Z"]] = scaler.transform(cleaned[["X", "Y", "Z"]])

            X, y = _get_frames(cleaned, frame_size, step_size)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
        except Exception as e:
            print(f"  [ERROR] Subject {i}: {e}")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    print(f"  Total frames: {X_all.shape[0]}  |  Shape per frame: {X_all.shape[1:]}")

    return X_all, y_all, label_encoder, scaler


def _get_frames(df, frame_size, step_size):
    """Segment a single DataFrame into fixed-size windows."""
    frames, labels = [], []
    for i in range(0, len(df) - frame_size, step_size):
        x = df["X"].values[i: i + frame_size]
        y = df["Y"].values[i: i + frame_size]
        z = df["Z"].values[i: i + frame_size]
        label = stats.mode(
            df["ActivityCode"].values[i: i + frame_size], keepdims=True
        )[0][0]
        frames.append([x, y, z])
        labels.append(label)

    frames = np.asarray(frames).reshape(-1, frame_size, 3)
    labels = np.asarray(labels)
    return frames, labels


def create_model(input_shape, num_classes):
    """Build and compile the CNN model."""
    model = Sequential([
        Conv2D(64, (2, 2), activation="relu", input_shape=input_shape),
        MaxPooling2D(pool_size=2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_model(model, X_test, y_test, label_encoder, unique_codes, output_dir):
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

    # Class names in encoded order
    target_names = [
        ACTIVITY_MAP.get(code, code)
        for code in unique_codes
    ]

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
    print(f"\n  Report saved → {report_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title("Confusion Matrix — HAR CNN", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {cm_path}")

    return acc


def save_artifacts(model, scaler, label_encoder, unique_codes, output_dir):
    """Save model, scaler, encoder, and activity name map."""
    print("\n" + "=" * 60)
    print("  SAVING ARTIFACTS")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Model
    model_path = os.path.join(output_dir, "har_model.keras")
    model.save(model_path)
    print(f"  Model saved          → {model_path}")

    # Scaler
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved         → {scaler_path}")

    # Label encoder
    encoder_path = os.path.join(output_dir, "label_encoder.pkl")
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)
    print(f"  Label encoder saved  → {encoder_path}")

    # Activity name map
    activity_names = {}
    for code in unique_codes:
        idx = int(label_encoder.transform([code])[0])
        activity_names[idx] = ACTIVITY_MAP.get(code, code)

    names_path = os.path.join(output_dir, "activity_names.pkl")
    with open(names_path, "wb") as f:
        pickle.dump(activity_names, f)
    print(f"  Activity names saved → {names_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load
    dataframes, unique_codes = load_data(BASE_PATH)

    # 2. Preprocess
    X_all, y_all, label_encoder, scaler = preprocess(
        dataframes, unique_codes, FRAME_SIZE, STEP_SIZE
    )

    # Reshape to 4D for Conv2D: (samples, timesteps, 3, 1)
    X_all = X_all.reshape(X_all.shape[0], FRAME_SIZE, 3, 1)

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
    model = create_model(input_shape=(FRAME_SIZE, 3, 1), num_classes=NUM_CLASSES)
    model.summary()

    print(f"\n  Training for {EPOCHS} epochs...\n")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
        verbose=1,
    )

    # 5. Evaluate
    evaluate_model(model, X_test, y_test, label_encoder, unique_codes, OUTPUT_DIR)

    # 6. Save everything
    save_artifacts(model, scaler, label_encoder, unique_codes, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("  ALL DONE!")
    print("  Run the app:  python backend/app.py")
    print("=" * 60 + "\n")