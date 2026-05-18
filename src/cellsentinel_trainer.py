"""
CellSentinel — Fault Classifier Trainer v2
Lightweight CNN trained from scratch — no MySQL dependency.
Labels generated directly from .mat files.

Run  : python src/cellsentinel_trainer.py
Saves: models/best_cellsentinel_v2.h5
       results/confusion_matrix.png
       results/roc_curve.png
       results/training_history.png
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_curve, auc)

import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping,
                                        ReduceLROnPlateau)

# ── Path setup ────────────────────────────────────────────────
SRC_DIR     = os.path.dirname(os.path.abspath(__file__))
BASE_DIR    = os.path.dirname(SRC_DIR)
DATA_DIR    = os.path.join(BASE_DIR, 'data')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
CACHE_PATH  = os.path.join(DATA_DIR, 'X_cache_v2.pkl')
MODEL_PATH  = os.path.join(MODELS_DIR, 'best_cellsentinel_v2.h5')

os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────
IMG_H      = 150
IMG_W      = 150
CHANNELS   = 3
N_CLASSES  = 3
BATCH_SIZE = 16     # smaller batch — better gradient estimates
EPOCHS     = 80     # train longer
SEED       = 42

BATTERIES  = ['B0005', 'B0006', 'B0007', 'B0018']

# ── Fault thresholds — tighter for better class balance ───────
SOH_S0 = 90.0   # >= 90% → Normal
SOH_S1 = 75.0   # 75–90% → Warning
                 # <  75% → Fault

# ── GPU ───────────────────────────────────────────────────────
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"  GPU detected: {gpus[0].name}")
    else:
        print("  No GPU — using CPU")

# ── Fault label from SOH ──────────────────────────────────────
def get_fault_label(soh):
    if soh >= SOH_S0: return 0   # S0 Normal
    if soh >= SOH_S1: return 1   # S1 Warning
    return 2                      # S2 Fault

# ── Convert discharge cycle → (H, W, 3) image ────────────────
def cycle_to_image(data, h=IMG_H, w=IMG_W):
    def norm_resize(sig, n):
        sig = np.array(sig, dtype=np.float32)
        mn, mx = sig.min(), sig.max()
        if mx - mn > 1e-8:
            sig = (sig - mn) / (mx - mn)
        else:
            sig = np.zeros_like(sig)
        idx = np.linspace(0, len(sig)-1, n)
        return np.interp(idx, np.arange(len(sig)), sig)

    V = norm_resize(data['Voltage_measured'],     h * w).reshape(h, w)
    I = norm_resize(data['Current_measured'],     h * w).reshape(h, w)
    T = norm_resize(data['Temperature_measured'], h * w).reshape(h, w)
    return np.stack([V, I, T], axis=-1).astype(np.float32)

# ── Load .mat and build dataset ───────────────────────────────
def build_or_load_dataset():
    if os.path.exists(CACHE_PATH):
        print(f"  Loading cache: {CACHE_PATH}")
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)

    print("  Building images from .mat files...")
    X, y, batteries = [], [], []

    for bat in BATTERIES:
        path   = os.path.join(DATA_DIR, f'{bat}.mat')
        mat    = scipy.io.loadmat(path, simplify_cells=True)
        cycles = mat[bat]['cycle']
        discharge = [c for c in cycles if c['type'] == 'discharge']
        capacities = [float(c['data']['Capacity']) for c in discharge]
        initial    = capacities[0]

        for i, c in enumerate(discharge):
            soh   = capacities[i] / initial * 100
            label = get_fault_label(soh)
            img   = cycle_to_image(c['data'])
            X.append(img)
            y.append(label)
            batteries.append(bat)

        # Print per-battery stats
        labels = [get_fault_label(q/initial*100) for q in capacities]
        print(f"    {bat}: {len(discharge)} cycles  "
              f"S0={labels.count(0)}  S1={labels.count(1)}  S2={labels.count(2)}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    batteries = np.array(batteries)

    with open(CACHE_PATH, 'wb') as f:
        pickle.dump((X, y, batteries), f)
    print(f"  Cached: {CACHE_PATH}")
    return X, y, batteries

# ── Lightweight CNN (better than ResNet50 for small datasets) ─
def build_model():
    inp = Input(shape=(IMG_H, IMG_W, CHANNELS), name='input')

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding='same', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    # Head
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(N_CLASSES, activation='softmax', name='output')(x)

    return Model(inp, out, name='CellSentinel_CNN')

# ── 4-panel training history plot ────────────────────────────
def plot_history(history):
    train_acc  = history.history['accuracy']
    val_acc    = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    epochs     = range(1, len(train_acc) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('CellSentinel — Training History\n'
                 'Lightweight CNN · NASA Battery Fault Detection',
                 fontsize=13, fontweight='bold')

    style = dict(linewidth=2, marker='o', markersize=3)

    # Training Accuracy
    ax = axes[0, 0]
    ax.plot(epochs, train_acc, color='#1a7abf',
            label='Training accuracy', **style)
    best_ep = int(np.argmax(train_acc)) + 1
    ax.set_title('Training Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)
    ax.annotate(f"Final: {train_acc[-1]:.3f}",
                xy=(len(epochs), train_acc[-1]),
                xytext=(-50, 10), textcoords='offset points',
                fontsize=9, color='#1a7abf')

    # Validation Accuracy
    ax = axes[0, 1]
    ax.plot(epochs, val_acc, color='#27ae60',
            label='Validation accuracy', linestyle='--', **style)
    best_ep  = int(np.argmax(val_acc)) + 1
    best_val = max(val_acc)
    ax.scatter([best_ep], [best_val], color='#27ae60', s=80, zorder=5)
    ax.annotate(f"Best: {best_val:.3f} (ep {best_ep})",
                xy=(best_ep, best_val),
                xytext=(8, -15), textcoords='offset points',
                fontsize=9, color='#27ae60')
    ax.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)

    # Training Loss
    ax = axes[1, 0]
    ax.plot(epochs, train_loss, color='#c0392b',
            label='Training loss', **style)
    ax.set_title('Training Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.annotate(f"Final: {train_loss[-1]:.3f}",
                xy=(len(epochs), train_loss[-1]),
                xytext=(-50, 10), textcoords='offset points',
                fontsize=9, color='#c0392b')

    # Validation Loss
    ax = axes[1, 1]
    ax.plot(epochs, val_loss, color='#e67e22',
            label='Validation loss', linestyle='--', **style)
    best_loss_ep  = int(np.argmin(val_loss)) + 1
    best_loss_val = min(val_loss)
    ax.scatter([best_loss_ep], [best_loss_val],
               color='#e67e22', s=80, zorder=5)
    ax.annotate(f"Best: {best_loss_val:.3f} (ep {best_loss_ep})",
                xy=(best_loss_ep, best_loss_val),
                xytext=(8, 8), textcoords='offset points',
                fontsize=9, color='#e67e22')
    ax.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'training_history.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

# ── Confusion matrix ──────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred):
    cm     = confusion_matrix(y_true, y_pred)
    labels = ['S0 Normal', 'S1 Warning', 'S2 Fault']
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title('CellSentinel — Confusion Matrix')
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

# ── ROC curves ────────────────────────────────────────────────
def plot_roc(y_true, y_prob):
    colors = ['#1a7abf', '#27ae60', '#c0392b']
    labels = ['S0 Normal', 'S1 Warning', 'S2 Fault']
    y_bin  = tf.keras.utils.to_categorical(y_true, N_CLASSES)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(N_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f'{labels[i]} (AUC={auc(fpr,tpr):.3f})')
    ax.plot([0,1],[0,1],'k--',lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('CellSentinel — ROC Curves')
    ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'roc_curve.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

# ── Main ──────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🔋 CellSentinel Trainer v2")
    print("=" * 50)
    setup_gpu()

    # ── Build dataset ─────────────────────────────────────────
    print("\n🖼️  Preparing images...")
    X, y, batteries = build_or_load_dataset()
    print(f"\n  Total  : {len(X)} samples")
    print(f"  S0     : {np.sum(y==0)}  S1: {np.sum(y==1)}  S2: {np.sum(y==2)}")

    # ── Split — stratified, use all batteries ─────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
    )

    print(f"\n  Train  : {len(X_train)} samples")
    print(f"  Val    : {len(X_val)} samples")
    print(f"  Test   : {len(X_test)} samples")

    # ── Manual class weights — heavily penalize S2 misses ─────
    class_weights = {
        0: 1.0,   # S0 Normal   — most common
        1: 2.0,   # S1 Warning  — moderate
        2: 6.0,   # S2 Fault    — rare but critical
    }
    print(f"\n  Class weights: {class_weights}")

    # ── Build model ───────────────────────────────────────────
    print("\n🏗️  Building lightweight CNN...")
    model = build_model()
    model.summary(line_length=70)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=12, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=20,
                      restore_best_weights=True, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────
    print(f"\n🚀 Training ({EPOCHS} epochs, batch={BATCH_SIZE})...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # ── Evaluate ──────────────────────────────────────────────
    print("\n📈 Evaluating on test set...")
    y_prob = model.predict(X_test, batch_size=BATCH_SIZE)
    y_pred = np.argmax(y_prob, axis=1)

    print("\n── Classification Report ────────────────────")
    print(classification_report(
        y_test, y_pred,
        target_names=['S0 Normal', 'S1 Warning', 'S2 Fault'],
        zero_division=0
    ))

    # ── Save plots ────────────────────────────────────────────
    print("\n💾 Saving results...")
    plot_history(history)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc(y_test, y_prob)

    print(f"\n  Model  : {MODEL_PATH}")
    print(f"  Results: {RESULTS_DIR}")
    print("\n🔋 CellSentinel training complete!\n")
