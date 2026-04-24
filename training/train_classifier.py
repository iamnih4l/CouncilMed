"""
=============================================================================
CouncilMed — Brain Tumor Classifier Training
=============================================================================
Trains a MobileNetV2-based transfer learning classifier on Brain Tumor MRI
images for 4-class classification:
  0: glioma      → Glioblastoma (Grade IV)
  1: meningioma  → Meningioma (Grade I)  
  2: notumor     → Normal
  3: pituitary   → Pituitary / Metastatic Lesion

Architecture: MobileNetV2 (ImageNet pretrained) + Global Avg Pool + Dense(256) + Dense(4, softmax)
Target: >90% validation accuracy
Export: Keras .h5 + TensorFlow.js model.json
=============================================================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# --- Configuration ---
BASE_DIR = Path(__file__).parent / "data"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR   = BASE_DIR / "val"
TEST_DIR  = BASE_DIR / "test"
OUTPUT_DIR = Path(__file__).parent / "output"
MODEL_DIR = OUTPUT_DIR / "classifier"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16  # Small batch for CPU training
EPOCHS = 20
NUM_CLASSES = 4
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

# Maps to CouncilMed's PathologyClass
CLASS_TO_PATHOLOGY = {
    0: "Glioblastoma (Grade IV)",
    1: "Meningioma (Grade I)",
    2: "Normal",
    3: "Metastatic Lesion",
}


def create_data_generators():
    """Create train/val/test data generators with augmentation."""
    print("\n→ Creating data generators...")
    
    # Training augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest',
    )
    
    # Validation/Test: only rescale (no augmentation)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    
    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True,
    )
    
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,
    )
    
    test_gen = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=False,
    )
    
    print(f"  Train: {train_gen.samples} images")
    print(f"  Val:   {val_gen.samples} images")
    print(f"  Test:  {test_gen.samples} images")
    
    return train_gen, val_gen, test_gen


def build_model():
    """Build MobileNetV2 transfer learning model."""
    print("\n→ Building MobileNetV2 classifier...")
    
    # Load MobileNetV2 pretrained on ImageNet, without the top classification layer
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3),
    )
    
    # Freeze the base model initially
    base_model.trainable = False
    
    # Build the classification head
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax', name='classification_output'),
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    model.summary()
    print(f"\n  Total params: {model.count_params():,}")
    print(f"  Trainable params: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    
    return model, base_model


def train_model(model, base_model, train_gen, val_gen):
    """Train in two phases: frozen base → fine-tune top layers."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # --- Phase 1: Train classification head only (base frozen) ---
    print("\n" + "=" * 60)
    print("PHASE 1: Training classification head (base frozen)")
    print("=" * 60)
    
    phase1_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]
    
    history1 = model.fit(
        train_gen,
        epochs=10,
        validation_data=val_gen,
        callbacks=phase1_callbacks,
        verbose=1,
    )
    
    # --- Phase 2: Fine-tune top layers of MobileNetV2 ---
    print("\n" + "=" * 60)
    print("PHASE 2: Fine-tuning top layers of MobileNetV2")
    print("=" * 60)
    
    # Unfreeze the top ~30% of the base model
    base_model.trainable = True
    fine_tune_at = int(len(base_model.layers) * 0.7)
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    
    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Fine-tuning {trainable_count}/{len(base_model.layers)} base layers")
    
    # Recompile with lower learning rate for fine-tuning
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    
    phase2_callbacks = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.ModelCheckpoint(
            str(MODEL_DIR / "best_classifier.keras"),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
    ]
    
    history2 = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=phase2_callbacks,
        verbose=1,
    )
    
    return {**{k: history1.history[k] + history2.history[k] for k in history1.history}}


def evaluate_model(model, test_gen):
    """Evaluate model on test set."""
    print("\n→ Evaluating on test set...")
    results = model.evaluate(test_gen, verbose=1)
    print(f"\n  Test Loss:     {results[0]:.4f}")
    print(f"  Test Accuracy: {results[1] * 100:.2f}%")
    
    # Get predictions for confusion matrix
    from sklearn.metrics import classification_report, confusion_matrix
    
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(
        true_classes, pred_classes,
        target_names=CLASS_NAMES,
        digits=4,
    ))
    
    cm = confusion_matrix(true_classes, pred_classes)
    print("Confusion Matrix:")
    print(cm)
    
    return results[1], cm, pred_classes, true_classes


def save_plots(history, cm, output_dir):
    """Save training history and confusion matrix plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Training accuracy/loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history.get('accuracy', []), label='Train', linewidth=2)
    ax1.plot(history.get('val_accuracy', []), label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history.get('loss', []), label='Train', linewidth=2)
    ax2.plot(history.get('val_loss', []), label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()
    
    # Confusion matrix
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_title('Confusion Matrix — Brain Tumor Classifier', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    
    print(f"\n✓ Plots saved to {output_dir}")


def export_model(model, output_dir):
    """Save model in Keras format for later TF.js conversion."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as Keras format
    keras_path = output_dir / "classifier.keras"
    model.save(keras_path)
    print(f"\n✓ Keras model saved: {keras_path}")
    
    # Save also as SavedModel format for tfjs conversion
    saved_model_path = output_dir / "saved_model"
    model.export(str(saved_model_path))
    print(f"✓ SavedModel exported: {saved_model_path}")
    
    # Save class mapping metadata
    meta = {
        "model_name": "CouncilMed_Classifier_v1",
        "architecture": "MobileNetV2 + Custom Head",
        "input_shape": [224, 224, 3],
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_to_pathology": {str(k): v for k, v in CLASS_TO_PATHOLOGY.items()},
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"✓ Metadata saved: {output_dir / 'model_metadata.json'}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CouncilMed — Brain Tumor Classifier Training")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # 1. Create data generators
    train_gen, val_gen, test_gen = create_data_generators()
    
    # 2. Build model
    model, base_model = build_model()
    
    # 3. Train
    history = train_model(model, base_model, train_gen, val_gen)
    
    # 4. Evaluate
    accuracy, cm, pred_classes, true_classes = evaluate_model(model, test_gen)
    
    # 5. Save plots
    save_plots(history, cm, OUTPUT_DIR)
    
    # 6. Export
    export_model(model, MODEL_DIR)
    
    print("\n" + "=" * 60)
    if accuracy >= 0.90:
        print(f"✓ TARGET MET — Test accuracy: {accuracy * 100:.2f}% (≥ 90%)")
    else:
        print(f"✘ TARGET MISSED — Test accuracy: {accuracy * 100:.2f}% (< 90%)")
        print("  Consider increasing epochs or unfreezing more layers.")
    print("=" * 60)
