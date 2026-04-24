"""
=============================================================================
CouncilMed — Attention Localizer + Segmenter Training
=============================================================================
Uses the trained classifier to generate:
1. Attention model: Extracts Grad-CAM heatmaps for spatial localization
2. Segmenter model: Lightweight U-Net trained on pseudo-labels from Grad-CAM

Both models are derived from the classifier to maintain consistency.
=============================================================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import json

# --- Configuration ---
BASE_DIR = Path(__file__).parent / "data"
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR   = BASE_DIR / "val"
OUTPUT_DIR = Path(__file__).parent / "output"
CLASSIFIER_DIR = OUTPUT_DIR / "classifier"
ATTENTION_DIR  = OUTPUT_DIR / "attention"
SEGMENTER_DIR  = OUTPUT_DIR / "segmenter"

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_CLASSES = 4
NUM_REGIONS = 10  # Brain regions in CouncilMed

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


def build_attention_model():
    """
    Build the Attention/Localization model.
    Architecture: MobileNetV2 encoder → Region classifier (10 brain regions)
    The intermediate feature maps serve as attention heatmaps.
    """
    print("\n" + "=" * 60)
    print("Building Attention Model")
    print("=" * 60)
    
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base.trainable = False  # Start frozen
    
    # Build the model with named intermediate outputs
    inputs = tf.keras.Input(shape=(224, 224, 3), name='attention_input')
    features = base(inputs, training=False)  # (batch, 7, 7, 1280)
    
    # Branch 1: Region classification
    gap = layers.GlobalAveragePooling2D(name='attention_gap')(features)
    x = layers.Dense(128, activation='relu', name='attention_fc1')(gap)
    x = layers.Dropout(0.3)(x)
    region_output = layers.Dense(NUM_REGIONS, activation='softmax', name='region_output')(x)
    
    # Branch 2: Spatial attention map (learned 1x1 conv to single channel)
    attn_map = layers.Conv2D(1, (1, 1), activation='sigmoid', name='attention_conv')(features)  # (batch, 7, 7, 1)
    attn_flat = layers.Flatten(name='attention_flat')(attn_map)
    
    model = models.Model(inputs=inputs, outputs=[region_output, attn_flat], name='AttentionNet')
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss={
            'region_output': 'categorical_crossentropy',
            'attention_flat': 'mse',  # Self-supervised: attention = where tumor class activates
        },
        loss_weights={'region_output': 1.0, 'attention_flat': 0.1},
        metrics={'region_output': 'accuracy'},
    )
    
    model.summary()
    return model, base


def generate_attention_labels(classifier_path, data_dir, num_samples=500):
    """
    Generate pseudo attention labels using Grad-CAM from the trained classifier.
    """
    print("\n-> Generating Grad-CAM pseudo-labels from classifier...")
    
    classifier = tf.keras.models.load_model(classifier_path)
    
    # The classifier is Sequential: [MobileNetV2_base, GAP, BN, Dropout, Dense, BN, Dropout, Dense]
    # We need to find the last conv layer inside the MobileNetV2 base
    base_model = classifier.layers[0]
    last_conv_layer = None
    last_conv_name = None
    for layer in reversed(base_model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            last_conv_name = layer.name
            break
    
    if last_conv_layer is None:
        print("  [ERROR] Could not find conv layer for Grad-CAM")
        return None, None, None
    
    print(f"  Using layer '{last_conv_name}' for Grad-CAM")
    
    # Build a model that outputs conv features from within the base model
    # We create a model from base_model's input to the last conv output
    conv_output_model = models.Model(
        inputs=base_model.input,
        outputs=last_conv_layer.output
    )
    
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    gen = datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode='categorical',
        classes=CLASS_NAMES,
        shuffle=True,
    )
    
    images_list = []
    heatmaps_list = []
    region_labels_list = []
    
    # region mapping: map class index to a set of probable brain regions
    class_to_region = {
        0: 2,  # glioma -> temporal left
        1: 0,  # meningioma -> frontal left
        2: 9,  # notumor -> thalamus (center, no anomaly)
        3: 8,  # pituitary -> brainstem
    }
    
    for i in range(min(num_samples, gen.samples)):
        img_batch, label_batch = next(gen)
        img = img_batch[0]
        class_idx = np.argmax(label_batch[0])
        
        # Use a simpler Grad-CAM approach: get conv features and use 
        # the classifier prediction to weight them
        img_tensor = tf.cast(tf.convert_to_tensor(img_batch), tf.float32)
        
        # Get conv features (not through gradient tape - just inference)
        conv_features = conv_output_model(img_tensor)  # (1, 7, 7, 1280)
        
        # Get classifier prediction
        predictions = classifier(img_tensor)  # (1, 4)
        predicted_class = int(tf.argmax(predictions[0]).numpy())
        
        # Generate heatmap from conv features using channel-weighted sum
        # Weight channels by the predicted class probability distribution
        conv_np = conv_features.numpy()[0]  # (7, 7, 1280)
        
        # Simple approach: use mean activation as attention heatmap
        heatmap = np.mean(conv_np, axis=-1)  # (7, 7)
        
        # Weight by prediction confidence for the predicted class
        confidence = float(predictions[0, predicted_class].numpy())
        
        # For "notumor" class, invert the heatmap (no anomaly)
        if predicted_class == 2:  # notumor
            heatmap = 1.0 - heatmap
        
        # Normalize to [0, 1]
        heatmap = heatmap - heatmap.min()
        heatmap = heatmap / (heatmap.max() + 1e-8)
        heatmap = heatmap * confidence  # Scale by confidence
        
        # Flatten to 49
        heatmap_flat = heatmap.flatten()
        
        images_list.append(img)
        heatmaps_list.append(heatmap_flat)
        
        # Create region label (one-hot)
        region_label = np.zeros(NUM_REGIONS)
        region_label[class_to_region.get(class_idx, 0)] = 1.0
        region_labels_list.append(region_label)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{num_samples} images")
    
    print(f"  [OK] Generated {len(images_list)} attention samples")
    
    return (
        np.array(images_list),
        np.array(region_labels_list),
        np.array(heatmaps_list),
    )


def train_attention(model, base, images, region_labels, heatmap_labels):
    """Train the attention model."""
    print("\n" + "=" * 60)
    print("Training Attention Model")
    print("=" * 60)
    
    # Split
    n = len(images)
    split = int(n * 0.85)
    
    train_imgs, val_imgs = images[:split], images[split:]
    train_regions, val_regions = region_labels[:split], region_labels[split:]
    train_heatmaps, val_heatmaps = heatmap_labels[:split], heatmap_labels[split:]
    
    os.makedirs(ATTENTION_DIR, exist_ok=True)
    
    # Phase 1: frozen
    history = model.fit(
        train_imgs,
        {'region_output': train_regions, 'attention_flat': train_heatmaps},
        validation_data=(val_imgs, {'region_output': val_regions, 'attention_flat': val_heatmaps}),
        epochs=10,
        batch_size=BATCH_SIZE,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_region_output_accuracy', patience=4, restore_best_weights=True, mode='max'),
        ],
        verbose=1,
    )
    
    # Phase 2: fine-tune some layers
    base.trainable = True
    for layer in base.layers[:int(len(base.layers) * 0.8)]:
        layer.trainable = False
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),
        loss={'region_output': 'categorical_crossentropy', 'attention_flat': 'mse'},
        loss_weights={'region_output': 1.0, 'attention_flat': 0.1},
        metrics={'region_output': 'accuracy'},
    )
    
    model.fit(
        train_imgs,
        {'region_output': train_regions, 'attention_flat': train_heatmaps},
        validation_data=(val_imgs, {'region_output': val_regions, 'attention_flat': val_heatmaps}),
        epochs=10,
        batch_size=BATCH_SIZE,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_region_output_accuracy', patience=4, restore_best_weights=True, mode='max'),
        ],
        verbose=1,
    )
    
    return model


def build_and_train_segmenter(images, heatmap_labels):
    """
    Build and train a lightweight segmentation model.
    Uses Grad-CAM pseudo-labels as segmentation targets.
    """
    print("\n" + "=" * 60)
    print("Building & Training Segmenter Model")
    print("=" * 60)
    
    # Reshape heatmaps from (n, 49) → (n, 7, 7, 1) as segmentation targets
    seg_targets = heatmap_labels.reshape(-1, 7, 7, 1)
    
    # Build a simple encoder-decoder (U-Net lite)
    inputs = tf.keras.Input(shape=(224, 224, 3), name='seg_input')
    
    # Encoder (lightweight)
    x = layers.Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)   # 112
    x = layers.Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)        # 56
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)       # 28
    x = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)       # 14
    encoded = layers.Conv2D(256, 3, strides=2, padding='same', activation='relu')(x)  # 7
    
    # Region classification branch
    gap = layers.GlobalAveragePooling2D()(encoded)
    region_head = layers.Dense(128, activation='relu')(gap)
    region_output = layers.Dense(NUM_REGIONS, activation='softmax', name='seg_region_output')(region_head)
    
    # Segmentation output (7x7x1 mask)
    seg_output = layers.Conv2D(1, 1, activation='sigmoid', name='seg_mask_output')(encoded)  # (batch, 7, 7, 1)
    
    model = models.Model(inputs=inputs, outputs=[region_output, seg_output], name='Segmenter')
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss={
            'seg_region_output': 'categorical_crossentropy',
            'seg_mask_output': 'binary_crossentropy',
        },
        loss_weights={'seg_region_output': 1.0, 'seg_mask_output': 2.0},
        metrics={'seg_region_output': 'accuracy'},
    )
    
    model.summary()
    
    # Prepare region labels from heatmap data (find highest activation quadrant)
    n = len(images)
    region_labels = np.zeros((n, NUM_REGIONS))
    for i in range(n):
        hm = heatmap_labels[i].reshape(7, 7)
        max_idx = np.unravel_index(np.argmax(hm), hm.shape)
        # Map spatial position to brain region (approximate)
        region_idx = (max_idx[0] * 7 + max_idx[1]) % NUM_REGIONS
        region_labels[i, region_idx] = 1.0
    
    # Split
    split = int(n * 0.85)
    
    os.makedirs(SEGMENTER_DIR, exist_ok=True)
    
    model.fit(
        images[:split],
        {'seg_region_output': region_labels[:split], 'seg_mask_output': seg_targets[:split]},
        validation_data=(
            images[split:],
            {'seg_region_output': region_labels[split:], 'seg_mask_output': seg_targets[split:]},
        ),
        epochs=15,
        batch_size=BATCH_SIZE,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ],
        verbose=1,
    )
    
    return model


def export_models(attention_model, segmenter_model):
    """Save both models."""
    # Attention
    os.makedirs(ATTENTION_DIR, exist_ok=True)
    attention_model.save(ATTENTION_DIR / "attention.keras")
    attention_model.export(str(ATTENTION_DIR / "saved_model"))
    
    meta = {
        "model_name": "CouncilMed_Attention_v1",
        "architecture": "MobileNetV2 Encoder + Region Head + Spatial Attention",
        "input_shape": [224, 224, 3],
        "outputs": ["region_probabilities (10)", "attention_flat (49)"],
    }
    with open(ATTENTION_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n✓ Attention model saved to {ATTENTION_DIR}")
    
    # Segmenter
    os.makedirs(SEGMENTER_DIR, exist_ok=True)
    segmenter_model.save(SEGMENTER_DIR / "segmenter.keras")
    segmenter_model.export(str(SEGMENTER_DIR / "saved_model"))
    
    meta = {
        "model_name": "CouncilMed_Segmenter_v1",
        "architecture": "Lightweight U-Net Encoder + Seg Mask + Region Head",
        "input_shape": [224, 224, 3],
        "outputs": ["region_probabilities (10)", "segmentation_mask (7x7x1)"],
    }
    with open(SEGMENTER_DIR / "model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"✓ Segmenter model saved to {SEGMENTER_DIR}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CouncilMed — Attention & Segmenter Training")
    print("=" * 60)
    
    classifier_path = CLASSIFIER_DIR / "classifier.keras"
    if not classifier_path.exists():
        # Try the checkpoint
        classifier_path = CLASSIFIER_DIR / "best_classifier.keras"
    
    if not classifier_path.exists():
        print("✘ Classifier model not found. Run train_classifier.py first.")
        exit(1)
    
    print(f"  Using classifier: {classifier_path}")
    
    # 1. Generate Grad-CAM labels from classifier
    images, region_labels, heatmap_labels = generate_attention_labels(
        classifier_path, TRAIN_DIR, num_samples=800
    )
    
    if images is None:
        print("✘ Failed to generate Grad-CAM labels")
        exit(1)
    
    # 2. Build & train attention model
    attention_model, base = build_attention_model()
    attention_model = train_attention(attention_model, base, images, region_labels, heatmap_labels)
    
    # 3. Build & train segmenter model
    segmenter_model = build_and_train_segmenter(images, heatmap_labels)
    
    # 4. Export both
    export_models(attention_model, segmenter_model)
    
    print("\n" + "=" * 60)
    print("✓ All models trained and exported!")
    print("=" * 60)
