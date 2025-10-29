import os
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from data_preprocessing import preprocess_age_data

# ===============================================================
# CONFIGURATION
# ===============================================================
IMG_SIZE = 224
INITIAL_EPOCHS = 20      # Phase 1: train top layers
FINE_TUNE_EPOCHS = 10    # Phase 2: fine-tune backbone
LEARNING_RATE = 1e-4
FINE_TUNE_LR = 1e-5
BATCH_SIZE = 32
MODEL_PATH = "models/age_classifier_v2.h5"

# ===============================================================
# LOAD DATA
# ===============================================================
print("[INFO] Loading and preprocessing UTKFace dataset...")
train_gen, val_gen, test_gen = preprocess_age_data(limit=None)

# Compute class weights to handle imbalance
class_labels = list(train_gen.class_indices.keys())
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
class_weights = dict(enumerate(class_weights))
print(f"[INFO] Computed class weights: {class_weights}")

# ===============================================================
# BUILD MODEL
# ===============================================================
print("[INFO] Building ResNet50 base model...")

base_model = ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze all layers initially
for layer in base_model.layers:
    layer.trainable = False

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(5, activation="softmax")(x)  # 5 age groups

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ===============================================================
# TRAIN (PHASE 1: TRAIN TOP LAYERS)
# ===============================================================
print("[INFO] Training top layers (frozen backbone)...")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
]

history_1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=INITIAL_EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# ===============================================================
# FINE-TUNING (PHASE 2: UNFREEZE TOP LAYERS)
# ===============================================================
print("[INFO] Fine-tuning top layers of ResNet50...")

# Unfreeze top 30 layers for fine-tuning
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=FINE_TUNE_LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_ft = [
    EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
]

history_2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=FINE_TUNE_EPOCHS,
    callbacks=callbacks_ft,
    class_weight=class_weights
)

# ===============================================================
# EVALUATE MODEL
# ===============================================================
print("[INFO] Evaluating fine-tuned model...")
loss, acc = model.evaluate(test_gen)
print(f"✅ Test Accuracy after fine-tuning: {acc*100:.2f}%")

# ===============================================================
# SAVE MODEL
# ===============================================================
model.save(MODEL_PATH)
print(f"[INFO] Final model saved to {MODEL_PATH}")
