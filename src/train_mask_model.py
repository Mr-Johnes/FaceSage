import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_preprocessing import preprocess_mask_data

# ===============================================================
# CONFIGURATION
# ===============================================================
IMG_SIZE = 224
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
MODEL_PATH = "models/mask_detector.h5"

# ===============================================================
# LOAD DATA
# ===============================================================
print("[INFO] Loading and preprocessing mask dataset...")
train_gen, val_gen, test_gen = preprocess_mask_data()

# ===============================================================
# BUILD MODEL
# ===============================================================
print("[INFO] Building MobileNetV2 model...")

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ===============================================================
# TRAIN MODEL
# ===============================================================
print("[INFO] Training model...")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ===============================================================
# EVALUATE MODEL
# ===============================================================
print("[INFO] Evaluating model...")
loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy: {acc*100:.2f}%")

# ===============================================================
# SAVE MODEL
# ===============================================================
model.save(MODEL_PATH)
print(f"[INFO] Model saved to {MODEL_PATH}")
