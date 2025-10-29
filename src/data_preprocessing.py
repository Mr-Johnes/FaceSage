import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


# ===============================================================
# CONFIGURATION
# ===============================================================
IMG_SIZE = 224  # image size for CNNs
MASK_DATA_DIR = "datasets/maskAndNoMask"
AGE_DATA_DIR = "datasets/faceToAge"

# ===============================================================
# MASK DATA PREPROCESSING
# ===============================================================
def preprocess_mask_data():
    """
    Prepares training and validation data generators for mask detection.
    Assumes directory structure:
        /datasets/Mask/train/mask/
        /datasets/Mask/train/no_mask/
        /datasets/Mask/test/mask/
        /datasets/Mask/test/no_mask/
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        MASK_DATA_DIR + "/train",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        MASK_DATA_DIR + "/train",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    test_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        MASK_DATA_DIR + "/test",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=32,
        class_mode='binary'
    )

    return train_gen, val_gen, test_gen


# ===============================================================
# AGE DATA PREPROCESSING
# ===============================================================
def parse_utkface_filename(filename):
    """
    UTKFace filename format: [age]_[gender]_[race]_[date&time].jpg
    Example: 25_0_0_20170116174525125.jpg.chip.jpg
    """
    try:
        age = int(filename.split("_")[0])
    except:
        return None
    if age <= 12:
        return "Child"
    elif age <= 19:
        return "Teen"
    elif age <= 39:
        return "Adult"
    elif age <= 59:
        return "Middle"
    else:
        return "Senior"

def preprocess_age_data(limit=None):
    """
    Preprocess UTKFace dataset: create dataframe of image paths and labels.
    """
    image_paths = []
    labels = []

    for img_name in os.listdir(AGE_DATA_DIR):
        label = parse_utkface_filename(img_name)
        if label:
            image_paths.append(os.path.join(AGE_DATA_DIR, img_name))
            labels.append(label)
        if limit and len(image_paths) >= limit:
            break

    df = pd.DataFrame({"image": image_paths, "label": labels})
    print(f"Loaded {len(df)} labeled images from UTKFace")

    # Split train-test
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='categorical',
        batch_size=32,
        subset='training'
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        x_col='image',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='categorical',
        batch_size=32,
        subset='validation'
    )

    test_gen = ImageDataGenerator(rescale=1./255).flow_from_dataframe(
        dataframe=test_df,
        x_col='image',
        y_col='label',
        target_size=(IMG_SIZE, IMG_SIZE),
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    return train_gen, val_gen, test_gen



