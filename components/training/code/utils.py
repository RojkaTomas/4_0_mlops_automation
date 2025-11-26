import os
import json
from datetime import datetime

import tensorflow as tf


def create_datasets(train_dir, test_dir, img_size=(64, 64), batch_size=32):
    img_height, img_width = img_size

    # DEBUG: show what Azure mounted
    print("=== DEBUG: listing training folder ===")
    for root, dirs, files in os.walk(train_dir):
        print(f"[TRAIN] {root} | dirs={dirs} | files={files[:5]}")
    print("=== DEBUG: listing testing folder ===")
    for root, dirs, files in os.walk(test_dir):
        print(f"[TEST]  {root} | dirs={dirs} | files={files[:5]}")

    # (existing code below)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=True,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        image_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    return train_ds, test_ds, class_names



def build_model(input_shape=(64, 64, 3), num_classes=3):
    """
    Simple CNN for small 64x64 images.
    """
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255, input_shape=input_shape),
            tf.keras.layers.Conv2D(32, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    return model


def compile_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(model.summary())


def train_model(model, train_ds, test_ds, epochs=5):
    """
    Trains the model and returns the History object.
    """
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=callbacks,
    )
    return history


def save_model_and_metadata(model, output_folder, class_names, history):
    """
    Saves:
      - Keras model to output_folder/model.keras
      - class names to output_folder/class_indices.json
      - simple training metrics to output_folder/metrics.json

    The pipeline's `register` step uses this folder as `model_path`.
    """
    os.makedirs(output_folder, exist_ok=True)

    # 1. Save model
    model_path = os.path.join(output_folder, "model.keras")
    print(f"Saving model to {model_path}")
    model.save(model_path)

    # 2. Save class names
    class_file = os.path.join(output_folder, "class_indices.json")
    with open(class_file, "w", encoding="utf-8") as f:
        json.dump({"class_names": list(class_names)}, f, indent=2)
    print(f"Saved class names to {class_file}")

    # 3. Save basic metrics
    metrics = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
    }
    metrics_file = os.path.join(output_folder, "metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved training metrics to {metrics_file}")
