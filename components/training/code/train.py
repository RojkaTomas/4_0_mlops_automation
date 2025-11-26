import argparse
import os
import json
from datetime import datetime

import tensorflow as tf

from utils import (
    create_datasets,
    build_model,
    compile_model,
    train_model,
    save_model_and_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train animals classifier")

    parser.add_argument(
        "--training_folder",
        type=str,
        required=True,
        help="Path to training data folder (uri_folder mounted by Azure ML).",
    )
    parser.add_argument(
        "--testing_folder",
        type=str,
        required=True,
        help="Path to testing data folder (uri_folder mounted by Azure ML).",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Output folder where the trained model will be saved.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("==== Training parameters ====")
    print(f"Training folder : {args.training_folder}")
    print(f"Testing folder  : {args.testing_folder}")
    print(f"Output folder   : {args.output_folder}")
    print(f"Epochs          : {args.epochs}")

    # Ensure output directory exists
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 1. Create datasets from folders
    train_ds, test_ds, class_names = create_datasets(
        train_dir=args.training_folder,
        test_dir=args.testing_folder,
        img_size=(64, 64),
        batch_size=32,
    )

    num_classes = len(class_names)
    print(f"Detected classes ({num_classes}): {class_names}")

    # 2. Build & compile model
    model = build_model(input_shape=(64, 64, 3), num_classes=num_classes)
    compile_model(model, learning_rate=1e-3)

    # 3. Train model
    history = train_model(
        model,
        train_ds=train_ds,
        test_ds=test_ds,
        epochs=args.epochs,
    )

    # 4. Save model + metadata into output_folder
    save_model_and_metadata(
        model=model,
        output_folder=args.output_folder,
        class_names=class_names,
        history=history,
    )

    print("Training finished successfully.")
    print(f"Model artifacts written to: {args.output_folder}")


if __name__ == "__main__":
    # Enable memory growth for GPU if present (avoids some CUDA issues)
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:  # noqa: BLE001
            print(f"Could not set memory growth on GPU {gpu}: {e}")

    main()
