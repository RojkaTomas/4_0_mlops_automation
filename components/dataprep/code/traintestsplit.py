import argparse
import os
import random
import shutil
from typing import List

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train/test split for 3 image datasets")

    parser.add_argument(
        "--datasets",
        nargs=3,
        required=True,
        help="Paths to the three preprocessed datasets (uri_folder inputs).",
    )
    parser.add_argument(
        "--split_size",
        type=float,
        required=True,
        help="Percentage of data to use for TEST set (e.g. 20 -> 80/20 split).",
    )
    parser.add_argument(
        "--training_data_output",
        type=str,
        required=True,
        help="Output folder for training data (uri_folder).",
    )
    parser.add_argument(
        "--testing_data_output",
        type=str,
        required=True,
        help="Output folder for testing data (uri_folder).",
    )
    return parser.parse_args()


def is_image_file(filename: str) -> bool:
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTS


def collect_images(root: str) -> List[str]:
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if is_image_file(fname):
                files.append(os.path.join(dirpath, fname))
    return files


def generate_dummy_images(train_dir: str, test_dir: str, n_train=40, n_test=20, color=(255, 0, 0)):
    """Fallback: create simple coloured 64x64 JPGs if no real images are visible."""
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f"[FALLBACK] Generating {n_train} train + {n_test} test dummy images in:")
    print(f"          train_dir={train_dir}")
    print(f"          test_dir={test_dir}")

    for i in range(n_train):
        img = Image.new("RGB", (64, 64), color=color)
        img.save(os.path.join(train_dir, f"dummy_train_{i:03d}.jpg"), format="JPEG", quality=90)

    for i in range(n_test):
        img = Image.new("RGB", (64, 64), color=color)
        img.save(os.path.join(test_dir, f"dummy_test_{i:03d}.jpg"), format="JPEG", quality=90)


def main():
    args = parse_args()

    print("=== data_split parameters ===")
    print(f"Datasets       : {args.datasets}")
    print(f"Test percentage: {args.split_size}")
    print(f"Train out      : {args.training_data_output}")
    print(f"Test out       : {args.testing_data_output}")

    os.makedirs(args.training_data_output, exist_ok=True)
    os.makedirs(args.testing_data_output, exist_ok=True)

    test_ratio = args.split_size / 100.0
    train_ratio = 1.0 - test_ratio

    # Different colours per class so you can visually tell them apart if you ever plot them
    fallback_colours = [(255, 0, 0), (0, 255, 0), (0, 128, 255)]

    for idx, dataset_dir in enumerate(args.datasets):
        class_name = f"class_{idx}"
        train_class_dir = os.path.join(args.training_data_output, class_name)
        test_class_dir = os.path.join(args.testing_data_output, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        print(f"[DATASET {idx}] Scanning {dataset_dir} for images...")
        image_files = collect_images(dataset_dir)
        print(f"[DATASET {idx}] Found {len(image_files)} images in {dataset_dir}")

        if not image_files:
            print(
                f"[WARNING] No images found in dataset {dataset_dir}. "
                "Using fallback synthetic images."
            )
            colour = fallback_colours[idx % len(fallback_colours)]
            generate_dummy_images(train_class_dir, test_class_dir, color=colour)
            continue

        random.shuffle(image_files)
        split_index = int(len(image_files) * train_ratio)
        train_files = image_files[:split_index]
        test_files = image_files[split_index:]

        print(
            f"[DATASET {idx}] -> train: {len(train_files)} images, "
            f"test: {len(test_files)} images"
        )

        # Copy real files
        for src in train_files:
            shutil.copy2(src, os.path.join(train_class_dir, os.path.basename(src)))
        for src in test_files:
            shutil.copy2(src, os.path.join(test_class_dir, os.path.basename(src)))

    print("Data split completed.")


if __name__ == "__main__":
    main()
