import argparse
import os
from pathlib import Path
from typing import Iterable

from PIL import Image

# Allowed image extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def parse_args():
    parser = argparse.ArgumentParser(description="Resize images to 64x64")

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Input data folder (mounted uri_folder from Azure ML).",
    )
    parser.add_argument(
        "--output_data",
        type=str,
        required=True,
        help="Output folder for resized images (uri_folder).",
    )

    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def iter_image_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p):
            yield p


def resize_and_copy_images(input_dir: Path, output_dir: Path, size=(64, 64)) -> None:
    """
    Walk through input_dir, resize all images to `size`,
    and write them to output_dir preserving the relative folder structure.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    os.makedirs(output_dir, exist_ok=True)

    num_total = 0
    num_ok = 0
    num_failed = 0

    print(f"Scanning for images under: {input_dir}")
    for img_path in iter_image_files(input_dir):
        num_total += 1
        rel_path = img_path.relative_to(input_dir)

        # Preserve folder structure under output_dir
        target_path = output_dir / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im = im.resize(size, Image.BILINEAR)
                # Save as JPEG (or keep original extension if you prefer)
                target_path = target_path.with_suffix(".jpg")
                im.save(target_path, format="JPEG", quality=90)
            num_ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"[WARNING] Failed to process {img_path}: {e}")
            num_failed += 1

    print("==== Dataprep summary ====")
    print(f"Input dir   : {input_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Total files : {num_total}")
    print(f"Resized ok  : {num_ok}")
    print(f"Failed      : {num_failed}")


def main():
    args = parse_args()

    input_dir = Path(args.data)
    output_dir = Path(args.output_data)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    resize_and_copy_images(input_dir, output_dir, size=(64, 64))


if __name__ == "__main__":
    main()
