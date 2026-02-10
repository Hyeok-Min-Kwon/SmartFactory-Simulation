"""
Kaggle Magnetic Tile Surface Defects 데이터셋 준비 스크립트

다운로드한 데이터를 train/val 세트로 분할합니다.

사용법:
    python prepare_dataset.py --input datasets --output datasets/magnetic_tile_defects --val-ratio 0.2
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    val_ratio: float = 0.2,
    seed: int = 42
):
    """
    데이터셋을 train/val로 분할

    Args:
        input_dir: 원본 데이터셋 폴더 (클래스별 폴더 구조)
        output_dir: 출력 폴더
        val_ratio: 검증 세트 비율
        seed: 랜덤 시드
    """
    random.seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # 출력 폴더 생성
    train_dir = output_path / "train"
    val_dir = output_path / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # 이미지 확장자
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

    # 클래스별 이미지 수집
    class_images = defaultdict(list)

    # 원본 데이터셋 구조 탐색 (MT_*/Imgs/ 구조 지원)
    # test_data, magnetic_tile_defects 폴더는 제외
    exclude_dirs = {"test_data", "magnetic_tile_defects", ".DS_Store"}
    for class_folder in input_path.iterdir():
        if class_folder.is_dir() and class_folder.name not in exclude_dirs and not class_folder.name.startswith("."):
            class_name = class_folder.name

            # MT_ 접두사 제거 (MT_Blowhole -> Blowhole)
            if class_name.startswith("MT_"):
                class_name = class_name[3:]

            # Imgs 하위 폴더 확인
            imgs_folder = class_folder / "Imgs"
            search_folder = imgs_folder if imgs_folder.exists() else class_folder

            for img_file in search_folder.iterdir():
                # jpg 파일만 사용 (png는 마스크 이미지)
                if img_file.suffix.lower() == ".jpg":
                    class_images[class_name].append(img_file)

    if not class_images:
        print(f"Error: No images found in {input_dir}")
        print("Expected structure:")
        print("  input_dir/")
        print("    MT_Blowhole/ or Blowhole/")
        print("    MT_Break/ or Break/")
        print("    ...")
        return

    # 통계 출력
    print("Dataset Statistics:")
    print("=" * 40)
    total_images = 0
    for class_name, images in sorted(class_images.items()):
        print(f"  {class_name}: {len(images)} images")
        total_images += len(images)
    print(f"  Total: {total_images} images")
    print("=" * 40)

    # train/val 분할
    print(f"\nSplitting dataset (val_ratio={val_ratio})...")

    for class_name, images in class_images.items():
        # 클래스별 폴더 생성
        (train_dir / class_name).mkdir(exist_ok=True)
        (val_dir / class_name).mkdir(exist_ok=True)

        # 셔플 후 분할
        random.shuffle(images)
        val_count = int(len(images) * val_ratio)

        val_images = images[:val_count]
        train_images = images[val_count:]

        # 파일 복사
        for img in train_images:
            shutil.copy2(img, train_dir / class_name / img.name)

        for img in val_images:
            shutil.copy2(img, val_dir / class_name / img.name)

        print(f"  {class_name}: train={len(train_images)}, val={len(val_images)}")

    print(f"\nDataset prepared at: {output_path}")
    print(f"  Train: {train_dir}")
    print(f"  Val: {val_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Magnetic Tile Defects Dataset")
    parser.add_argument("--input", type=str, default="datasets",
                        help="Input directory (raw dataset)")
    parser.add_argument("--output", type=str, default="datasets/magnetic_tile_defects",
                        help="Output directory")
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="Validation set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    prepare_dataset(
        input_dir=args.input,
        output_dir=args.output,
        val_ratio=args.val_ratio,
        seed=args.seed
    )