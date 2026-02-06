"""
PPE Detection 모델 학습 스크립트 (로컬 실행용)

사용법:
    python train.py --epochs 100 --batch 16

학습 완료 후:
    - best.pt 파일을 서버의 server/app/models/weights/ 폴더로 복사
"""

import argparse
from ultralytics import YOLO
import os


def train_ppe_model(
    data_yaml: str = "data.yaml",
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    model_name: str = "yolov8n.pt"
):
    """
    PPE 감지 모델 학습

    Args:
        data_yaml: 데이터셋 설정 파일 경로
        epochs: 학습 에포크 수
        batch_size: 배치 크기
        img_size: 입력 이미지 크기
        model_name: 기본 모델 (yolov8n, yolov8s, yolov8m 등)
    """
    # 기본 YOLO 모델 로드
    model = YOLO(model_name)

    # 학습 실행 (MPS 디바이스 명시적 지정)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device="mps",  # Apple Silicon GPU 사용
        project="runs/ppe_detection",
        name="train",
        exist_ok=True,
        pretrained=True,
        optimizer="Adam",
        lr0=0.001,
        lrf=0.01,
        patience=50,
        save=True,
        save_period=10,
        val=True,
        plots=True
    )

    print("\n" + "=" * 50)
    print("학습 완료!")
    print("=" * 50)
    print(f"Best 모델: runs/ppe_detection/train/weights/best.pt")
    print(f"Last 모델: runs/ppe_detection/train/weights/last.pt")
    print("\n서버 배포 방법:")
    print("1. best.pt 파일을 서버로 복사")
    print("2. server/app/models/weights/ 폴더에 저장")
    print("3. 서버 재시작")

    return results


def validate_model(model_path: str, data_yaml: str = "data.yaml"):
    """학습된 모델 검증"""
    model = YOLO(model_path)
    results = model.val(data=data_yaml)
    return results


def export_model(model_path: str, format: str = "onnx"):
    """모델 내보내기 (ONNX, TensorRT 등)"""
    model = YOLO(model_path)
    model.export(format=format)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPE Detection 모델 학습")
    parser.add_argument("--data", type=str, default="data.yaml", help="데이터셋 YAML 파일")
    parser.add_argument("--epochs", type=int, default=100, help="학습 에포크 수")
    parser.add_argument("--batch", type=int, default=16, help="배치 크기")
    parser.add_argument("--img-size", type=int, default=640, help="입력 이미지 크기")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="기본 모델")
    parser.add_argument("--validate", type=str, default=None, help="검증할 모델 경로")
    parser.add_argument("--export", type=str, default=None, help="내보낼 모델 경로")

    args = parser.parse_args()

    if args.validate:
        validate_model(args.validate, args.data)
    elif args.export:
        export_model(args.export)
    else:
        train_ppe_model(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            model_name=args.model
        )
