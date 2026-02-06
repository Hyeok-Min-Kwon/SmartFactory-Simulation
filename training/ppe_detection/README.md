# PPE Detection 모델 학습

안경/모자 감지를 위한 YOLOv8 모델 학습 가이드

## 환경 설정

```bash
cd training/ppe_detection
pip install -r requirements.txt
```

## 데이터셋 준비

### 방법 1: Roboflow에서 다운로드 (추천)

1. [Roboflow](https://roboflow.com)에서 PPE 관련 데이터셋 검색
2. YOLO 형식으로 다운로드
3. `datasets/` 폴더에 압축 해제

추천 데이터셋:
- Safety Helmet Detection
- Glasses Detection
- PPE Detection Dataset

### 방법 2: 직접 라벨링

1. [LabelImg](https://github.com/heartexlabs/labelImg) 또는 [CVAT](https://cvat.org) 사용
2. 이미지에 안경(glasses), 모자(hat) 바운딩 박스 라벨링
3. YOLO 형식으로 저장

### 폴더 구조

```
datasets/
├── images/
│   ├── train/
│   │   ├── image001.jpg
│   │   └── ...
│   └── val/
│       ├── image001.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── image001.txt  # YOLO 형식 라벨
    │   └── ...
    └── val/
        ├── image001.txt
        └── ...
```

### YOLO 라벨 형식

```
# class_id center_x center_y width height (정규화된 값 0-1)
0 0.5 0.3 0.2 0.1  # glasses
1 0.5 0.1 0.3 0.15  # hat
```

## 학습 실행

```bash
# 기본 학습
python train.py --epochs 100 --batch 16

# GPU 메모리가 부족하면 배치 크기 줄이기
python train.py --epochs 100 --batch 8

# 더 정확한 모델 (느림)
python train.py --model yolov8s.pt --epochs 150
```

## 학습 결과 확인

- 결과 폴더: `runs/ppe_detection/train/`
- 학습 그래프: `results.png`
- 모델 가중치: `weights/best.pt`

## 서버 배포

1. `runs/ppe_detection/train/weights/best.pt` 파일 복사
2. 서버의 `server/app/models/weights/ppe_best.pt`로 저장
3. 서버 재시작

```bash
scp runs/ppe_detection/train/weights/best.pt user@server:/path/to/server/app/models/weights/ppe_best.pt
```
