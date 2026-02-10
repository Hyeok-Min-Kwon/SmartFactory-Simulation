# Magnetic Tile Defect Classification

자기 타일 표면 결함 분류 모델 학습 가이드

## 데이터셋

**Kaggle**: [Magnetic Tile Surface Defects](https://www.kaggle.com/datasets/alex000kim/magnetic-tile-surface-defects)

### 결함 클래스 (6종)

| 클래스 | 설명 |
|--------|------|
| Blowhole | 기공 결함 |
| Break | 파손 |
| Crack | 균열 |
| Fray | 해어짐 |
| Free | 정상 (결함 없음) |
| Uneven | 불균일 |

## 환경 설정

```bash
cd training/classification
pip install -r requirements.txt
```

## 데이터셋 준비

### 1. Kaggle에서 다운로드

```bash
# Kaggle CLI 설치 (필요시)
pip install kaggle

# 데이터셋 다운로드
kaggle datasets download -d alex000kim/magnetic-tile-surface-defects

# 압축 해제
unzip magnetic-tile-surface-defects.zip -d datasets/raw
```

### 2. 데이터셋 구조 변환

다운로드한 데이터를 학습/검증 세트로 분리합니다:

```bash
python prepare_dataset.py
```

또는 수동으로 다음 구조로 정리:

```
datasets/magnetic_tile_defects/
├── train/
│   ├── Blowhole/
│   │   ├── image001.jpg
│   │   └── ...
│   ├── Break/
│   ├── Crack/
│   ├── Fray/
│   ├── Free/
│   └── Uneven/
└── val/
    ├── Blowhole/
    ├── Break/
    ├── Crack/
    ├── Fray/
    ├── Free/
    └── Uneven/
```

## 학습 실행

```bash
# 기본 학습 (ResNet18)
python train.py --epochs 50 --batch 32

# GPU 메모리가 부족하면 배치 크기 줄이기
python train.py --epochs 50 --batch 16

# 더 정확한 모델 (ResNet50)
python train.py --model resnet50 --epochs 100

# EfficientNet 사용
python train.py --model efficientnet_b0 --epochs 50

# MobileNet (경량 모델)
python train.py --model mobilenet_v3 --epochs 50
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--data` | 데이터셋 폴더 | `datasets/magnetic_tile_defects` |
| `--epochs` | 학습 에포크 수 | 50 |
| `--batch` | 배치 크기 | 32 |
| `--img-size` | 입력 이미지 크기 | 224 |
| `--model` | 모델 아키텍처 | resnet18 |
| `--lr` | 학습률 | 0.001 |
| `--output` | 결과 저장 폴더 | `runs/classification` |

## 모델 평가

```bash
python train.py --evaluate runs/classification/weights/best.pt --data datasets/magnetic_tile_defects
```

## 학습 결과 확인

- 결과 폴더: `runs/classification/`
- 학습 그래프: `training_history.png`
- 모델 가중치: `weights/best.pt`
- 혼동 행렬: `confusion_matrix.png` (평가 시 생성)

## 서버 배포

1. `runs/classification/weights/best.pt` 파일 복사
2. 서버의 `server/app/models/weights/defect_classifier.pt`로 저장
3. 서버 재시작

```bash
scp runs/classification/weights/best.pt user@server:/path/to/server/app/models/weights/defect_classifier.pt
```

## 지원 모델

| 모델 | 파라미터 수 | 추론 속도 | 정확도 |
|------|------------|-----------|--------|
| MobileNetV3 | ~2.5M | 빠름 | 중 |
| ResNet18 | ~11M | 중간 | 중상 |
| ResNet34 | ~21M | 중간 | 상 |
| EfficientNet-B0 | ~5.3M | 중간 | 상 |
| ResNet50 | ~25M | 느림 | 최상 |