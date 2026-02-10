import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt


# 클래스 정의
CLASSES = ["Blowhole", "Break", "Crack", "Fray", "Free", "Uneven"]
NUM_CLASSES = len(CLASSES)


def get_transforms(img_size: int = 224):
    """데이터 변환 정의"""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def create_model(model_name: str = "resnet18", num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """분류 모델 생성"""
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet34":
        model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "mobilenet_v3":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """한 에포크 학습"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            "loss": f"{running_loss/len(pbar):.4f}",
            "acc": f"{100.*correct/total:.2f}%"
        })

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """검증"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def plot_training_history(history: dict, save_path: str):
    """학습 히스토리 시각화"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy plot
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Training and Validation Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training history saved to {save_path}")


def train_classification_model(
    data_dir: str = "datasets/magnetic_tile_defects",
    epochs: int = 50,
    batch_size: int = 32,
    img_size: int = 224,
    model_name: str = "resnet18",
    learning_rate: float = 0.001,
    output_dir: str = "runs/classification"
):
    """
    결함 분류 모델 학습

    Args:
        data_dir: 데이터셋 폴더 경로 (train/, val/ 하위 폴더 필요)
        epochs: 학습 에포크 수
        batch_size: 배치 크기
        img_size: 입력 이미지 크기
        model_name: 모델 종류 (resnet18, resnet34, resnet50, efficientnet_b0, mobilenet_v3)
        learning_rate: 학습률
        output_dir: 결과 저장 폴더
    """
    # 출력 폴더 생성
    output_path = Path(output_dir)
    weights_path = output_path / "weights"
    weights_path.mkdir(parents=True, exist_ok=True)

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # 데이터 변환
    train_transform, val_transform = get_transforms(img_size)

    # 데이터셋 로드
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(train_dir):
        print(f"Error: {train_dir} not found!")
        print("Please prepare your dataset with the following structure:")
        print(f"  {data_dir}/")
        print("    train/")
        print("      Blowhole/")
        print("      Break/")
        print("      Crack/")
        print("      Fray/")
        print("      Free/")
        print("      Uneven/")
        print("    val/")
        print("      (same structure)")
        return None

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 모델 생성
    model = create_model(model_name, num_classes=len(train_dataset.classes))
    model = model.to(device)

    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 학습 히스토리
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_acc = 0.0

    print(f"\nStarting training with {model_name}...")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, LR: {learning_rate}")
    print("=" * 60)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # 학습
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 검증
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 스케줄러 업데이트
        scheduler.step()

        # 히스토리 저장
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # Best 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": train_dataset.classes
            }, weights_path / "best.pt")
            print(f"Best model saved! (Val Acc: {val_acc:.2f}%)")

        # 주기적 저장
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": train_dataset.classes
            }, weights_path / f"epoch_{epoch+1}.pt")

    # 최종 모델 저장
    torch.save({
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "classes": train_dataset.classes
    }, weights_path / "last.pt")

    # 학습 히스토리 시각화
    plot_training_history(history, str(output_path / "training_history.png"))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Val Accuracy: {best_val_acc:.2f}%")
    print(f"Best model: {weights_path / 'best.pt'}")
    print(f"Last model: {weights_path / 'last.pt'}")
    print("\nServer deployment:")
    print("1. Copy best.pt to server")
    print("2. Save to server/app/models/weights/defect_classifier.pt")
    print("3. Restart server")

    return history


def evaluate_model(model_path: str, data_dir: str, batch_size: int = 32, img_size: int = 224):
    """학습된 모델 평가 및 혼동 행렬 생성"""
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns

    # 디바이스 설정
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # 모델 로드
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint["classes"]

    model = create_model("resnet18", num_classes=len(classes), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 데이터 로드
    _, val_transform = get_transforms(img_size)
    val_dir = os.path.join(data_dir, "val")
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # 예측
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 분류 리포트
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # 혼동 행렬
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("Confusion matrix saved to confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Defect Classification Model Training")
    parser.add_argument("--data", type=str, default="datasets/magnetic_tile_defects", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0", "mobilenet_v3"],
                        help="Model architecture")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--output", type=str, default="runs/classification", help="Output directory")
    parser.add_argument("--evaluate", type=str, default=None, help="Path to model for evaluation")

    args = parser.parse_args()

    if args.evaluate:
        evaluate_model(args.evaluate, args.data, args.batch, args.img_size)
    else:
        train_classification_model(
            data_dir=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.img_size,
            model_name=args.model,
            learning_rate=args.lr,
            output_dir=args.output
        )