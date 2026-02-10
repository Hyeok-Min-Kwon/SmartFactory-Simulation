"""
Defect Classifier - 서버 Inference 전용
학습된 PyTorch 모델을 로드하여 결함 분류 수행
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
import os

import torch
import torch.nn as nn
from torchvision import transforms, models


class DefectClassifier:
    """
    PyTorch 기반 결함 분류 클래스 (Inference 전용)

    클래스:
        - Blowhole (기공)
        - Break (파손)
        - Crack (균열)
        - Fray (해어짐)
        - Free (정상)
        - Uneven (불균일)
    """

    DEFAULT_WEIGHTS_PATH = os.path.join(
        os.path.dirname(__file__), "weights", "defect_classifier.pt"
    )

    DEFECT_CLASSES = ["Blowhole", "Break", "Crack", "Fray", "Free", "Uneven"]

    def __init__(self, model_path: Optional[str] = None, img_size: int = 224):
        self.img_size = img_size
        self.model = None
        self.model_loaded = False
        self.classes = self.DEFECT_CLASSES
        self.device = self._get_device()
        self.model_path = model_path if model_path else self.DEFAULT_WEIGHTS_PATH

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self._load_model()

    def _get_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self) -> bool:
        if not os.path.exists(self.model_path):
            print(f"[Defect Classifier] Model not found: {self.model_path}")
            self.model_loaded = False
            return False

        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)

            if "classes" in checkpoint:
                self.classes = checkpoint["classes"]

            num_classes = len(self.classes)

            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()

            self.model_loaded = True
            print(f"[Defect Classifier] Model loaded: {self.model_path}, Device: {self.device}")
            return True

        except Exception as e:
            print(f"[Defect Classifier] Failed to load model: {e}")
            self.model_loaded = False
            return False

    def is_ready(self) -> bool:
        return self.model_loaded and self.model is not None

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        """이미지 결함 분류"""
        if not self.is_ready():
            return {"error": "Model not loaded"}

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            return {
                "class_name": self.classes[predicted.item()],
                "confidence": round(confidence.item(), 4)
            }

        except Exception as e:
            return {"error": f"Classification failed: {str(e)}"}

    def classify_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """바이트 데이터에서 결함 분류"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Failed to decode image"}

        return self.classify(image)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.model_loaded,
            "classes": self.classes,
            "device": str(self.device)
        }


_classifier_instance: Optional[DefectClassifier] = None


def get_defect_classifier(model_path: Optional[str] = None) -> DefectClassifier:
    """Defect Classifier 싱글톤 인스턴스 반환"""
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = DefectClassifier(model_path=model_path)

    return _classifier_instance
