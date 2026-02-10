"""
Defect Classifier - 서버 Inference 전용
학습된 PyTorch 모델을 로드하여 결함 분류 수행
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, List
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

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        img_size: int = 224
    ):
        """
        Defect Classifier 초기화

        Args:
            model_path: 학습된 모델 경로 (.pt 파일)
            confidence_threshold: 분류 신뢰도 임계값
            img_size: 입력 이미지 크기
        """
        self.confidence_threshold = confidence_threshold
        self.img_size = img_size
        self.model = None
        self.model_loaded = False
        self.classes = self.DEFECT_CLASSES
        self.device = self._get_device()

        # 모델 경로 결정
        self.model_path = model_path if model_path else self.DEFAULT_WEIGHTS_PATH

        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 모델 로드
        self._load_model()

    def _get_device(self) -> torch.device:
        """사용 가능한 디바이스 반환"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_model(self) -> bool:
        """학습된 모델 로드"""
        if not os.path.exists(self.model_path):
            print(f"[Defect Classifier] Model not found: {self.model_path}")
            print("[Defect Classifier] Please train and deploy the model first.")
            self.model_loaded = False
            return False

        try:
            # 체크포인트 로드
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # 클래스 정보 로드
            if "classes" in checkpoint:
                self.classes = checkpoint["classes"]

            num_classes = len(self.classes)

            # 모델 생성 (ResNet18 기본)
            self.model = models.resnet18(weights=None)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

            # 가중치 로드
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model = self.model.to(self.device)
            self.model.eval()

            self.model_loaded = True
            print(f"[Defect Classifier] Model loaded: {self.model_path}")
            print(f"[Defect Classifier] Classes: {self.classes}")
            print(f"[Defect Classifier] Device: {self.device}")
            return True

        except Exception as e:
            print(f"[Defect Classifier] Failed to load model: {e}")
            self.model_loaded = False
            return False

    def is_ready(self) -> bool:
        """모델이 로드되어 사용 가능한지 확인"""
        return self.model_loaded and self.model is not None

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        """
        이미지 결함 분류

        Args:
            image: OpenCV 이미지 (BGR 형식)

        Returns:
            분류 결과 딕셔너리
        """
        if not self.is_ready():
            return {
                "error": "Model not loaded",
                "class_name": None,
                "confidence": 0.0,
                "is_defect": False
            }

        try:
            # BGR -> RGB 변환
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # 전처리
            input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

            # 추론
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            class_idx = predicted.item()
            class_name = self.classes[class_idx]
            conf = confidence.item()

            # 모든 클래스의 확률
            all_probs = {
                cls: round(prob, 4)
                for cls, prob in zip(self.classes, probabilities[0].cpu().tolist())
            }

            # 결함 여부 (Free가 아니면 결함)
            is_defect = class_name != "Free"

            return {
                "class_name": class_name,
                "class_id": class_idx,
                "confidence": round(conf, 4),
                "is_defect": is_defect,
                "all_probabilities": all_probs
            }

        except Exception as e:
            return {
                "error": f"Classification failed: {str(e)}",
                "class_name": None,
                "confidence": 0.0,
                "is_defect": False
            }

    def classify_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        바이트 데이터에서 결함 분류

        Args:
            image_bytes: 이미지 바이트 데이터

        Returns:
            분류 결과 딕셔너리
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {
                "error": "Failed to decode image",
                "class_name": None,
                "confidence": 0.0,
                "is_defect": False
            }

        return self.classify(image)

    def classify_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """
        여러 이미지 일괄 분류

        Args:
            images: OpenCV 이미지 리스트

        Returns:
            분류 결과 리스트
        """
        if not self.is_ready():
            return [{"error": "Model not loaded"} for _ in images]

        results = []
        for image in images:
            result = self.classify(image)
            results.append(result)

        return results

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_path": self.model_path,
            "model_loaded": self.model_loaded,
            "confidence_threshold": self.confidence_threshold,
            "classes": self.classes,
            "device": str(self.device),
            "img_size": self.img_size
        }


# 싱글톤 인스턴스
_classifier_instance: Optional[DefectClassifier] = None


def get_defect_classifier(
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.5
) -> DefectClassifier:
    """Defect Classifier 싱글톤 인스턴스 반환"""
    global _classifier_instance

    if _classifier_instance is None:
        _classifier_instance = DefectClassifier(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

    return _classifier_instance