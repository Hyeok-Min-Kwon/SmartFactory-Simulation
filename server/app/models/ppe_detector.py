"""
PPE Detector - 서버 Inference 전용
학습된 .pt 파일을 로드하여 예측만 수행
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from ultralytics import YOLO
import os


class PPEDetector:
    """YOLO 기반 PPE(마스크) 감지 클래스 (Inference 전용)"""

    DEFAULT_WEIGHTS_PATH = os.path.join(
        os.path.dirname(__file__), "weights", "ppe_best.pt"
    )

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_loaded = False
        self.model_path = model_path if model_path else self.DEFAULT_WEIGHTS_PATH

        self._load_model()

    def _load_model(self) -> bool:
        if not os.path.exists(self.model_path):
            print(f"[PPE Detector] Model not found: {self.model_path}")
            self.model_loaded = False
            return False

        try:
            self.model = YOLO(self.model_path)
            self.model_loaded = True
            print(f"[PPE Detector] Model loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"[PPE Detector] Failed to load model: {e}")
            self.model_loaded = False
            return False

    def is_ready(self) -> bool:
        return self.model_loaded and self.model is not None

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """이미지에서 마스크 감지"""
        if not self.is_ready():
            return {"error": "Model not loaded", "mask_detected": False}

        results = self.model(image, verbose=False)
        mask_detected = False

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence < self.confidence_threshold:
                    continue

                class_name = self.model.names.get(cls_id, "unknown")
                if class_name == "mask":
                    mask_detected = True
                    break

            if mask_detected:
                break

        return {"mask_detected": mask_detected}

    def detect_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """바이트 데이터에서 마스크 감지"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Failed to decode image", "mask_detected": False}

        return self.detect(image)

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path
        }


_detector_instance: Optional[PPEDetector] = None


def get_ppe_detector(model_path: Optional[str] = None, confidence_threshold: float = 0.5) -> PPEDetector:
    """PPE Detector 싱글톤 인스턴스 반환"""
    global _detector_instance

    if _detector_instance is None:
        _detector_instance = PPEDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

    return _detector_instance
