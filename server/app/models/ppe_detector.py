"""
PPE Detector - 서버 Inference 전용
학습된 .pt 파일을 로드하여 예측만 수행
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ultralytics import YOLO
import os


class PPEDetector:
    """
    YOLO 기반 PPE(개인보호장비) 감지 클래스 (Inference 전용)
    - 마스크(mask)
    """

    # 기본 모델 가중치 경로
    DEFAULT_WEIGHTS_PATH = os.path.join(
        os.path.dirname(__file__), "weights", "ppe_best.pt"
    )

    # PPE 클래스 (학습된 모델의 클래스와 일치해야 함)
    PPE_CLASSES = {
        0: "mask",
        1: "no-mask"
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5
    ):
        """
        PPE Detector 초기화

        Args:
            model_path: 학습된 YOLO 모델 경로 (.pt 파일)
            confidence_threshold: 감지 신뢰도 임계값
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.model_loaded = False

        # 모델 경로 결정
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = self.DEFAULT_WEIGHTS_PATH

        # 모델 로드 시도
        self._load_model()

    def _load_model(self) -> bool:
        """학습된 YOLO 모델 로드"""
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                self.model_loaded = True
                print(f"[PPE Detector] Model loaded: {self.model_path}")
                return True
            except Exception as e:
                print(f"[PPE Detector] Failed to load model: {e}")
                self.model_loaded = False
                return False
        else:
            print(f"[PPE Detector] Model not found: {self.model_path}")
            print("[PPE Detector] Please train and deploy the model first.")
            self.model_loaded = False
            return False

    def is_ready(self) -> bool:
        """모델이 로드되어 사용 가능한지 확인"""
        return self.model_loaded and self.model is not None

    def detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        이미지에서 PPE 감지

        Args:
            image: OpenCV 이미지 (BGR 형식)

        Returns:
            감지 결과 딕셔너리
        """
        if not self.is_ready():
            return {
                "error": "Model not loaded",
                "detections": [],
                "mask_detected": False,
                "ppe_compliant": False
            }

        # YOLO 추론 실행
        results = self.model(image, verbose=False)

        detections = []
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

                # 클래스 이름 가져오기
                class_name = self.PPE_CLASSES.get(cls_id, self.model.names.get(cls_id, "unknown"))

                # 바운딩 박스 좌표
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detection = {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": [x1, y1, x2, y2]
                }
                detections.append(detection)

                # PPE 타입 확인
                if class_name == "mask":
                    mask_detected = True

        return {
            "detections": detections,
            "mask_detected": mask_detected,
            "ppe_compliant": mask_detected,
            "total_detections": len(detections)
        }

    def detect_from_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        바이트 데이터에서 PPE 감지

        Args:
            image_bytes: 이미지 바이트 데이터

        Returns:
            감지 결과 딕셔너리
        """
        # 바이트 -> numpy 배열 변환
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {
                "error": "Failed to decode image",
                "detections": [],
                "mask_detected": False,
                "ppe_compliant": False
            }

        return self.detect(image)

    def detect_with_visualization(
        self,
        image: np.ndarray
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        PPE 감지 및 결과 시각화

        Args:
            image: OpenCV 이미지 (BGR 형식)

        Returns:
            (감지 결과, 시각화된 이미지)
        """
        result = self.detect(image)
        visualized = image.copy()

        if "error" in result:
            return result, visualized

        # 색상 정의
        COLORS = {
            "mask": (0, 255, 0),      # 녹색
            "default": (255, 0, 0)    # 파란색
        }

        for detection in result["detections"]:
            x1, y1, x2, y2 = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]

            color = COLORS.get(class_name, COLORS["default"])

            # 바운딩 박스
            cv2.rectangle(visualized, (x1, y1), (x2, y2), color, 2)

            # 라벨
            label = f"{class_name}: {confidence:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                visualized,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 5, y1),
                color,
                -1
            )
            cv2.putText(
                visualized,
                label,
                (x1 + 2, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        # PPE 상태 표시
        status_color = (0, 255, 0) if result["ppe_compliant"] else (0, 0, 255)
        status_text = "PPE OK" if result["ppe_compliant"] else "PPE NOT DETECTED"
        cv2.putText(
            visualized,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            status_color,
            2
        )

        return result, visualized

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_path": self.model_path,
            "model_loaded": self.model_loaded,
            "confidence_threshold": self.confidence_threshold,
            "classes": self.PPE_CLASSES
        }


# 싱글톤 인스턴스 (서버 시작 시 한 번만 로드)
_detector_instance: Optional[PPEDetector] = None


def get_ppe_detector(
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.5
) -> PPEDetector:
    """PPE Detector 싱글톤 인스턴스 반환"""
    global _detector_instance

    if _detector_instance is None:
        _detector_instance = PPEDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

    return _detector_instance
