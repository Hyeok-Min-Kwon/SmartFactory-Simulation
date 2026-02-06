import cv2
import numpy as np
from typing import Tuple, Optional


class ImageProcessor:
    """OpenCV 기반 이미지 전처리 클래스"""

    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """바이트 데이터에서 이미지 로드"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image

    @staticmethod
    def resize_image(
        image: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
        keep_aspect_ratio: bool = True
    ) -> np.ndarray:
        """이미지 리사이즈"""
        if keep_aspect_ratio:
            h, w = image.shape[:2]
            scale = min(target_size[0] / w, target_size[1] / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            # 패딩 추가
            canvas = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
            x_offset = (target_size[0] - new_w) // 2
            y_offset = (target_size[1] - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
            return canvas
        else:
            return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """이미지 정규화 (0-1 범위)"""
        return image.astype(np.float32) / 255.0

    @staticmethod
    def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
        """노이즈 제거"""
        return cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)

    @staticmethod
    def adjust_brightness_contrast(
        image: np.ndarray,
        brightness: int = 0,
        contrast: int = 0
    ) -> np.ndarray:
        """밝기/대비 조절"""
        if brightness != 0:
            if brightness > 0:
                shadow = brightness
                highlight = 255
            else:
                shadow = 0
                highlight = 255 + brightness
            alpha = (highlight - shadow) / 255
            gamma = shadow
            image = cv2.addWeighted(image, alpha, image, 0, gamma)

        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha = f
            gamma = 127 * (1 - f)
            image = cv2.addWeighted(image, alpha, image, 0, gamma)

        return image

    @staticmethod
    def convert_to_rgb(image: np.ndarray) -> np.ndarray:
        """BGR -> RGB 변환"""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    @staticmethod
    def convert_to_bgr(image: np.ndarray) -> np.ndarray:
        """RGB -> BGR 변환"""
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    @staticmethod
    def draw_bounding_box(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str,
        confidence: float,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """바운딩 박스와 라벨 그리기"""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        label_text = f"{label}: {confidence:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        cv2.putText(
            image,
            label_text,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        return image

    @staticmethod
    def encode_image_to_bytes(
        image: np.ndarray,
        format: str = ".jpg",
        quality: int = 90
    ) -> bytes:
        """이미지를 바이트로 인코딩"""
        encode_params = []
        if format.lower() in [".jpg", ".jpeg"]:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif format.lower() == ".png":
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 11)]

        _, buffer = cv2.imencode(format, image, encode_params)
        return buffer.tobytes()

    @staticmethod
    def crop_roi(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """관심 영역(ROI) 크롭"""
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2]
