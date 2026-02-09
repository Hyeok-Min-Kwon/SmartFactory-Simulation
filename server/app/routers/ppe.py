"""
PPE Detection API Router
작업자의 마스크 착용 여부 확인
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import Response
import os
from datetime import datetime
import asyncio
from pathlib import Path

from ..models.ppe_detector import get_ppe_detector, PPEDetector
from ..services.image_processor import ImageProcessor


router = APIRouter(prefix="/api/v1/ppe", tags=["PPE Detection"])

# API Key 검증
API_KEY = os.getenv("API_KEY")


async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key


@router.get("/status")
async def get_ppe_status():
    """PPE 감지 모델 상태 확인"""
    detector = get_ppe_detector()
    info = detector.get_model_info()

    return {
        "service": "PPE Detection (Mask)",
        "model_ready": detector.is_ready(),
        "model_info": info,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/detect", dependencies=[Depends(verify_api_key)])
async def detect_ppe(
    image: UploadFile = File(..., description="작업자 얼굴 이미지 (JPG, PNG)")
):
    """
    작업자 이미지에서 마스크 감지

    Returns:
        - mask_detected: 마스크 착용 여부
        - ppe_compliant: PPE 규정 준수 여부
        - detections: 감지된 객체 목록
    """
    detector = get_ppe_detector()

    if not detector.is_ready():
        raise HTTPException(
            status_code=503,
            detail="PPE detection model not loaded. Please deploy the trained model."
        )

    try:
        # 이미지 읽기
        contents = await image.read()

        # PPE 감지
        result = detector.detect_from_bytes(contents)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "mask_detected": result["mask_detected"],
            "ppe_compliant": result["ppe_compliant"],
            "detections": result["detections"],
            "total_detections": result["total_detections"],
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/detect-visualize", dependencies=[Depends(verify_api_key)])
async def detect_ppe_with_visualization(
    image: UploadFile = File(..., description="작업자 얼굴 이미지 (JPG, PNG)")
):
    """
    PPE 감지 및 시각화된 이미지 반환

    Returns:
        시각화된 이미지 (바운딩 박스 포함)
    """
    detector = get_ppe_detector()

    if not detector.is_ready():
        raise HTTPException(
            status_code=503,
            detail="PPE detection model not loaded. Please deploy the trained model."
        )

    try:
        # 이미지 읽기
        contents = await image.read()

        # OpenCV로 이미지 로드
        image_np = ImageProcessor.load_image_from_bytes(contents)

        if image_np is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")

        # PPE 감지 및 시각화
        result, visualized = detector.detect_with_visualization(image_np)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # 이미지 인코딩
        image_bytes = ImageProcessor.encode_image_to_bytes(visualized, ".jpg", 90)

        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={
                "X-PPE-Compliant": str(result["ppe_compliant"]).lower(),
                "X-Mask-Detected": str(result["mask_detected"]).lower()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/check")
async def check_ppe_from_camera():
    """
    시뮬레이션에서 호출: 카메라로 PPE 체크

    플로우:
    1. 아두이노에 사진 촬영 요청 (MQTT)
    2. 아두이노가 촬영한 이미지 대기
    3. PPE 감지 수행
    4. 결과 반환

    현재는 임시로 테스트 이미지 사용
    """
    detector = get_ppe_detector()

    if not detector.is_ready():
        return {
            "status": "error",
            "message": "PPE detection model not loaded",
            "mask_detected": False
        }

    try:
        # TODO: MQTT로 아두이노에 사진 촬영 요청
        # mqtt_client.publish("arduino/camera/trigger", "capture")

        # TODO: 아두이노로부터 이미지 수신 대기
        # await asyncio.sleep(2)  # 촬영 대기
        # image_path = await get_latest_captured_image()

        # 임시: uploaded_images 디렉토리에서 최신 이미지 사용
        upload_dir = Path("uploaded_images")
        if not upload_dir.exists():
            return {
                "status": "error",
                "message": "No images available. Please upload an image first.",
                "mask_detected": False
            }

        # 가장 최근 이미지 찾기
        image_files = list(upload_dir.glob("*.jpg")) + list(upload_dir.glob("*.png"))
        if not image_files:
            return {
                "status": "error",
                "message": "No images found in uploaded_images directory",
                "mask_detected": False
            }

        latest_image = max(image_files, key=lambda p: p.stat().st_mtime)
        print(f"Using latest image: {latest_image}")

        # 이미지 읽기 및 PPE 감지
        with open(latest_image, "rb") as f:
            image_bytes = f.read()

        result = detector.detect_from_bytes(image_bytes)

        if "error" in result:
            return {
                "status": "error",
                "message": result["error"],
                "mask_detected": False
            }

        return {
            "status": "success",
            "mask_detected": result["mask_detected"],
            "ppe_compliant": result["ppe_compliant"],
            "detections_count": result["total_detections"],
            "message": "마스크 착용 확인" if result["mask_detected"] else "마스크 미착용",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"PPE check error: {e}")
        return {
            "status": "error",
            "message": f"PPE check failed: {str(e)}",
            "mask_detected": False
        }


@router.post("/check-compliance", dependencies=[Depends(verify_api_key)])
async def check_ppe_compliance(
    image: UploadFile = File(..., description="작업자 얼굴 이미지")
):
    """
    간단한 PPE 규정 준수 여부 확인

    공정 시작 가능 여부를 빠르게 확인할 때 사용

    Returns:
        - compliant: 공정 시작 가능 여부 (True/False)
        - reason: 판단 근거
    """
    detector = get_ppe_detector()

    if not detector.is_ready():
        raise HTTPException(
            status_code=503,
            detail="PPE detection model not loaded"
        )

    try:
        contents = await image.read()
        result = detector.detect_from_bytes(contents)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        # 규정 준수 판단
        compliant = result["ppe_compliant"]

        if compliant:
            reason = "마스크 착용 확인"
        else:
            reason = "마스크 미착용"

        return {
            "compliant": compliant,
            "can_start_process": compliant,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Check failed: {str(e)}")
