"""
PPE Detection API Router
작업자의 마스크 착용 여부 확인
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Header
import os

from ..models.ppe_detector import get_ppe_detector


router = APIRouter(prefix="/api/v1/ppe", tags=["PPE Detection"])

API_KEY = os.getenv("API_KEY")


async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key


@router.get("/status")
async def get_status():
    """PPE 감지 모델 상태 확인"""
    detector = get_ppe_detector()
    return {
        "ready": detector.is_ready(),
        "model_info": detector.get_model_info()
    }


@router.post("/detect", dependencies=[Depends(verify_api_key)])
async def detect(image: UploadFile = File(...)):
    """
    이미지에서 마스크 착용 여부 감지

    Returns:
        - mask_detected: 마스크 착용 여부 (True/False)
    """
    detector = get_ppe_detector()

    if not detector.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await image.read()
    result = detector.detect_from_bytes(contents)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {"mask_detected": result["mask_detected"]}
