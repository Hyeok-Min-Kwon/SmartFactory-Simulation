"""
Defect Classification API Router
자기 타일 표면 결함 분류
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Header
import os

from ..models.defect_classifier import get_defect_classifier


router = APIRouter(prefix="/api/v1/defect", tags=["Defect Classification"])

API_KEY = os.getenv("API_KEY")


async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key


@router.get("/status")
async def get_status():
    """결함 분류 모델 상태 확인"""
    classifier = get_defect_classifier()
    return {
        "ready": classifier.is_ready(),
        "model_info": classifier.get_model_info()
    }


@router.post("/classify", dependencies=[Depends(verify_api_key)])
async def classify(image: UploadFile = File(...)):
    """
    타일 표면 이미지 결함 분류

    Returns:
        - class_name: 분류된 결함 종류
        - confidence: 분류 신뢰도
    """
    classifier = get_defect_classifier()

    if not classifier.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded")

    contents = await image.read()
    result = classifier.classify_from_bytes(contents)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return {
        "class_name": result["class_name"],
        "confidence": result["confidence"]
    }
