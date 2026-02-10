"""
Defect Classification API Router
자기 타일 표면 결함 분류
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Header
from fastapi.responses import JSONResponse
import os
from datetime import datetime
from pathlib import Path
from typing import List

from ..models.defect_classifier import get_defect_classifier, DefectClassifier


router = APIRouter(prefix="/api/v1/defect", tags=["Defect Classification"])

# API Key 검증
API_KEY = os.getenv("API_KEY")


async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key


@router.get("/status")
async def get_defect_status():
    """결함 분류 모델 상태 확인"""
    classifier = get_defect_classifier()
    info = classifier.get_model_info()

    return {
        "service": "Defect Classification (Magnetic Tile)",
        "model_ready": classifier.is_ready(),
        "model_info": info,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/classify", dependencies=[Depends(verify_api_key)])
async def classify_defect(
    image: UploadFile = File(..., description="타일 표면 이미지 (JPG, PNG)")
):
    """
    타일 표면 이미지에서 결함 분류

    Returns:
        - class_name: 분류된 결함 종류
        - confidence: 분류 신뢰도
        - is_defect: 결함 여부 (Free가 아니면 True)
        - all_probabilities: 모든 클래스의 확률
    """
    classifier = get_defect_classifier()

    if not classifier.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Defect classification model not loaded. Please deploy the trained model."
        )

    try:
        contents = await image.read()
        result = classifier.classify_from_bytes(contents)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return {
            "status": "success",
            "class_name": result["class_name"],
            "class_id": result["class_id"],
            "confidence": result["confidence"],
            "is_defect": result["is_defect"],
            "all_probabilities": result["all_probabilities"],
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.post("/classify-batch", dependencies=[Depends(verify_api_key)])
async def classify_defects_batch(
    images: List[UploadFile] = File(..., description="타일 표면 이미지들 (JPG, PNG)")
):
    """
    여러 이미지 일괄 분류

    Returns:
        - results: 각 이미지의 분류 결과 리스트
        - summary: 결함 통계
    """
    classifier = get_defect_classifier()

    if not classifier.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Defect classification model not loaded."
        )

    try:
        results = []
        defect_count = 0
        normal_count = 0

        for idx, image in enumerate(images):
            contents = await image.read()
            result = classifier.classify_from_bytes(contents)

            result["filename"] = image.filename
            result["index"] = idx
            results.append(result)

            if result.get("is_defect"):
                defect_count += 1
            else:
                normal_count += 1

        return {
            "status": "success",
            "total_images": len(images),
            "results": results,
            "summary": {
                "defect_count": defect_count,
                "normal_count": normal_count,
                "defect_rate": round(defect_count / len(images) * 100, 2) if images else 0
            },
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch classification failed: {str(e)}")


@router.post("/check")
async def check_defect():
    """
    시뮬레이션에서 호출: 최신 업로드 이미지로 결함 검사

    uploaded_images 디렉토리의 최신 이미지를 분류
    """
    classifier = get_defect_classifier()

    if not classifier.is_ready():
        return {
            "status": "error",
            "message": "Defect classification model not loaded",
            "is_defect": False
        }

    try:
        upload_dir = Path("uploaded_images")
        if not upload_dir.exists():
            return {
                "status": "error",
                "message": "No images available. Please upload an image first.",
                "is_defect": False
            }

        image_files = list(upload_dir.glob("*.jpg")) + list(upload_dir.glob("*.png"))
        if not image_files:
            return {
                "status": "error",
                "message": "No images found in uploaded_images directory",
                "is_defect": False
            }

        latest_image = max(image_files, key=lambda p: p.stat().st_mtime)
        print(f"[Defect Check] Using latest image: {latest_image}")

        with open(latest_image, "rb") as f:
            image_bytes = f.read()

        result = classifier.classify_from_bytes(image_bytes)

        if "error" in result:
            return {
                "status": "error",
                "message": result["error"],
                "is_defect": False
            }

        defect_type = result["class_name"]
        is_defect = result["is_defect"]

        if is_defect:
            message = f"결함 발견: {defect_type} (신뢰도: {result['confidence']:.2%})"
        else:
            message = f"정상 (신뢰도: {result['confidence']:.2%})"

        return {
            "status": "success",
            "class_name": defect_type,
            "confidence": result["confidence"],
            "is_defect": is_defect,
            "message": message,
            "image_path": str(latest_image),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        print(f"[Defect Check] Error: {e}")
        return {
            "status": "error",
            "message": f"Defect check failed: {str(e)}",
            "is_defect": False
        }


@router.post("/quality-check", dependencies=[Depends(verify_api_key)])
async def quality_check(
    image: UploadFile = File(..., description="타일 표면 이미지")
):
    """
    품질 검사 - 합격/불합격 판정

    Returns:
        - passed: 품질 검사 통과 여부
        - defect_type: 발견된 결함 종류 (있을 경우)
        - confidence: 분류 신뢰도
    """
    classifier = get_defect_classifier()

    if not classifier.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Defect classification model not loaded"
        )

    try:
        contents = await image.read()
        result = classifier.classify_from_bytes(contents)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        passed = not result["is_defect"]
        defect_type = None if passed else result["class_name"]

        return {
            "passed": passed,
            "quality_status": "PASS" if passed else "FAIL",
            "defect_type": defect_type,
            "confidence": result["confidence"],
            "message": "품질 검사 통과" if passed else f"결함 발견: {defect_type}",
            "timestamp": datetime.now().isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality check failed: {str(e)}")
