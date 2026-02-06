from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from datetime import datetime
import uuid

# 환경변수 로드
load_dotenv()

# 라우터 임포트
from .routers import ppe

app = FastAPI(
    title="Smart Factory API",
    description="스마트 팩토리 - PPE 감지 및 결함 검사 시스템",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key 로드
API_KEY = os.getenv("API_KEY")

# API Key 검증 함수
async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key


# PPE 라우터 등록
app.include_router(ppe.router)

# 이미지 저장 디렉토리
UPLOAD_DIR = "uploaded_images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 공개 엔드포인트
@app.get("/")
def read_root():
    return {
        "message": "Defect Detection API is running!",
        "status": "OK",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "server": "EC2"
    }

# 이미지 업로드 (임시로 해놓은 버전)
@app.post("/api/v1/upload-image", dependencies=[Depends(verify_api_key)])
async def upload_image(
    image: UploadFile = File(..., description="이미지 파일 (JPG, PNG)"),
    product_id: str = Form(None, description="제품 ID (선택)")
):
    """
    Arduino에서 이미지 업로드
    - 이미지를 서버에 저장만 함
    - YOLO 분석은 나중에 추가
    """
    try:
        # Product ID 생성
        if not product_id:
            product_id = str(uuid.uuid4())[:8]
        
        # 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = image.filename.split(".")[-1]
        filename = f"{product_id}_{timestamp}.{extension}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        
        # 이미지 저장
        contents = await image.read()
        with open(filepath, "wb") as f:
            f.write(contents)
        
        # 로그 출력
        print(f"📸 Image uploaded:")
        print(f"   - Product ID: {product_id}")
        print(f"   - Filename: {filename}")
        print(f"   - Size: {len(contents)} bytes")
        
        return {
            "status": "success",
            "product_id": product_id,
            "filename": filename,
            "size_bytes": len(contents),
            "saved_path": filepath,
            "timestamp": datetime.now().isoformat(),
            "message": "Image uploaded successfully"
        }
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 업로드된 이미지 목록
@app.get("/api/v1/images", dependencies=[Depends(verify_api_key)])
async def list_images():
    """
    업로드된 이미지 목록 확인
    """
    try:
        images = os.listdir(UPLOAD_DIR)
        images.sort(reverse=True)  # 최신순
        
        return {
            "total": len(images),
            "images": images[:20]  # 최근 20개만
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# API Key 테스트
@app.get("/api/v1/test", dependencies=[Depends(verify_api_key)])
async def test_api_key():
    """
    API Key가 제대로 작동하는지 테스트
    """
    return {
        "status": "success",
        "message": "API Key is valid! ✅"
    }