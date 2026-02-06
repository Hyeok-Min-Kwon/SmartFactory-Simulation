"""
Defect Detection API
제조 공정 결함 검사 시스템
"""

''''
---

## **📡 API 엔드포인트 요약**

공개 엔드포인트:
├─ GET  /                      홈 (API 정보)
└─ GET  /health                서버 상태 확인

보호된 엔드포인트 (API Key 필요):
├─ POST /api/v1/upload-image   이미지 업로드
├─ GET  /api/v1/images         업로드된 이미지 목록
└─ GET  /api/v1/test           API Key 테스트
```

---

## **🔑 중요 정보**
```
서버 주소: http://13.125.121.143:8000
API Key: defect-2024-secret-key-abc123xyz789
Swagger UI: http://13.125.121.143:8000/docs

이미지 업로드:
POST http://13.125.121.143:8000/api/v1/upload-image
Header: X-API-Key
Form-data: image (파일), product_id (선택)
'''