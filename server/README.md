## 여기는 서버에 올라갈 파일들

```
app/
├── __init__.py
├── main.py
├── routers/
│   ├── __init__.py
│   ├── detect.py      (이미지 분석 API)
│   └── trigger.py     (시뮬레이션 API)
├── services/
│   ├── __init__.py
│   ├── yolo.py        (YOLO 로직)
│   └── mqtt.py        (MQTT 통신)
└── schemas/
    ├── __init__.py
    └── defect.py      (데이터 모델)
```