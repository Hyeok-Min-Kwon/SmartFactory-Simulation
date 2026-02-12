import requests
import os
import random
import time
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

# 서버 설정
SERVER_URL = os.getenv("SERVER_URL", "http://13.125.121.143:8000")
API_KEY = os.getenv("API_KEY", "")

# 테스트 이미지 폴더 경로
TEST_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "training",
    "classification",
    "datasets",
    "test_data"
)

# 클래스명 → 라벨 매핑 (1~6)
CLASS_TO_LABEL = {
    "Blowhole": 1,  # 기공
    "Break": 2,     # 파손
    "Crack": 3,     # 균열
    "Fray": 4,      # 해어짐
    "Uneven": 5,    # 불균일
    "Free": 6       # 양품
}


def get_random_image_path():
    """
    test_data 폴더에서 랜덤하게 이미지 경로 선택

    Returns:
        str: 이미지 파일 경로
    """
    # 모든 카테고리 폴더 가져오기
    categories = [d for d in os.listdir(TEST_DATA_PATH)
                  if os.path.isdir(os.path.join(TEST_DATA_PATH, d))]

    if not categories:
        return None

    # 랜덤 카테고리 선택
    category = random.choice(categories)
    category_path = os.path.join(TEST_DATA_PATH, category)

    # 해당 카테고리의 이미지 파일들
    images = [f for f in os.listdir(category_path)
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not images:
        return None

    # 랜덤 이미지 선택
    image_file = random.choice(images)
    return os.path.join(category_path, image_file)


def classify_image(image_path=None):
    """
    이미지를 서버로 전송하여 분류 결과(라벨)를 반환

    Args:
        image_path: 이미지 파일 경로 (None이면 랜덤 선택)

    Returns:
        int: 분류 라벨 (1~6), 실패 시 6(양품) 반환
    """
    # 이미지 경로가 없으면 랜덤 선택
    if image_path is None:
        image_path = get_random_image_path()

    if image_path is None or not os.path.exists(image_path):
        print(f"[imageSending] 이미지를 찾을 수 없음, 기본값 6(양품) 반환")
        return 6

    try:
        # 이미지 파일 열기
        with open(image_path, 'rb') as f:
            files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
            headers = {'X-API-Key': API_KEY}

            # 서버에 분류 요청
            response = requests.post(
                f"{SERVER_URL}/api/v1/defect/classify",
                files=files,
                headers=headers,
                timeout=10
            )

        if response.status_code == 200:
            result = response.json()
            class_name = result.get("class_name", "Free")
            confidence = result.get("confidence", 0)

            label = CLASS_TO_LABEL.get(class_name, 6)
            print(f"[imageSending] 분류 결과: {class_name} (라벨: {label}, 신뢰도: {confidence:.2%})")
            return label
        else:
            print(f"[imageSending] 서버 오류: HTTP {response.status_code}")
            return 6

    except requests.exceptions.ConnectionError:
        print("[imageSending] 서버 연결 실패")
        return 6
    except requests.exceptions.Timeout:
        print("[imageSending] 요청 시간 초과")
        return 6
    except Exception as e:
        print(f"[imageSending] 오류 발생: {e}")
        return 6


def get_product_label():
    """
    시뮬레이션에서 호출할 메인 함수
    랜덤 이미지를 서버로 보내서 분류 결과 반환

    Returns:
        int: 분류 라벨 (1~5=불량, 6=양품)
    """
    return classify_image()


# 테스트용
if __name__ == "__main__":
    print("이미지 분류 테스트...")
    for i in range(3):
        label = get_product_label()
        print(f"  테스트 {i+1}: 라벨 = {label}")
        print()
        # time.sleep(3)  # 1초 대기
