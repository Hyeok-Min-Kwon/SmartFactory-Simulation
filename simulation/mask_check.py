import requests
import time


def check_ppe_detection(server_url="http://13.125.121.143:8000", max_wait=10):
    """
    ESP32-CAM에 촬영 요청을 보내고 PPE 감지 결과를 받아옵니다.

    Args:
        server_url: 서버 URL (기본값: http://13.125.121.143:8000)
        max_wait: ESP32 응답 최대 대기 시간 (초)

    Returns:
        bool: 마스크 착용이 확인되면 True, 아니면 False
    """
    try:
        print("\n" + "="*50)
        print("PPE(마스크) 감지 시작...")

        # 1. ESP32에 사진 촬영 요청
        print("ESP32-CAM에 촬영 요청 전송 중...")
        trigger_response = requests.post(
            f"{server_url}/trigger-capture",
            timeout=5
        )

        if trigger_response.status_code != 200:
            print(f"❌ 촬영 트리거 실패: HTTP {trigger_response.status_code}")
            print("="*50 + "\n")
            return False

        print("📸 촬영 요청 전송 완료. ESP32 응답 대기 중...")

        # 2. ESP32가 사진 업로드할 때까지 대기 (폴링)
        start_time = time.time()
        result = None

        while time.time() - start_time < max_wait:
            time.sleep(1)  # 1초 간격으로 체크

            response = requests.post(
                f"{server_url}/api/v1/ppe/check",
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                # pending이 아니면 결과가 업데이트된 것
                if result.get("status") == "success":
                    break

            elapsed = int(time.time() - start_time)
            print(f"   대기 중... ({elapsed}초)")

        # 3. 결과 확인
        if result is None:
            print("❌ ESP32 응답 시간 초과")
            print("="*50 + "\n")
            return False

        print(f"서버 응답: {result}")

        if result.get("status") == "success" and result.get("mask_detected"):
            print("✅ 마스크 착용 확인! 시뮬레이션을 시작합니다.")
            print("="*50 + "\n")
            return True
        else:
            print("❌ 마스크 미착용 또는 감지 실패!")
            print(f"사유: {result.get('message', '알 수 없음')}")
            print("="*50 + "\n")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        print("="*50 + "\n")
        return False
    except requests.exceptions.Timeout:
        print("❌ 요청 시간 초과")
        print("="*50 + "\n")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        print("="*50 + "\n")
        return False
