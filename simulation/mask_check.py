import requests


def check_ppe_detection(server_url="http://13.125.121.143:8000"):
    """
    서버에 PPE 감지 요청을 보내고 결과를 받아옵니다.

    Args:
        server_url: 서버 URL (기본값: http://13.125.121.143:8000)

    Returns:
        bool: 마스크 착용이 확인되면 True, 아니면 False
    """
    try:
        print("\n" + "="*50)
        print("PPE(마스크) 감지 시작...")
        print("서버에 얼굴인식 요청 전송 중...")

        # 서버에 PPE 감지 요청 (POST 방식으로 트리거)
        response = requests.post(
            f"{server_url}/api/v1/ppe/check",
            timeout=15  # 10초 타임아웃
        )

        if response.status_code == 200:
            result = response.json()
            print(f"서버 응답: {result}")

            # 결과 확인
            if result.get("status") == "success" and result.get("mask_detected"):
                print("✅ 마스크 착용 확인! 시뮬레이션을 시작합니다.")
                print("="*50 + "\n")
                return True
            else:
                print("❌ 마스크 미착용 또는 감지 실패!")
                print(f"사유: {result.get('message', '알 수 없음')}")
                print("="*50 + "\n")
                return False
        else:
            print(f"❌ 서버 오류: HTTP {response.status_code}")
            print("="*50 + "\n")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
        print("="*50 + "\n")
        return False
    except requests.exceptions.Timeout:
        print("❌ 요청 시간 초과 (15초)")
        print("="*50 + "\n")
        return False
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")
        print("="*50 + "\n")
        return False
