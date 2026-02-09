"""
CoppeliaSim에서 바닥과 공이 있는 간단한 환경을 생성하는 스크립트
(create_sphere.py 구조 기반)
"""
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time


def main():
    print("=" * 50)
    print("CoppeliaSim 간단한 환경 생성")
    print("=" * 50)

    print("\nCoppeliaSim에 연결 중...")
    client = RemoteAPIClient()
    sim = client.require('sim')
    print("연결 성공!")

    # 시뮬레이션 정지
    sim.stopSimulation()
    time.sleep(0.5)

    print("\n=== 1. 바닥 생성 ===")
    # 바닥 생성 (큰 직육면체)
    floor_handle = sim.createPrimitiveShape(
        0,  # cuboid type
        [2.0, 2.0, 0.1]  # 2m x 2m x 0.1m
    )
    print(f"바닥 생성 완료! Handle: {floor_handle}")

    # 바닥 위치 설정
    sim.setObjectPosition(floor_handle, -1, [0.0, 0.0, -0.05])
    print("바닥 위치: [0, 0, -0.05]")

    # 바닥 색상 (회색)
    sim.setShapeColor(floor_handle, None, 0, [0.7, 0.7, 0.7])
    print("바닥 색상: 회색")

    # 바닥을 정적 객체로 설정
    sim.setObjectInt32Param(floor_handle, sim.shapeintparam_static, 1)
    sim.setObjectInt32Param(floor_handle, sim.shapeintparam_respondable, 1)
    print("바닥 물리 속성 설정 완료")

    print("\n=== 2. 빨간 공 생성 ===")
    # 빨간 공
    red_sphere = sim.createPrimitiveShape(1, [0.2, 0.2, 0.2])
    sim.setObjectPosition(red_sphere, -1, [0.0, 0.0, 0.5])
    sim.setShapeColor(red_sphere, None, 0, [1.0, 0.0, 0.0])
    sim.setObjectInt32Param(red_sphere, sim.shapeintparam_respondable, 1)
    sim.setShapeMass(red_sphere, 0.5)
    print(f"빨간 공 생성 완료! Handle: {red_sphere}")

    print("\n=== 3. 녹색 공 생성 ===")
    # 녹색 공
    green_sphere = sim.createPrimitiveShape(1, [0.16, 0.16, 0.16])
    sim.setObjectPosition(green_sphere, -1, [0.3, 0.0, 0.6])
    sim.setShapeColor(green_sphere, None, 0, [0.0, 1.0, 0.0])
    sim.setObjectInt32Param(green_sphere, sim.shapeintparam_respondable, 1)
    sim.setShapeMass(green_sphere, 0.4)
    print(f"녹색 공 생성 완료! Handle: {green_sphere}")

    print("\n=== 4. 파란 공 생성 ===")
    # 파란 공
    blue_sphere = sim.createPrimitiveShape(1, [0.18, 0.18, 0.18])
    sim.setObjectPosition(blue_sphere, -1, [-0.3, 0.0, 0.7])
    sim.setShapeColor(blue_sphere, None, 0, [0.0, 0.0, 1.0])
    sim.setObjectInt32Param(blue_sphere, sim.shapeintparam_respondable, 1)
    sim.setShapeMass(blue_sphere, 0.45)
    print(f"파란 공 생성 완료! Handle: {blue_sphere}")

    print("\n" + "=" * 50)
    print("환경 생성 완료!")
    print("=" * 50)
    print(f"\n생성된 객체:")
    print(f"  - 바닥 (회색): Handle {floor_handle}")
    print(f"  - 빨간 공: Handle {red_sphere}")
    print(f"  - 녹색 공: Handle {green_sphere}")
    print(f"  - 파란 공: Handle {blue_sphere}")
    print("\n시뮬레이션을 시작하려면 CoppeliaSim에서")
    print("재생 버튼(▶)을 클릭하세요!")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        print("\nCoppeliaSim이 실행 중인지 확인하세요!")
