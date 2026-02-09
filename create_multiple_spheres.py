"""
CoppeliaSim에서 여러 개의 공(sphere)을 생성하는 Python 스크립트
"""
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import random


def create_sphere(sim, position, radius=0.1, color=None, mass=0.5):
    """
    공을 생성하는 헬퍼 함수

    Args:
        sim: CoppeliaSim API 객체
        position: [x, y, z] 위치
        radius: 공의 반지름 (기본값: 0.1m)
        color: RGB 색상 [r, g, b] (기본값: 랜덤)
        mass: 질량 (kg)

    Returns:
        sphere_handle: 생성된 공의 핸들
    """
    # 공 생성
    sphere_handle = sim.createPrimitiveShape(
        1,  # sphere type
        [radius * 2, radius * 2, radius * 2]
    )

    # 위치 설정
    sim.setObjectPosition(sphere_handle, -1, position)

    # 색상 설정 (기본값: 랜덤)
    if color is None:
        color = [random.random(), random.random(), random.random()]

    sim.setShapeColor(sphere_handle, None, 0, color)

    # 동적 속성 활성화
    sim.setObjectInt32Param(
        sphere_handle,
        sim.shapeintparam_respondable,
        1
    )

    # 질량 설정
    sim.setShapeMass(sphere_handle, mass)

    return sphere_handle


def main():
    print("CoppeliaSim에 연결 중...")

    # CoppeliaSim에 연결
    client = RemoteAPIClient()
    sim = client.require('sim')

    print("연결 성공!")

    # 시뮬레이션 정지
    sim.stopSimulation()
    time.sleep(0.5)

    # 여러 개의 공 생성
    num_spheres = 5
    sphere_handles = []

    print(f"\n{num_spheres}개의 공을 생성합니다...")

    for i in range(num_spheres):
        # 위치를 격자 형태로 배치
        x = (i % 3 - 1) * 0.3  # -0.3, 0, 0.3
        y = (i // 3) * 0.3
        z = 0.5 + i * 0.2  # 높이를 다르게 설정

        position = [x, y, z]

        # 반지름을 다양하게 설정
        radius = 0.05 + (i * 0.02)

        # 공 생성
        handle = create_sphere(
            sim,
            position=position,
            radius=radius,
            mass=0.3 + i * 0.1
        )

        sphere_handles.append(handle)

        print(f"  공 {i+1}: 위치={position}, 반지름={radius:.3f}m")

    print(f"\n=== {num_spheres}개의 공 생성 완료 ===")
    print(f"핸들 목록: {sphere_handles}")
    print("\nCoppeliaSim에서 시뮬레이션을 시작하면 공들이 떨어집니다!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"오류 발생: {e}")
        print("\nCoppeliaSim이 실행 중인지 확인하세요!")
