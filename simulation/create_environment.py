"""
CoppeliaSim에서 바닥과 공이 있는 완전한 환경을 생성하는 스크립트
"""
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time


def create_floor(sim):
    """바닥 생성"""
    # 큰 직육면체를 바닥으로 사용
    floor_handle = sim.createPrimitiveShape(
        0,  # cuboid type
        [2.0, 2.0, 0.1]  # 2m x 2m x 0.1m
    )

    # 바닥 위치 설정
    sim.setObjectPosition(floor_handle, -1, [0.0, 0.0, 0.0])

    # 바닥 색상 (회색)
    sim.setShapeColor(floor_handle, None, 0, [0.5, 0.5, 0.5])

    # 바닥을 정적 객체로 설정
    sim.setObjectInt32Param(
        floor_handle,
        sim.shapeintparam_static,
        1
    )

    # 바닥을 respondable로 설정 (다른 객체와 충돌 가능)
    sim.setObjectInt32Param(
        floor_handle,
        sim.shapeintparam_respondable,
        1
    )

    print("바닥 생성 완료")
    return floor_handle


def create_sphere(sim, position, radius=0.1, color=None):
    """공 생성"""
    sphere_handle = sim.createPrimitiveShape(
        1,  # sphere type
        [radius * 2, radius * 2, radius * 2]
    )

    sim.setObjectPosition(sphere_handle, -1, position)

    if color is None:
        color = [1.0, 0.0, 0.0]  # 기본 빨간색

    sim.setShapeColor(sphere_handle, None, 0, color)

    # 동적 객체로 설정
    sim.setObjectInt32Param(
        sphere_handle,
        sim.shapeintparam_respondable,
        1
    )

    sim.setShapeMass(sphere_handle, 0.5)

    return sphere_handle


def main():
    print("=" * 50)
    print("CoppeliaSim 환경 생성 스크립트")
    print("=" * 50)

    print("\nCoppeliaSim에 연결 중...")

    # CoppeliaSim에 연결
    client = RemoteAPIClient()
    sim = client.require('sim')

    print("연결 성공!")

    # 시뮬레이션 정지
    sim.stopSimulation()
    time.sleep(1.0)  # 충분한 대기 시간

    print("\n환경을 생성합니다...")
    print("(기존 객체들은 유지되며 새로운 환경이 추가됩니다)")

    # 1. 바닥 생성
    floor_handle = create_floor(sim)

    # 2. 공 생성
    sphere_handle = create_sphere(
        sim,
        position=[0.0, 0.0, 1.0],  # 바닥에서 1m 높이
        radius=0.1,
        color=[1.0, 0.0, 0.0]  # 빨간색
    )

    print(f"공 생성 완료 (Handle: {sphere_handle})")

    # 3. 추가 공들 생성
    additional_spheres = []
    positions = [
        [0.3, 0.3, 0.8],
        [-0.3, 0.3, 0.9],
        [0.3, -0.3, 1.1],
        [-0.3, -0.3, 0.7]
    ]

    colors = [
        [0.0, 1.0, 0.0],  # 녹색
        [0.0, 0.0, 1.0],  # 파란색
        [1.0, 1.0, 0.0],  # 노란색
        [1.0, 0.0, 1.0]   # 자홍색
    ]

    for i, (pos, color) in enumerate(zip(positions, colors)):
        handle = create_sphere(sim, pos, radius=0.08, color=color)
        additional_spheres.append(handle)

    print(f"추가 공 {len(additional_spheres)}개 생성 완료")

    print("\n" + "=" * 50)
    print("환경 생성 완료!")
    print("=" * 50)
    print(f"\n생성된 객체:")
    print(f"  - 바닥: {floor_handle}")
    print(f"  - 메인 공: {sphere_handle}")
    print(f"  - 추가 공들: {additional_spheres}")
    print("\n시뮬레이션을 시작하려면 CoppeliaSim에서")
    print("재생 버튼(▶)을 클릭하세요!")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n오류 발생: {e}")
        print("\n해결 방법:")
        print("1. CoppeliaSim이 실행되어 있는지 확인하세요")
        print("2. CoppeliaSim의 ZMQ Remote API가 활성화되어 있는지 확인하세요")
        print("3. 방화벽이 localhost 연결을 차단하지 않는지 확인하세요")
