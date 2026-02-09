"""
Franka 로봇 IK 제어 스크립트
- Dummy를 엔드이펙터 위치에 생성
- 마우스로 Dummy를 움직이면 IK가 풀리며 로봇이 따라감
"""

from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
from ik_solver import IKSolver


def create_target_dummy(sim, tip_handle, dummy_name="FrankaTarget"):
    """
    엔드이펙터(tip) 위치에 타겟 Dummy 생성

    Args:
        sim: CoppeliaSim sim 객체
        tip_handle: 엔드이펙터 핸들
        dummy_name: 생성할 dummy 이름

    Returns:
        dummy_handle: 생성된 dummy 핸들
    """
    # 엔드이펙터의 현재 위치와 자세 가져오기
    tip_position = sim.getObjectPosition(tip_handle, sim.handle_world)
    tip_orientation = sim.getObjectOrientation(tip_handle, sim.handle_world)

    # Dummy 생성
    dummy_handle = sim.createDummy(0.05)  # 크기 0.05m

    # Dummy 이름 설정
    sim.setObjectAlias(dummy_handle, dummy_name)

    # Dummy를 엔드이펙터 위치로 이동
    sim.setObjectPosition(dummy_handle, tip_position, sim.handle_world)
    sim.setObjectOrientation(dummy_handle, tip_orientation, sim.handle_world)

    # Dummy 색상 설정 (빨간색으로 구분)
    sim.setObjectColor(dummy_handle, 0, sim.colorcomponent_ambient_diffuse, [1, 0, 0])

    print(f"타겟 Dummy 생성 완료: {dummy_name}")
    print(f"  - 위치: {tip_position}")

    return dummy_handle


def get_franka_handles(sim):
    """
    Franka 로봇의 주요 핸들 가져오기

    Returns:
        dict: 로봇 관련 핸들들
    """
    handles = {}

    # 로봇 베이스 (Franka 로봇 루트)
    handles['base'] = sim.getObject(":/Franka")
    print(f"로봇 베이스 핸들: {handles['base']}")

    # 엔드이펙터 (tip) - Franka의 끝점
    # Franka 로봇에서 tip은 보통 "connection" 또는 "tip" 이름을 가짐
    try:
        handles['tip'] = sim.getObject(":/Franka/connection")
    except Exception:
        # 다른 이름으로 시도
        try:
            handles['tip'] = sim.getObject(":/Franka/Franka_tip")
        except Exception:
            # 로봇 트리에서 마지막 링크 찾기
            all_objects = sim.getObjectsInTree(handles['base'], sim.handle_all, 0)
            # 가장 마지막 더미나 연결점 찾기
            for obj in all_objects:
                alias = sim.getObjectAlias(obj, 0)
                if 'tip' in alias.lower() or 'connection' in alias.lower():
                    handles['tip'] = obj
                    break

    print(f"엔드이펙터(tip) 핸들: {handles['tip']}")

    return handles


def print_joint_positions(sim, robot_handle):
    """로봇의 모든 조인트 위치 출력"""
    all_objects = sim.getObjectsInTree(robot_handle, sim.handle_all, 0)
    joint_type = sim.sceneobject_joint

    print("\n--- Joint Positions ---")
    for obj in all_objects:
        if sim.getObjectType(obj) == joint_type:
            name = sim.getObjectAlias(obj, 0)
            pos = sim.getJointPosition(obj)
            print(f"  {name}: {pos:.4f} rad")


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("Franka 로봇 IK 제어 시작")
    print("=" * 50)

    # 1. CoppeliaSim 연결
    client = RemoteAPIClient()
    sim = client.require('sim')

    # 2. 시뮬레이션 시작 (IK가 동작하려면 필수)
    sim.startSimulation()
    print("시뮬레이션 시작됨")

    # 잠시 대기 (시뮬레이션 안정화)
    time.sleep(0.5)

    # 3. Franka 로봇 핸들 가져오기
    handles = get_franka_handles(sim)

    # 4. 타겟 Dummy 생성 (엔드이펙터 위치에)
    target_handle = create_target_dummy(sim, handles['tip'])

    # 5. IK 솔버 초기화
    ik_solver = IKSolver(sim)
    ik_solver.setup_ik_from_scene(client)

    # 6. IK 그룹 생성
    ik_solver.create_ik_group(
        base_handle=handles['base'],
        tip_handle=handles['tip'],
        target_handle=target_handle
    )
    print("IK 설정 완료")

    # 7. 메인 루프 - Dummy 위치를 추적하며 IK 계산
    print("\n" + "=" * 50)
    print("마우스로 'FrankaTarget' Dummy를 움직여보세요!")
    print("CoppeliaSim에서 Dummy를 선택 후 드래그하면 됩니다.")
    print("종료하려면 Ctrl+C를 누르세요.")
    print("=" * 50 + "\n")

    try:
        loop_count = 0
        while True:
            # IK 계산 (Dummy 위치가 변경되면 로봇이 따라감)
            success = ik_solver.solve()

            # 1초마다 상태 출력
            loop_count += 1
            if loop_count % 50 == 0:
                target_pos = sim.getObjectPosition(target_handle, sim.handle_world)
                tip_pos = sim.getObjectPosition(handles['tip'], sim.handle_world)
                print(f"타겟 위치: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                print(f"Tip 위치:  [{tip_pos[0]:.3f}, {tip_pos[1]:.3f}, {tip_pos[2]:.3f}]")
                print(f"IK 상태: {'성공' if success else '실패'}")
                print("-" * 30)

            # 시뮬레이션 스텝 대기
            time.sleep(0.02)  # 50Hz

    except KeyboardInterrupt:
        print("\n\n종료 중...")

    finally:
        # 정리 작업
        ik_solver.cleanup()
        sim.stopSimulation()
        print("시뮬레이션 종료됨")


if __name__ == "__main__":
    main()
