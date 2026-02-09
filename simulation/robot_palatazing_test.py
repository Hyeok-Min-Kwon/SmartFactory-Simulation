from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import json

client = RemoteAPIClient()
sim = client.require('sim')
def _find_tip(robot_path='/Franka[2]'):
    base_handle = sim.getObject(robot_path)
    base_handle = sim.getObject(robot_path)
    """로봇 tip(엔드이펙터) 핸들 탐색"""
    objects = sim.getObjectsInTree(
        base_handle, sim.handle_all, 0
    )

    # forceSensor를 tip으로 사용
    for obj in objects[::-1]:
        try:
            alias = sim.getObjectAlias(obj, 0)
            if 'forcesensor' in alias.lower():
                print(f"[FrankaRobot] ForceSensor를 Tip으로 사용: {alias}")
                return obj
        except Exception:
            continue

    # 그래도 없으면 마지막 link
    last_link = None
    last_link_name = ""
    for obj in objects:
        try:
            alias = sim.getObjectAlias(obj, 0)
            if 'link' in alias.lower():
                last_link = obj
                last_link_name = alias
        except Exception:
            continue
    if last_link is not None:
        print(f"[FrankaRobot] Tip 대신 마지막 link 사용: {last_link_name}")
        return last_link

    raise Exception(f"Tip을 찾을 수 없습니다: {robot_path}")

def _get_joints(robot_path = '/Franka[2]'):
    """로봇의 revolute joint 핸들 목록 반환 (최대 7개)"""
    base_handle = sim.getObject(robot_path)
    objects = sim.getObjectsInTree(
        base_handle, sim.handle_all, 0
    )
    joints = []
    for obj in objects:
        try:
            if sim.getObjectType(obj) == sim.sceneobject_joint:
                joints.append(obj)
        except Exception:
            continue
    return joints[:7]
def _create_target_dummy():
        """IK 타겟 더미 생성 (tip 위치에 배치)"""
        robot_path = '/Franka[2]'
        dummy = sim.createDummy(0.01)
        tip_handle = _find_tip()
        joint_handles = _get_joints()
        sim.setObjectAlias(dummy, f'{robot_path}_IK_Target')
        pos = sim.getObjectPosition(tip_handle, sim.handle_world)
        ori = sim.getObjectOrientation(tip_handle, sim.handle_world)
        sim.setObjectPosition(dummy, pos, sim.handle_world)
        sim.setObjectOrientation(dummy, ori, sim.handle_world)
        return dummy
target_dummy = _create_target_dummy()