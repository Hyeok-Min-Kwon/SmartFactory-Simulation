"""
Franka 로봇 제어 모듈
CoppeliaSim ZMQ Remote API를 사용하여 Franka 로봇의 더미 기반 이동,
블록 파지/놓기, 픽앤플레이스 시퀀스를 수행하는 클래스를 제공합니다.

IK는 CoppeliaSim 내부 스크립트에서 처리하고, Python에서는 target dummy만 이동합니다.
"""

import math


class FrankaRobot:
    """CoppeliaSim Franka 로봇 제어 클래스 - 최적화 버전"""

    # Franka Panda 로봇의 조인트 속도 제한 (rad/s)
    JOINT_MAX_VEL = [10,10,10,10,10,10,10]
    # 조인트 가속도 제한 (rad/s^2)
    JOINT_MAX_ACC = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]

    # 시뮬레이션 타임스텝 (50Hz)
    SIM_DT = 0.05

    def __init__(self, sim, client, robot_path='/Franka[1]'):
        """
        Franka 로봇 초기화

        Args:
            sim: CoppeliaSim sim 객체
            client: RemoteAPIClient 객체
            robot_path: 로봇 경로 (기본값: '/Franka[1]')
        """
        self.sim = sim
        self.client = client
        self.robot_path = robot_path
        self.is_busy = False
        self.cuboidCount = 0

        # 로봇 핸들 가져오기
        self.base_handle = sim.getObject(robot_path)
        self.joint_handles = self._get_joints()

        # Target dummy 가져오기 (CoppeliaSim 내부 스크립트에서 IK 처리)
        # Franka[0] -> /_Franka_0__IK_Target, Franka[1] -> /_Franka_1__IK_Target, ...
        robot_index = robot_path.split('[')[1].split(']')[0]  # '/Franka[1]' -> '1'
        target_dummy_name = f'/_Franka_{robot_index}__IK_Target'
        self.target_dummy = sim.getObject(target_dummy_name)

        # ForceSensor 핸들 가져오기 (블록 파지용)
        self.force_sensor = self._find_force_sensor()

        # 초기 상태 저장
        self.initial_tip_position = sim.getObjectPosition(self.target_dummy, sim.handle_world)
        self.initial_tip_orientation = sim.getObjectOrientation(self.target_dummy, sim.handle_world)
        self.initial_joint_positions = [
            sim.getJointPosition(j) for j in self.joint_handles
        ]
        if robot_path == "/Franka[0]":
            
            self.IK_status_object = "Franka0"
        elif robot_path == "/Franka[1]":
            
            self.IK_status_object = "Franka1"
        elif robot_path == "/Franka[2]":
            self.IK_status_object = "Franka2"

        # 조인트 동적 제어 모드 설정 및 속도 제한
        self._setup_joint_control()
        

        print(f"[FrankaRobot] 초기화 완료: {robot_path}")
        print(f"[FrankaRobot] Target Dummy 위치: {[round(v, 3) for v in self.initial_tip_position]}")

    def _setup_joint_control(self):
        """조인트 제어 모드 및 속도/가속도 제한 설정"""
        for i, joint in enumerate(self.joint_handles):
            # 동적 위치 제어 모드 설정
            self.sim.setObjectInt32Param(
                joint,
                self.sim.jointintparam_dynctrlmode,
                self.sim.jointdynctrl_position
            )

            # 조인트 최대 속도 설정
            if i < len(self.JOINT_MAX_VEL):
                self.sim.setJointTargetVelocity(joint, self.JOINT_MAX_VEL[i])

            # 조인트 최대 힘/토크 설정 (더 부드러운 움직임)
            self.sim.setJointTargetForce(joint, 87.0)  # Franka 기본 토크 제한

    def _get_joints(self):
        """로봇의 revolute joint 핸들 목록 반환 (최대 7개)"""
        objects = self.sim.getObjectsInTree(
            self.base_handle, self.sim.handle_all, 0
        )
        joints = []
        for obj in objects:
            try:
                if self.sim.getObjectType(obj) == self.sim.sceneobject_joint:
                    joints.append(obj)
            except Exception:
                continue
        return joints[:7]

    def _find_force_sensor(self):
        """로봇 트리에서 ForceSensor 핸들 탐색"""
        objects = self.sim.getObjectsInTree(
            self.base_handle, self.sim.handle_all, 0
        )
        for obj in objects:
            try:
                alias = self.sim.getObjectAlias(obj, 0)
                if 'forcesensor' in alias.lower():
                    print(f"[FrankaRobot] ForceSensor 발견: {alias}")
                    return obj
            except Exception:
                continue
        print(f"[FrankaRobot] 경고: ForceSensor를 찾을 수 없습니다")
        return None

    @staticmethod
    def _ease_in_out(t):
        """
        부드러운 가감속을 위한 ease-in-out 함수 (smoothstep)

        Args:
            t: 0~1 사이의 진행률
        Returns:
            변환된 진행률 (시작과 끝에서 부드러움)
        """
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0
        # Smootherstep (Ken Perlin)
        return t * t * t * (t * (t * 6 - 15) + 10)

    @staticmethod
    def _ease_out(t):
        """
        빠른 시작, 부드러운 감속 (ease-out)

        Args:
            t: 0~1 사이의 진행률
        Returns:
            변환된 진행률
        """
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0
        return 1 - (1 - t) ** 3

    @staticmethod
    def _ease_in(t):
        """
        부드러운 시작, 빠른 가속 (ease-in)

        Args:
            t: 0~1 사이의 진행률
        Returns:
            변환된 진행률
        """
        if t <= 0:
            return 0.0
        if t >= 1:
            return 1.0
        return t ** 3


    def _step_simulation(self, steps=1):
        """
        시뮬레이션 스텝 진행

        Args:
            steps: 진행할 스텝 수
        """
        for _ in range(steps):
            self.client.step()

    def _get_distance(self, pos1, pos2):
        """두 위치 사이의 유클리드 거리 계산"""
        return math.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2 +
            (pos1[2] - pos2[2]) ** 2
        )

    def _interpolate_position(self, start, end, t):
        """두 위치 사이를 선형 보간"""
        return [
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
            start[2] + (end[2] - start[2]) * t,
        ]

    def move_to_position(self, target_pos, speed=2.0, ease_type='smooth'):
        """
        [최적화됨] 통신 빈도를 줄여 이동 속도 향상
        """
        start_pos = self.sim.getObjectPosition(self.target_dummy, self.sim.handle_world)
        distance = self._get_distance(start_pos, target_pos)

        if distance < 0.001:
            return

        # 이동 시간 계산
        duration = distance / speed

        # 최소 스텝 수 보장 (IK가 따라갈 수 있도록 충분히)
        total_steps = max(int(duration / self.SIM_DT), 10)

        # ease 함수 선택 (기존 코드 유지)
        if ease_type == 'smooth':
            ease_func = self._ease_in_out
        elif ease_type == 'fast_start':
            ease_func = self._ease_out
        elif ease_type == 'fast_end':
            ease_func = self._ease_in
        else:
            ease_func = lambda t: t

        for step in range(1, total_steps + 1):
            t_linear = step / total_steps
            t_eased = ease_func(t_linear)

            interp_pos = self._interpolate_position(start_pos, target_pos, t_eased)
            
            # 타겟 더미 이동 (CoppeliaSim 내부 스크립트에서 IK 처리)
            self.sim.setObjectPosition(self.target_dummy, interp_pos, self.sim.handle_world)

            # 매 스텝마다 시뮬레이션 진행 (IK가 따라갈 수 있도록)
            self._step_simulation(1)

    def move_to_position_fast(self, target_pos, max_speed=4.0):
        """
        빠른 이동 - 최대 속도로 목표까지 이동

        Args:
            target_pos: [x, y, z] 월드 좌표
            max_speed: 최대 속도 (m/s)
        """
        self.move_to_position(target_pos, speed=max_speed, ease_type='smooth')

    def descend_to_position(self, target_pos, speed=1.2):
        """
        하강 전용 - 부드럽게 하강

        Args:
            target_pos: [x, y, z] 월드 좌표
            speed: 하강 속도 (m/s) - 기본값 1.2m/s
        """
        self.move_to_position(target_pos, speed=speed, ease_type='fast_start')

    def ascend_to_position(self, target_pos, speed=2.4):
        """
        상승 전용 - 빠르게 상승

        Args:
            target_pos: [x, y, z] 월드 좌표
            speed: 상승 속도 (m/s) - 기본값 2.4m/s
        """
        self.move_to_position(target_pos, speed=speed, ease_type='fast_end')

    def grasp(self, block_handle):
        """
        블록 파지 - 블록을 force_sensor의 자식으로 설정

        Args:
            block_handle: 파지할 블록 핸들
        """
        # 충돌 비활성화
        self.sim.setObjectInt32Param(
            block_handle,
            self.sim.shapeintparam_respondable,
            0
        )
        self.sim.resetDynamicObject(block_handle)
        # force_sensor가 있으면 force_sensor에, 없으면 target_dummy에 부착
        parent_handle = self.force_sensor if self.force_sensor else self.target_dummy
        self.sim.setObjectParent(block_handle, parent_handle, True)
        print("[FrankaRobot] 블록 파지 완료")

    def release(self, block_handle):
        """
        블록 놓기 - 부모 해제, dynamic으로 복원

        Args:
            block_handle: 놓을 블록 핸들
        """
        # 부모에서 분리
        self.sim.setObjectParent(block_handle, -1, True)

        # dynamic + respondable 복원
        self.sim.setObjectInt32Param(
            block_handle, self.sim.shapeintparam_static, 0
        )
        self.sim.setObjectInt32Param(
            block_handle, self.sim.shapeintparam_respondable, 1
        )
        self.sim.resetDynamicObject(block_handle)
        print("[FrankaRobot] 블록 놓기 완료")

    def return_to_initial_pose(self, speed=2.0):
        """초기 dummy 위치/자세로 복귀"""
        self.move_to_position(self.initial_tip_position, speed=speed, ease_type='smooth')

        # target_dummy orientation 복원
        self.sim.setObjectOrientation(
            self.target_dummy, self.initial_tip_orientation, self.sim.handle_world
        )
        self._step_simulation(15)
        print("[FrankaRobot] 초기 자세 복귀 완료")

    def track_to_object(self, block_handle, z_offset=0.15, max_speed=0.5,
                        threshold=0.05, timeout=10.0):
        """
        이동 중인 블록을 실시간 추적하며 블록 위(z_offset)까지 접근

        Args:
            block_handle: 추적할 블록 핸들
            z_offset: 블록 위 접근 높이(m)
            max_speed: 최대 추적 속도(m/s)
            threshold: 도착 판정 거리(m)
            timeout: 최대 추적 시간(초)
        Returns:
            블록의 최종 위치 [x, y, z]
        """
        max_steps = int(timeout / self.SIM_DT)

        for step in range(max_steps):
            # 블록 현재 위치
            block_pos = self.sim.getObjectPosition(block_handle, self.sim.handle_world)
            goal = [block_pos[0], block_pos[1], block_pos[2] + z_offset]

            # 현재 타겟 위치
            current = self.sim.getObjectPosition(self.target_dummy, self.sim.handle_world)

            # 목표까지 거리
            dist_to_goal = self._get_distance(current, goal)

            if dist_to_goal < threshold:
                # 도착 - IK 안정화 대기
                self._step_simulation(15)
                print(f"[FrankaRobot] 추적 도달 (step={step})")
                return block_pos

            # 속도 계산 - 거리에 비례하되 최대값 제한
            # 가까워질수록 천천히 (proportional control)
            speed_factor = min(1.0, dist_to_goal / 0.3)  # 30cm 이내에서 감속
            current_speed = max_speed * (0.3 + 0.7 * speed_factor)

            # 이번 스텝에서 이동할 최대 거리
            max_step_dist = current_speed * self.SIM_DT

            if dist_to_goal <= max_step_dist:
                new_pos = goal
            else:
                ratio = max_step_dist / dist_to_goal
                new_pos = self._interpolate_position(current, goal, ratio)

            # 타겟 더미 업데이트
            self.sim.setObjectPosition(self.target_dummy, new_pos, self.sim.handle_world)
            self._step_simulation()

        print(f"[FrankaRobot] 추적 타임아웃")
        return self.sim.getObjectPosition(block_handle, self.sim.handle_world)

    def descend_to_object(self, block_handle, z_offset=0.08, max_speed=0.3,
                          threshold=0.03, timeout=10.0):
        """
        블록으로 하강 - x, y 추적하며 z 방향 하강

        Args:
            block_handle: 대상 블록 핸들
            z_offset: 블록 중심으로부터의 z 오프셋(m)
            max_speed: 최대 하강 속도(m/s)
            threshold: 도착 판정 거리(m)
            timeout: 최대 시간(초)
        Returns:
            최종 tip 위치 [x, y, z]
        """
        max_steps = int(timeout / self.SIM_DT)

        for step in range(max_steps):
            # 블록 현재 위치
            block_pos = self.sim.getObjectPosition(block_handle, self.sim.handle_world)
            goal = [block_pos[0], block_pos[1], block_pos[2] + z_offset]

            # 현재 타겟 위치
            current = self.sim.getObjectPosition(self.target_dummy, self.sim.handle_world)

            # 목표까지 거리
            dist_to_goal = self._get_distance(current, goal)

            if dist_to_goal < threshold:
                # 도착 - IK 안정화 대기
                self._step_simulation(15)
                print(f"[FrankaRobot] 하강 도달 (step={step})")
                return self.sim.getObjectPosition(self.target_dummy, self.sim.handle_world)

            # 하강 시 더 부드러운 감속 (거리에 따라)
            speed_factor = min(1.0, dist_to_goal / 0.15)  # 15cm 이내에서 감속
            current_speed = max_speed * (0.2 + 0.8 * speed_factor)

            # 이번 스텝에서 이동할 최대 거리
            max_step_dist = current_speed * self.SIM_DT

            if dist_to_goal <= max_step_dist:
                new_pos = goal
            else:
                ratio = max_step_dist / dist_to_goal
                new_pos = self._interpolate_position(current, goal, ratio)

            # 타겟 더미 업데이트
            self.sim.setObjectPosition(self.target_dummy, new_pos, self.sim.handle_world)
            self._step_simulation()

        print(f"[FrankaRobot] 하강 타임아웃")
        return self.sim.getObjectPosition(self.target_dummy, self.sim.handle_world)

    def direct_approach_to_object(self, block_handle, z_offset=0.08, max_speed=0.5,
                                   threshold=0.02, timeout=5.0):
        """
        블록으로 직접 빠르게 접근 - 중간 단계 없이 바로 블록 위치로 이동

        Args:
            block_handle: 대상 블록 핸들
            z_offset: 블록 중심으로부터의 z 오프셋(m)
            max_speed: 최대 속도(m/s) - 매우 빠름
            threshold: 도착 판정 거리(m)
            timeout: 최대 시간(초)
        Returns:
            블록의 최종 위치 [x, y, z]
        """
        max_steps = int(timeout / self.SIM_DT)

        for step in range(max_steps):
            # 블록 현재 위치
            block_pos = self.sim.getObjectPosition(block_handle, self.sim.handle_world)
            goal = [block_pos[0], block_pos[1], block_pos[2] + z_offset]

            # 현재 타겟 위치
            current = self.sim.getObjectPosition(self.target_dummy, self.sim.handle_world)

            # 목표까지 거리
            dist_to_goal = self._get_distance(current, goal)

            if dist_to_goal < threshold:
                # 도착 - IK 안정화 대기
                self._step_simulation(1)
                print(f"[FrankaRobot] 블록 도달 (step={step}, dist={dist_to_goal:.3f})")
                return block_pos

            # 거리에 관계없이 최대 속도로 이동 (감속 없음)
            max_step_dist = max_speed * self.SIM_DT

            if dist_to_goal <= max_step_dist:
                new_pos = goal
            else:
                ratio = max_step_dist / dist_to_goal
                new_pos = self._interpolate_position(current, goal, ratio)

            # 타겟 더미 업데이트
            self.sim.setObjectPosition(self.target_dummy, new_pos, self.sim.handle_world)
            self._step_simulation()

        print(f"[FrankaRobot] 접근 타임아웃 - 강제 파지 시도")
        return self.sim.getObjectPosition(block_handle, self.sim.handle_world)

    def rotate_joint1(self, angle_deg):
        """
        첫 번째 조인트를 지정 각도로 회전 후 dummy 위치 동기화

        Args:
            angle_deg: 회전 각도 (도 단위, 90 또는 -90)
        """
        if len(self.joint_handles) < 1:
            print("[FrankaRobot] 경고: 조인트를 찾을 수 없습니다")
            return

        angle_rad = math.radians(angle_deg)
        joint1 = self.joint_handles[0]

        # 현재 조인트 위치에서 목표 각도까지 점진적으로 이동
        current_angle = self.sim.getJointPosition(joint1)
        target_angle = angle_rad

        # 스텝 수 계산 (부드러운 회전을 위해)
        steps = 20
        for step in range(1, steps + 1):
            t = step / steps
            interp_angle = current_angle + (target_angle - current_angle) * t
            self.sim.setJointTargetPosition(joint1, interp_angle)
            self._step_simulation(1)

        # 회전 후 dummy 위치를 force_sensor 위치로 동기화
        if self.force_sensor:
            new_pos = self.sim.getObjectPosition(self.force_sensor, self.sim.handle_world)
            new_ori = self.sim.getObjectOrientation(self.force_sensor, self.sim.handle_world)
            self.sim.setObjectPosition(self.target_dummy, new_pos, self.sim.handle_world)
            self.sim.setObjectOrientation(self.target_dummy, new_ori, self.sim.handle_world)
            self._step_simulation(5)

        print(f"[FrankaRobot] 조인트1 회전 완료: {angle_deg}도, dummy 동기화됨")
        
    # def init_position()

    def pick_and_place(self, block_handle, block_pos, place_pos, place_drop_z=-0.25, cuboid_Count=0, block_number=0):
        """
        블록 픽앤플레이스 전체 시퀀스

        Args:
            block_handle: 블록 핸들
            block_pos: 블록 감지 시점 위치 [x, y, z] (로그용)
            place_pos: 놓을 기준 위치 리스트
            place_drop_z: 놓을 때 z 방향 추가 이동량
            cuboid_Count: 현재 큐보이드 카운터
            block_number: 불량 종류 (1,3,6은 +90도, 2,4,5는 -90도)
        """
        self.is_busy = True

        try:
            # 1) 블록으로 접근 (블록 바로 위 0.08m)
            print(f"[FrankaRobot] 블록 접근 시작")
            block_pos = self.direct_approach_to_object(
                block_handle,
                z_offset=0.08,
                max_speed=1.0,
                threshold=0.02,
                timeout=5.0
            )
            print(f"[FrankaRobot] 블록 도착: {[round(v, 3) for v in block_pos]}")

            # IK 안정화 대기
            # self._step_simulation(3)

            # 2) 블록 파지
            self.grasp(block_handle)

            # 3) 위로 들어올림
            current_pos = self.sim.getObjectPosition(self.target_dummy, self.sim.handle_world)
            lift_pos = [current_pos[0], current_pos[1], current_pos[2] + 0.15]
            self.ascend_to_position(lift_pos, speed=0.5)

            # 4) 불량 종류에 따라 첫 번째 조인트 회전 (IK 실패 방지)
            self.sim.setInt32Signal(self.IK_status_object,0)
            if block_number in [1, 3, 5]:
                print(f"[FrankaRobot] 불량 {block_number}: 조인트1 +90도 회전")
                self.rotate_joint1(90)
            elif block_number in [2, 4, 6]:
                print(f"[FrankaRobot] 불량 {block_number}: 조인트1 -90도 회전")
                self.rotate_joint1(-90)
            self.sim.setInt32Signal(self.IK_status_object, 1)
            
            # 5) 목표 위치로 이동
            index = cuboid_Count % 9
            target_place = place_pos[index]
            print(f"[FrankaRobot] 목표 위치 이동: {[round(v, 3) for v in target_place]}")
            self.move_to_position_fast(target_place, max_speed=2.0)

            # 5) z 방향 하강
            drop_pos = [target_place[0], target_place[1], target_place[2] + place_drop_z]
            print(f"[FrankaRobot] 하강: {[round(v, 3) for v in drop_pos]}")
            self.descend_to_position(drop_pos, speed=0.7)

            # 6) 블록 놓기
            self.release(block_handle)

            # 잠시 대기 (블록 안정화)
            # 7) 약간 후퇴(위로)
            retreat_pos = [drop_pos[0], drop_pos[1], drop_pos[2] + 0.3]
            self.ascend_to_position(retreat_pos, speed=0.5)
            
            
            self.sim.setInt32Signal(self.IK_status_object,0)
            
            self.sim.setInt32Signal(self.IK_status_object,1)
            # 8) 초기 자세 복귀
            self.return_to_initial_pose(speed=2.0)

            print("[FrankaRobot] 픽앤플레이스 완료")

        except Exception as e:
            print(f"[FrankaRobot] 오류: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_busy = False

    def cleanup(self):
        """정리 (target_dummy는 씬에 있으므로 삭제하지 않음)"""
        self.target_dummy = None
