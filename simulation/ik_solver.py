"""
IK Solver 모듈
CoppeliaSim의 simIK API를 사용하여 역기구학을 계산하는 모듈
"""


class IKSolver:
    """역기구학 솔버 클래스"""

    def __init__(self, sim):
        """
        IK 솔버 초기화

        Args:
            sim: CoppeliaSim의 sim 객체
        """
        self.sim = sim
        # simIK 모듈 로드
        self.simIK = sim.getObject  # simIK는 별도로 require해야 함
        self.ik_env = None
        self.ik_group = None

    def setup_ik_from_scene(self, client, ik_group_name="ik"):
        """
        씬에 이미 설정된 IK 그룹을 사용하여 IK 환경 설정

        Args:
            client: RemoteAPIClient 객체
            ik_group_name: IK 그룹 이름 (기본값: "ik")
        """
        # simIK 모듈 가져오기
        self.simIK = client.require('simIK')

        # IK 환경 생성
        self.ik_env = self.simIK.createEnvironment()

        return self.ik_env

    def create_ik_group(self, base_handle, tip_handle, target_handle):
        """
        IK 그룹 생성

        Args:
            base_handle: 로봇 베이스 핸들
            tip_handle: 엔드이펙터(tip) 핸들
            target_handle: 타겟(dummy) 핸들

        Returns:
            ik_group: 생성된 IK 그룹 핸들
        """
        if self.ik_env is None:
            raise Exception("IK 환경이 초기화되지 않았습니다. setup_ik_from_scene()을 먼저 호출하세요.")

        # IK 그룹 생성
        self.ik_group = self.simIK.createGroup(self.ik_env)

        # IK 요소 추가 (씬에서 가져오기)
        # constraints: position + orientation (X, Y, Z, Alpha, Beta, Gamma)
        self.simIK.addElementFromScene(
            self.ik_env,
            self.ik_group,
            base_handle,
            tip_handle,
            target_handle,
            self.simIK.constraint_position | self.simIK.constraint_orientation
        )

        return self.ik_group

    def solve(self):
        """
        IK 계산 실행

        Returns:
            bool: IK 계산 성공 여부
        """
        if self.ik_env is None or self.ik_group is None:
            raise Exception("IK가 설정되지 않았습니다.")

        # IK 그룹 핸들링 (IK 계산 수행)
        result = self.simIK.handleGroup(
            self.ik_env,
            self.ik_group,
            {"syncWorlds": True}
        )

        return result == self.simIK.result_success

    def solve_with_target_position(self, target_handle, position):
        """
        타겟 위치를 설정하고 IK 계산

        Args:
            target_handle: 타겟(dummy) 핸들
            position: [x, y, z] 목표 위치

        Returns:
            bool: IK 계산 성공 여부
        """
        # 타겟 위치 설정
        self.sim.setObjectPosition(target_handle, position, self.sim.handle_world)

        # IK 계산
        return self.solve()

    def solve_with_target_pose(self, target_handle, position, orientation):
        """
        타겟 위치와 자세를 설정하고 IK 계산

        Args:
            target_handle: 타겟(dummy) 핸들
            position: [x, y, z] 목표 위치
            orientation: [alpha, beta, gamma] 목표 자세 (오일러 각도)

        Returns:
            bool: IK 계산 성공 여부
        """
        # 타겟 위치 설정
        self.sim.setObjectPosition(target_handle, position, self.sim.handle_world)
        # 타겟 자세 설정
        self.sim.setObjectOrientation(target_handle, orientation, self.sim.handle_world)

        # IK 계산
        return self.solve()

    def cleanup(self):
        """IK 환경 정리"""
        if self.ik_env is not None:
            self.simIK.eraseEnvironment(self.ik_env)
            self.ik_env = None
            self.ik_group = None
