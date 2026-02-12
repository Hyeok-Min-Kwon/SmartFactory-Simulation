# CoppeliaSim ZMQ Remote API 클라이언트 라이브러리를 임포트합니다
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

# 시간 관련 함수를 사용하기 위해 time 모듈을 임포트합니다
import time
import random
# import requests
from AGV  import AGV
# Franka 로봇 제어 모듈 임포트
# from simulation.franka_robot import FrankaRobot
from franka_robot import FrankaRobot
#최적화에 필요한 툴(AGV생성, 랙 위치 소환) 
#AGV생성을 위해 simulation폴더안에 있는 AGV.ttm파일 경로를 확인해주어야 함.
from optimization_tool import Optimization
# CoppeliaSim과 통신할 RemoteAPIClient 객체를 생성합니다
from mask_check import check_ppe_detection

client = RemoteAPIClient()

# sim 모듈을 요청하여 시뮬레이션 API에 접근합니다
sim = client.require('sim')

# 시뮬레이션 stepping 모드 활성화 (로봇 부드러운 움직임을 위해 필수)
client.setStepping(True)

# 시뮬레이션을 시작합니다
if check_ppe_detection():
    sim.startSimulation()
else : 
    raise KeyError


# 시뮬레이션 시작 메시지를 출력합니다
print("시뮬레이션 시작 (Stepping 모드)")

# ========== PPE(마스크) 감지 체크 ==========
# 서버에 요청하여 마스크 착용 여부 확인
# if not check_ppe_detection():
#     print("\n⚠️  마스크 착용이 확인되지 않아 시뮬레이션을 종료합니다.")
#     sim.stopSimulation()
#     exit(1)

# 시뮬레이션 안정화 대기 - stepping 모드에서는 직접 step 호출
for _ in range(25):  # 0.5초 = 25 스텝 (0.02초 * 25)
    client.step()

# Franka로봇 3개 다 불러오기
franka1 = FrankaRobot(sim, client, '/Franka[0]')
franka2 = FrankaRobot(sim, client, '/Franka[1]')
franka3 = FrankaRobot(sim, client, '/Franka[2]')


#AGV 등록
agv1= AGV(sim, client, 0)
agv1.stop()
# 이미 처리한 블록을 추적하는 집합
processed_blocks = set()

# ========== 상자별 블록 데이터 관리 ==========
box_data = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}
defect_to_rack = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}
rack_dummies = {}
for i in range(3):
    rack_dummies[i] = sim.getObject(f'/rack[{i}]/Cuboid/dummy')

# CoppeliaSim 씬에서 'Proximity_sensor'라는 이름의 센서 핸들을 가져옵니다
# 씬에 있는 proximity sensor의 이름을 정확히 입력해야 합니다
proximSensorHandle = sim.getObject('/proximitySensor')
r1proximsensor = sim.getObject('/Franka[0]/proximitySensor')
r2proximsensor = sim.getObject('/Franka[1]/proximitySensor')
r3proximsensor = sim.getObject('/Franka[2]/proximitySensor')
# 센서 핸들을 성공적으로 가져왔음을 출력합니다
# print(f"Proximity Sensor 핸들: {proximSensorHandle}")
# print(f"r1핸들 성공")

# 센서의 현재 위치를 확인합니다 (디버깅용)
sensorPosition = sim.getObjectPosition(proximSensorHandle, sim.handle_world)
# print(f"Proximity Sensor 위치: x={sensorPosition[0]:.3f}, y={sensorPosition[1]:.3f}, z={sensorPosition[2]:.3f}")

# 생성된 cuboid 객체들을 저장할 리스트를 초기화합니다
createdCuboids = {}

# cuboid 생성 카운터를 초기화합니다
cuboidCount = 0

# 마지막으로 cuboid를 생성한 시간을 기록합니다 (초기값은 현재 시간에서 2초를 뺀 값)
lastCuboidTime = time.time() - 10

def place_position_calculator(i,palatizing_position):
    position = palatizing_position[i]
    data = []
    for i in range(3):
        for j in range(3):
            current_x = position[0] + (0.1 * j)
            current_y = position[1] + (0.1 * i)

            new_position = [current_x, current_y, position[2]]

            data.append(new_position)
    return data

# ========== 팔레타이징 설정 ==========
# 0=양품, 1~6=불량종류
palatizing_position = {
    # 0: [4.075, 0.825, 1.25],  # 양품 위치 (Franka1 담당)
    1: [4.075, 0.525, 1.25],
    2: [2.725, 0.525, 1.25],
    3: [1.5, 0.525, 1.25],
    4: [0.075, 0.525, 1.25],
    5: [1.325, -1.025, 1.25],
    6: [2.625, -1.025, 1.25]
}

# 각 키별 9개 좌표 미리 계산
place_positions_data = {}
for key in range(1, 7):  # 0부터 시작 (양품 포함)
    place_positions_data[key] = place_position_calculator(key, palatizing_position)

# 각 키별 배치 카운터 (몇 개 놓았는지 추적)
place_counters = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}

# 각 키별 z축 레이어 오프셋 (9개 채워지면 0.1씩 증가)
z_layer_offsets = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0}

# 로봇별 담당 키 매핑 
robot_config = [
    (franka1, r1proximsensor, 'Franka0', [1, 2]),
    (franka2, r2proximsensor, 'Franka1', [3, 4]),
    (franka3, r3proximsensor, 'Franka2', [5, 6]),
]
multi_interrupt = ''
# 메인 루프: 무한 반복하면서 센서 값을 읽고 cuboid를 생성합니다
try:
    # 무한 루프를 시작합니다
    while True:
        # 현재 시간을 가져옵니다
        currentTime = time.time()

        # ========== Proximity Sensor 값 읽기 ==========
        result = sim.readProximitySensor(proximSensorHandle)
        detectionState = result[0]
        detectedPoint = result[1]

        # ========== 각 로봇 proximity 센서로 블록 감지 (idle 로봇만) ==========
        for robot, sensor_handle, robot_name, assigned_keys in robot_config:
            if not robot.is_busy:
                sensor_result = sim.readProximitySensor(sensor_handle)
                if sensor_result[0]:
                    detected_handle = sensor_result[3]
                    if isinstance(detected_handle, int) and detected_handle in createdCuboids and detected_handle not in processed_blocks:
                        block_number = createdCuboids[detected_handle]
                        if block_number in assigned_keys:
                            block_pos = sim.getObjectPosition(detected_handle, sim.handle_world)
                            processed_blocks.add(detected_handle)

                            key = block_number
                            counter = place_counters[key]
                            z_offset = z_layer_offsets[key]

                            positions = place_positions_data[key]
                            place_pos_with_z = [[p[0], p[1], p[2] + z_offset] for p in positions]

                            # 비동기 pick_and_place 시작 (블로킹하지 않음)
                            robot.start_pick_and_place(
                                block_handle=detected_handle,
                                block_pos=block_pos,
                                place_pos=place_pos_with_z,
                                place_drop_z=-0.25,
                                cuboid_Count=counter,
                                block_number=block_number
                            )

                            place_counters[key] += 1
                            if place_counters[key] % 9 == 0:
                                z_layer_offsets[key] += 0.1

                            box_data[block_number].append(detected_handle)
                            print(f"[box_data] 불량{block_number} 상자: {len(box_data[block_number])}개")

        # ========== Cuboid 생성 ==========
        if currentTime - lastCuboidTime >= 5.0:
            control_signal = sim.getInt32Signal('cube_create')
            if control_signal is not None and multi_interrupt != control_signal:
                cuboidHandle = control_signal
                cuboidCount += 1
                rand_product = random.randint(0, 6)
                createdCuboids[cuboidHandle] = rand_product
                print(createdCuboids)
                lastCuboidTime = currentTime
            multi_interrupt = cuboidHandle

        # ========== AGV 이송 체크 (불량별 5개 이상이면 rack으로 이송) ==========
        for defect_key in range(1, 7):
            if len(box_data[defect_key]) >= 5:
                rack_idx = defect_to_rack[defect_key]
                dummy_handle = rack_dummies[rack_idx]
                transfer_count = len(box_data[defect_key])
                """
                최적화 알고리즘 들어가서 물체를 옮긴 후 아래 transfer_to_rack으로 물체를 옮김
                """
                agv1.transfer_to_rack(box_data[defect_key], dummy_handle)
                box_data[defect_key] = []
                place_counters[defect_key] = 0
                z_layer_offsets[defect_key] = 0.0
                print(f"[AGV] 불량{defect_key} 블록 {transfer_count}개 rack[{rack_idx}]로 이송 완료, 데이터 초기화")

        # ========== 모든 로봇 업데이트 + 시뮬레이션 스텝 (동시 작업) ==========
        for robot, _, _, _ in robot_config:
            robot.update()
        client.step()
        

# Ctrl+C가 눌리면 KeyboardInterrupt 예외가 발생합니다
except KeyboardInterrupt:
    # 종료 메시지를 출력합니다
    print("\n프로그램 종료 요청됨...")

# 프로그램 종료 시 정리 작업을 수행합니다
finally:
    # 모든 Franka 로봇 리소스 정리
    franka1.cleanup()
    franka2.cleanup()
    franka3.cleanup()
    # 시뮬레이션을 중지합니다

    sim.stopSimulation()

    # 최종 종료 메시지를 출력합니다
    print(f"시뮬레이션 종료. 총 {cuboidCount}개의 cuboid가 생성되었습니다.")
    print(f"팔레타이징 현황: {place_counters}")