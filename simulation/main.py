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
# CoppeliaSim과 통신할 RemoteAPIClient 객체를 생성합니다
client = RemoteAPIClient()

# sim 모듈을 요청하여 시뮬레이션 API에 접근합니다
sim = client.require('sim')

# 시뮬레이션 stepping 모드 활성화 (로봇 부드러운 움직임을 위해 필수)
client.setStepping(True)

# 시뮬레이션을 시작합니다
sim.startSimulation()

# def check_ppe_detection(server_url="13.125.121.143:8000"):
#     """
#     서버에 PPE 감지 요청을 보내고 결과를 받아옵니다.

#     Args:
#         server_url: 서버 URL (기본값: http://13.125.121.143:8000)

#     Returns:
#         bool: 마스크 착용이 확인되면 True, 아니면 False
#     """
#     try:
#         print("\n" + "="*50)
#         print("PPE(마스크) 감지 시작...")
#         print("서버에 얼굴인식 요청 전송 중...")

#         # 서버에 PPE 감지 요청 (POST 방식으로 트리거)
#         response = requests.post(
#             f"{server_url}/api/v1/ppe/check",
#             timeout=15  # 10초 타임아웃
#         )

#         if response.status_code == 200:
#             result = response.json()
#             print(f"서버 응답: {result}")

#             # 결과 확인
#             if result.get("status") == "success" and result.get("mask_detected"):
#                 print("✅ 마스크 착용 확인! 시뮬레이션을 시작합니다.")
#                 print("="*50 + "\n")
#                 return True
#             else:
#                 print("❌ 마스크 미착용 또는 감지 실패!")
#                 print(f"사유: {result.get('message', '알 수 없음')}")
#                 print("="*50 + "\n")
#                 return False
#         else:
#             print(f"❌ 서버 오류: HTTP {response.status_code}")
#             print("="*50 + "\n")
#             return False

#     except requests.exceptions.ConnectionError:
#         print("❌ 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
#         print("="*50 + "\n")
#         return False
#     except requests.exceptions.Timeout:
#         print("❌ 요청 시간 초과 (15초)")
#         print("="*50 + "\n")
#         return False
#     except Exception as e:
#         print(f"❌ 예상치 못한 오류: {e}")
#         print("="*50 + "\n")
#         return False

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
    # while cuboidCount <1:
        # 현재 시간을 가져옵니다
        currentTime = time.time()

        # ========== Proximity Sensor 값 읽기 ==========
        result = sim.readProximitySensor(proximSensorHandle)
        detectionState = result[0]
        detectedPoint = result[1]

        # ========== 각 로봇 proximity 센서로 블록 감지 및 팔레타이징 ==========
        for robot, sensor_handle, robot_name, assigned_keys in robot_config:
            sensor_result = sim.readProximitySensor(sensor_handle)
            if sensor_result[0] and not robot.is_busy:
                detected_handle = sensor_result[3]
                if isinstance(detected_handle, int) and detected_handle in createdCuboids and detected_handle not in processed_blocks:
                    block_number = createdCuboids[detected_handle]
                    # 블록 번호가 이 로봇의 담당 키에 해당하는지 확인
                    if block_number in assigned_keys:
                        block_pos = sim.getObjectPosition(detected_handle, sim.handle_world)
                        # print(f"[{robot_name}] 블록 감지! 핸들={detected_handle}, 번호={block_number}, 위치={[round(v,3) for v in block_pos]}")
                        processed_blocks.add(detected_handle)

                        key = block_number
                        counter = place_counters[key]
                        index = counter % 9
                        z_offset = z_layer_offsets[key]

                        # 해당 키의 9개 좌표에 z 레이어 오프셋 적용
                        positions = place_positions_data[key]
                        place_pos_with_z = [[p[0], p[1], p[2] + z_offset] for p in positions]

                        robot.pick_and_place(
                            block_handle=detected_handle,
                            block_pos=block_pos,
                            place_pos=place_pos_with_z,
                            place_drop_z=-0.25,
                            cuboid_Count=counter,
                            block_number=block_number
                        )

                        # 카운터 증가 및 레이어 체크
                        place_counters[key] += 1
                        if place_counters[key] % 9 == 0:
                            z_layer_offsets[key] += 0.1
                            # print(f"[{robot_name}] 키 {key}: 9개 완료! z 레이어 오프셋 → {z_layer_offsets[key]:.1f}")
        
        # 컨베이어 센서 감지 로그
        if detectionState:
            if isinstance(detectedPoint, (list, tuple)) and len(detectedPoint) >= 3:
                distance = (detectedPoint[0]**2 + detectedPoint[1]**2 + detectedPoint[2]**2)**0.5
                # print(f"[센서] 객체 감지! 거리: {distance:.4f}m")


        # ========== Cuboid 생성 (2초마다) ==========
        # 마지막 생성 이후 2초가 경과했는지 확인합니다
        if currentTime - lastCuboidTime >= 5.0:
            # cuboid 카운터를 증가시킵니다
            cuboidCount += 1

            control_signal = sim.getInt32Signal('cube_create')
            if control_signal is not None and multi_interrupt!= control_signal:
                cuboidHandle = control_signal
            
            # sim.setShapeMass(cuboidHandle, 0.5)
            # # createPrimitiveShape는 기본 static → dynamic으로 전환
            # sim.setObjectInt32Param(cuboidHandle, sim.shapeintparam_static, 0)
            # sim.setObjectInt32Param(cuboidHandle, sim.shapeintparam_respondable, 1)
            # sim.resetDynamicObject(cuboidHandle)
            

            # # 생성된 cuboid의 위치를 설정합니다
            # # setObjectPosition(objectHandle, relativeToObjectHandle, position)
            # # relativeToObjectHandle: sim.handle_world는 월드 좌표계 기준을 의미합니다
            # sim.setObjectPosition(
            #     cuboidHandle,           # 위치를 설정할 객체의 핸들
            #     sim.handle_world,       # 월드 좌표계 기준
            #     [6.325, -0.1, 0.8+0.3]    # 목표 위치 [x, y, z] (미터 단위)
            # )

            # # cuboid를 proximity sensor가 감지할 수 있도록 설정합니다
            # # setObjectSpecialProperty로 객체의 특수 속성을 설정합니다
            # # sim.objectspecialproperty_detectable_all: 모든 센서 타입에 의해 감지 가능하게 설정
            # sim.setObjectSpecialProperty(
            #     cuboidHandle,                                    # 대상 객체 핸들
            #     sim.objectspecialproperty_detectable_all         # 모든 센서에 의해 감지 가능
            # )

            # 생성된 cuboid 핸들을 리스트에 추가합니다
            #랜덤으로 상품의 불량종류 및 양품값 추출하여 cuboid의 value값으로 저장
            #0번은 양품 1~6은 불량종류

                rand_product = random.randint(0,6)
                # print(rand_product)
                createdCuboids[cuboidHandle]=rand_product
                print(createdCuboids)
            

            # 마지막 생성 시간을 현재 시간으로 업데이트합니다
                lastCuboidTime = currentTime
            multi_interrupt = cuboidHandle
        
        # 시뮬레이션 스텝 진행 (5 스텝 = 0.1초)
        for _ in range(5):
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