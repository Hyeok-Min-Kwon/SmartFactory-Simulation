
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import math
class AGV:
    v=80*2.398795*math.pi/180
    def __init__(self,sim, client, robot_num):
        self.sim = sim
        self.client = client
        self.robot_num = robot_num
        self.omniPads = []
        self.speed = 0
        self.x_value = 0
        self.y_value = 0
        for i in range(4):
            if i == 0:
                self.omniPads.append(self.sim.getObject(f'/OmniPlatform[{self.robot_num}]/regularRotation'))
            else:
                self.omniPads.append(self.sim.getObject(f'/OmniPlatform[{self.robot_num}]/link[{i}]/regularRotation'))
    
        self.omniPads.append(self.sim.getObject(f"/OmniPlatform[{self.robot_num}]"))
        
    def get_position(self):
        return self.sim.getObjectPosition(self.omniPads[4])
        
        
    def move_xplus(self,speed=1):
        # self.get_move_joint()
        self.sim.setJointTargetVelocity(self.omniPads[0],-self.v*speed)
        self.sim.setJointTargetVelocity(self.omniPads[1],-self.v*speed)
        self.sim.setJointTargetVelocity(self.omniPads[2],self.v*speed)
        self.sim.setJointTargetVelocity(self.omniPads[3],self.v*speed)
    #move right
    def move_xminus(self,speed=1):
        # self.get_move_joint()
        self.sim.setJointTargetVelocity(self.omniPads[0],self.v*speed)
        self.sim.setJointTargetVelocity(self.omniPads[1],self.v*speed)
        self.sim.setJointTargetVelocity(self.omniPads[2],-self.v*speed)
        self.sim.setJointTargetVelocity(self.omniPads[3],-self.v*speed)
        
    def move_yminus(self,speed=1):
        # self.get_move_joint()
        self.sim.setJointTargetVelocity(self.omniPads[0],-self.v*speed)
        self.sim.setJointTargetVelocity(self.omniPads[1],self.v*speed)
        self.sim.setJointTargetVelocity(self.omniPads[2],self.v*speed)
        self.sim.setJointTargetVelocity(self.omniPads[3],-self.v*speed)
        
    def move_yplus(self,speed=1):
        # self.get_move_joint()
        self.sim.setJointTargetVelocity(self.omniPads[0],self.v*speed)
        self.sim.setJointTargetVelocity(self.omniPads[1],-self.v*speed)
        self.sim.setJointTargetVelocity(self.omniPads[2],-self.v*speed)
        self.sim.setJointTargetVelocity(self.omniPads[3],self.v*speed)
    def stop(self):
        # self.get_move_joint()
        self.sim.setJointTargetVelocity(self.omniPads[0],0)
        self.sim.setJointTargetVelocity(self.omniPads[1],0)
        self.sim.setJointTargetVelocity(self.omniPads[2],0)
        self.sim.setJointTargetVelocity(self.omniPads[3],0)
    
    def move_x(self, x_value,speed):
        first_position = self.get_position()[0]
        update_position = first_position
        
        if x_value<0:
            while update_position>=first_position+x_value:
                self.move_xminus(speed)
                update_position = self.get_position()[0]
        else:
            while update_position<=first_position+x_value:
                self.move_xplus(speed)
                update_position = self.get_position()[0]
        self.stop()
        
    def move_y(self, y_value,speed):
        first_position = self.get_position()[1]
        update_position = first_position

        if y_value<0:
            while update_position>=first_position+y_value:
                self.move_yminus(speed)
                update_position = self.get_position()[1]
        else:
            while update_position<=first_position+y_value:
                self.move_yplus(speed)
                update_position = self.get_position()[1]
        self.stop()

    def move_to_position(self, target_x, target_y, speed=1, tolerance=0.05):
        """
        목표 좌표로 AGV 이동

        Args:
            target_x: 목표 x 좌표
            target_y: 목표 y 좌표
            speed: 이동 속도 (기본값 1)
            tolerance: 도착 허용 오차 (미터)
        """
        current_pos = self.get_position()

        # X축 이동
        x_diff = target_x - current_pos[0]
        if abs(x_diff) > tolerance:
            self.move_x(x_diff, speed)

        # Y축 이동
        current_pos = self.get_position()
        y_diff = target_y - current_pos[1]
        if abs(y_diff) > tolerance:
            self.move_y(y_diff, speed)

        self.stop()
        final_pos = self.get_position()
        print(f"[AGV] 이동 완료: ({final_pos[0]:.3f}, {final_pos[1]:.3f})")
        return final_pos

    def execute_route(self, route_commands, on_pickup=None, on_dropoff=None):
        """
        최적화된 경로 명령 리스트 수행

        Args:
            route_commands: AGVPickupOptimizer.get_full_route() 반환값
            on_pickup: 픽업 시 호출될 콜백 함수 (category, count)
            on_dropoff: 하차 시 호출될 콜백 함수 (category, count)
        """
        print(f"[AGV] 경로 수행 시작 (총 {len(route_commands)}개 명령)")

        for i, cmd in enumerate(route_commands):
            action = cmd['action']
            desc = cmd.get('description', '')
            print(f"[AGV] [{i+1}/{len(route_commands)}] {desc}")

            if action == 'move':
                target = cmd['target']
                self.move_to_position(target[0], target[1], speed=2)

            elif action == 'pickup':
                category = cmd['category']
                count = cmd['count']
                if on_pickup:
                    on_pickup(category, count)
                print(f"[AGV] 분류 {category} 물품 {count}개 적재 완료")

            elif action == 'dropoff':
                category = cmd['category']
                count = cmd['count']
                if on_dropoff:
                    on_dropoff(category, count)
                print(f"[AGV] 분류 {category} 물품 {count}개 하차 완료")

        print("[AGV] 경로 수행 완료")

        
    
    

# client = RemoteAPIClient()

# sim = client.require('sim')
# dummy_object=sim.getObject('/dummy')
# dummy_position = sim.getObjectPosition(dummy_object,sim.handle_world)
# print(dummy_position)
# sim.startSimulation()
# omni1 = AGV(sim,client,0)
# omni1.move_y(1,3)
# omni1.stop()
# omni1.move_x(1,3)

# omni1.stop()
# omni1.move_x(-2)
# omni1.move_y(2,2)
# omni1.stop()
# omni1.move_left(1)
# omni1.move_backward(1)
# omni1.move_forward(2)
# omni1.stop()
# omniPads=[]
# for i in range(4):
#     if i == 0:
#         omniPads.append(sim.getObject('/OmniPlatform[6]/regularRotation'))
#     else:
#         omniPads.append(sim.getObject(f'/OmniPlatform[6]/link[{i}]/regularRotation'))

# v=80*2.398795*math.pi/180*5
# #move left