
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

        
    
    

client = RemoteAPIClient()

sim = client.require('sim')
dummy_object=sim.getObject('/dummy')
dummy_position = sim.getObjectPosition(dummy_object,sim.handle_world)
print(dummy_position)
sim.startSimulation()
omni1 = AGV(sim,client,0)
omni1.move_y(1,3)
omni1.stop()
omni1.move_x(1,3)

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