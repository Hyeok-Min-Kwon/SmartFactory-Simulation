
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
        self.init_position = self.getObjectPosition(self.omniPads[4])        
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
    def transfer_to_rack(self, block_handles, rack_dummy_handle):
        dummy_pos = self.sim.getObjectPosition(rack_dummy_handle, self.sim.handle_world)
        for i, handle in enumerate(block_handles):
            offset_x = (i % 3) * 0.08
            offset_y = (i // 3) * 0.08
            pos = [dummy_pos[0] + offset_x, dummy_pos[1] + offset_y, dummy_pos[2]]
            self.sim.setObjectPosition(handle, pos, self.sim.handle_world)
            self.sim.setObjectInt32Param(handle, self.sim.shapeintparam_static, 1)
            self.sim.setObjectInt32Param(handle, self.sim.shapeintparam_respondable, 1)
            self.sim.setObjectParent(handle, rack_dummy_handle, True)
