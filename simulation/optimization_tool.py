class Optimization:
    def __init__(self,sim,client):
        self.sim = sim
        self.client = client
    def get_rack_position(self):
        rack_position=[]
        for i in range(3):
            object=self.sim.getObject(f"/rack[{i}]")
            position = self.sim.getObjectPosition(object)
            rack_position.append(position)
        return rack_position
    def create_AGV(self,position = [3.075, -6.325, 0.53215]):
        new_agv = self.sim.loadModel("C:/Program Files/CoppeliaRobotics/CoppeliaSimEdu/models/AGV.ttm")
        self.sim.setObjectPosition(new_agv, position)
        return new_agv




