
import math

class AGV:
    v = 80 * 2.398795 * math.pi / 180

    def __init__(self, sim, client, robot_num):
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
        self.init_position = self.sim.getObjectPosition(self.omniPads[4])

    @classmethod
    def from_handle(cls, sim, client, platform_handle):
        instance = cls.__new__(cls)
        instance.sim = sim
        instance.client = client
        instance.robot_num = None
        instance.omniPads = []
        instance.speed = 0
        instance.x_value = 0
        instance.y_value = 0

        all_objects = sim.getObjectsInTree(platform_handle, sim.handle_all, 0)

        # 조인트 분류: pad0 = 플랫폼 직속 regularRotation, pad1~3 = link 하위 regularRotation
        # 원본 __init__ 기준: pad0=/OmniPlatform/regularRotation, pad1~3=/OmniPlatform/link[i]/regularRotation
        direct_joint = None   # pad 0 (link 없이 직접 연결)
        link_joints = []      # pad 1~3 (link 하위, 트리 순서 = link 순서)

        for obj in all_objects:
            try:
                if sim.getObjectType(obj) != sim.sceneobject_joint:
                    continue
                alias = sim.getObjectAlias(obj, 0)
                if 'regularRotation' not in alias:
                    continue

                # 부모 체인을 따라 올라가며 'link' 조상이 있는지 확인
                has_link_ancestor = False
                parent = sim.getObjectParent(obj)
                while parent != -1 and parent != platform_handle:
                    try:
                        p_alias = sim.getObjectAlias(parent, 0)
                        if 'link' in p_alias:
                            has_link_ancestor = True
                            break
                    except:
                        pass
                    parent = sim.getObjectParent(parent)

                if has_link_ancestor:
                    link_joints.append(obj)
                else:
                    direct_joint = obj
            except Exception:
                continue

        # pad 순서: [pad0(직속), pad1, pad2, pad3(link 순서)]
        instance.omniPads = []
        if direct_joint is not None:
            instance.omniPads.append(direct_joint)
        instance.omniPads.extend(link_joints)
        instance.omniPads = instance.omniPads[:4]

        instance.omniPads.append(platform_handle)
        instance.init_position = sim.getObjectPosition(platform_handle, sim.handle_world)

        # 모든 조인트 정지 (잔여 속도 방지)
        for i in range(min(4, len(instance.omniPads))):
            try:
                sim.setJointTargetVelocity(instance.omniPads[i], 0)
            except:
                pass

        return instance

    def get_position(self):
        return self.sim.getObjectPosition(self.omniPads[4])

    def get_dummy_handle(self):
        all_objects = self.sim.getObjectsInTree(self.omniPads[4], self.sim.handle_all, 0)
        for obj in all_objects:
            try:
                alias = self.sim.getObjectAlias(obj, 0)
                if alias.lower() == 'dummy':
                    return obj
            except Exception:
                continue
        return None

    def move_xplus(self, speed=2):
        self.sim.setJointTargetVelocity(self.omniPads[0], -self.v * speed)
        self.sim.setJointTargetVelocity(self.omniPads[1], -self.v * speed)
        self.sim.setJointTargetVelocity(self.omniPads[2], self.v * speed)
        self.sim.setJointTargetVelocity(self.omniPads[3], self.v * speed)

    def move_xminus(self, speed=2):
        self.sim.setJointTargetVelocity(self.omniPads[0], self.v * speed)
        self.sim.setJointTargetVelocity(self.omniPads[1], self.v * speed)
        self.sim.setJointTargetVelocity(self.omniPads[2], -self.v * speed)
        self.sim.setJointTargetVelocity(self.omniPads[3], -self.v * speed)

    def move_yminus(self, speed=2):
        self.sim.setJointTargetVelocity(self.omniPads[0], -self.v * speed)
        self.sim.setJointTargetVelocity(self.omniPads[1], self.v * speed)
        self.sim.setJointTargetVelocity(self.omniPads[2], self.v * speed)
        self.sim.setJointTargetVelocity(self.omniPads[3], -self.v * speed)

    def move_yplus(self, speed=2):
        self.sim.setJointTargetVelocity(self.omniPads[0], self.v * speed)
        self.sim.setJointTargetVelocity(self.omniPads[1], -self.v * speed)
        self.sim.setJointTargetVelocity(self.omniPads[2], -self.v * speed)
        self.sim.setJointTargetVelocity(self.omniPads[3], self.v * speed)

    def stop(self):
        self.sim.setJointTargetVelocity(self.omniPads[0], 0)
        self.sim.setJointTargetVelocity(self.omniPads[1], 0)
        self.sim.setJointTargetVelocity(self.omniPads[2], 0)
        self.sim.setJointTargetVelocity(self.omniPads[3], 0)

    def move_x(self, x_value, speed):
        first_position = self.get_position()[0]
        update_position = first_position

        if x_value < 0:
            while update_position >= first_position + x_value:
                self.move_xminus(speed)
                update_position = self.get_position()[0]
        else:
            while update_position <= first_position + x_value:
                self.move_xplus(speed)
                update_position = self.get_position()[0]
        self.stop()

    def move_y(self, y_value, speed):
        first_position = self.get_position()[1]
        update_position = first_position

        if y_value < 0:
            while update_position >= first_position + y_value:
                self.move_yminus(speed)
                update_position = self.get_position()[1]
        else:
            while update_position <= first_position + y_value:
                self.move_yplus(speed)
                update_position = self.get_position()[1]
        self.stop()

    def _move_x_gen(self, x_value, speed=1):
        first_position = self.get_position()[0]
        if x_value < 0:
            while self.get_position()[0] >= first_position + x_value:
                self.move_xminus(speed)
                yield
        else:
            while self.get_position()[0] <= first_position + x_value:
                self.move_xplus(speed)
                yield
        self.stop()

    def _move_y_gen(self, y_value, speed=1):
        first_position = self.get_position()[1]
        if y_value < 0:
            while self.get_position()[1] >= first_position + y_value:
                self.move_yminus(speed)
                yield
        else:
            while self.get_position()[1] <= first_position + y_value:
                self.move_yplus(speed)
                yield
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
