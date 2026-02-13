"""
AGV Dispatcher - OmniPlatform 블록 수거/배분 시스템

트리거: 로봇팔이 5번째 블록 배치 완료 후 add_block → 즉시 출발
동시작업: 그룹A / 그룹B 각각 독립적으로 1대씩 동시 운행 가능
이동규칙:
  - 그룹A (Platform 0~3): +Y 방향으로 퇴출 후 안전 차선
  - 그룹B (Platform 4~5): -Y 방향으로 퇴출 후 안전 차선
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from AGV import AGV
from AGV_optimize import SimulatedAnnealing


@dataclass
class PlatformInfo:
    handle: int
    group: str              # 'A' 또는 'B'
    defect_type: int        # 1~6
    capacity: int           # 100(A) 또는 50(B)
    dummy_handle: int
    init_position: list
    agv: object
    blocks: list = field(default_factory=list)
    block_defect_map: dict = field(default_factory=dict)
    is_dispatched: bool = False


class AGVDispatcher:
    TRIGGER_COUNT = 5
    ITEM_SIZES = {1: 2, 2: 10, 3: 4, 4: 8, 5: 6, 6: 7}
    DEFECT_TO_RACK = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2}

    SAFE_OFFSET = 1.0
    APPROACH_OFFSET = 0.7

    def __init__(self, sim, client, optimization_tool, box_data, place_counters, z_layer_offsets):
        self.sim = sim
        self.client = client
        self.optimization_tool = optimization_tool
        self.box_data = box_data
        self.place_counters = place_counters
        self.z_layer_offsets = z_layer_offsets

        self.platforms: Dict[int, PlatformInfo] = {}
        # (handle, group, generator, respawn_info)
        self.active_tasks: list = []
        self.pending_respawns: list = []

        self.rack_dummies: Dict[int, int] = {}
        self.rack_positions: Dict[int, Tuple[float, float]] = {}
        self._load_rack_info()

        self.sa = SimulatedAnnealing(
            initial_temp=100.0, cooling_rate=0.95,
            min_temp=1.0, max_iterations=500
        )

    def _load_rack_info(self):
        for i in range(3):
            self.rack_dummies[i] = self.sim.getObject(f'/rack[{i}]/Cuboid/dummy')
            rack_handle = self.sim.getObject(f'/rack[{i}]')
            pos = self.sim.getObjectPosition(rack_handle, self.sim.handle_world)
            self.rack_positions[i] = (pos[0], pos[1])

    # ========== 플랫폼 등록 ==========

    def register_platform(self, handle, group, defect_type):
        agv = AGV.from_handle(self.sim, self.client, handle)
        dummy_handle = agv.get_dummy_handle()
        position = list(self.sim.getObjectPosition(handle, self.sim.handle_world))

        self.platforms[handle] = PlatformInfo(
            handle=handle,
            group=group,
            defect_type=defect_type,
            capacity=100 if group == 'A' else 50,
            dummy_handle=dummy_handle,
            init_position=position,
            agv=agv,
        )
        print(f"[Dispatcher] 플랫폼 등록: 불량{defect_type}, 그룹{group}, handle={handle}")

    # ========== 블록 관리 ==========

    def add_block(self, platform_handle, block_handle, defect_type):
        if platform_handle not in self.platforms:
            return
        info = self.platforms[platform_handle]
        if info.is_dispatched:
            return
        info.blocks.append(block_handle)
        info.block_defect_map[block_handle] = defect_type
        count = len(info.blocks)
        print(f"[Dispatcher] 불량{defect_type} 블록 추가 → handle={platform_handle} ({count}개)")
        self._check_trigger(platform_handle)

    def find_platform_for_defect(self, defect_type) -> Optional[int]:
        for handle, info in self.platforms.items():
            if info.defect_type == defect_type and not info.is_dispatched:
                return handle
        return None

    # ========== 트리거 & 디스패치 ==========

    def _check_trigger(self, platform_handle):
        if platform_handle not in self.platforms:
            return
        info = self.platforms[platform_handle]
        if len(info.blocks) < self.TRIGGER_COUNT:
            return
        if info.is_dispatched:
            return
        # 같은 그룹에 이미 활성 태스크가 있으면 대기 (다른 그룹은 허용)
        for _, task_group, _, _ in self.active_tasks:
            if task_group == info.group:
                return
        self._start_dispatch(platform_handle)

    def _start_dispatch(self, activated_handle):
        info = self.platforms[activated_handle]
        info.is_dispatched = True

        # 재생성용 정보 저장 (generator 오류 시에도 복구 가능)
        respawn_info = {
            'position': list(info.init_position),
            'group': info.group,
            'defect_type': info.defect_type,
        }

        pickup_plan, pickup_order = self._plan_pickup(activated_handle)

        all_defect_types = {info.defect_type}
        for target_handle in pickup_plan:
            all_defect_types.add(self.platforms[target_handle].defect_type)

        rack_indices = list(set(self.DEFECT_TO_RACK[dt] for dt in all_defect_types))
        delivery_order = self._plan_delivery(rack_indices, info.init_position)

        gen = self._dispatch_gen(activated_handle, pickup_plan, pickup_order, delivery_order)
        self.active_tasks.append((activated_handle, info.group, gen, respawn_info))

        print(f"[Dispatcher] 디스패치 시작: 불량{info.defect_type} (handle={activated_handle})")
        print(f"  수거 계획: { {self.platforms[h].defect_type: len(blocks) for h, blocks in pickup_plan.items()} }")
        print(f"  배분 rack: {delivery_order}")

    # ========== 수거 계획 (체적 효율 최적화) ==========

    def _plan_pickup(self, activated_handle):
        info = self.platforms[activated_handle]
        group = info.group

        own_size = len(info.blocks) * self.ITEM_SIZES.get(info.defect_type, 1)
        remaining_capacity = info.capacity - own_size

        group_others = {
            h: p for h, p in self.platforms.items()
            if p.group == group and h != activated_handle and not p.is_dispatched and len(p.blocks) > 0
        }

        if remaining_capacity <= 0 or not group_others:
            return {}, []

        candidates = []
        for h, p in group_others.items():
            item_size = self.ITEM_SIZES.get(p.defect_type, 1)
            candidates.append((h, len(p.blocks), item_size))
        candidates.sort(key=lambda x: x[2])

        pickup_plan = {}
        for h, count, item_size in candidates:
            if remaining_capacity <= 0:
                break
            max_take = remaining_capacity // item_size
            take = min(count, max_take)
            if take > 0:
                target_blocks = self.platforms[h].blocks[:take]
                pickup_plan[h] = target_blocks
                remaining_capacity -= take * item_size

        if len(pickup_plan) <= 1:
            pickup_order = list(pickup_plan.keys())
        else:
            handles = list(pickup_plan.keys())
            positions = [(self.platforms[h].init_position[0], self.platforms[h].init_position[1]) for h in handles]
            agv_pos = info.agv.get_position()
            order_idx, _ = self.sa.optimize(positions, (agv_pos[0], agv_pos[1]), return_to_start=False)
            pickup_order = [handles[i] for i in order_idx]

        return pickup_plan, pickup_order

    def _plan_delivery(self, rack_indices, start_position):
        if len(rack_indices) <= 1:
            return rack_indices
        positions = [self.rack_positions[ri] for ri in rack_indices]
        start = (start_position[0], start_position[1])
        order_idx, _ = self.sa.optimize(positions, start, return_to_start=False)
        return [rack_indices[i] for i in order_idx]

    # ========== 안전 Y 좌표 ==========

    def _get_safe_y(self, group, base_y):
        if group == 'A':
            return base_y + self.SAFE_OFFSET
        else:
            return base_y - self.SAFE_OFFSET

    # ========== 디스패치 Generator ==========

    def _dispatch_gen(self, activated_handle, pickup_plan, pickup_order, delivery_order):
        info = self.platforms[activated_handle]
        agv = info.agv
        group = info.group
        saved_position = list(info.init_position)
        saved_group = info.group
        saved_defect_type = info.defect_type

        safe_y = self._get_safe_y(group, info.init_position[1])

        # 블록은 이미 로봇팔이 배치 완료한 상태 → 즉시 수집 확정
        collected = {}
        collected[info.defect_type] = list(info.blocks)

        # 자신의 블록을 dummy에 고정
        for bh in info.blocks:
            try:
                self.sim.setObjectInt32Param(bh, self.sim.shapeintparam_static, 1)
                self.sim.setObjectParent(bh, info.dummy_handle, True)
            except Exception:
                pass

        # ===== 출발 즉시 재생성 예약 (2초 후) =====
        self.pending_respawns.append({
            'countdown': 100,  # 2초 (50Hz 기준)
            'position': saved_position,
            'group': saved_group,
            'defect_type': saved_defect_type,
        })
        print(f"[Dispatcher] 불량{saved_defect_type} 재생성 예약 (2초 후)")

        # ===== Phase 0: 안전 차선으로 퇴출 =====
        print(f"[Dispatcher] Phase 0: 안전 차선 이동 (safe_y={safe_y:.2f})")
        cur = agv.get_position()
        yield from self._move_to_gen(agv, cur[0], safe_y)

        # ===== Phase 1: 같은 그룹 플랫폼 방문하여 수거 =====
        # 이동 순서: 안전차선(Y) → X이동 → 접근(Y) → 텔레포트 → 복귀(Y)
        for target_handle in pickup_order:
            if target_handle not in self.platforms:
                continue
            target_info = self.platforms[target_handle]
            target_pos = target_info.init_position

            if group == 'A':
                approach_y = target_pos[1] + self.APPROACH_OFFSET
            else:
                approach_y = target_pos[1] - self.APPROACH_OFFSET

            # 1) 안전 차선에서 X 이동 (이 시점에서 safe_y에 있음)
            cur = agv.get_position()
            dx = target_pos[0] - cur[0]
            if abs(dx) > 0.05:
                yield from agv._move_x_gen(dx, 2)

            # 2) 접근: safe_y → approach_y (±0.7 앞)
            cur = agv.get_position()
            dy = approach_y - cur[1]
            if abs(dy) > 0.05:
                yield from agv._move_y_gen(dy, 2)

            # 3) 블록 즉시 텔레포트
            blocks_to_take = pickup_plan[target_handle]
            dummy_pos = self.sim.getObjectPosition(info.dummy_handle, self.sim.handle_world)
            existing_count = sum(len(v) for v in collected.values())

            for i, bh in enumerate(blocks_to_take):
                idx = existing_count + i
                offset_x = (idx % 3) * 0.08
                offset_y = (idx // 3) * 0.08
                pos = [dummy_pos[0] + offset_x, dummy_pos[1] + offset_y, dummy_pos[2]]
                try:
                    self.sim.setObjectPosition(bh, pos, self.sim.handle_world)
                    self.sim.setObjectInt32Param(bh, self.sim.shapeintparam_static, 1)
                    self.sim.setObjectParent(bh, info.dummy_handle, True)
                except Exception:
                    pass

            for bh in blocks_to_take:
                if bh in target_info.blocks:
                    target_info.blocks.remove(bh)
                dt = target_info.block_defect_map.pop(bh, target_info.defect_type)
                collected.setdefault(dt, []).append(bh)

            print(f"[Dispatcher] 플랫폼 handle={target_handle}에서 {len(blocks_to_take)}개 수거 완료")

            # 4) 안전 차선으로 Y 복귀 (approach_y → safe_y, 충돌 방지)
            cur = agv.get_position()
            dy = safe_y - cur[1]
            if abs(dy) > 0.05:
                yield from agv._move_y_gen(dy, 2)

        # ===== Phase 2: rack으로 배분 =====
        # 이동 순서: 안전차선(Y) → X이동(+0.2 뒤) → Y이동(rack Y) → 배분 → 복귀(Y)
        RACK_X_MARGIN = 0.2  # rack 상자 충돌 방지 여유
        for rack_idx in delivery_order:
            rack_pos = self.rack_positions[rack_idx]
            approach_x = rack_pos[0] + self.APPROACH_OFFSET + RACK_X_MARGIN

            # 1) 안전 차선에서 X 이동 (rack보다 +0.2 더 뒤)
            cur = agv.get_position()
            dx = approach_x - cur[0]
            if abs(dx) > 0.05:
                yield from agv._move_x_gen(dx, 2)

            # 2) Y 이동: safe_y → rack Y (충돌 없는 X 위치에서)
            cur = agv.get_position()
            dy = rack_pos[1] - cur[1]
            if abs(dy) > 0.05:
                yield from agv._move_y_gen(dy, 2)

            # 3) 블록 즉시 배분
            blocks_for_rack = []
            for dt, blocks in collected.items():
                if self.DEFECT_TO_RACK.get(dt) == rack_idx:
                    blocks_for_rack.extend(blocks)

            if blocks_for_rack:
                rack_dummy = self.rack_dummies[rack_idx]
                agv.transfer_to_rack(blocks_for_rack, rack_dummy)

                for dt in list(collected.keys()):
                    if self.DEFECT_TO_RACK.get(dt) == rack_idx:
                        collected.pop(dt, None)

                print(f"[Dispatcher] rack[{rack_idx}]에 {len(blocks_for_rack)}개 블록 배분 완료")

            # 4) 안전 차선으로 Y 복귀 (충돌 방지)
            cur = agv.get_position()
            dy = safe_y - cur[1]
            if abs(dy) > 0.05:
                yield from agv._move_y_gen(dy, 2)

        # ===== Phase 3: 정지 & 플랫폼 삭제 =====
        agv.stop()

        try:
            self.sim.removeModel(activated_handle)
        except Exception:
            try:
                self.sim.removeObjects([activated_handle])
            except Exception:
                print(f"[Dispatcher] 플랫폼 삭제 실패: handle={activated_handle}")

        if activated_handle in self.platforms:
            del self.platforms[activated_handle]

        print(f"[Dispatcher] 불량{saved_defect_type} 플랫폼 삭제 완료")

    # ========== 이동 Generator ==========

    def _move_to_gen(self, agv, target_x, target_y, speed=2):
        current = agv.get_position()
        dx = target_x - current[0]
        dy = target_y - current[1]

        if abs(dx) > 0.05:
            yield from agv._move_x_gen(dx, speed)
        if abs(dy) > 0.05:
            yield from agv._move_y_gen(dy, speed)

    # ========== 재생성 처리 ==========

    def _process_respawns(self):
        still_pending = []
        for entry in self.pending_respawns:
            entry['countdown'] -= 1
            if entry['countdown'] <= 0:
                try:
                    self._respawn_platform(entry)
                except Exception as e:
                    print(f"[Dispatcher] 재생성 실패: {e}")
            else:
                still_pending.append(entry)
        self.pending_respawns = still_pending

    def _respawn_platform(self, entry):
        position = entry['position']
        group = entry['group']
        defect_type = entry['defect_type']

        # 카운터 초기화 (새 플랫폼이 블록을 처음부터 받을 수 있도록)
        self.box_data[defect_type] = []
        self.place_counters[defect_type] = 0
        self.z_layer_offsets[defect_type] = 0.0

        new_handle = self.optimization_tool.create_AGV(position)
        self.sim.setObjectPosition(new_handle, position, self.sim.handle_world)
        self.register_platform(new_handle, group, defect_type)
        print(f"[Dispatcher] 불량{defect_type} 플랫폼 재생성 완료 (handle={new_handle})")

    # ========== 매 스텝 업데이트 ==========

    def update(self):
        completed = []
        for i, (handle, _, gen, respawn_info) in enumerate(self.active_tasks):
            try:
                next(gen)
            except StopIteration:
                completed.append((i, False))
            except Exception as e:
                print(f"[Dispatcher] 태스크 오류: {e}")
                completed.append((i, True))

        for idx, is_error in reversed(completed):
            handle, _, gen, respawn_info = self.active_tasks.pop(idx)

            # 오류로 종료된 경우: 플랫폼 정리 + 강제 재생성
            if is_error and respawn_info:
                if handle in self.platforms:
                    try:
                        self.platforms[handle].agv.stop()
                        self.sim.removeModel(handle)
                    except Exception:
                        try:
                            self.sim.removeObjects([handle])
                        except Exception:
                            pass
                    del self.platforms[handle]

                dt = respawn_info['defect_type']
                self.box_data[dt] = []
                self.place_counters[dt] = 0
                self.z_layer_offsets[dt] = 0.0

                already = any(r['defect_type'] == dt for r in self.pending_respawns)
                if not already:
                    self.pending_respawns.append({
                        'countdown': 50,
                        **respawn_info,
                    })
                    print(f"[Dispatcher] 오류 복구: 불량{dt} 재생성 예약")

        # 태스크 완료 후 대기 중인 플랫폼 트리거 재확인
        if completed:
            for h in list(self.platforms.keys()):
                self._check_trigger(h)

        self._process_respawns()
