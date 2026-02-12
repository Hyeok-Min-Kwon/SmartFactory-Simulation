"""
AGV 경로 최적화 모듈
- Simulated Annealing: 다중 적재 위치 방문 순서 최적화
- 물건 분류 (1~6번)에 따른 최적 배송 경로 계산
"""

import math
import random
from typing import List, Tuple, Dict


class SimulatedAnnealing:
    """Simulated Annealing - 다중 목적지 방문 순서 최적화 (TSP)"""

    def __init__(self,
                 initial_temp: float = 1000.0,
                 cooling_rate: float = 0.995,
                 min_temp: float = 0.1,
                 max_iterations: int = 10000):
        """
        Args:
            initial_temp: 초기 온도
            cooling_rate: 냉각률 (0~1)
            min_temp: 최소 온도
            max_iterations: 최대 반복 횟수
        """
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """두 점 사이 유클리드 거리"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _total_distance(self,
                        route: List[int],
                        positions: List[Tuple[float, float]],
                        start: Tuple[float, float],
                        return_to_start: bool = True) -> float:
        """경로 총 거리 계산"""
        if not route:
            return 0.0

        total = self._distance(start, positions[route[0]])

        for i in range(len(route) - 1):
            total += self._distance(positions[route[i]], positions[route[i + 1]])

        if return_to_start:
            total += self._distance(positions[route[-1]], start)

        return total

    def _swap(self, route: List[int]) -> List[int]:
        """두 위치 스왑"""
        new_route = route[:]
        i, j = random.sample(range(len(route)), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

    def _reverse_segment(self, route: List[int]) -> List[int]:
        """구간 역순 (2-opt)"""
        new_route = route[:]
        i, j = sorted(random.sample(range(len(route)), 2))
        new_route[i:j+1] = reversed(new_route[i:j+1])
        return new_route

    def _insert(self, route: List[int]) -> List[int]:
        """한 요소를 다른 위치로 이동"""
        new_route = route[:]
        i = random.randrange(len(route))
        j = random.randrange(len(route))
        if i != j:
            elem = new_route.pop(i)
            new_route.insert(j, elem)
        return new_route

    def _nearest_neighbor(self,
                          positions: List[Tuple[float, float]],
                          start: Tuple[float, float]) -> List[int]:
        """Nearest Neighbor 휴리스틱으로 초기해 생성"""
        n = len(positions)
        visited = [False] * n
        route = []
        current = start

        for _ in range(n):
            nearest = -1
            min_dist = float('inf')

            for i in range(n):
                if not visited[i]:
                    dist = self._distance(current, positions[i])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = i

            if nearest >= 0:
                visited[nearest] = True
                route.append(nearest)
                current = positions[nearest]

        return route

    def optimize(self,
                 positions: List[Tuple[float, float]],
                 start: Tuple[float, float],
                 return_to_start: bool = True) -> Tuple[List[int], float]:
        """
        방문 순서 최적화

        Args:
            positions: 목적지 좌표 리스트
            start: 시작 위치
            return_to_start: 시작점으로 복귀 여부

        Returns:
            (최적 순서 인덱스 리스트, 총 거리)
        """
        n = len(positions)
        if n == 0:
            return [], 0.0
        if n == 1:
            dist = self._distance(start, positions[0])
            if return_to_start:
                dist *= 2
            return [0], dist

        # 초기해: Nearest Neighbor
        current_route = self._nearest_neighbor(positions, start)
        current_dist = self._total_distance(current_route, positions, start, return_to_start)

        best_route = current_route[:]
        best_dist = current_dist

        temp = self.initial_temp

        for _ in range(self.max_iterations):
            if temp < self.min_temp:
                break

            # 변이 연산 선택 (33% 확률씩)
            r = random.random()
            if r < 0.33:
                new_route = self._swap(current_route)
            elif r < 0.66:
                new_route = self._reverse_segment(current_route)
            else:
                new_route = self._insert(current_route)

            new_dist = self._total_distance(new_route, positions, start, return_to_start)
            delta = new_dist - current_dist

            # 수락 여부 결정
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_route = new_route
                current_dist = new_dist

                if current_dist < best_dist:
                    best_route = current_route[:]
                    best_dist = current_dist

            temp *= self.cooling_rate

        return best_route, best_dist


class MultiAGVOptimizer:
    """다중 AGV 경로 최적화 (mTSP/VRP)"""

    def __init__(self,
                 num_agvs: int = 1,
                 initial_temp: float = 1000.0,
                 cooling_rate: float = 0.995,
                 min_temp: float = 0.1,
                 max_iterations: int = 15000):
        """
        Args:
            num_agvs: AGV 대수
            initial_temp: SA 초기 온도
            cooling_rate: SA 냉각률
            min_temp: SA 최소 온도
            max_iterations: SA 최대 반복 횟수
        """
        self.num_agvs = num_agvs
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.max_iterations = max_iterations

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """두 점 사이 유클리드 거리"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _route_distance(self,
                        route: List[int],
                        positions: List[Tuple[float, float]],
                        start: Tuple[float, float],
                        return_to_start: bool) -> float:
        """단일 AGV 경로 거리 계산"""
        if not route:
            return 0.0

        total = self._distance(start, positions[route[0]])
        for i in range(len(route) - 1):
            total += self._distance(positions[route[i]], positions[route[i + 1]])
        if return_to_start:
            total += self._distance(positions[route[-1]], start)
        return total

    def _total_cost(self,
                    assignments: List[List[int]],
                    positions: List[Tuple[float, float]],
                    starts: List[Tuple[float, float]],
                    return_to_start: bool,
                    minimize_makespan: bool = False) -> float:
        """
        전체 비용 계산

        Args:
            assignments: 각 AGV별 목적지 인덱스 리스트
            positions: 목적지 좌표 리스트
            starts: 각 AGV 시작 위치
            return_to_start: 복귀 여부
            minimize_makespan: True면 최대 거리(makespan) 최소화, False면 총 거리 최소화
        """
        distances = []
        for agv_idx, route in enumerate(assignments):
            start = starts[agv_idx] if agv_idx < len(starts) else starts[0]
            dist = self._route_distance(route, positions, start, return_to_start)
            distances.append(dist)

        if minimize_makespan:
            return max(distances) if distances else 0.0
        else:
            return sum(distances)

    def _init_assignments(self,
                          n_destinations: int,
                          positions: List[Tuple[float, float]],
                          starts: List[Tuple[float, float]]) -> List[List[int]]:
        """초기 할당 생성 (Greedy 기반)"""
        assignments = [[] for _ in range(self.num_agvs)]
        assigned = [False] * n_destinations

        # 각 AGV에서 가장 가까운 목적지부터 할당
        for _ in range(n_destinations):
            best_agv = -1
            best_dest = -1
            best_dist = float('inf')

            for agv_idx in range(self.num_agvs):
                start = starts[agv_idx] if agv_idx < len(starts) else starts[0]
                current_pos = start
                if assignments[agv_idx]:
                    current_pos = positions[assignments[agv_idx][-1]]

                for dest_idx in range(n_destinations):
                    if not assigned[dest_idx]:
                        dist = self._distance(current_pos, positions[dest_idx])
                        # 균형 잡힌 분배를 위해 이미 많이 할당된 AGV에 패널티
                        penalty = len(assignments[agv_idx]) * 0.5
                        total = dist + penalty
                        if total < best_dist:
                            best_dist = total
                            best_agv = agv_idx
                            best_dest = dest_idx

            if best_dest >= 0:
                assigned[best_dest] = True
                assignments[best_agv].append(best_dest)

        return assignments

    def _mutate_move(self, assignments: List[List[int]]) -> List[List[int]]:
        """목적지를 다른 AGV로 이동"""
        new_assignments = [route[:] for route in assignments]

        # 비어있지 않은 AGV 찾기
        non_empty = [i for i, route in enumerate(new_assignments) if route]
        if not non_empty:
            return new_assignments

        # 랜덤 AGV에서 목적지 하나 선택
        src_agv = random.choice(non_empty)
        src_idx = random.randrange(len(new_assignments[src_agv]))
        dest = new_assignments[src_agv].pop(src_idx)

        # 다른 AGV에 삽입
        dst_agv = random.randrange(self.num_agvs)
        insert_pos = random.randint(0, len(new_assignments[dst_agv]))
        new_assignments[dst_agv].insert(insert_pos, dest)

        return new_assignments

    def _mutate_swap_between(self, assignments: List[List[int]]) -> List[List[int]]:
        """두 AGV 간 목적지 교환"""
        new_assignments = [route[:] for route in assignments]

        non_empty = [i for i, route in enumerate(new_assignments) if route]
        if len(non_empty) < 2:
            return new_assignments

        agv1, agv2 = random.sample(non_empty, 2)
        idx1 = random.randrange(len(new_assignments[agv1]))
        idx2 = random.randrange(len(new_assignments[agv2]))

        new_assignments[agv1][idx1], new_assignments[agv2][idx2] = \
            new_assignments[agv2][idx2], new_assignments[agv1][idx1]

        return new_assignments

    def _mutate_swap_within(self, assignments: List[List[int]]) -> List[List[int]]:
        """AGV 내 순서 스왑"""
        new_assignments = [route[:] for route in assignments]

        # 2개 이상 목적지가 있는 AGV 찾기
        valid = [i for i, route in enumerate(new_assignments) if len(route) >= 2]
        if not valid:
            return new_assignments

        agv_idx = random.choice(valid)
        i, j = random.sample(range(len(new_assignments[agv_idx])), 2)
        new_assignments[agv_idx][i], new_assignments[agv_idx][j] = \
            new_assignments[agv_idx][j], new_assignments[agv_idx][i]

        return new_assignments

    def _mutate_reverse(self, assignments: List[List[int]]) -> List[List[int]]:
        """AGV 내 구간 역순 (2-opt)"""
        new_assignments = [route[:] for route in assignments]

        valid = [i for i, route in enumerate(new_assignments) if len(route) >= 2]
        if not valid:
            return new_assignments

        agv_idx = random.choice(valid)
        i, j = sorted(random.sample(range(len(new_assignments[agv_idx])), 2))
        new_assignments[agv_idx][i:j+1] = reversed(new_assignments[agv_idx][i:j+1])

        return new_assignments

    def optimize(self,
                 positions: List[Tuple[float, float]],
                 starts: List[Tuple[float, float]],
                 return_to_start: bool = True,
                 minimize_makespan: bool = False) -> Tuple[List[List[int]], float]:
        """
        다중 AGV 경로 최적화

        Args:
            positions: 목적지 좌표 리스트
            starts: 각 AGV 시작 위치 (부족하면 첫 번째 위치 사용)
            return_to_start: 시작점 복귀 여부
            minimize_makespan: True면 최대 이동시간 최소화

        Returns:
            (각 AGV별 목적지 인덱스 리스트, 비용)
        """
        n = len(positions)
        if n == 0:
            return [[] for _ in range(self.num_agvs)], 0.0

        # 시작 위치 보정
        while len(starts) < self.num_agvs:
            starts = list(starts) + [starts[0]]

        # 초기 할당
        current = self._init_assignments(n, positions, starts)
        current_cost = self._total_cost(current, positions, starts, return_to_start, minimize_makespan)

        best = [route[:] for route in current]
        best_cost = current_cost

        temp = self.initial_temp

        for _ in range(self.max_iterations):
            if temp < self.min_temp:
                break

            # 변이 연산 선택
            r = random.random()
            if r < 0.25:
                new = self._mutate_move(current)
            elif r < 0.5:
                new = self._mutate_swap_between(current)
            elif r < 0.75:
                new = self._mutate_swap_within(current)
            else:
                new = self._mutate_reverse(current)

            new_cost = self._total_cost(new, positions, starts, return_to_start, minimize_makespan)
            delta = new_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / temp):
                current = new
                current_cost = new_cost

                if current_cost < best_cost:
                    best = [route[:] for route in current]
                    best_cost = current_cost

            temp *= self.cooling_rate

        return best, best_cost


class AGVRouteOptimizer:
    """AGV 배송 경로 최적화기 - 물건 분류별 적재 위치"""

    def __init__(self, sim=None, rack_mapping: Dict[int, int] = None):
        """
        Args:
            sim: CoppeliaSim sim 객체 (None이면 기본 좌표 사용)
            rack_mapping: 분류 번호 -> rack 인덱스 매핑
                          기본값: {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        """
        self.sim = sim
        self.positions: Dict[int, Tuple[float, float]] = {}
        self.rack_handles: Dict[int, int] = {}  # rack 인덱스 -> handle
        self.sa = SimulatedAnnealing()

        # 분류 -> rack 매핑 (1:1 매핑)
        # 분류 1 -> rack[0], 분류 2 -> rack[1], ... 분류 6 -> rack[5]
        self.rack_mapping = rack_mapping if rack_mapping else {
            1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5
        }

        if sim:
            self._load_rack_positions()
        else:
            # sim 없으면 기본 좌표 사용
            self._set_default_positions()

    def _load_rack_positions(self):
        """CoppeliaSim에서 rack 위치 로드"""
        print("[AGVRouteOptimizer] CoppeliaSim에서 rack 위치 로드 중...")

        for rack_idx in set(self.rack_mapping.values()):
            try:
                rack_handle = self.sim.getObject(f'/rack[{rack_idx}]')
                self.rack_handles[rack_idx] = rack_handle

                pos = self.sim.getObjectPosition(rack_handle, self.sim.handle_world)
                print(f"  rack[{rack_idx}]: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

                # 해당 rack에 매핑된 분류들에 좌표 할당
                for category, mapped_rack in self.rack_mapping.items():
                    if mapped_rack == rack_idx:
                        self.positions[category] = (pos[0], pos[1])

            except Exception as e:
                print(f"  [경고] rack[{rack_idx}] 로드 실패: {e}")

        print(f"[AGVRouteOptimizer] {len(self.positions)}개 분류 위치 로드 완료")

    def _set_default_positions(self):
        """기본 좌표 설정 (테스트용)"""
        default = {
            1: (4.075, 0.525),
            2: (2.725, 0.525),
            3: (1.5, 0.525),
            4: (0.075, 0.525),
            5: (1.325, -1.025),
            6: (2.625, -1.025),
        }
        self.positions = default.copy()

    def reload_positions(self):
        """rack 위치 다시 로드 (씬 변경 시 호출)"""
        if self.sim:
            self._load_rack_positions()
        else:
            print("[경고] sim 객체가 없어 위치를 다시 로드할 수 없습니다.")

    def set_position(self, category: int, x: float, y: float):
        """특정 분류의 적재 위치 설정"""
        self.positions[category] = (x, y)

    def optimize_delivery(self,
                          items: List[int],
                          start: Tuple[float, float],
                          return_to_start: bool = True) -> Dict:
        """
        배송 물품 목록에 대한 최적 경로 계산

        Args:
            items: 배송할 물품 분류 번호 리스트 (예: [1, 3, 5, 2])
            start: AGV 시작 위치
            return_to_start: 시작점 복귀 여부

        Returns:
            {
                'order': 최적 방문 순서 (물품 분류 번호),
                'positions': 방문 순서대로의 좌표,
                'total_distance': 총 이동 거리,
                'path': 전체 경로 [(x, y), ...]
            }
        """
        if not items:
            return {
                'order': [],
                'positions': [],
                'total_distance': 0.0,
                'path': [start]
            }

        # 중복 제거 및 위치 리스트 생성
        unique_items = list(dict.fromkeys(items))  # 순서 유지하며 중복 제거
        dest_positions = [self.positions[item] for item in unique_items if item in self.positions]

        if not dest_positions:
            return {
                'order': [],
                'positions': [],
                'total_distance': 0.0,
                'path': [start]
            }

        # SA 최적화
        order_indices, total_dist = self.sa.optimize(dest_positions, start, return_to_start)

        # 결과 구성
        ordered_items = [unique_items[i] for i in order_indices]
        ordered_positions = [dest_positions[i] for i in order_indices]

        path = [start] + ordered_positions
        if return_to_start:
            path.append(start)

        return {
            'order': ordered_items,
            'positions': ordered_positions,
            'total_distance': total_dist,
            'path': path
        }

    def get_next_destination(self,
                             current_pos: Tuple[float, float],
                             remaining_items: List[int]) -> Tuple[int, Tuple[float, float], float]:
        """
        현재 위치에서 가장 가까운 다음 목적지 반환 (Greedy)

        Args:
            current_pos: 현재 위치
            remaining_items: 남은 배송 물품 분류 번호

        Returns:
            (분류 번호, 좌표, 거리)
        """
        if not remaining_items:
            return None, None, 0.0

        min_dist = float('inf')
        nearest_item = remaining_items[0]
        nearest_pos = self.positions.get(nearest_item, (0, 0))

        for item in remaining_items:
            if item in self.positions:
                pos = self.positions[item]
                dist = math.sqrt((pos[0] - current_pos[0])**2 +
                                (pos[1] - current_pos[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_item = item
                    nearest_pos = pos

        return nearest_item, nearest_pos, min_dist

    def optimize_multi_agv(self,
                           items: List[int],
                           agv_starts: List[Tuple[float, float]],
                           num_agvs: int = None,
                           return_to_start: bool = True,
                           minimize_makespan: bool = False) -> Dict:
        """
        다중 AGV 배송 경로 최적화

        Args:
            items: 배송할 물품 분류 번호 리스트
            agv_starts: 각 AGV 시작 위치 리스트
            num_agvs: AGV 대수 (None이면 agv_starts 길이 사용)
            return_to_start: 시작점 복귀 여부
            minimize_makespan: True면 가장 늦게 끝나는 AGV 시간 최소화

        Returns:
            {
                'agv_routes': [
                    {'agv_id': 0, 'order': [1,3], 'positions': [...], 'distance': 5.2, 'path': [...]},
                    ...
                ],
                'total_distance': 총 이동 거리,
                'makespan': 최대 이동 거리
            }
        """
        if num_agvs is None:
            num_agvs = len(agv_starts)

        if not items:
            return {
                'agv_routes': [{'agv_id': i, 'order': [], 'positions': [], 'distance': 0.0, 'path': [agv_starts[i] if i < len(agv_starts) else agv_starts[0]]} for i in range(num_agvs)],
                'total_distance': 0.0,
                'makespan': 0.0
            }

        # 중복 제거 및 위치 리스트 생성
        unique_items = list(dict.fromkeys(items))
        dest_positions = [self.positions[item] for item in unique_items if item in self.positions]

        if not dest_positions:
            return {
                'agv_routes': [{'agv_id': i, 'order': [], 'positions': [], 'distance': 0.0, 'path': [agv_starts[i] if i < len(agv_starts) else agv_starts[0]]} for i in range(num_agvs)],
                'total_distance': 0.0,
                'makespan': 0.0
            }

        # 다중 AGV 최적화
        multi_optimizer = MultiAGVOptimizer(num_agvs=num_agvs)
        assignments, _ = multi_optimizer.optimize(dest_positions, list(agv_starts), return_to_start, minimize_makespan)

        # 결과 구성
        agv_routes = []
        distances = []

        for agv_idx, route_indices in enumerate(assignments):
            start = agv_starts[agv_idx] if agv_idx < len(agv_starts) else agv_starts[0]
            order = [unique_items[i] for i in route_indices]
            positions = [dest_positions[i] for i in route_indices]

            # 거리 계산
            dist = 0.0
            path = [start]
            current = start
            for pos in positions:
                dist += math.sqrt((pos[0] - current[0])**2 + (pos[1] - current[1])**2)
                path.append(pos)
                current = pos
            if return_to_start and positions:
                dist += math.sqrt((start[0] - current[0])**2 + (start[1] - current[1])**2)
                path.append(start)

            distances.append(dist)
            agv_routes.append({
                'agv_id': agv_idx,
                'order': order,
                'positions': positions,
                'distance': dist,
                'path': path
            })

        return {
            'agv_routes': agv_routes,
            'total_distance': sum(distances),
            'makespan': max(distances) if distances else 0.0
        }


class AGVPickupOptimizer:
    """
    AGV 물품 선택 및 경로 최적화기
    - 10개 도달한 상자는 필수 적재
    - AGV 용량 내에서 다른 상자의 물품 추가 선택
    - 상자 픽업 순서 + 보관 장소 배송 순서 최적화
    """

    def __init__(self, sim=None, agv_capacity: int = 100, trigger_count: int = 10):
        """
        Args:
            sim: CoppeliaSim sim 객체
            agv_capacity: AGV 최대 적재 용량 (크기 합계 기준)
            trigger_count: 트리거 발동 기준 (상자에 쌓인 물품 수)
        """
        self.sim = sim
        self.agv_capacity = agv_capacity
        self.trigger_count = trigger_count
        # 빠른 최적화를 위해 파라미터 조정 (6개 이하 위치는 빠르게 수렴)
        self.sa = SimulatedAnnealing(
            initial_temp=100.0,      # 낮은 초기 온도
            cooling_rate=0.95,       # 빠른 냉각
            min_temp=1.0,            # 빠른 종료
            max_iterations=500       # 적은 반복 (6개 위치에 충분)
        )

        # 각 분류별 물품 크기
        self.item_sizes: Dict[int, int] = {
            1: 2,
            2: 10,
            3: 4,
            4: 8,
            5: 6,
            6: 7
        }

        # 상자 위치 (Franka가 물건을 놓는 곳) - 분류 1~6
        self.box_positions: Dict[int, Tuple[float, float, float]] = {
            1: (4.075, 0.525, 1.25),
            2: (2.725, 0.525, 1.25),
            3: (1.5, 0.525, 1.25),
            4: (0.075, 0.525, 1.25),
            5: (1.325, -1.025, 1.25),
            6: (2.625, -1.025, 1.25)
        }

        # 최종 보관 장소 위치 (rack[0]~[5]) - 분류 1→rack[0], ..., 분류 6→rack[5]
        self.storage_positions: Dict[int, Tuple[float, float, float]] = {}

        if sim:
            self._load_rack_positions()
        else:
            self._set_default_storage_positions()

    def _load_rack_positions(self):
        """CoppeliaSim에서 rack 위치 로드"""
        print("[AGVPickupOptimizer] CoppeliaSim에서 rack 위치 로드 중...")
        for category in range(1, 7):
            rack_idx = category - 1  # 분류 1 → rack[0]
            try:
                rack_handle = self.sim.getObject(f'/rack[{rack_idx}]')
                pos = self.sim.getObjectPosition(rack_handle, self.sim.handle_world)
                self.storage_positions[category] = (pos[0], pos[1], pos[2])
                print(f"  분류 {category} → rack[{rack_idx}]: ({pos[0]:.3f}, {pos[1]:.3f})")
            except Exception as e:
                print(f"  [경고] rack[{rack_idx}] 로드 실패: {e}")
                # 기본값 사용
                self.storage_positions[category] = self.box_positions[category]
    # 
    def _set_default_storage_positions(self):
        """기본 보관 장소 좌표 설정 (테스트용)"""
        self.storage_positions = {
            1: (5.0, 2.0, 0.0),
            2: (4.0, 2.0, 0.0),
            3: (3.0, 2.0, 0.0),
            4: (2.0, 2.0, 0.0),
            5: (1.0, 2.0, 0.0),
            6: (0.0, 2.0, 0.0)
        }

    def _distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """두 점 사이 유클리드 거리 (2D)"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def _optimize_route_fast(self,
                              positions: List[Tuple[float, float]],
                              start: Tuple[float, float],
                              return_to_start: bool = True) -> Tuple[List[int], float]:
        """
        빠른 경로 최적화 (6개 이하: 브루트포스, 그 이상: SA)
        6! = 720가지는 즉시 계산 가능
        """
        from itertools import permutations

        n = len(positions)
        if n == 0:
            return [], 0.0
        if n == 1:
            dist = self._distance(start, positions[0])
            if return_to_start:
                dist *= 2
            return [0], dist

        # 6개 이하: 브루트포스 (최적해 보장, 매우 빠름)
        if n <= 6:
            best_order = None
            best_dist = float('inf')

            for perm in permutations(range(n)):
                dist = self._distance(start, positions[perm[0]])
                for i in range(len(perm) - 1):
                    dist += self._distance(positions[perm[i]], positions[perm[i + 1]])
                if return_to_start:
                    dist += self._distance(positions[perm[-1]], start)

                if dist < best_dist:
                    best_dist = dist
                    best_order = list(perm)

            return best_order, best_dist

        # 7개 이상: SA 사용
        return self.sa.optimize(positions, start, return_to_start)

    def check_trigger(self, box_counts: Dict[int, int]) -> List[int]:
        """
        트리거 발동 조건 확인

        Args:
            box_counts: 각 상자별 물품 수 {분류: 개수}

        Returns:
            10개 이상 도달한 상자 분류 리스트
        """
        triggered = [cat for cat, count in box_counts.items() if count >= self.trigger_count]
        return triggered

    def optimize_pickup(self,
                        box_counts: Dict[int, int],
                        triggered_boxes: List[int],
                        agv_start: Tuple[float, float]) -> Dict:
        """
        물품 선택 및 경로 최적화

        Args:
            box_counts: 각 상자별 물품 수 {분류: 개수}
            triggered_boxes: 트리거된 상자 분류 리스트 (필수 적재)
            agv_start: AGV 시작 위치 (x, y)

        Returns:
            {
                'pickup_plan': {분류: 가져갈 개수},
                'pickup_order': 상자 방문 순서 [분류, ...],
                'pickup_path': 상자 방문 경로 [(x, y), ...],
                'delivery_order': 보관장소 방문 순서 [분류, ...],
                'delivery_path': 보관장소 방문 경로 [(x, y), ...],
                'total_items': 총 적재 물품 수,
                'total_distance': 총 이동 거리
            }
        """
        pickup_plan = {}
        remaining_capacity = self.agv_capacity  # 크기 합계 기준

        # 1. 트리거된 상자는 전부 적재 (필수)
        for cat in triggered_boxes:
            count = box_counts.get(cat, 0)
            item_size = self.item_sizes.get(cat, 1)
            if count > 0:
                # 용량 내에서 최대한 가져감
                max_take = remaining_capacity // item_size
                take = min(count, max_take)
                if take > 0:
                    pickup_plan[cat] = take
                    remaining_capacity -= take * item_size

        # 2. 남은 용량으로 다른 상자에서 추가 적재 (효율 기반 Greedy)
        if remaining_capacity > 0:
            # 이미 방문할 보관 장소 목록
            committed_storages = set(pickup_plan.keys())

            # 후보: 트리거되지 않은 상자들
            candidates = [(cat, count) for cat, count in box_counts.items()
                          if cat not in triggered_boxes and count > 0]

            # 효율 점수 계산: 같은 보관 장소면 높은 점수, 새 장소면 낮은 점수
            # 크기가 작을수록 효율적 (같은 용량에 더 많이 담을 수 있음)
            scored_candidates = []
            for cat, count in candidates:
                item_size = self.item_sizes.get(cat, 1)
                if cat in committed_storages:
                    # 이미 방문할 보관 장소 → 추가 이동 비용 낮음
                    efficiency = (count / item_size) * 10  # 높은 점수
                else:
                    # 새 보관 장소 추가 → 상자→보관장소 거리 고려
                    box_pos = self.box_positions[cat][:2]
                    storage_pos = self.storage_positions[cat][:2]
                    extra_dist = self._distance(box_pos, storage_pos)
                    efficiency = (count / item_size) / (1 + extra_dist * 0.5)
                scored_candidates.append((cat, count, efficiency))

            # 효율 높은 순 정렬
            scored_candidates.sort(key=lambda x: x[2], reverse=True)

            # 남은 용량 내에서 추가
            for cat, count, _ in scored_candidates:
                if remaining_capacity <= 0:
                    break
                item_size = self.item_sizes.get(cat, 1)
                max_take = remaining_capacity // item_size
                take = min(count, max_take)
                if take > 0:
                    pickup_plan[cat] = take
                    remaining_capacity -= take * item_size

        # 3. 픽업 경로 최적화 (상자 방문 순서) - 빠른 브루트포스 사용
        pickup_categories = list(pickup_plan.keys())
        if pickup_categories:
            pickup_positions = [(self.box_positions[cat][0], self.box_positions[cat][1])
                                for cat in pickup_categories]
            pickup_order_idx, _ = self._optimize_route_fast(pickup_positions, agv_start, return_to_start=False)
            pickup_order = [pickup_categories[i] for i in pickup_order_idx]
            pickup_path = [agv_start] + [pickup_positions[i] for i in pickup_order_idx]
        else:
            pickup_order = []
            pickup_path = [agv_start]

        # 4. 배송 경로 최적화 (보관 장소 방문 순서) - 빠른 브루트포스 사용
        # 마지막 상자 위치에서 시작
        delivery_start = pickup_path[-1] if len(pickup_path) > 1 else agv_start

        delivery_categories = list(set(pickup_plan.keys()))  # 중복 제거
        if delivery_categories:
            delivery_positions = [(self.storage_positions[cat][0], self.storage_positions[cat][1])
                                  for cat in delivery_categories]
            delivery_order_idx, _ = self._optimize_route_fast(delivery_positions, delivery_start, return_to_start=True)
            delivery_order = [delivery_categories[i] for i in delivery_order_idx]
            delivery_path = [delivery_positions[i] for i in delivery_order_idx] + [agv_start]
        else:
            delivery_order = []
            delivery_path = [agv_start]

        # 5. 총 거리 계산
        total_distance = 0.0
        full_path = pickup_path + delivery_path[:-1]  # 마지막 복귀점 제외 (중복)
        for i in range(len(full_path) - 1):
            total_distance += self._distance(full_path[i], full_path[i + 1])
        # 마지막 복귀
        if delivery_path:
            total_distance += self._distance(delivery_path[-2] if len(delivery_path) > 1 else delivery_start, agv_start)

        # 총 적재량 계산 (개수 및 크기)
        total_items = sum(pickup_plan.values())
        total_size = sum(count * self.item_sizes.get(cat, 1) for cat, count in pickup_plan.items())

        return {
            'pickup_plan': pickup_plan,
            'pickup_order': pickup_order,
            'pickup_path': pickup_path,
            'delivery_order': delivery_order,
            'delivery_path': delivery_path,
            'total_items': total_items,
            'total_size': total_size,
            'remaining_capacity': remaining_capacity,
            'total_distance': total_distance
        }

    def get_full_route(self, optimization_result: Dict) -> List[Dict]:
        """
        최적화 결과를 AGV 이동 명령 리스트로 변환

        Args:
            optimization_result: optimize_pickup() 반환값

        Returns:
            [
                {'action': 'move', 'target': (x, y), 'description': '상자 1로 이동'},
                {'action': 'pickup', 'category': 1, 'count': 10, 'description': '분류 1 물품 10개 적재'},
                ...
                {'action': 'dropoff', 'category': 1, 'count': 10, 'description': '분류 1 물품 보관장소에 하차'},
                ...
            ]
        """
        route = []
        pickup_plan = optimization_result['pickup_plan']
        pickup_order = optimization_result['pickup_order']
        delivery_order = optimization_result['delivery_order']

        # 상자 방문 및 픽업
        for cat in pickup_order:
            box_pos = self.box_positions[cat]
            route.append({
                'action': 'move',
                'target': (box_pos[0], box_pos[1]),
                'description': f'상자 {cat}로 이동'
            })
            route.append({
                'action': 'pickup',
                'category': cat,
                'count': pickup_plan[cat],
                'description': f'분류 {cat} 물품 {pickup_plan[cat]}개 적재'
            })

        # 보관 장소 방문 및 하차
        for cat in delivery_order:
            storage_pos = self.storage_positions[cat]
            route.append({
                'action': 'move',
                'target': (storage_pos[0], storage_pos[1]),
                'description': f'보관장소 {cat}(rack[{cat-1}])로 이동'
            })
            route.append({
                'action': 'dropoff',
                'category': cat,
                'count': pickup_plan[cat],
                'description': f'분류 {cat} 물품 {pickup_plan[cat]}개 하차'
            })

        return route


# ============ 테스트 코드 ============
if __name__ == "__main__":
    print("=" * 60)
    print("AGV 경로 최적화 테스트 (Simulated Annealing)")
    print("=" * 60)

    # CoppeliaSim 연결 시도
    try:
        from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        client = RemoteAPIClient()
        sim = client.require('sim')
        print("\n[CoppeliaSim 연결 성공]")
        optimizer = AGVRouteOptimizer(sim=sim)
    except Exception as e:
        print(f"\n[CoppeliaSim 연결 실패: {e}]")
        print("기본 좌표로 테스트를 진행합니다.\n")
        optimizer = AGVRouteOptimizer()

    # 적재 위치 출력
    print("\n[적재 위치]")
    for cat, pos in sorted(optimizer.positions.items()):
        print(f"  분류 {cat}: ({pos[0]:.3f}, {pos[1]:.3f})")

    # 테스트 1: 전체 분류 배송
    print("\n[테스트 1] 모든 분류 위치 방문")
    print("-" * 40)

    start = (3.0, 0.0)  # AGV 시작 위치
    items = [1, 2, 3, 4, 5, 6]

    result = optimizer.optimize_delivery(items, start)

    print(f"시작 위치: {start}")
    print(f"배송 물품: {items}")
    print(f"최적 순서: {result['order']}")
    print(f"총 거리: {result['total_distance']:.2f}m")
    print(f"경로:")
    for i, pos in enumerate(result['path']):
        print(f"  {i}: ({pos[0]:.3f}, {pos[1]:.3f})")

    # 테스트 2: 일부 분류만 배송
    print("\n[테스트 2] 일부 분류만 배송")
    print("-" * 40)

    items = [2, 5, 1]
    result = optimizer.optimize_delivery(items, start)

    print(f"배송 물품: {items}")
    print(f"최적 순서: {result['order']}")
    print(f"총 거리: {result['total_distance']:.2f}m")

    # 테스트 3: Greedy 방식 비교
    print("\n[테스트 3] Greedy vs SA 비교")
    print("-" * 40)

    items = [1, 2, 3, 4, 5, 6]

    # Greedy 경로
    greedy_dist = 0.0
    greedy_order = []
    current = start
    remaining = items[:]

    while remaining:
        item, pos, dist = optimizer.get_next_destination(current, remaining)
        greedy_order.append(item)
        greedy_dist += dist
        current = pos
        remaining.remove(item)

    # 복귀 거리 추가
    greedy_dist += math.sqrt((start[0] - current[0])**2 + (start[1] - current[1])**2)

    # SA 결과
    sa_result = optimizer.optimize_delivery(items, start)

    print(f"Greedy 순서: {greedy_order}, 거리: {greedy_dist:.2f}m")
    print(f"SA 순서:     {sa_result['order']}, 거리: {sa_result['total_distance']:.2f}m")
    print(f"개선율: {((greedy_dist - sa_result['total_distance']) / greedy_dist * 100):.1f}%")

    # 테스트 4: 다중 AGV
    print("\n[테스트 4] 다중 AGV (2대)")
    print("-" * 40)

    items = [1, 2, 3, 4, 5, 6]
    agv_starts = [(1.0, 0.0), (5.0, 0.0)]  # 2대의 AGV 시작 위치

    result = optimizer.optimize_multi_agv(items, agv_starts, num_agvs=2)

    print(f"배송 물품: {items}")
    print(f"AGV 시작 위치: {agv_starts}")
    print(f"총 거리: {result['total_distance']:.2f}m")
    print(f"Makespan: {result['makespan']:.2f}m")

    for route in result['agv_routes']:
        print(f"  AGV {route['agv_id']}: 순서={route['order']}, 거리={route['distance']:.2f}m")

    # 테스트 5: 다중 AGV (3대)
    print("\n[테스트 5] 다중 AGV (3대)")
    print("-" * 40)

    agv_starts_3 = [(0.0, 0.0), (3.0, 0.0), (6.0, 0.0)]
    result = optimizer.optimize_multi_agv(items, agv_starts_3, num_agvs=3)

    print(f"AGV 시작 위치: {agv_starts_3}")
    print(f"총 거리: {result['total_distance']:.2f}m")
    print(f"Makespan: {result['makespan']:.2f}m")

    for route in result['agv_routes']:
        print(f"  AGV {route['agv_id']}: 순서={route['order']}, 거리={route['distance']:.2f}m")

    print("\n" + "=" * 60)
    print("테스트 완료!")
