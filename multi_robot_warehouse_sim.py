import pygame
import random
import time
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# =========================================================
# Configuration
# =========================================================
GRID_SIZE = 30
CELL_SIZE = 24
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 12

NUM_OBSTACLES = 110
NUM_ROBOTS = 3
NUM_BENCHMARK_RUNS = 200

# Colors
WHITE = (245, 245, 245)
BLACK = (40, 40, 40)
GRAY = (200, 200, 200)
YELLOW = (255, 230, 120)
START_GOAL_OUTLINE = (80, 80, 80)

ROBOT_COLORS = [
    (0, 120, 215),   # Blue
    (220, 20, 60),   # Crimson
    (34, 177, 76),   # Green
    (255, 140, 0),   # Orange
    (140, 82, 255),  # Purple
]

# =========================================================
# Data structures
# =========================================================
Position = Tuple[int, int]


@dataclass
class Robot:
    robot_id: int
    start: Position
    goal: Position
    color: Tuple[int, int, int]
    position: Position = field(init=False)
    path: List[Position] = field(default_factory=list)
    path_index: int = 0
    completed: bool = False
    wait_steps: int = 0
    replans: int = 0

    def __post_init__(self):
        self.position = self.start

    def reset(self):
        self.position = self.start
        self.path = []
        self.path_index = 0
        self.completed = False
        self.wait_steps = 0
        self.replans = 0

    def next_position(self) -> Position:
        if self.completed or not self.path:
            return self.position
        if self.path_index + 1 < len(self.path):
            return self.path[self.path_index + 1]
        return self.position


# =========================================================
# Grid / warehouse generation
# =========================================================
def create_empty_grid(size: int) -> List[List[int]]:
    return [[0 for _ in range(size)] for _ in range(size)]


def is_valid_cell(x: int, y: int, size: int) -> bool:
    return 0 <= x < size and 0 <= y < size


def add_structured_shelves(grid: List[List[int]]):
    size = len(grid)
    # Shelf rows with aisle gaps
    for row in range(3, size - 3, 5):
        for col in range(2, size - 2):
            if col % 7 not in (0, 1):  # aisle openings
                grid[row][col] = 1


def place_random_obstacles(
    grid: List[List[int]],
    num_obstacles: int,
    reserved_cells: Set[Position],
):
    size = len(grid)
    placed = 0
    attempts = 0
    max_attempts = num_obstacles * 30

    while placed < num_obstacles and attempts < max_attempts:
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        if grid[y][x] == 0 and (x, y) not in reserved_cells:
            grid[y][x] = 1
            placed += 1
        attempts += 1


def random_free_cell(grid: List[List[int]], reserved: Set[Position]) -> Position:
    size = len(grid)
    while True:
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        if grid[y][x] == 0 and (x, y) not in reserved:
            return (x, y)


def create_warehouse_with_robots(
    size: int,
    num_obstacles: int,
    num_robots: int,
) -> Tuple[List[List[int]], List[Robot]]:
    grid = create_empty_grid(size)
    add_structured_shelves(grid)

    robots: List[Robot] = []
    reserved: Set[Position] = set()

    # Temporary obstacle placement after reserving robot starts/goals
    for i in range(num_robots):
        # choose start
        start = random_free_cell(grid, reserved)
        reserved.add(start)

        # choose goal
        goal = random_free_cell(grid, reserved)
        reserved.add(goal)

        robot = Robot(
            robot_id=i,
            start=start,
            goal=goal,
            color=ROBOT_COLORS[i % len(ROBOT_COLORS)],
        )
        robots.append(robot)

    place_random_obstacles(grid, num_obstacles, reserved)

    # ensure reserved cells remain free
    for cell in reserved:
        grid[cell[1]][cell[0]] = 0

    return grid, robots


# =========================================================
# Pathfinding
# =========================================================
def heuristic(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(node: Position, grid: List[List[int]]) -> List[Position]:
    x, y = node
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    neighbors = []
    for dx, dy in moves:
        nx, ny = x + dx, y + dy
        if is_valid_cell(nx, ny, len(grid)) and grid[ny][nx] == 0:
            neighbors.append((nx, ny))
    return neighbors


def reconstruct_path(came_from: Dict[Position, Position], current: Position) -> List[Position]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(
    grid: List[List[int]],
    start: Position,
    goal: Position,
    blocked_cells: Optional[Set[Position]] = None,
) -> Optional[List[Position]]:
    if blocked_cells is None:
        blocked_cells = set()

    open_heap: List[Tuple[int, Position]] = []
    heapq.heappush(open_heap, (0, start))

    came_from: Dict[Position, Position] = {}
    g_score: Dict[Position, int] = {start: 0}
    f_score: Dict[Position, int] = {start: heuristic(start, goal)}
    open_set = {start}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current not in open_set:
            continue
        open_set.remove(current)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in get_neighbors(current, grid):
            if neighbor in blocked_cells and neighbor != goal:
                continue

            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                if neighbor not in open_set:
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))
                    open_set.add(neighbor)

    return None


# =========================================================
# Multi-robot planning and simulation logic
# =========================================================
def initial_plan_all_robots(grid: List[List[int]], robots: List[Robot]) -> bool:
    for robot in robots:
        blocked = {r.start for r in robots if r.robot_id != robot.robot_id}
        path = astar(grid, robot.start, robot.goal, blocked_cells=blocked)
        if path is None:
            return False
        robot.path = path
        robot.path_index = 0
        robot.position = robot.start
        robot.completed = (robot.start == robot.goal)
    return True


def all_completed(robots: List[Robot]) -> bool:
    return all(robot.completed for robot in robots)


def simulate_multi_robot_run(
    grid: List[List[int]],
    robots: List[Robot],
    max_steps: int = 300,
) -> Dict[str, float]:
    """
    Collision rules:
    1. Two robots cannot move into the same cell in the same timestep.
    2. Two robots cannot swap positions in the same timestep.
    3. Lower-priority robot waits; if repeated conflict persists, it replans.
    Priority = lower robot_id wins.
    """
    total_steps = 0
    collision_conflicts = 0
    successful = True

    if not initial_plan_all_robots(grid, robots):
        return {
            "success": 0,
            "steps": 0,
            "wait_steps": 0,
            "replans": 0,
            "conflicts": 0,
        }

    for _ in range(max_steps):
        total_steps += 1

        if all_completed(robots):
            break

        current_positions = {r.robot_id: r.position for r in robots}
        desired_moves: Dict[int, Position] = {}

        # Phase 1: propose next positions
        for robot in robots:
            if robot.completed:
                desired_moves[robot.robot_id] = robot.position
                continue

            next_pos = robot.next_position()
            desired_moves[robot.robot_id] = next_pos

        # Phase 2: detect same-cell conflicts
        target_to_robot_ids: Dict[Position, List[int]] = {}
        for rid, target in desired_moves.items():
            target_to_robot_ids.setdefault(target, []).append(rid)

        blocked_this_step: Set[int] = set()

        for target, robot_ids in target_to_robot_ids.items():
            if len(robot_ids) > 1:
                collision_conflicts += 1
                robot_ids_sorted = sorted(robot_ids)
                # allow lowest id, others wait
                for loser_id in robot_ids_sorted[1:]:
                    blocked_this_step.add(loser_id)

        # Phase 3: detect swap conflicts
        robot_map = {r.robot_id: r for r in robots}
        for i in range(len(robots)):
            for j in range(i + 1, len(robots)):
                r1 = robots[i]
                r2 = robots[j]

                if r1.robot_id in blocked_this_step or r2.robot_id in blocked_this_step:
                    continue

                if (
                    desired_moves[r1.robot_id] == current_positions[r2.robot_id]
                    and desired_moves[r2.robot_id] == current_positions[r1.robot_id]
                    and desired_moves[r1.robot_id] != current_positions[r1.robot_id]
                    and desired_moves[r2.robot_id] != current_positions[r2.robot_id]
                ):
                    collision_conflicts += 1
                    loser = max(r1.robot_id, r2.robot_id)
                    blocked_this_step.add(loser)

        # Phase 4: apply moves, wait, or replan
        occupied_now = {r.position for r in robots}
        for robot in sorted(robots, key=lambda r: r.robot_id):
            if robot.completed:
                continue

            if robot.robot_id in blocked_this_step:
                robot.wait_steps += 1

                # If blocked repeatedly, try replanning
                if robot.wait_steps % 3 == 0:
                    other_positions = {
                        r.position for r in robots if r.robot_id != robot.robot_id
                    }
                    new_path = astar(
                        grid,
                        robot.position,
                        robot.goal,
                        blocked_cells=other_positions,
                    )
                    if new_path is not None:
                        robot.path = new_path
                        robot.path_index = 0
                        robot.replans += 1
                continue

            next_pos = desired_moves[robot.robot_id]

            # Stay if already at destination
            if robot.position == robot.goal:
                robot.completed = True
                continue

            # Sanity check for blocked grid cell
            if grid[next_pos[1]][next_pos[0]] == 1:
                robot.wait_steps += 1
                continue

            # Move if next cell is not occupied by someone who is staying there
            staying_positions = {
                current_positions[r.robot_id]
                for r in robots
                if desired_moves[r.robot_id] == current_positions[r.robot_id]
            }
            if next_pos in staying_positions and next_pos != robot.goal:
                robot.wait_steps += 1
                continue

            # Advance along path if next step is valid
            if robot.path_index + 1 < len(robot.path) and robot.path[robot.path_index + 1] == next_pos:
                robot.position = next_pos
                robot.path_index += 1

                if robot.position == robot.goal:
                    robot.completed = True
            else:
                # Path got stale, try replan
                other_positions = {r.position for r in robots if r.robot_id != robot.robot_id}
                new_path = astar(grid, robot.position, robot.goal, blocked_cells=other_positions)
                if new_path is not None:
                    robot.path = new_path
                    robot.path_index = 0
                    robot.replans += 1
                else:
                    robot.wait_steps += 1

    if not all_completed(robots):
        successful = False

    return {
        "success": 1 if successful else 0,
        "steps": total_steps,
        "wait_steps": sum(r.wait_steps for r in robots),
        "replans": sum(r.replans for r in robots),
        "conflicts": collision_conflicts,
    }


# =========================================================
# Benchmarking
# =========================================================
def benchmark_multi_robot(runs: int = NUM_BENCHMARK_RUNS) -> Dict[str, float]:
    success_count = 0
    total_steps = 0
    total_wait_steps = 0
    total_replans = 0
    total_conflicts = 0
    total_planning_ms = 0.0

    valid_runs = 0

    for _ in range(runs):
        grid, robots = create_warehouse_with_robots(GRID_SIZE, NUM_OBSTACLES, NUM_ROBOTS)

        t0 = time.perf_counter()
        ok = initial_plan_all_robots(grid, robots)
        t1 = time.perf_counter()

        if not ok:
            continue

        planning_ms = (t1 - t0) * 1000

        # reset planned robots to run clean simulation
        for r in robots:
            r.reset()

        metrics = simulate_multi_robot_run(grid, robots)
        valid_runs += 1

        total_planning_ms += planning_ms
        success_count += metrics["success"]
        total_steps += metrics["steps"]
        total_wait_steps += metrics["wait_steps"]
        total_replans += metrics["replans"]
        total_conflicts += metrics["conflicts"]

    if valid_runs == 0:
        return {
            "valid_runs": 0
        }

    return {
        "valid_runs": valid_runs,
        "success_rate_pct": (success_count / valid_runs) * 100,
        "avg_steps": total_steps / valid_runs,
        "avg_wait_steps": total_wait_steps / valid_runs,
        "avg_replans": total_replans / valid_runs,
        "avg_conflicts": total_conflicts / valid_runs,
        "avg_initial_planning_ms": total_planning_ms / valid_runs,
    }


# =========================================================
# Visualization
# =========================================================
def draw_grid(
    screen: pygame.Surface,
    grid: List[List[int]],
    robots: List[Robot],
    show_paths: bool = True,
):
    screen.fill(WHITE)

    # draw paths first
    if show_paths:
        for robot in robots:
            for cell in robot.path:
                rect = pygame.Rect(
                    cell[0] * CELL_SIZE,
                    cell[1] * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                )
                pygame.draw.rect(screen, YELLOW, rect)

    # draw cells
    for y in range(len(grid)):
        for x in range(len(grid[0])):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if grid[y][x] == 1:
                pygame.draw.rect(screen, BLACK, rect)
            pygame.draw.rect(screen, GRAY, rect, 1)

    # draw starts and goals
    for robot in robots:
        sx, sy = robot.start
        gx, gy = robot.goal

        start_rect = pygame.Rect(sx * CELL_SIZE, sy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        goal_rect = pygame.Rect(gx * CELL_SIZE, gy * CELL_SIZE, CELL_SIZE, CELL_SIZE)

        pygame.draw.rect(screen, robot.color, start_rect, border_radius=3)
        pygame.draw.rect(screen, robot.color, goal_rect, 3, border_radius=3)

    # draw robots on top
    for robot in robots:
        x, y = robot.position
        cx = x * CELL_SIZE + CELL_SIZE // 2
        cy = y * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, robot.color, (cx, cy), CELL_SIZE // 3)

    pygame.display.flip()


def visualize_simulation(screen: pygame.Surface, clock: pygame.time.Clock):
    grid, robots = create_warehouse_with_robots(GRID_SIZE, NUM_OBSTACLES, NUM_ROBOTS)
    planned = initial_plan_all_robots(grid, robots)

    if not planned:
        return

    running = True
    paused = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    return "reset"
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused and not all_completed(robots):
            simulate_one_visual_step(grid, robots)

        draw_grid(screen, grid, robots, show_paths=True)
        clock.tick(FPS)

    return None


def simulate_one_visual_step(grid: List[List[int]], robots: List[Robot]):
    current_positions = {r.robot_id: r.position for r in robots}
    desired_moves: Dict[int, Position] = {}

    for robot in robots:
        if robot.completed:
            desired_moves[robot.robot_id] = robot.position
        else:
            desired_moves[robot.robot_id] = robot.next_position()

    blocked_this_step: Set[int] = set()

    # same-target conflicts
    target_to_robot_ids: Dict[Position, List[int]] = {}
    for rid, target in desired_moves.items():
        target_to_robot_ids.setdefault(target, []).append(rid)

    for _, robot_ids in target_to_robot_ids.items():
        if len(robot_ids) > 1:
            robot_ids_sorted = sorted(robot_ids)
            for loser_id in robot_ids_sorted[1:]:
                blocked_this_step.add(loser_id)

    # swap conflicts
    for i in range(len(robots)):
        for j in range(i + 1, len(robots)):
            r1 = robots[i]
            r2 = robots[j]
            if r1.robot_id in blocked_this_step or r2.robot_id in blocked_this_step:
                continue
            if (
                desired_moves[r1.robot_id] == current_positions[r2.robot_id]
                and desired_moves[r2.robot_id] == current_positions[r1.robot_id]
                and desired_moves[r1.robot_id] != current_positions[r1.robot_id]
                and desired_moves[r2.robot_id] != current_positions[r2.robot_id]
            ):
                loser = max(r1.robot_id, r2.robot_id)
                blocked_this_step.add(loser)

    for robot in sorted(robots, key=lambda r: r.robot_id):
        if robot.completed:
            continue

        if robot.robot_id in blocked_this_step:
            robot.wait_steps += 1
            if robot.wait_steps % 3 == 0:
                other_positions = {r.position for r in robots if r.robot_id != robot.robot_id}
                new_path = astar(grid, robot.position, robot.goal, blocked_cells=other_positions)
                if new_path is not None:
                    robot.path = new_path
                    robot.path_index = 0
                    robot.replans += 1
            continue

        next_pos = desired_moves[robot.robot_id]

        if robot.position == robot.goal:
            robot.completed = True
            continue

        if robot.path_index + 1 < len(robot.path) and robot.path[robot.path_index + 1] == next_pos:
            robot.position = next_pos
            robot.path_index += 1
            if robot.position == robot.goal:
                robot.completed = True
        else:
            other_positions = {r.position for r in robots if r.robot_id != robot.robot_id}
            new_path = astar(grid, robot.position, robot.goal, blocked_cells=other_positions)
            if new_path is not None:
                robot.path = new_path
                robot.path_index = 0
                robot.replans += 1
            else:
                robot.wait_steps += 1


# =========================================================
# Main
# =========================================================
def main():
    random.seed(42)

    print("\nRunning multi-robot benchmark...")
    results = benchmark_multi_robot(NUM_BENCHMARK_RUNS)

    if results["valid_runs"] == 0:
        print("No valid runs found.")
        return

    print("\n===== Multi-Robot Benchmark Results =====")
    print(f"Valid runs: {results['valid_runs']}")
    print(f"Success rate: {results['success_rate_pct']:.2f}%")
    print(f"Average completion steps: {results['avg_steps']:.2f}")
    print(f"Average wait steps: {results['avg_wait_steps']:.2f}")
    print(f"Average replans: {results['avg_replans']:.2f}")
    print(f"Average conflicts detected: {results['avg_conflicts']:.2f}")
    print(f"Average initial planning time: {results['avg_initial_planning_ms']:.2f} ms")

    print("\nControls:")
    print("  R      -> generate a new warehouse run")
    print("  SPACE  -> pause / resume")
    print("  Close window to exit")

    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Multi-Robot Warehouse Navigation Simulator")
    clock = pygame.time.Clock()

    while True:
        result = visualize_simulation(screen, clock)
        if result == "quit":
            break

    pygame.quit()


if __name__ == "__main__":
    main()