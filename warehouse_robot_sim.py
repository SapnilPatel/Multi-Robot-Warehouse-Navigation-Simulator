import pygame
import random
import time
import heapq
from collections import deque

# =========================
# Configuration
# =========================
GRID_SIZE = 30
CELL_SIZE = 24
WINDOW_SIZE = GRID_SIZE * CELL_SIZE
FPS = 60

NUM_OBSTACLES = 110
NUM_TEST_RUNS = 200

# Colors
WHITE = (245, 245, 245)
BLACK = (30, 30, 30)
GRAY = (180, 180, 180)
GREEN = (34, 177, 76)
RED = (220, 20, 60)
BLUE = (0, 120, 215)
YELLOW = (255, 215, 0)
PURPLE = (140, 82, 255)

# =========================
# Grid / Warehouse Helpers
# =========================
def create_empty_grid(size: int):
    return [[0 for _ in range(size)] for _ in range(size)]


def is_valid_cell(x: int, y: int, size: int) -> bool:
    return 0 <= x < size and 0 <= y < size


def place_random_obstacles(grid, num_obstacles, start, goal):
    """
    Places random obstacles while keeping start and goal free.
    0 = free cell, 1 = obstacle
    """
    size = len(grid)
    placed = 0
    attempts = 0
    max_attempts = num_obstacles * 20

    while placed < num_obstacles and attempts < max_attempts:
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        if (x, y) != start and (x, y) != goal and grid[y][x] == 0:
            grid[y][x] = 1
            placed += 1
        attempts += 1


def create_warehouse_grid(size, num_obstacles, start, goal):
    """
    Creates a warehouse-like layout:
    - random obstacles
    - some structured shelf rows
    """
    grid = create_empty_grid(size)

    # Add shelf-like blocks in rows
    for row in range(3, size - 3, 5):
        for col in range(2, size - 2):
            # Leave aisle gaps
            if col % 7 not in (0, 1):
                if (col, row) != start and (col, row) != goal:
                    grid[row][col] = 1

    # Add more random obstacles to exceed 100+
    place_random_obstacles(grid, num_obstacles, start, goal)

    # Ensure start/goal are free
    grid[start[1]][start[0]] = 0
    grid[goal[1]][goal[0]] = 0
    return grid


def get_neighbors(node, grid):
    x, y = node
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    neighbors = []

    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if is_valid_cell(nx, ny, len(grid)) and grid[ny][nx] == 0:
            neighbors.append((nx, ny))

    return neighbors


# =========================
# Pathfinding
# =========================
def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(grid, start, goal):
    open_heap = []
    heapq.heappush(open_heap, (0, start))

    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    open_set = {start}
    expanded_nodes = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current not in open_set:
            continue

        open_set.remove(current)
        expanded_nodes += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, expanded_nodes

        for neighbor in get_neighbors(current, grid):
            tentative_g = g_score[current] + 1

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)

                if neighbor not in open_set:
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))
                    open_set.add(neighbor)

    return None, expanded_nodes


def bfs(grid, start, goal):
    queue = deque([start])
    came_from = {}
    visited = {start}
    expanded_nodes = 0

    while queue:
        current = queue.popleft()
        expanded_nodes += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            return path, expanded_nodes

        for neighbor in get_neighbors(current, grid):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return None, expanded_nodes


# =========================
# Metrics / Benchmarking
# =========================
def random_free_cell(grid):
    size = len(grid)
    while True:
        x = random.randint(0, size - 1)
        y = random.randint(0, size - 1)
        if grid[y][x] == 0:
            return (x, y)


def run_benchmark(num_runs=200):
    astar_total_time = 0.0
    bfs_total_time = 0.0
    astar_total_len = 0
    bfs_total_len = 0
    astar_success = 0
    bfs_success = 0
    astar_total_expanded = 0
    bfs_total_expanded = 0

    valid_runs = 0

    for _ in range(num_runs):
        start = (1, 1)
        goal = (GRID_SIZE - 2, GRID_SIZE - 2)
        grid = create_warehouse_grid(GRID_SIZE, NUM_OBSTACLES, start, goal)

        # Pick random free start/goal to make benchmark more realistic
        start = random_free_cell(grid)
        goal = random_free_cell(grid)
        while goal == start:
            goal = random_free_cell(grid)

        t0 = time.perf_counter()
        astar_path, astar_expanded = astar(grid, start, goal)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        bfs_path, bfs_expanded = bfs(grid, start, goal)
        t3 = time.perf_counter()

        # Only count runs where both found a path
        if astar_path and bfs_path:
            valid_runs += 1

            astar_total_time += (t1 - t0)
            bfs_total_time += (t3 - t2)

            astar_total_len += len(astar_path)
            bfs_total_len += len(bfs_path)

            astar_total_expanded += astar_expanded
            bfs_total_expanded += bfs_expanded

            astar_success += 1
            bfs_success += 1

    if valid_runs == 0:
        return None

    avg_astar_time_ms = (astar_total_time / valid_runs) * 1000
    avg_bfs_time_ms = (bfs_total_time / valid_runs) * 1000
    avg_astar_len = astar_total_len / valid_runs
    avg_bfs_len = bfs_total_len / valid_runs
    avg_astar_expanded = astar_total_expanded / valid_runs
    avg_bfs_expanded = bfs_total_expanded / valid_runs

    # Since BFS on an unweighted grid often finds the same shortest path length,
    # we quantify "efficiency" using node expansions.
    # This gives a more meaningful difference for A* vs BFS.
    reduction_in_expanded = 0.0
    if avg_bfs_expanded > 0:
        reduction_in_expanded = ((avg_bfs_expanded - avg_astar_expanded) / avg_bfs_expanded) * 100

    return {
        "valid_runs": valid_runs,
        "avg_astar_time_ms": avg_astar_time_ms,
        "avg_bfs_time_ms": avg_bfs_time_ms,
        "avg_astar_len": avg_astar_len,
        "avg_bfs_len": avg_bfs_len,
        "avg_astar_expanded": avg_astar_expanded,
        "avg_bfs_expanded": avg_bfs_expanded,
        "expanded_reduction_pct": reduction_in_expanded,
    }


# =========================
# Visualization
# =========================
def draw_grid(screen, grid, start, goal, path, robot_pos):
    screen.fill(WHITE)

    size = len(grid)
    path_set = set(path) if path else set()

    for y in range(size):
        for x in range(size):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)

            if grid[y][x] == 1:
                pygame.draw.rect(screen, BLACK, rect)
            elif (x, y) in path_set:
                pygame.draw.rect(screen, YELLOW, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)

            pygame.draw.rect(screen, GRAY, rect, 1)

    # Start
    pygame.draw.rect(
        screen,
        GREEN,
        pygame.Rect(start[0] * CELL_SIZE, start[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
    )

    # Goal
    pygame.draw.rect(
        screen,
        RED,
        pygame.Rect(goal[0] * CELL_SIZE, goal[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE),
    )

    # Robot
    if robot_pos:
        cx = robot_pos[0] * CELL_SIZE + CELL_SIZE // 2
        cy = robot_pos[1] * CELL_SIZE + CELL_SIZE // 2
        pygame.draw.circle(screen, BLUE, (cx, cy), CELL_SIZE // 3)

    pygame.display.flip()


def animate_robot(screen, clock, grid, start, goal, path):
    if not path:
        return

    for pos in path:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        draw_grid(screen, grid, start, goal, path, pos)
        clock.tick(12)  # animation speed


# =========================
# Main
# =========================
def main():
    random.seed(42)

    # Benchmark first
    print("\nRunning benchmark...")
    results = run_benchmark(NUM_TEST_RUNS)

    if results is None:
        print("Could not find valid paths during benchmark.")
        return

    print("\n===== Benchmark Results =====")
    print(f"Valid runs: {results['valid_runs']}")
    print(f"A* average time:  {results['avg_astar_time_ms']:.2f} ms")
    print(f"BFS average time: {results['avg_bfs_time_ms']:.2f} ms")
    print(f"A* average path length:  {results['avg_astar_len']:.2f}")
    print(f"BFS average path length: {results['avg_bfs_len']:.2f}")
    print(f"A* avg expanded nodes:  {results['avg_astar_expanded']:.2f}")
    print(f"BFS avg expanded nodes: {results['avg_bfs_expanded']:.2f}")
    print(
        f"A* reduced node expansions by "
        f"{results['expanded_reduction_pct']:.2f}% vs BFS"
    )

    # Build one demo environment for visualization
    start = (1, 1)
    goal = (GRID_SIZE - 2, GRID_SIZE - 2)
    grid = create_warehouse_grid(GRID_SIZE, NUM_OBSTACLES, start, goal)

    astar_start = time.perf_counter()
    path, expanded = astar(grid, start, goal)
    astar_end = time.perf_counter()

    if not path:
        print("No path found in visualization demo.")
        return

    print("\n===== Demo Run =====")
    print(f"Path length: {len(path)}")
    print(f"Expanded nodes: {expanded}")
    print(f"A* computation time: {(astar_end - astar_start) * 1000:.2f} ms")

    # Visualization
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Warehouse Robot Path Planning Simulator")
    clock = pygame.time.Clock()

    running = True
    first_animation_done = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Press R to regenerate another warehouse demo
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                start = (1, 1)
                goal = (GRID_SIZE - 2, GRID_SIZE - 2)
                grid = create_warehouse_grid(GRID_SIZE, NUM_OBSTACLES, start, goal)
                path, expanded = astar(grid, start, goal)
                first_animation_done = False

        if not first_animation_done:
            animate_robot(screen, clock, grid, start, goal, path)
            first_animation_done = True
        else:
            final_robot_pos = path[-1] if path else None
            draw_grid(screen, grid, start, goal, path, final_robot_pos)
            clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()