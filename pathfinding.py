"""
童话镇寻路系统 —— 障碍物定义、A* 寻路、碰撞检测。
"""

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

# ── 常量 ────────────────────────────────────────────────

CELL_SIZE = 20
GRID_COLS = 40   # 800 / 20
GRID_ROWS = 30   # 600 / 20
AGENT_RADIUS = 12

# ── 障碍物数据结构 ──────────────────────────────────────

@dataclass
class CircleObstacle:
    kind: str       # "tree" | "rock"
    cx: float
    cy: float
    r: float


@dataclass
class RectObstacle:
    kind: str       # "house"
    x: float
    y: float
    w: float
    h: float


Obstacle = Union[CircleObstacle, RectObstacle]


def obstacle_to_dict(obs: Obstacle) -> dict:
    if isinstance(obs, CircleObstacle):
        return {"type": obs.kind, "cx": obs.cx, "cy": obs.cy, "r": obs.r}
    else:
        return {"type": obs.kind, "x": obs.x, "y": obs.y, "w": obs.w, "h": obs.h}


# ── 坐标/格子互转 ──────────────────────────────────────

def pos_to_cell(x: float, y: float) -> Tuple[int, int]:
    return (int(x // CELL_SIZE), int(y // CELL_SIZE))


def cell_to_pos(cx: int, cy: int) -> Tuple[float, float]:
    return (cx * CELL_SIZE + CELL_SIZE / 2, cy * CELL_SIZE + CELL_SIZE / 2)


# ── 碰撞检测 ────────────────────────────────────────────

def is_position_blocked(x: float, y: float, obstacles: List[Obstacle],
                        margin: float = AGENT_RADIUS) -> bool:
    for obs in obstacles:
        if isinstance(obs, CircleObstacle):
            if math.hypot(x - obs.cx, y - obs.cy) < obs.r + margin:
                return True
        else:
            if (obs.x - margin < x < obs.x + obs.w + margin and
                    obs.y - margin < y < obs.y + obs.h + margin):
                return True
    return False


# ── 网格构建 ────────────────────────────────────────────

def build_grid(obstacles: List[Obstacle]) -> List[List[bool]]:
    """将障碍物转换为网格碰撞图，True 表示不可通行。"""
    grid = [[False] * GRID_COLS for _ in range(GRID_ROWS)]
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            cx, cy = cell_to_pos(col, row)
            if is_position_blocked(cx, cy, obstacles):
                grid[row][col] = True
    return grid


# ── A* 寻路 ─────────────────────────────────────────────

_NEIGHBORS = [
    (-1,  0, 1.0), (1,  0, 1.0), (0, -1, 1.0), (0,  1, 1.0),
    (-1, -1, math.sqrt(2)), (1, -1, math.sqrt(2)),
    (-1,  1, math.sqrt(2)), (1,  1, math.sqrt(2)),
]


def _nearest_unblocked(grid: List[List[bool]],
                       start: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """BFS 寻找最近的可行走格子。"""
    if not grid[start[1]][start[0]]:
        return start
    q = deque([start])
    visited = {start}
    while q:
        cx, cy = q.popleft()
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0),
                        (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS and (nx, ny) not in visited:
                if not grid[ny][nx]:
                    return (nx, ny)
                visited.add((nx, ny))
                q.append((nx, ny))
    return None


def _astar(grid: List[List[bool]],
           start: Tuple[int, int],
           goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* 寻路，返回格子坐标列表或 None。"""
    if grid[start[1]][start[0]] or grid[goal[1]][goal[0]]:
        return None

    open_set = [(0.0, start[0], start[1])]  # (f, col, row)
    came_from: dict = {}
    g_score = {start: 0.0}

    while open_set:
        _, cx, cy = heapq.heappop(open_set)
        current = (cx, cy)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        current_g = g_score[current]

        for dx, dy, cost in _NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < GRID_COLS and 0 <= ny < GRID_ROWS):
                continue
            if grid[ny][nx]:
                continue
            new_g = current_g + cost
            neighbor = (nx, ny)
            if new_g < g_score.get(neighbor, float('inf')):
                g_score[neighbor] = new_g
                h = math.hypot(nx - goal[0], ny - goal[1])
                heapq.heappush(open_set, (new_g + h, nx, ny))
                came_from[neighbor] = current

    return None


def find_path(obstacles: List[Obstacle],
              start_pos: Tuple[float, float],
              goal_pos: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
    """给定障碍物列表和起止像素坐标，返回路径点列表或 None。"""
    grid = build_grid(obstacles)

    start_cell = pos_to_cell(start_pos[0], start_pos[1])
    goal_cell = pos_to_cell(goal_pos[0], goal_pos[1])

    # clamp to grid bounds
    start_cell = (max(0, min(GRID_COLS - 1, start_cell[0])),
                  max(0, min(GRID_ROWS - 1, start_cell[1])))
    goal_cell = (max(0, min(GRID_COLS - 1, goal_cell[0])),
                 max(0, min(GRID_ROWS - 1, goal_cell[1])))

    if grid[start_cell[1]][start_cell[0]]:
        start_cell = _nearest_unblocked(grid, start_cell)
        if start_cell is None:
            return None
    if grid[goal_cell[1]][goal_cell[0]]:
        goal_cell = _nearest_unblocked(grid, goal_cell)
        if goal_cell is None:
            return None

    cell_path = _astar(grid, start_cell, goal_cell)
    if cell_path is None:
        return None

    pixel_path = [cell_to_pos(c[0], c[1]) for c in cell_path]
    # 追加精确目标点，避免角色停在格子中心而非用户点击位置
    if pixel_path:
        last = pixel_path[-1]
        if math.hypot(last[0] - goal_pos[0], last[1] - goal_pos[1]) > 1:
            pixel_path.append(goal_pos)
    return pixel_path
