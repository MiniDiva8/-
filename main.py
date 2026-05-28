"""
童话镇 Web 服务 —— FastAPI 后端。
管理智能体在 800x600 二维空间中的位置、触发近距离互动，
并通过 REST API 将状态暴露给前端渲染。
"""

import math
import os
import random
import time

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import Agent
from pathfinding import (
    CircleObstacle, RectObstacle, Obstacle,
    find_path, is_position_blocked, obstacle_to_dict,
)
from script_parser import Character

# ── 常量 ────────────────────────────────────────────────

MAP_W, MAP_H = 800, 600
INTERACT_DIST = 100   # 触发交互的距离阈值（像素）
MOVE_RANGE = 25       # 每步最大随机移动像素
MANUAL_STEP = 40      # 手动模式每步移动像素（配合 500ms tick，约 80px/s）
INTERACT_COOLDOWN = 4.0  # 交互冷却时间（秒），防止 LLM 调用阻塞移动

_last_interact_time = 0.0

# ── 障碍物定义 ──────────────────────────────────────────

obstacles: list[Obstacle] = [
    # ── 左上角小树林 ──
    CircleObstacle("tree", 70, 60, 38),
    CircleObstacle("tree", 140, 45, 32),
    CircleObstacle("tree", 55, 120, 35),
    CircleObstacle("tree", 130, 110, 40),
    CircleObstacle("tree", 180, 80, 30),

    # ── 右上角小树林 ──
    CircleObstacle("tree", 680, 55, 35),
    CircleObstacle("tree", 740, 70, 38),
    CircleObstacle("tree", 660, 130, 32),
    CircleObstacle("tree", 720, 120, 40),

    # ── 右下角小树林 ──
    CircleObstacle("tree", 690, 510, 36),
    CircleObstacle("tree", 750, 530, 34),
    CircleObstacle("tree", 700, 560, 38),
    CircleObstacle("tree", 760, 555, 30),

    # ── 中央林带（分隔上下区域）──
    CircleObstacle("tree", 290, 205, 30),
    CircleObstacle("tree", 360, 195, 35),
    CircleObstacle("tree", 430, 210, 32),
    CircleObstacle("tree", 500, 200, 34),

    # ── 零散树木 ──
    CircleObstacle("tree", 340, 380, 28),
    CircleObstacle("tree", 620, 380, 32),
    CircleObstacle("tree", 55, 520, 30),

    # ── 石头 ──
    CircleObstacle("rock", 230, 100, 14),
    CircleObstacle("rock", 550, 320, 16),
    CircleObstacle("rock", 400, 450, 13),
    CircleObstacle("rock", 640, 240, 15),
    CircleObstacle("rock", 170, 350, 12),
    CircleObstacle("rock", 470, 510, 14),

    # ── 房屋 ──
    RectObstacle("house", 115, 435, 65, 55),   # 奶奶的小屋
    RectObstacle("house", 515, 125, 60, 48),   # 伐木工小屋
    RectObstacle("house", 580, 495, 65, 55),   # 村舍
]

# ── 游戏智能体封装 ──────────────────────────────────────

class GameAgent:
    """将 Agent 绑定到二维坐标和显示属性上，构成可渲染的游戏角色。"""

    def __init__(self, character: Character, color: str, x: float, y: float):
        self.agent = Agent(character)
        self.name = character.name
        self.color = color
        self.x = x
        self.y = y
        self.latest_message = ""   # 当前对话气泡内容
        self.mode = "autonomous"  # autonomous | manual
        self.target_x = None
        self.target_y = None
        self.path: list = []                    # 缓存的路径点
        self._last_target: tuple | None = None  # 上次计算路径时的目标

    def move_random(self) -> None:
        """随机移动，避开障碍物。尝试 10 个随机方向，取第一个不碰撞的。"""
        for _ in range(10):
            nx = max(15, min(MAP_W - 15,
                       self.x + random.randint(-MOVE_RANGE, MOVE_RANGE)))
            ny = max(15, min(MAP_H - 15,
                       self.y + random.randint(-MOVE_RANGE, MOVE_RANGE)))
            if not is_position_blocked(nx, ny, obstacles):
                self.x = nx
                self.y = ny
                return

    def move_toward_target(self) -> bool:
        """沿 A* 路径向目标移动一步，到达返回 True。"""
        if self.target_x is None or self.target_y is None:
            return True

        target = (self.target_x, self.target_y)

        # 目标改变或路径过期 → 重新计算
        if self._last_target != target or not self.path:
            self._last_target = target
            self.path = find_path(obstacles, (self.x, self.y), target) or []

        if not self.path:
            return True  # 无路可走，放弃

        # 沿路径前进
        remaining = MANUAL_STEP
        while remaining > 0 and self.path:
            wp_x, wp_y = self.path[0]
            dx = wp_x - self.x
            dy = wp_y - self.y
            dist = math.hypot(dx, dy)
            if dist <= remaining:
                self.x = wp_x
                self.y = wp_y
                remaining -= dist
                self.path.pop(0)
            else:
                self.x += (dx / dist) * remaining
                self.y += (dy / dist) * remaining
                remaining = 0

        # 检查是否到达最终目标
        if not self.path:
            if math.hypot(self.target_x - self.x, self.target_y - self.y) < 5:
                self.x = self.target_x
                self.y = self.target_y
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "color": self.color,
            "x": round(self.x, 1),
            "y": round(self.y, 1),
            "mode": self.mode,
            "message": self.latest_message,
        }


# ── 初始化角色 ──────────────────────────────────────────

little_hat = Character(
    name="小红帽",
    personality="开朗善良，但有些粗心大意",
    initial_motivation="提着一篮子浆果去看望奶奶",
)
big_wolf = Character(
    name="大灰狼",
    personality="狡猾又充满耐心，饥肠辘辘",
    initial_motivation="骗走小红帽的浆果作为开胃菜",
)

agents = [
    GameAgent(little_hat, "crimson", random.uniform(100, 300), random.uniform(150, 450)),
    GameAgent(big_wolf, "dimgray", random.uniform(500, 700), random.uniform(150, 450)),
]

# ── 室内场景 ────────────────────────────────────────────

GRANDMA_HOUSE = RectObstacle("house", 115, 435, 65, 55)
indoor_scene = False

grandma_character = Character(
    name="奶奶",
    personality="表面上是慈祥的老奶奶，实际是大灰狼乔装假扮的。声音沙哑低沉，偶尔露出破绽（比如露出爪子、说漏嘴），但会立刻掩饰",
    initial_motivation="假装生病卧床，骗小红帽放下篮子靠近床边，然后抓住机会吃掉她",
)
grandma_agent = Agent(grandma_character)


def _is_in_house(x: float, y: float) -> bool:
    """检测小红帽是否靠近奶奶小屋门口（南侧）。"""
    door_x = GRANDMA_HOUSE.x + GRANDMA_HOUSE.w / 2
    door_y = GRANDMA_HOUSE.y + GRANDMA_HOUSE.h + 8
    return math.hypot(x - door_x, y - door_y) < 25


# ── FastAPI 应用 ────────────────────────────────────────

app = FastAPI(title="童话镇")


# ── 请求模型 ────────────────────────────────────────────

class InteractRequest(BaseModel):
    name: str
    message: str


class MoveRequest(BaseModel):
    name: str
    x: float
    y: float


class ModeRequest(BaseModel):
    name: str
    mode: str


@app.get("/api/scene")
def get_scene():
    """返回当前场景类型。"""
    return {"scene": "indoor" if indoor_scene else "outdoor"}


@app.post("/api/indoor/interact")
def indoor_interact(req: InteractRequest):
    """室内场景中与奶奶（假狼）对话。"""
    if not indoor_scene:
        return {"error": "不在室内"}
    reaction = grandma_agent.perceive_and_act(req.message)
    return {
        "name": "奶奶",
        "action_type": reaction["action_type"],
        "content": reaction["content"],
    }


@app.post("/api/indoor/leave")
def leave_house():
    """离开小屋，回到室外。"""
    global indoor_scene
    indoor_scene = False
    a_hat = agents[0]
    a_hat.x = GRANDMA_HOUSE.x + GRANDMA_HOUSE.w / 2
    a_hat.y = GRANDMA_HOUSE.y + GRANDMA_HOUSE.h + 35
    a_hat.mode = "autonomous"
    a_hat.target_x = None
    a_hat.target_y = None
    a_hat.latest_message = ""
    return {"ok": True}


@app.get("/api/state")
def get_state():
    """返回地图尺寸及所有智能体的名称、颜色、坐标、对话气泡。"""
    return {
        "map": {"width": MAP_W, "height": MAP_H},
        "agents": [a.to_dict() for a in agents],
        "obstacles": [obstacle_to_dict(o) for o in obstacles],
        "scene": "indoor" if indoor_scene else "outdoor",
    }


@app.post("/api/step")
def step():
    """
    推进一个时间步：
    1. 每个角色随机移动
    2. 如果两人距离 < 阈值，触发交互（大灰狼先说话 → 小红帽回应）
    返回距离和是否发生了交互。
    """
    # 1. 移动
    a_hat, a_wolf = agents[0], agents[1]

    # 室内场景：检测小红帽是否离开门口
    global indoor_scene
    if indoor_scene:
        if _is_in_house(a_hat.x, a_hat.y):
            return {"distance": 0, "interaction": False, "scene": "indoor"}
        else:
            indoor_scene = False

    for a in agents:
        if a.mode == "manual":
            a.move_toward_target()
        elif a.name == "大灰狼":
            # 追逐小红帽，但保持 WOLF_APPROACH_DIST 距离
            dx = a_wolf.x - a_hat.x
            dy = a_wolf.y - a_hat.y
            chase_dist = math.hypot(dx, dy)
            if chase_dist < 1:
                a.target_x = a_hat.x + 60
                a.target_y = a_hat.y
            else:
                a.target_x = a_hat.x + (dx / chase_dist) * 60
                a.target_y = a_hat.y + (dy / chase_dist) * 60
            a.move_toward_target()
        elif a.name == "小红帽":
            # 逃离大灰狼
            dx = a_hat.x - a_wolf.x
            dy = a_hat.y - a_wolf.y
            flee_dist = math.hypot(dx, dy)
            if flee_dist < 250:
                if flee_dist > 1:
                    evade_x = a_hat.x + (dx / flee_dist) * 180
                    evade_y = a_hat.y + (dy / flee_dist) * 180
                else:
                    evade_x = a_hat.x + 180
                    evade_y = a_hat.y
                a.target_x = max(15, min(MAP_W - 15, evade_x))
                a.target_y = max(15, min(MAP_H - 15, evade_y))
                a.move_toward_target()
            else:
                a.move_random()
        else:
            a.move_random()

    # 计算当前距离
    dx = a_hat.x - a_wolf.x
    dy = a_hat.y - a_wolf.y
    dist = math.hypot(dx, dy)

    # 室内场景检测：小红帽进入奶奶小屋
    if _is_in_house(a_hat.x, a_hat.y):
        if not indoor_scene:
            indoor_scene = True
            a_hat.latest_message = ""
            a_wolf.latest_message = ""
            return {
                "distance": round(dist, 1),
                "interaction": False,
                "scene": "indoor",
                "greeting": "咳咳...谁啊？是小红帽吗？快进来吧，奶奶生病了躺在床上呢...把门关上，走近一点让奶奶看看你...",
            }
    else:
        if indoor_scene:
            indoor_scene = False

    # 防止重叠：强制推开
    if dist < 30:
        if dist > 0.1:
            push = (30 - dist) / 2 + 1
            nx = dx / dist * push
            ny = dy / dist * push
        else:
            nx, ny = 15, 0
        a_hat.x = max(15, min(MAP_W - 15, a_hat.x + nx))
        a_hat.y = max(15, min(MAP_H - 15, a_hat.y + ny))
        a_wolf.x = max(15, min(MAP_W - 15, a_wolf.x - nx))
        a_wolf.y = max(15, min(MAP_H - 15, a_wolf.y - ny))
        dist = math.hypot(a_hat.x - a_wolf.x, a_hat.y - a_wolf.y)

    # 2. 距离检测 & 交互（带冷却，防止 LLM 调用阻塞移动）
    global _last_interact_time
    now = time.time()
    if dist < INTERACT_DIST and (now - _last_interact_time) >= INTERACT_COOLDOWN:
        _last_interact_time = now
        # 大灰狼观察环境 → 行动
        wolf_obs = (
            f"你悄悄靠近了{a_hat.name}，她就在你面前不到几步远的地方。"
            f"现在是说话的好时机。"
        )
        wolf_reaction = a_wolf.agent.perceive_and_act(wolf_obs)
        a_wolf.latest_message = (
            wolf_reaction["content"]
            if wolf_reaction["action_type"] == "speak"
            else ""
        )

        # 小红帽收到大灰狼的话作为观察 → 行动
        if wolf_reaction["action_type"] == "speak":
            hat_obs = f"{a_wolf.name}靠近了你，对你说：「{wolf_reaction['content']}」"
        else:
            hat_obs = f"{a_wolf.name}在附近徘徊，似乎不怀好意。"

        hat_reaction = a_hat.agent.perceive_and_act(hat_obs)
        a_hat.latest_message = (
            hat_reaction["content"]
            if hat_reaction["action_type"] == "speak"
            else ""
        )
    else:
        # 距离不够近，清除气泡
        a_hat.latest_message = ""
        a_wolf.latest_message = ""

    return {"distance": round(dist, 1), "interaction": dist < INTERACT_DIST, "scene": "outdoor"}


# ── 用户交互 ────────────────────────────────────────────

@app.post("/api/interact")
def interact(req: InteractRequest):
    """用户点击角色并输入文字后，作为 observation 传入，返回角色反应。"""
    target = next((a for a in agents if a.name == req.name), None)
    if not target:
        return {"error": f"未找到角色: {req.name}"}

    reaction = target.agent.perceive_and_act(req.message)
    target.latest_message = (
        reaction["content"]
        if reaction["action_type"] == "speak"
        else ""
    )
    return {
        "name": target.name,
        "action_type": reaction["action_type"],
        "content": reaction["content"],
    }


# ── 移动控制 ────────────────────────────────────────────

@app.post("/api/move")
def move_to(req: MoveRequest):
    """设置角色目标点，切换为手动模式。"""
    target = next((a for a in agents if a.name == req.name), None)
    if not target:
        return {"error": f"未找到角色: {req.name}"}
    target.mode = "manual"
    target.target_x = max(15, min(MAP_W - 15, req.x))
    target.target_y = max(15, min(MAP_H - 15, req.y))
    return {"ok": True}


@app.post("/api/mode")
def set_mode(req: ModeRequest):
    """切换角色控制模式。"""
    target = next((a for a in agents if a.name == req.name), None)
    if not target:
        return {"error": f"未找到角色: {req.name}"}
    target.mode = req.mode
    target.target_x = None
    target.target_y = None
    return {"ok": True}


# ── 前端入口 ────────────────────────────────────────────

@app.get("/")
async def get_index():
    return FileResponse("web/index.html")
