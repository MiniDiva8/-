"""
童话镇 Web 服务 —— FastAPI 后端。
管理智能体在 800x600 二维空间中的位置、触发近距离互动，
并通过 REST API 将状态暴露给前端渲染。
"""

import math
import os
import random

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from agent import Agent
from script_parser import Character

# ── 常量 ────────────────────────────────────────────────

MAP_W, MAP_H = 800, 600
INTERACT_DIST = 100   # 触发交互的距离阈值（像素）
MOVE_RANGE = 25       # 每步最大随机移动像素
MANUAL_STEP = 120     # 手动模式每步移动像素

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

    def move_random(self) -> None:
        """在限定范围内随机移动一小步。"""
        self.x = max(15, min(MAP_W - 15, self.x + random.randint(-MOVE_RANGE, MOVE_RANGE)))
        self.y = max(15, min(MAP_H - 15, self.y + random.randint(-MOVE_RANGE, MOVE_RANGE)))

    def move_toward_target(self) -> bool:
        """向目标点移动一步，到达返回 True。"""
        if self.target_x is None or self.target_y is None:
            return True
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.hypot(dx, dy)
        if dist < 3:
            self.x = self.target_x
            self.y = self.target_y
            return True
        step = min(MANUAL_STEP, dist)
        self.x += (dx / dist) * step
        self.y += (dy / dist) * step
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

# ── FastAPI 应用 ────────────────────────────────────────

app = FastAPI(title="童话镇")


@app.get("/api/state")
def get_state():
    """返回地图尺寸及所有智能体的名称、颜色、坐标、对话气泡。"""
    return {
        "map": {"width": MAP_W, "height": MAP_H},
        "agents": [a.to_dict() for a in agents],
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
    for a in agents:
        if a.mode == "manual":
            a.move_toward_target()
        else:
            a.move_random()

    a_hat, a_wolf = agents[0], agents[1]
    dist = math.hypot(a_hat.x - a_wolf.x, a_hat.y - a_wolf.y)

    # 2. 距离检测 & 交互
    if dist < INTERACT_DIST:
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

    return {"distance": round(dist, 1), "interaction": dist < INTERACT_DIST}


# ── 用户交互 ────────────────────────────────────────────

class InteractRequest(BaseModel):
    name: str
    message: str


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

class MoveRequest(BaseModel):
    name: str
    x: float
    y: float


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


class ModeRequest(BaseModel):
    name: str
    mode: str


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
