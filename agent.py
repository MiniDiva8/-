"""
童话镇多智能体系统 —— 智能体核心逻辑。
每个 Agent 由 Character 数据驱动，具备记忆、感知和行动能力。
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

import httpx
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

# 复用 script_parser 中的数据模型（不重复定义）
from script_parser import Character


# ── 工具函数 ──────────────────────────────────────────────

def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ── Agent 类 ──────────────────────────────────────────────

class Agent:
    """童话镇智能体：基于角色设定 + 历史记忆 + 当前感知 → 决定行动。"""

    def __init__(
        self,
        character: Character,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ):
        self.character = character
        self.model = model
        self.memory: List[Dict[str, str]] = []  # 每条: {"role": "...", "content": "...", "time": "..."}

        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量")
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(proxy=None, trust_env=False),  # 绕过系统代理
        )

    # ── 记忆管理 ──────────────────────────────────────────

    def _add_to_memory(self, role: str, content: str) -> None:
        self.memory.append({
            "role": role,
            "content": content,
            "time": _now(),
        })

    def _format_memory(self, limit: int = 20) -> str:
        """将记忆格式化为 LLM 可读的上下文，保留最近 N 条。"""
        recent = self.memory[-limit:] if len(self.memory) > limit else self.memory
        if not recent:
            return "（暂无历史记录）"
        lines = []
        for i, m in enumerate(recent, 1):
            label = {"observe": "观察到", "speak": "说", "think": "内心想", "act": "行动"}
            role_label = label.get(m["role"], m["role"])
            lines.append(f"  {i}. [{m['time']}] {role_label}: {m['content']}")
        return "\n".join(lines)

    # ── 核心行动循环 ──────────────────────────────────────

    def perceive_and_act(self, observation: str) -> dict:
        """
        接收外部观察，存入记忆，并让 LLM 结合角色设定和历史决定下一步行动。

        Args:
            observation: 外部输入（他人说的话、环境变化等）。

        Returns:
            {"action_type": "speak" | "think", "content": "具体内容"}
        """
        self._add_to_memory("observe", observation)

        system_prompt = (
            f"你是童话镇中的一个角色，请严格扮演你的设定。\n\n"
            f"【你的名字】{self.character.name}\n"
            f"【你的性格】{self.character.personality}\n"
            f"【你的当前动机】{self.character.initial_motivation}\n\n"
            f"【最近的经历（按时间顺序）】\n{self._format_memory()}\n\n"
            "请基于以上信息，决定你下一步的反应。\n"
            "注意：\n"
            "- 如果观察到有人对你说话，通常应该用 speak 回应。\n"
            "- 如果是环境描述或内心感触，可以用 think 表达内心活动。\n"
            "- 反应要符合你的性格和动机，保持角色一致性。\n"
            "- 不要说其他角色的话，只表达你自己的言行。\n"
            '输出格式（严格 JSON）：\n'
            '{"action_type": "speak", "content": "你要说的话"}\n'
            '或\n'
            '{"action_type": "think", "content": "你的内心想法"}'
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"刚刚发生的事情：{observation}\n请决定你的反应。"},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )

        raw = response.choices[0].message.content
        result = json.loads(raw)
        action_type = result.get("action_type", "think")
        content = result.get("content", "")

        self._add_to_memory(action_type, content)
        return {"action_type": action_type, "content": content}


# ── 测试入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    # 手动构造角色数据（模拟 script_parser 的解析结果）
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

    # 实例化智能体
    red = Agent(little_hat)
    wolf = Agent(big_wolf)

    # 模拟：大灰狼对小红帽说话
    wolf_line = "小姑娘，这森林里可不太平，你提着这么重的篮子要去哪呀？"

    print("=" * 56)
    print("  童话镇 · 智能体模拟")
    print("=" * 56)
    print(f"\n  [大灰狼]  {wolf_line}\n")

    reaction = red.perceive_and_act(wolf_line)

    action_label = "[说出]" if reaction["action_type"] == "speak" else "[心想]"
    print(f"  [小红帽] {action_label}: {reaction['content']}")

    # 打印小红帽的完整记忆
    print(f"\n  --- 小红帽的记忆 (共 {len(red.memory)} 条) ---")
    for m in red.memory:
        print(f"  [{m['time']}] {m['role']}: {m['content']}")

    # 保存结果到 UTF-8 文件（解决终端 GBK 编码问题）
    result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump({
            "observation": wolf_line,
            "reaction": reaction,
            "memory": red.memory,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[结果已写入] {result_path}")

    print("\n" + "=" * 56)
