"""
童话镇多智能体系统 —— 剧本解析模块。
使用 OpenAI 兼容接口（默认 DeepSeek）将非结构化文本解析为
Character 和 Scene 的 Pydantic 结构化对象。
"""

import json
import os
from typing import List, Tuple

from openai import OpenAI
from pydantic import BaseModel, Field


# ── 数据模型 ──────────────────────────────────────────────

class Character(BaseModel):
    """剧本角色"""
    name: str = Field(..., description="角色姓名")
    personality: str = Field(..., description="性格描述")
    initial_motivation: str = Field(..., description="初始动机或目标")


class Scene(BaseModel):
    """剧本场景"""
    location_name: str = Field(..., description="地点名称")
    description: str = Field(..., description="环境描述")
    current_time: str = Field(..., description="当前时间")


# ── 核心解析函数 ──────────────────────────────────────────

# JSON 模板，注入到 system prompt 中约束 LLM 输出格式
_OUTPUT_SCHEMA = """
{
  "characters": [
    {"name": "角色名", "personality": "性格", "initial_motivation": "动机"}
  ],
  "scenes": [
    {"location_name": "地点", "description": "环境", "current_time": "时间"}
  ]
}
"""


def parse_script(
    text: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
) -> Tuple[List[Scene], List[Character]]:
    """
    将非结构化剧本文本解析为 Scene 和 Character 列表。

    Args:
        text:       原始剧本/故事文本（支持中文）。
        model:      LLM 模型名称，默认 deepseek-chat。
        base_url:   API 端点地址，可替换为任意 OpenAI 兼容服务。

    Returns:
        (scenes, characters) 元组。
    """
    # 优先读 DEEPSEEK_API_KEY，其次 OPENAI_API_KEY
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY 环境变量")

    client = OpenAI(api_key=api_key, base_url=base_url)

    system_prompt = (
        "你是一个童话世界剧本解析器。从用户提供的文本中提取所有角色和场景。\n"
        "角色需要包含 name、personality、initial_motivation 三个字段。\n"
        "场景需要包含 location_name、description、current_time 三个字段。\n"
        f"请严格按照以下 JSON 格式输出，只输出 JSON，不要包含其他文字：\n{_OUTPUT_SCHEMA}"
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)

    characters = [Character(**item) for item in data.get("characters", [])]
    scenes = [Scene(**item) for item in data.get("scenes", [])]
    return scenes, characters


# ── 测试入口 ──────────────────────────────────────────────

if __name__ == "__main__":
    test_text = """
    在一个阳光明媚的清晨，童话镇的中心广场上。小红帽正提着一篮子新鲜采摘的浆果，
    准备去看望奶奶。她性格开朗善良，但有些粗心大意，甚至忘了带防身的小刀。
    与此同时，大灰狼躲在广场边缘的灌木丛里，他饥肠辘辘，狡猾又充满耐心的他正盘算着
    怎么骗走小红帽的浆果作为开胃菜。
    """

    try:
        scenes, characters = parse_script(test_text)

        print("=" * 60)
        print("  童话镇 · 剧本解析结果")
        print("=" * 60)

        print(f"\n角色列表（共 {len(characters)} 个）：")
        print("-" * 40)
        for i, c in enumerate(characters, 1):
            print(f"  [{i}] {c.name}")
            print(f"      性格：{c.personality}")
            print(f"      初始动机：{c.initial_motivation}")

        print(f"\n场景列表（共 {len(scenes)} 个）：")
        print("-" * 40)
        for i, s in enumerate(scenes, 1):
            print(f"  [{i}] {s.location_name}")
            print(f"      时间：{s.current_time}")
            print(f"      描述：{s.description}")

        output = {
            "characters": [c.model_dump() for c in characters],
            "scenes": [s.model_dump() for s in scenes],
        }

        # 屏幕打印
        print("\n" + "=" * 60)
        print("  完整 JSON")
        print("=" * 60)
        print(json.dumps(output, ensure_ascii=False, indent=2))

        # 同时写入 UTF-8 文件以解决终端编码问题
        result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parse_result.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[结果已写入] {result_path}")

    except Exception as e:
        print(f"[错误] {e}")
