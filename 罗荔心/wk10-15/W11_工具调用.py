#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W11 作业 · 工具调用（Tool / Function Calling）—— 以 FieldWise 售后维修为例
============================================================================
【作业要求】
理解 Agent 如何"调用外部工具/函数"获取确定性的事实（而非靠模型生成）：
定义工具的描述（名称、用途、参数），由调度器根据意图选择并调用正确的工具。

【本提交实现】
1. 定义 3 个工具（schema 描述 + 真实函数）：查 SOP、查备件库存、派单。
2. 实现简易"工具路由/调度器"：根据用户意图匹配工具并调用，返回结构化结果。
3. 演示：Agent 的"手脚"来自工具调用——确定性事实应由工具返回，而非模型猜测。

【运行】python3 W11_工具调用.py
【数据】自生成模拟数据（示例），非真实生产数据。
"""

# ---------- 工具 1：检索 SOP（复用 W10 的 BM25 思路） ----------
_SKILLS = {
    "喷嘴堵塞": "升温空挤 → 通针清理 → 更换喷嘴",
    "打印层错位": "调紧同步带 → 校准步进 → 降低速度",
    "热失控": "断电冷却 → 更换热敏 → 重刷固件",
    "平台不粘": "清洁平台 → 升温60℃ → 重新调平",
}


def tool_lookup_skill(query: str) -> dict:
    """根据故障描述返回对应 SOP 步骤。"""
    for name, sop in _SKILLS.items():
        if name in query:
            return {"tool": "lookup_skill", "hit": name, "sop": sop}
    return {"tool": "lookup_skill", "hit": None, "sop": "未匹配到已知 SOP"}


# ---------- 工具 2：查询备件库存（BOM 检索原型） ----------
_BOM = {
    "0.4mm 喷嘴": {"stock": 38, "price": 12, "alt": "0.6mm 喷嘴"},
    "同步带 GT2": {"stock": 15, "price": 8, "alt": "GT2 钢芯带"},
    "热敏电阻": {"stock": 22, "price": 5, "alt": "100K 3950B"},
}


def tool_lookup_bom(part: str) -> dict:
    """根据备件名返回库存 / 价格 / 替代件。"""
    for name, info in _BOM.items():
        if name in part:
            return {"tool": "lookup_bom", "hit": name, **info}
    return {"tool": "lookup_bom", "hit": None, "msg": "BOM 无此件"}


# ---------- 工具 3：派单（技能匹配 + 同城） ----------
_TECHS = [
    {"id": "T1", "name": "张工", "city": "上海", "skills": ["喷嘴堵塞", "平台不粘"]},
    {"id": "T2", "name": "李工", "city": "上海", "skills": ["打印层错位", "热失控"]},
    {"id": "T3", "name": "王工", "city": "北京", "skills": ["打印层错位", "喷嘴堵塞"]},
]


def tool_dispatch(fault: str, city: str = "") -> dict:
    """按技能匹配 + 同城推荐维修工。"""
    cands = [t for t in _TECHS if fault in t["skills"]]
    cands.sort(key=lambda t: (t["city"] == city), reverse=True)
    best = cands[0] if cands else None
    return {"tool": "dispatch", "fault": fault,
            "recommend": best["name"] if best else "无可选", "city": best["city"] if best else ""}


# ---------- 工具注册表 + 路由调度器 ----------
TOOLS = {
    "lookup_skill": {"desc": "查故障对应的处理 SOP", "fn": tool_lookup_skill},
    "lookup_bom": {"desc": "查备件库存/价格/替代件", "fn": tool_lookup_bom},
    "dispatch": {"desc": "按技能+同城派单", "fn": tool_dispatch},
}


def route_and_call(user_msg: str, city: str = "") -> dict:
    """极简意图路由：命中关键词即调用对应工具（真实 LLM 场景下由模型输出 tool_call）。"""
    if any(k in user_msg for k in ("库存", "备件", "有没有", "价格")):
        part = next((p for p in _BOM if p in user_msg), "0.4mm 喷嘴")
        return TOOLS["lookup_bom"]["fn"](part)
    if any(k in user_msg for k in ("派", "谁", "安排", "维修工")):
        fault = next((f for f in _SKILLS if f in user_msg), "喷嘴堵塞")
        return TOOLS["dispatch"]["fn"](fault, city)
    return TOOLS["lookup_skill"]["fn"](user_msg)


def main():
    cases = [
        ("打印层错位了怎么办", ""),
        ("0.4mm 喷嘴还有库存吗", ""),
        ("喷嘴堵了，安排个上海的维修工", "上海"),
    ]
    for msg, city in cases:
        r = route_and_call(msg, city)
        print("用户：%-28s → 调用工具[%s] → %s" % (msg, r.get("tool"), r))


if __name__ == "__main__":
    main()
