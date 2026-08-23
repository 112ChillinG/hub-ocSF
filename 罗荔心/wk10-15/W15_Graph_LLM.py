#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W15 作业 · 知识图谱 × LLM（Graph × LLM）—— 以 FieldWise 备件-故障关系网为例
============================================================================
【作业要求】
理解如何用"实体-关系"图谱增强 LLM：把结构化关系（故障↔备件、备件↔替代件）建成图，
通过图遍历做确定性检索/推理，补 LLM 在关系与一致性上的短板（即 GraphRAG 雏形）。

【本提交实现】
1. 建一张"故障—备件"多对多图（一个故障用多个备件，一个备件被多个故障复用）。
2. 实现两类图遍历：故障→所需备件；备件→被哪些故障使用(used_by 反向边)。
3. 演示：BOM 独立索引正是图谱的实体关系雏形，可平滑升级为 GraphRAG 根因推理。

【运行】python3 W15_Graph_LLM.py
【数据】自生成模拟数据（示例），非真实生产数据。
"""

# ---------- 故障 → 备件（多对多） ----------
GRAPH = {
    "喷嘴堵塞": ["0.4mm 喷嘴", "通针", "扳手"],
    "打印层错位": ["同步带 GT2", "同步轮 20齿", "直线轴承"],
    "挤出不足": ["0.4mm 喷嘴", "喉管", "步进电机"],
    "Z轴波纹": ["直线轴承", "同步轮 20齿", "导轨"],
}
# 备件 → 替代件（另一类关系边）
ALT = {
    "0.4mm 喷嘴": ["0.6mm 喷嘴"],
    "同步带 GT2": ["GT2 钢芯带"],
    "直线轴承": ["LM8UU"],
}


class FaultPartGraph:
    def __init__(self, graph, alt):
        self.graph = graph
        self.alt = alt
        # 反向边：备件 → 被哪些故障使用（used_by）
        self.used_by = {}
        for fault, parts in graph.items():
            for p in parts:
                self.used_by.setdefault(p, []).append(fault)

    def parts_of(self, fault):
        return self.graph.get(fault, [])

    def faults_using(self, part):
        return self.used_by.get(part, [])

    def alternatives(self, part):
        return self.alt.get(part, [])

    def related_faults_via_parts(self, fault):
        """给定故障，经"备件"桥接找到"也用同一备件的其他故障"（图谱推理）。"""
        out = set()
        for p in self.parts_of(fault):
            for f in self.faults_using(p):
                if f != fault:
                    out.add(f)
        return sorted(out)


def main():
    g = FaultPartGraph(GRAPH, ALT)
    print("【故障→备件】 喷嘴堵塞 需要：", g.parts_of("喷嘴堵塞"))
    print("【备件→被使用】 直线轴承 被这些故障复用：", g.faults_using("直线轴承"))
    print("【替代件】 同步带 GT2 的替代：", g.alternatives("同步带 GT2"))
    print("【图谱推理】 与『打印层错位』共享备件的其他故障：",
          g.related_faults_via_parts("打印层错位"))
    print("\n→ 这类确定性的「关系检索」正是 GraphRAG 的基础：比纯文本检索更准、可解释、易维护。")


if __name__ == "__main__":
    main()
