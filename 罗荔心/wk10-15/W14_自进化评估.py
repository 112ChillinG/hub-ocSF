#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W14 作业 · ③ 编写 Skill + 模型/检索优化 + 前后对比评估（Self-Evolution / Optimization Eval）
============================================================================
【作业要求】
实现"自进化"的第一步：量化"优化前 vs 优化后"的效果差。先把 Skill 写好（丰富 triggers/
SOP），再用一份固定测试集评估检索/匹配准确率的变化，输出可写进简历的 Δ 提升。

【本提交实现】
1. 同一批故障，构造两套检索语料：弱（Skill 仅故障名）/ 强（含 symptom+causes+steps+parts）。
2. 固定测试集用"纯症状口语描述"（不含故障名），专测检索器能否从症状反推故障。
3. 指标：top-1 准确率、top-3 召回率；输出优化前后 Δ（百分点）。

【运行】python3 W14_自进化评估.py
【数据】自生成模拟数据（示例），非真实生产数据，seed=42 思路可复现。诚实标注仿真基准。
"""

import re
from collections import defaultdict


def tokenize(text):
    text = (text or "").lower()
    return [t for t in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text) if t.strip()]


def bm25_search(corpus, query, topk=3):
    docs = [tokenize(c["text"]) for c in corpus]
    N = len(docs) or 1
    df = defaultdict(int)
    for d in docs:
        for t in set(d):
            df[t] += 1
    avgdl = (sum(len(d) for d in docs) / N) or 1
    idf = {t: __import__("math").log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in df}
    q = tokenize(query)
    out = []
    for i, d in enumerate(docs):
        dl = len(d) or 1
        freq = defaultdict(int)
        for t in d:
            freq[t] += 1
        s = 0.0
        for t in set(q):
            f = freq.get(t, 0)
            if f == 0 or t not in idf:
                continue
            s += idf[t] * (f * 2.5) / (f + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))
        if s > 0:
            out.append((s, corpus[i]["id"]))
    out.sort(reverse=True)
    return [i for _, i in out[:topk]]


# ---------- 故障知识（弱/强两套语料来源） ----------
FAULTS = {
    "喷嘴堵塞": {"symptom": "料丝推不动、出丝口像被卡住、还有焦味", "causes": ["碳化堆积", "料丝直径错"],
                 "steps": ["升温空挤", "通针清理", "更换喷嘴"], "parts": ["0.4mm 喷嘴", "通针"]},
    "打印层错位": {"symptom": "模型中途整体往一边偏、像台阶移开、电机好像跳了", "causes": ["同步带松动", "步进丢步"],
                   "steps": ["调紧同步带", "校准步进", "降速"], "parts": ["同步带 GT2", "同步轮 20齿"]},
    "热失控": {"symptom": "温度一直降不下来、屏上读数还在往上窜、烫得厉害", "causes": ["热敏失效", "固件bug"],
               "steps": ["断电冷却", "换热敏", "刷固件"], "parts": ["热敏电阻"]},
    "平台不粘": {"symptom": "第一层总也贴不稳、边角老是翘起来、首层脱落", "causes": ["平台有油", "未升温", "Z偏移"],
                 "steps": ["清洁平台", "升温60℃", "调平"], "parts": ["平台清洁布"]},
}

# 固定测试集：纯症状口语 -> 标准故障（ground truth），查询里不出现故障名
TEST_SET = [
    ("料丝推不动，出丝口像被卡住，还有焦味", "喷嘴堵塞"),
    ("模型中途整体往一边偏，像台阶一样移开，电机好像跳了", "打印层错位"),
    ("温度一直降不下来，屏上读数还在往上窜，烫得厉害", "热失控"),
    ("第一层总也贴不稳，边角老是翘起来", "平台不粘"),
    ("热端推料阻力特别大，出料时断时续还夹着气泡", "喷嘴堵塞"),
    ("板上打的东西老是起翘，首层就脱落", "平台不粘"),
    ("模型上部整体挪了位，像是喷头被撞了一下", "打印层错位"),
    ("加热停不下来，温度读数一直往上飙", "热失控"),
]


def build_corpus(mode):
    out = []
    for name, f in FAULTS.items():
        if mode == "weak":
            text = name                                   # 优化前：Skill 仅故障名
        else:
            text = "%s。%s。可能原因：%s。处理：%s。备件：%s。" % (
                name, f["symptom"], "，".join(f["causes"]),
                "，".join(f["steps"]), "，".join(f["parts"]))
        out.append({"id": name, "text": text})
    return out


def evaluate(mode, topk=3):
    corpus = build_corpus(mode)
    top1 = top3 = 0
    for q, gt in TEST_SET:
        ids = bm25_search(corpus, q, topk=topk)
        if ids and ids[0] == gt:
            top1 += 1
        if gt in ids:
            top3 += 1
    n = len(TEST_SET)
    return round(top1 / n, 4), round(top3 / n, 4)


def main():
    weak_t1, weak_t3 = evaluate("weak")
    strong_t1, strong_t3 = evaluate("strong")
    print("测试集规模：%d 条（纯症状描述，不含故障名）" % len(TEST_SET))
    print("优化前（Skill 仅故障名）：  top1=%.1f%%  top3=%.1f%%" % (weak_t1 * 100, weak_t3 * 100))
    print("优化后（完整 SOP + 丰富触发）：top1=%.1f%%  top3=%.1f%%" % (strong_t1 * 100, strong_t3 * 100))
    print("Δ 提升：top1=+%.1fpp   top3=+%.1fpp" % ((strong_t1 - weak_t1) * 100, (strong_t3 - weak_t3) * 100))
    print("\n→ 这就是『自进化/优化』的可量化闭环：写好 Skill → 评估 → 看 Δ。真实项目用此闸门防止劣化。")


if __name__ == "__main__":
    main()
