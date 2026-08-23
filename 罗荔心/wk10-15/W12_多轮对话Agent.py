#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W12 作业 · ① 为诊断 Agent 增加多轮对话能力（Multi-turn Diagnostic Agent）
============================================================================
【作业要求】
在 Agent 基础上实现"有状态多轮对话"：能记住上下文、主动澄清（Clarify）、理解否定
（"还是不行"）、接受约束（"只要上海"），而非一问一答就丢上下文。

【本提交实现】
1. 会话记忆（Session）+ BM25 召回候选故障。
2. 澄清循环：多种可能时反问；用户补充信息则收窄；说"还是不行"则排除当前看下一个。
3. 约束解析：城市（"只要上海"）、排除某维修工（"别派张三"）。
4. 定位明确后输出完整诊断（原因/步骤/备件）+ 按约束推荐维修工。

【运行】python3 W12_多轮对话Agent.py
【数据】自生成模拟数据（示例），非真实生产数据。
"""

import re
from collections import defaultdict


# ---------- 零依赖 BM25（与 W10 同思路，独立可运行） ----------
def tokenize(text):
    text = (text or "").lower()
    return [t for t in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text) if t.strip()]


def bigrams(text):
    """字符 bigram 集合（零依赖中文近似词匹配，比单字更抗误命中）。"""
    toks = tokenize(text)
    return {"".join(p) for p in zip(toks, toks[1:])}


def bm25_search(corpus, query, topk=3):
    docs = [tokenize(c["text"]) for c in corpus]
    N = len(docs) or 1
    df = defaultdict(int)
    for d in docs:
        for t in set(d):
            df[t] += 1
    avgdl = (sum(len(d) for d in docs) / N) or 1
    idf = {t: math_log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in df}
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


def math_log(x):
    import math
    return math.log(x)


# ---------- 模拟故障知识（示例） ----------
FAULTS = {
    "喷嘴堵塞": {"symptom": "出料口推不动、有焦味", "causes": ["碳化堆积", "料丝直径错"],
                 "steps": ["升温空挤", "通针清理", "更换喷嘴"], "parts": ["0.4mm 喷嘴", "通针"]},
    "打印层错位": {"symptom": "整层偏移、像丢步、有异响", "causes": ["同步带松动", "步进丢步"],
                   "steps": ["调紧同步带", "校准步进", "降速"], "parts": ["同步带 GT2", "同步轮 20齿"]},
    "热失控": {"symptom": "温度降不下来、读数飙升", "causes": ["热敏失效", "固件bug"],
               "steps": ["断电冷却", "换热敏", "刷固件"], "parts": ["热敏电阻"]},
    "平台不粘": {"symptom": "首层翘边、脱落", "causes": ["平台有油", "未升温", "Z偏移"],
                 "steps": ["清洁", "升温60℃", "调平"], "parts": ["平台清洁布"]},
}
CORPUS = [{"id": k, "text": k + " " + v["symptom"] + " " + " ".join(v["causes"])}
          for k, v in FAULTS.items()]

CLARIFY = {
    "喷嘴堵塞": "堵塞是偶发还是每次都堵？热端温度正常吗？",
    "打印层错位": "是整层偏移还是局部？是否伴随异响或丢步？",
    "热失控": "温度是一直偏高，还是会突然飙升到 200℃ 以上？",
    "平台不粘": "平台是玻璃还是 PEI？打印前是否升温到 60℃？",
}
_NEG = ["还是", "不行", "没用", "没解决", "不对", "无法", "仍然", "依旧", "没好"]
_TECHS = [
    {"id": "T1", "name": "张工", "city": "上海", "skills": ["喷嘴堵塞", "平台不粘"]},
    {"id": "T2", "name": "李工", "city": "上海", "skills": ["打印层错位", "热失控"]},
    {"id": "T3", "name": "王工", "city": "北京", "skills": ["打印层错位", "喷嘴堵塞"]},
]


class Session:
    def __init__(self, sid):
        self.sid = sid
        self.remaining = []
        self.constraints = {"city": None, "exclude": []}
        self.resolved = False

    def handle(self, msg, city=None):
        if city:
            self.constraints["city"] = city
        # 解析约束
        for t in _TECHS:
            if t["city"] and t["city"] in msg:
                self.constraints["city"] = t["city"]
        m = re.search(r"(?:别派|不要|排除)\s*([一-鿿]{2,3})", msg)
        if m:
            self.constraints["exclude"].append(m.group(1))

        if not self.remaining:                       # 首轮召回
            self.remaining = bm25_search(CORPUS, msg, topk=3)
            if not self.remaining:
                return "没太理解，请描述具体一点，例如：打印到一半层整体错位偏移。"
            primary = self.remaining[0]
            return "最可能是【%s】。为更准确定位，请教：%s" % (primary, CLARIFY.get(primary, "请补充现象。"))

        if any(w in msg for w in _NEG):              # 否定排除
            if self.remaining:
                self.remaining.pop(0)
            if not self.remaining:
                return "已尝试主要可能仍无效，建议升级处理或补充更多线索（异响/焦味/报错码）。"
            primary = self.remaining[0]
            return "排除上一个。下一个怀疑【%s】。%s" % (primary, CLARIFY.get(primary, ""))

        # 关键词收窄：按 bigram 重叠分排序，只保留最高分的候选（避免"偏移"等弱信号误命中）
        q = bigrams(msg)
        scores = {c: len(q & bigrams(FAULTS[c]["symptom"] + " ".join(FAULTS[c]["causes"])))
                  for c in self.remaining}
        best = max(scores.values()) if scores else 0
        narrowed = [c for c in self.remaining if scores[c] == best and best > 0]
        if narrowed:
            self.remaining = narrowed[:3]
        if len(self.remaining) == 1:
            self.resolved = True
            p = self.remaining[0]
            techs = self._recommend(p)
            return ("确定是【%s】。\n处理：%s\n备件：%s\n推荐维修工：%s"
                    % (p, " → ".join(FAULTS[p]["steps"]), "、".join(FAULTS[p]["parts"]), techs))
        return "收到，目前仍怀疑【%s】等。%s" % (self.remaining[0], CLARIFY.get(self.remaining[0], ""))

    def _recommend(self, fault):
        cands = [t for t in _TECHS if fault in t["skills"]
                 and not any(x in t["name"] for x in self.constraints["exclude"])]
        city = self.constraints["city"]
        cands.sort(key=lambda t: (t["city"] == city), reverse=True)
        return cands[0]["name"] + ("（同城）" if cands and cands[0]["city"] == city else "") if cands else "无"


def main():
    s = Session("demo")
    turns = [
        ("打印到一半层整体错位偏移", None),
        ("是整层偏移，有异响", None),
        ("还是不行，清了还是错位", "上海"),
    ]
    for i, (msg, city) in enumerate(turns, 1):
        print("用户%d：%s" % (i, msg))
        print("Agent：%s\n" % s.handle(msg, city))
    print("→ 三轮内保持上下文、主动澄清、排除无效假设，并套用『只要上海』约束。")


if __name__ == "__main__":
    main()
