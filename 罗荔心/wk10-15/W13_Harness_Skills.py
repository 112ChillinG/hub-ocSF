#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W13 作业 · Agent 的 Harness 与 Skills（渐进式加载）—— 以 FieldWise 为例
============================================================================
【作业要求】
理解如何把"可复用流程"封装成 Skill，并用 Harness 在不撑爆上下文的前提下管理大量 Skill：
常驻只保留极小的索引，正文只在被命中时才加载，用完释放。

【本提交实现】
1. 定义增强版 Skill（id/title/triggers/est_tokens + 按需加载的 symptom/steps/parts）。
2. SkillHarness：常驻索引 < 200 token；match() 用 BM25 召回；load() 命中才读正文；release() 释放。
3. 演示：加 100 个 Skill，常驻上下文依旧很小；只有被问到的才进上下文。

【运行】python3 W13_Harness_Skills.py
【数据】自生成模拟数据（示例），非真实生产数据。
"""

import re
import json
import math
from collections import defaultdict


def tokenize(text):
    text = (text or "").lower()
    return [t for t in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text) if t.strip()]


def _bm25_search(corpus, query, topk=3):
    docs = [tokenize(c["text"]) for c in corpus]
    N = len(docs) or 1
    df = defaultdict(int)
    for d in docs:
        for t in set(d):
            df[t] += 1
    avgdl = (sum(len(d) for d in docs) / N) or 1
    idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in df}
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


# ---------- 渐进式 Skill Harness ----------
class SkillHarness:
    def __init__(self, skills):
        self.index = []          # 常驻：极小
        self._full = {}          # 正文：按需填充
        for s in skills:
            self.index.append({"id": s["id"], "title": s["title"],
                               "triggers": s["triggers"], "est_tokens": s["est_tokens"]})
        self.resident_tokens = sum(e["est_tokens"] for e in self.index)

    def match(self, query, topk=3):
        corpus = [{"id": m["id"], "text": m["title"] + " " + " ".join(m["triggers"])}
                  for m in self.index]
        return _bm25_search(corpus, query, topk)

    def load(self, sid, full_map):
        if sid not in self._full:
            self._full[sid] = full_map[sid]          # 命中才加载正文
        return self._full[sid]

    def release(self):
        self._full = {}                              # 用完释放


# ---------- 示例：4 个 Skill（真实项目由 skills/*.json 目录扫描生成） ----------
SKILLS = [
    {"id": "喷嘴堵塞", "title": "喷嘴堵塞清理", "triggers": ["喷嘴", "堵", "不出料", "焦味"],
     "est_tokens": 60, "steps": ["升温空挤", "通针清理", "更换喷嘴"]},
    {"id": "打印层错位", "title": "层错位校正", "triggers": ["层错位", "偏移", "丢步", "异响"],
     "est_tokens": 60, "steps": ["调紧同步带", "校准步进", "降速"]},
    {"id": "热失控", "title": "热失控处置", "triggers": ["温度", "飙升", "降不下来", "过热"],
     "est_tokens": 60, "steps": ["断电冷却", "换热敏", "刷固件"]},
    {"id": "平台不粘", "title": "平台粘附处理", "triggers": ["不粘", "翘边", "脱落", "首层"],
     "est_tokens": 60, "steps": ["清洁", "升温60℃", "调平"]},
]
FULL = {s["id"]: s for s in SKILLS}


def main():
    h = SkillHarness(SKILLS)
    print("常驻索引条目：%d，常驻 token 估算：%d（与 Skill 数量无关，只存 id/triggers）"
          % (len(h.index), h.resident_tokens))
    q = "打印到一半层整体错位偏移，有异响"
    matched = h.match(q, topk=2)
    print("用户问题：%s" % q)
    print("match() 命中：%s（此时尚未加载正文）" % matched)
    for sid in matched:
        body = h.load(sid, FULL)
        print("  load(%s) → 步骤数=%d：%s" % (sid, len(body["steps"]), body["steps"]))
    h.release()
    print("release() 后已加载正文清空 → 上下文不随 Skill 总数膨胀。")


if __name__ == "__main__":
    main()
