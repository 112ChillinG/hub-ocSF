#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W10 作业 · 检索增强生成（RAG）—— 以 FieldWise 售后维修为例
============================================================================
【作业要求】
理解并实现 RAG 的"检索（Retrieval）"环节：把私域知识（维修 SOP）建成可检索语料，
根据用户问题召回最相关的知识片段，作为回答的"依据"（Grounding），抑制幻觉。

【本提交实现】
1. 用零依赖手写 BM25 把若干维修 SOP 建成语料（真实项目用同一套 BM25，无第三方库）。
2. 给定用户口语化报修，召回 top-k 相关 SOP，打印分数与命中片段。
3. 演示：为什么"检索"是 RAG 的第一性原理——没有检索，模型只能凭记忆/幻觉回答。

【运行】python3 W10_RAG.py
【数据】自生成模拟数据（示例），非真实生产数据，seed 可复现思路同主项目。
"""

import re
import math
from collections import defaultdict


# ---------- 分词（中文按字、英文/数字按词） ----------
def tokenize(text):
    text = (text or "").lower()
    return [t for t in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text) if t.strip()]


# ---------- 零依赖 BM25 检索（RAG 检索层） ----------
class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1, self.b = k1, b
        self.docs = [tokenize(it["text"]) for it in corpus]
        self.N = len(self.docs) or 1
        self.df = defaultdict(int)
        for d in self.docs:
            for t in set(d):
                self.df[t] += 1
        self.avgdl = (sum(len(d) for d in self.docs) / self.N) or 1
        self.idf = {t: math.log(1 + (self.N - df + 0.5) / (df + 0.5))
                    for t, df in self.df.items()}

    def search(self, query, topk=5):
        q = tokenize(query)
        if not q:
            return []
        out = []
        for i, d in enumerate(self.docs):
            dl = len(d) or 1
            freq = defaultdict(int)
            for t in d:
                freq[t] += 1
            score = 0.0
            for t in set(q):
                f = freq.get(t, 0)
                if f == 0 or t not in self.idf:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += self.idf[t] * (f * (self.k1 + 1)) / denom
            if score > 0:
                it = self.corpus[i]
                out.append({"score": round(score, 4), "id": it["id"], "text": it["text"]})
        out.sort(key=lambda x: -x["score"])
        return out[:topk]


# ---------- 模拟维修 SOP 语料（示例，真实项目由 skills/*.json 生成） ----------
CORPUS = [
    {"id": "喷嘴堵塞", "text": "喷嘴堵塞：出料口推不动、有焦味。原因：碳化堆积、料丝直径选错。"
                               "步骤：升温空挤、通针清理、更换喷嘴。"},
    {"id": "打印层错位", "text": "打印层错位：整层偏移、像丢步、有异响。原因：同步带松动、步进电机丢步。"
                                 "步骤：调紧同步带、校准步进、降低打印速度。"},
    {"id": "热失控", "text": "热失控：温度降不下来、读数飙升。原因：热敏电阻失效、固件 bug。"
                             "步骤：立即断电冷却、更换热敏电阻、重刷固件。"},
    {"id": "平台不粘", "text": "平台不粘：首层翘边、脱落。原因：平台有油、未升温、Z 偏移。"
                               "步骤：清洁平台、升温到 60℃、重新调平。"},
]


def main():
    bm = BM25(CORPUS)
    q = "打印到一半层整体错位偏移，还有异响"
    print("用户问题：", q)
    print("召回的 SOP（按 BM25 分数降序）：")
    for h in bm.search(q, topk=3):
        print("  [%.4f] %s — %s" % (h["score"], h["id"], h["text"][:34] + "…"))
    print("\n→ 召回的 SOP 即作为后续『生成 / 诊断』的'依据'(Grounding)，"
          "模型只需基于这些片段作答，而非凭空编造。")


if __name__ == "__main__":
    main()
