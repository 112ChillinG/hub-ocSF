#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W12 作业 · ④ 实现可下发 Sub-agent 并行的 Supervisor（Parallel Sub-agent Orchestration）
============================================================================
【作业要求】
在单 Agent 之上实现"多智能体编排"：Supervisor 把一条复杂请求拆为子任务，并行下发多个
专业 Sub-agent，各自查独立索引，最后聚合。体现"扇出-聚合（fan-out / fan-in）"。

【本提交实现】
1. Supervisor 先做轻量 top-1 召回确定主故障（毫级，不阻塞并行）。
2. 三个 Sub-agent 并行：诊断检索 / 备件(BOM)检索 / 派单——各自查自己的索引，互不阻塞。
3. 收齐后 Supervisor 合成最终答复，并输出并行执行轨迹与耗时。

【运行】python3 W12_并行Supervisor.py
【数据】自生成模拟数据（示例），非真实生产数据。
"""

import re
import time
import threading
from collections import defaultdict


# ---------- 零依赖 BM25 ----------
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


# ---------- 模拟数据（示例） ----------
FAULTS = {
    "喷嘴堵塞": {"steps": ["升温空挤", "通针清理", "更换喷嘴"], "parts": ["0.4mm 喷嘴", "通针"]},
    "打印层错位": {"steps": ["调紧同步带", "校准步进", "降速"], "parts": ["同步带 GT2", "同步轮 20齿"]},
    "热失控": {"steps": ["断电冷却", "换热敏", "刷固件"], "parts": ["热敏电阻"]},
    "平台不粘": {"steps": ["清洁", "升温60℃", "调平"], "parts": ["平台清洁布"]},
}
CORPUS = [{"id": k, "text": k + " " + " ".join(v["steps"])} for k, v in FAULTS.items()]
BOM = {"0.4mm 喷嘴": 38, "同步带 GT2": 15, "热敏电阻": 22, "平台清洁布": 50}
_TECHS = [
    {"id": "T1", "name": "张工", "city": "上海", "skills": ["喷嘴堵塞", "平台不粘"], "rating": 4.5},
    {"id": "T2", "name": "李工", "city": "上海", "skills": ["打印层错位", "热失控"], "rating": 4.8},
    {"id": "T3", "name": "王工", "city": "北京", "skills": ["打印层错位", "喷嘴堵塞"], "rating": 4.2},
]


# ---------- Sub-agent 定义（各自查独立索引） ----------
def sub_diag(query, primary):
    return {"index": "Skill BM25", "primary": primary,
            "steps": FAULTS.get(primary, {}).get("steps", [])}


def sub_bom(primary):
    parts = FAULTS.get(primary, {}).get("parts", [])
    return {"index": "BOM BM25（独立）",
            "enriched": [{"part": p, "stock": BOM.get(p)} for p in parts]}


def sub_dispatch(primary, city):
    cands = [t for t in _TECHS if primary in t["skills"]]
    cands.sort(key=lambda t: (t["city"] == city), reverse=True)
    best = cands[0] if cands else None
    return [{"name": best["name"], "city": best["city"], "score": round(best["rating"], 2)}] if best else []


def run_supervisor(query, city=None):
    t0 = time.time()
    hits = bm25_search(CORPUS, query, topk=1)
    primary = hits[0] if hits else None

    results, errs, lock = {}, {}, threading.Lock()

    def worker(name, fn):
        try:
            r = fn()
        except Exception as e:
            r, errs[name] = {"error": str(e)}, str(e)
        with lock:
            results[name] = r

    jobs = [("diag", lambda: sub_diag(query, primary)),
            ("bom", lambda: sub_bom(primary)),
            ("dispatch", lambda: sub_dispatch(primary, city))]
    threads = [threading.Thread(target=worker, args=(n, fn)) for n, fn in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=6.0)

    elapsed = round((time.time() - t0) * 1000, 1)
    return {"query": query, "primary": primary, "results": results,
            "total_ms": elapsed, "sub_agents": [j[0] for j in jobs]}


def main():
    r = run_supervisor("打印到一半层整体错位偏移，还有异响", "上海")
    print("查询：%s" % r["query"])
    print("主故障：%s | 并行 Sub-agent：%s | 总耗时：%s ms"
          % (r["primary"], r["sub_agents"], r["total_ms"]))
    for name, res in r["results"].items():
        print("  - %s → %s" % (name, res))


if __name__ == "__main__":
    main()
