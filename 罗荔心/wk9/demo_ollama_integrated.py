# -*- coding: utf-8 -*-
"""
集成版 Demo：Ollama 约束解码（choice 版 + json 版 合并在同一份文件里）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
当前状态：
  ✅ choice 版（意图路由）—— 已激活，直接运行即可
  #️⃣  json  版（结构化抽取）—— 已用 # 注释掉，需要时取消注释即可

学习路线：choice（单值约束）→ json（多字段约束）→ function call（调工具）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

运行环境：VSCode + conda py311 + Ollama 服务已启动
运行方式：
  conda activate py311
  pip install ollama jsonschema
  python demo_ollama_integrated.py

前置条件：
  1. Ollama app 已打开（后台服务在 localhost:11434）
  2. 已拉取模型：ollama pull qwen2:0.5b
"""

# ═══════════════════════════════════════════════════════════════════════
#  公共部分（choice 版和 json 版共用，无需注释）
# ═══════════════════════════════════════════════════════════════════════

import json      # 解析模型返回的 JSON 字符串
import time      # 给每次调用计时
import ollama    # 连本机 Ollama 服务的官方 Python 客户端

# 连上本机 Ollama（默认 11434 端口）。改过端口就把这里改掉。
client = ollama.Client(host="http://localhost:11434")
MODEL = "qwen2:0.5b"   # 0.5B 小模型，下载快；约束解码能让小模型也稳定输出


# ═══════════════════════════════════════════════════════════════════════
#  ▼▼▼  PART 1 / 2  —  choice 版（意图路由）  ✅ 当前激活  ▼▼▼
# ═══════════════════════════════════════════════════════════════════════
#
# 约束解码的入门形态：让模型从「固定的几个选项里挑一个」——只约束单个词。
# 场景：金融意图路由（分类）—— "这句话属于哪一类？"
# 输出样例：{"intent": "查股价"}
#
# 对比 2 种模式：
#   1. raw          —— 裸 prompt（不约束，靠文字指令，模型偶尔乱来）
#   2. guided_choice —— 加 format=INTENT_SCHEMA（强制输出合法）
# ───────────────────────────────────────────────────────────────────────

# ── 约束规则本体 ────────────────────────────────────────────────────────
# 5 个意图选项，模型只能从中选一个
INTENT_CHOICES = ["查股价", "查财报", "查新闻", "对比分析", "其他"]

# JSON Schema：只有一个 intent 字段，取值限定在 INTENT_CHOICES 里
INTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {                       # 只有一个字段
            "type": "string",
            "enum": INTENT_CHOICES,       # ★ 只能取这 5 个之一
        }
    },
    "required": ["intent"],
}

# 系统提示词：告诉模型"判断意图，从选项里选一个，输出纯 JSON"
# f-string 里 {{...}} 双花括号 = 字面的大括号（否则 Python 会当变量解析）
SYSTEM_PROMPT = f"""你是金融助手。判断用户问题的意图，从以下选项中选择最匹配的一个：
{', '.join(INTENT_CHOICES)}

输出纯 JSON，格式：{{"intent": "选择的意图"}}"""

# 测试用例：12 条 (用户问题, 标准答案) 元组
TEST_CASES = [
    ("招行现在股价多少",           "查股价"),
    ("贵州茅台2023年净利润",       "查财报"),
    ("今天有什么财经新闻",         "查新闻"),
    ("比亚迪和特斯拉哪个好",       "对比分析"),
    ("今天天气怎么样",             "其他"),
    ("帮我查一下中石油的股价",     "查股价"),
    ("宁德时代去年的营收是多少",   "查财报"),
    ("最近有什么财经热点",         "查新闻"),
    ("比较一下茅台和五粮液的估值", "对比分析"),
    ("你好，在吗",                 "其他"),
    ("隆基绿能的股票多少钱",       "查股价"),
    ("平安银行的ROE是多少",        "查财报"),
]


def evaluate_choice(output: str, expected: str) -> dict:
    """评估 choice 输出：简单校验（输出在不在枚举里 + 预测对不对）
    返回小字典记录各项是否通过。"""
    result = {
        "is_json": False,         # 能不能被 json.loads 解析
        "intent_valid": False,    # intent 是否在 5 个枚举里
        "correct": False,         # intent 是否 == 标准答案
        "parsed": None,           # 解析后的字典（调试用）
    }
    try:
        obj = json.loads(output)  # 尝试解析成字典
        result["is_json"] = True
        result["parsed"] = obj
    except json.JSONDecodeError:
        return result             # 解析失败直接返回

    intent = obj.get("intent")
    if intent in INTENT_CHOICES:         # 在枚举里 → 合法
        result["intent_valid"] = True
    if intent == expected:               # 和标准答案一致 → 正确
        result["correct"] = True
    return result


def run_choice(user_msg: str, mode: str) -> tuple:
    """根据 mode 调用模型。
    mode: 'raw'（裸 prompt）/ 'guided_choice'（format=schema 约束）"""
    t0 = time.time()

    fmt = INTENT_SCHEMA if mode == "guided_choice" else None  # raw 不加 format

    call = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "options": {"temperature": 0},  # 关掉随机性，输出稳定可复现
    }
    if fmt is not None:
        call["format"] = fmt           # guided_choice 模式才加 format=schema

    resp = client.chat(**call)
    return resp["message"]["content"].strip(), time.time() - t0


def main_choice():
    """choice 版主函数：遍历 12 条测试 × 2 种模式，打印明细 + 汇总表"""
    print("=" * 78)
    print("  Demo: Ollama 约束解码 —— choice 版（意图路由）")
    print(f"  Model: {MODEL}")
    print("  对比两种模式：裸 prompt / guided_choice (format=schema)")
    print("=" * 78)

    # 为两种模式各建计数器
    counters = {m: {"valid": 0, "correct": 0, "latency": 0.0}
                for m in ["raw", "guided_choice"]}
    n = len(TEST_CASES)

    for user, expected in TEST_CASES:
        print(f"\n▶ {user}  （期望：{expected}）")
        for mode in ["raw", "guided_choice"]:
            out, latency = run_choice(user, mode)
            ev = evaluate_choice(out, expected)
            c = counters[mode]
            if ev["intent_valid"]:  c["valid"] += 1
            if ev["correct"]:       c["correct"] += 1
            c["latency"] += latency
            # ✓ 正确 / ~ 合法但答错 / ✗ 不合法
            tag = "✓" if ev["correct"] else ("~" if ev["intent_valid"] else "✗")
            disp = out[:60] + "…" if len(out) > 60 else out
            print(f"  [{mode:<16}] {tag}  {disp}  ({latency:.2f}s)")

    # ── 中文宽度对齐小工具（中文算 2 宽，避免表格错位）──
    def _w(s, width):
        import unicodedata
        disp = sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(s))
        return str(s) + " " * max(0, width - disp)

    # 汇总表
    print("\n" + "=" * 78)
    print(f"  {n} 条测试结果汇总")
    print("=" * 78)
    print(_w("指标", 20) + _w("裸 prompt", 24) + _w("guided_choice", 24))
    print("-" * 78)
    for metric_name, key in [("输出合法(在枚举内)", "valid"),
                             ("预测正确", "correct")]:
        row = _w(metric_name, 20)
        for mode in ["raw", "guided_choice"]:
            v = counters[mode][key]
            row += _w(f"{v}/{n} ({100*v/n:.0f}%)", 24)
        print(row)
    # 平均延迟
    row = _w("平均延迟", 20)
    for mode in ["raw", "guided_choice"]:
        avg = counters[mode]["latency"] / n
        row += _w(f"{avg:.2f}s", 24)
    print(row)

    # ★ 最终评分（一眼看清模型效果，避免被上方明细刷走）
    print()
    print("=" * 78)
    print(f"  最终评分（准确率 = 预测正确 / 共 {n} 条）")
    print("=" * 78)
    for mode in ["raw", "guided_choice"]:
        c = counters[mode]
        print(f"    {_w(mode, 16)} 合法 {c['valid']}/{n} ({100*c['valid']/n:.0f}%)"
              f"  |  准确 {c['correct']}/{n} ({100*c['correct']/n:.0f}%)"
              f"  |  平均 {c['latency']/n:.2f}s")
    print()
    print("=" * 78)
    print("  结论：")
    print("    guided_choice (format=schema) 保证输出 100% 在枚举内")
    print("    但分类准不准仍取决于模型本身的智力")
    print("=" * 78)


# ═══════════════════════════════════════════════════════════════════════
#  ▼▼▼  PART 2 / 2  —  json 版（结构化抽取）  #️⃣ 已注释  ▼▼▼
# ═══════════════════════════════════════════════════════════════════════
#
# 约束解码的进阶形态：让模型填一张「多字段的表」——同时约束字段名、类型、取值范围。
# 它是 function call / tool call（让模型决定调哪个工具、填什么参数）的基础。
# 场景：财报结构化抽取 —— "这句话里公司/年度/指标分别是什么？"
# 输出样例：{"company":"招商银行","year":2023,"metric":"营收"}
#
# 对比 3 种模式：
#   1. raw             —— 裸 prompt（经常连合法 JSON 都不是）
#   2. response_format —— format="json"（保证是 JSON，但字段/类型/枚举不一定对）
#   3. guided_json     —— format=INTENT_SCHEMA_JSON（严格按 schema，100% 合法）
#
# ── 使用方法 ──────────────────────────────────────────────────────────
# 想运行 json 版时：
#   1. 把下面所有 # 开头的行去掉 # （VSCode 选中后按 Cmd+/ 可批量切换注释）
#   2. 把文件末尾的 main_choice() 改成 main_json()
#   3. 确保装了 jsonschema：pip install jsonschema
# ───────────────────────────────────────────────────────────────────────

# # jsonschema：用来严格校验输出是否符合 schema（choice 版不需要，json 版才用）
# from jsonschema import validate, ValidationError
#
# # ── JSON Schema 定义（多字段、带类型与取值范围）──────────────────────
# # 这段 schema 描述"我们希望模型输出长什么样"：一个对象，含 company/year/metric 三个字段。
# # 它会同时用在两处：① 作为 Ollama 的 format 参数（运行时硬约束）
# #                   ② 作为 jsonschema.validate 的校验标准（跑完再验证一遍）
# INTENT_SCHEMA_JSON = {
#     "type": "object",                       # 顶层是一个 JSON 对象
#     "properties": {                         # properties 描述对象里有哪些字段
#         "company": {
#             "type": "string",               # company 是字符串（公司名）
#             "description": "公司全称，如 招商银行、贵州茅台",
#         },
#         "year": {
#             "type": "integer",              # year 必须是整数
#             "minimum": 2015,                # 最小 2015
#             "maximum": 2025,                # 最大 2025（防止模型编出离谱年份）
#         },
#         "metric": {
#             "type": "string",               # metric 是字符串
#             "enum": ["营收", "净利润", "ROE", "毛利率", "总资产", "经营现金流"],  # ★ 6 选 1
#         },
#     },
#     "required": ["company", "year", "metric"],  # 三个字段都必填
#     "additionalProperties": False,          # 不允许输出 schema 之外的额外字段
# }
#
# # 系统提示词：告诉模型"从问题里抽这三个字段，输出纯 JSON"
# SYSTEM_PROMPT_JSON = f"""你是财报问答助手。从用户问题中提取结构化信息，输出纯 JSON，不要任何解释文字。
#
# 字段定义：
#   company: 公司全称
#   year: 年度（2015~2025 整数）
#   metric: 指标，必须是 ['营收', '净利润', 'ROE', '毛利率', '总资产', '经营现金流'] 之一
#
# 示例输出：
# {{"company": "招商银行", "year": 2023, "metric": "营收"}}"""
#
# # 测试用例：9 句真实风格提问。后几句是"坑"——测模型的理解与映射能力。
# TEST_CASES_JSON = [
#     "招行 2023 年营收多少",
#     "贵州茅台 2022 的净利润",
#     "平安银行去年（2024）的 ROE",
#     "2021 年五粮液毛利率",
#     "2023 宁德时代经营现金流",
#     "问一下比亚迪 2024 的总资产规模",
#     "茅台 2020 年利润情况",   # "利润"不在枚举里，模型要映射到"净利润"
#     "ICBC 2023 营收",           # 英文简称，测模型理解
#     "隆基绿能 22 年 roe",       # 简写年份+小写指标
# ]
#
#
# def evaluate_json(output: str) -> dict:
#     """评估 json 输出：分层校验（合法 JSON / 字段齐不齐 / 范围对不对 / schema 是否完全通过）
#     返回小字典记录各项是否通过。学这一关，重点就是看懂"分层校验"的思路。"""
#     result = {
#         "is_json": False,          # 能不能被 json.loads 解析
#         "has_all_fields": False,   # 三个必填字段是否都有
#         "year_in_range": False,    # year 是否在 2015~2025
#         "metric_in_enum": False,   # metric 是否在 6 个枚举内
#         "schema_valid": False,     # jsonschema 整体校验是否通过
#         "parsed": None,            # 解析后的字典（调试用）
#     }
#     try:
#         obj = json.loads(output)   # 尝试解析成字典
#         result["is_json"] = True
#         result["parsed"] = obj
#     except json.JSONDecodeError:
#         return result              # 解析失败直接返回
#
#     required = INTENT_SCHEMA_JSON["required"]   # ["company","year","metric"]
#     if all(k in obj for k in required):          # 三个字段都在 obj 里吗
#         result["has_all_fields"] = True
#
#     yr = obj.get("year")
#     if isinstance(yr, int) and 2015 <= yr <= 2025:  # 是 int 且落在范围内
#         result["year_in_range"] = True
#
#     if obj.get("metric") in INTENT_SCHEMA_JSON["properties"]["metric"]["enum"]:
#         result["metric_in_enum"] = True
#
#     try:
#         validate(instance=obj, schema=INTENT_SCHEMA_JSON)  # jsonschema 最严格校验
#         result["schema_valid"] = True
#     except ValidationError:
#         pass                          # 校验不过就保持 False，不中断程序
#
#     return result
#
#
# def run_json(user_msg: str, mode: str) -> tuple:
#     """根据 mode 用不同方式调用模型。
#     mode: 'raw' / 'response_format' / 'guided_json'"""
#     t0 = time.time()
#
#     fmt = None                        # 默认 None（raw 不加约束）
#     if mode == "guided_json":
#         fmt = INTENT_SCHEMA_JSON      # ★ 整段 schema 交给 Ollama（等价 vLLM guided_json）
#     elif mode == "response_format":
#         fmt = "json"                  # 只保证"输出是 JSON"，不保证字段/类型/枚举正确
#
#     call = {
#         "model": MODEL,
#         "messages": [
#             {"role": "system", "content": SYSTEM_PROMPT_JSON},
#             {"role": "user", "content": user_msg},
#         ],
#         "options": {"temperature": 0},
#     }
#     if fmt is not None:
#         call["format"] = fmt
#
#     resp = client.chat(**call)
#     return resp["message"]["content"].strip(), time.time() - t0
#
#
# def main_json():
#     """json 版主函数：遍历 9 条测试 × 3 种模式，打印明细 + 分层汇总表"""
#     print("=" * 78)
#     print("  Demo: Ollama 结构化输出（JSON Schema 约束，等价于 vLLM guided_json）")
#     print(f"  Model: {MODEL}")
#     print("  对比三种模式：裸 prompt / format='json' / format=schema")
#     print("=" * 78)
#
#     counters = {m: {"json": 0, "fields": 0, "year": 0, "enum": 0, "valid": 0}
#                 for m in ["raw", "response_format", "guided_json"]}
#     n = len(TEST_CASES_JSON)
#
#     for user in TEST_CASES_JSON:
#         print(f"\n▶ {user}")
#         for mode in ["raw", "response_format", "guided_json"]:
#             out, _ = run_json(user, mode)
#             ev = evaluate_json(out)
#             c = counters[mode]
#             if ev["is_json"]:         c["json"] += 1
#             if ev["has_all_fields"]:  c["fields"] += 1
#             if ev["year_in_range"]:   c["year"] += 1
#             if ev["metric_in_enum"]:  c["enum"] += 1
#             if ev["schema_valid"]:    c["valid"] += 1
#             tag = "✓" if ev["schema_valid"] else "✗"
#             disp = out[:80] + "…" if len(out) > 80 else out
#             print(f"  [{mode:<16}] {tag}  {disp}")
#
#     print("\n" + "=" * 78)
#     print(f"  {n} 条测试结果汇总")
#     print("=" * 78)
#     print(f"{'指标':<24}{'裸 prompt':<18}{'response_format':<20}{'guided_json':<15}")
#     print("-" * 78)
#     for metric_name, key in [("合法 JSON", "json"),
#                              ("字段齐全", "fields"),
#                              ("year 在 2015~2025", "year"),
#                              ("metric 在枚举内", "enum"),
#                              ("jsonschema 完全通过", "valid")]:
#         row = f"{metric_name:<22}"
#         for mode in ["raw", "response_format", "guided_json"]:
#             v = counters[mode][key]
#             row += f"{v}/{n} ({100*v/n:.0f}%)      "
#         print(row)
#
#     print()
#     print("=" * 78)
#     print("  结论：")
#     print("    format='json' 只保证是 JSON，不保证字段名、类型、枚举正确")
#     print("    format=schema 是唯一 100% 保证 schema 合法的方式（这就是 function call 的基础）")
#     print("=" * 78)


# ═══════════════════════════════════════════════════════════════════════
#  程序入口
# ═══════════════════════════════════════════════════════════════════════
#
# 直接运行本文件时执行这里。
# 当前跑 choice 版；想跑 json 版时：
#   1. 取消上方 json 版代码的注释
#   2. 把下面 main_choice() 换成 main_json()
#
if __name__ == "__main__":
    main_choice()
    # main_json()   # ← 取消注释以运行 json 版（需先取消上方 json 代码的注释）
