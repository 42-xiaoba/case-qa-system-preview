"""
Golden Set 评测脚本（P0 验收）
对 golden_qa.json 中的每道题走完整 RAG 管线（检索→组装→LLM），用程序化断言检查回答质量。

检查维度：
- must_contain: 所有指定关键词都必须出现
- any_of: 多组备选词，任一组命中即通过（组内为可替代表述）
- expect_sources: 至少出现一个来源标注
- forbid: 禁止出现（关键数据篡改/幻觉数字检测）

通过标准：
- 总体通过率 >= 80%
- critical 题目（优先级/关键数据/反幻觉）100% 通过
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")

from core.llm_client import llm_client
from core.pipeline import build_answer_messages

ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = ROOT / "scripts" / "golden_qa.json"
RESULT_PATH = ROOT / "scripts" / "eval_results.json"

EVAL_TEMPERATURE = 0.3
EVAL_MAX_TOKENS = 1024
EVAL_RETRIES = 6          # 批量评测穿透 429：3+6+12+24+48s 指数退避
PASS_RATE = 0.80
REQUEST_TIMEOUT_FALLBACK = 90


def check_answer(answer: str, item: dict) -> tuple[bool, list[str]]:
    """返回 (是否通过, 失败原因列表)"""
    reasons = []

    for kw in item.get("must_contain", []):
        if kw not in answer:
            reasons.append(f"缺少必须关键词: {kw}")

    any_groups = item.get("any_of", [])
    if any_groups and not any(any(alt in answer for alt in g) for g in any_groups):
        reasons.append(f"所有备选组均未命中: {any_groups}")

    sources = item.get("expect_sources", [])
    if sources and not any(s in answer for s in sources):
        reasons.append(f"未标注任何预期来源: {sources}")

    for bad in item.get("forbid", []):
        if bad in answer:
            reasons.append(f"出现禁止内容(数据篡改/幻觉): {bad}")

    return (not reasons), reasons


def main():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    questions = golden["questions"]

    # 断点续跑：沿用上一轮已 PASS 的结果，只重跑失败/调用异常的题（--full 强制全量）
    prev: dict[str, dict] = {}
    full_mode = "--full" in sys.argv
    if not full_mode and RESULT_PATH.exists():
        with open(RESULT_PATH, "r", encoding="utf-8") as f:
            prev = {r["id"]: r for r in json.load(f) if r.get("passed")}
    if prev:
        print(
            f"[续跑模式] 沿用上轮 {len(prev)} 道已通过题目，重跑其余 {len(questions) - len(prev)} 道"
            f"（--full 可强制全量）\n",
            flush=True,
        )

    results = []
    n_run = 0
    for i, item in enumerate(questions):
        qid = item["id"]
        q = item["q"]
        if qid in prev:
            results.append(prev[qid])
            print(f"[{i+1}/{len(questions)}] {qid} [SKIP] 沿用上轮 PASS 结果", flush=True)
            continue
        n_run += 1
        t0 = time.time()
        try:
            messages, docs = build_answer_messages(q)
            answer = llm_client.chat(
                messages,
                temperature=EVAL_TEMPERATURE,
                max_tokens=EVAL_MAX_TOKENS,
                retries=EVAL_RETRIES,
            )
        except Exception as e:
            answer = ""
            results.append({**item, "answer": "", "passed": False, "reasons": [f"调用失败: {e}"], "latency_s": round(time.time() - t0, 1), "n_docs": 0})
            print(f"[{i+1}/{len(questions)}] {qid} 调用失败: {e}", flush=True)
            time.sleep(5)
            continue

        ok, reasons = check_answer(answer, item)
        latency = round(time.time() - t0, 1)
        results.append({**item, "answer": answer, "passed": ok, "reasons": reasons, "latency_s": latency, "n_docs": len(docs)})
        mark = "PASS" if ok else "FAIL"
        print(f"[{i+1}/{len(questions)}] {qid} [{mark}] {item['category']} {latency}s | {q[:24]}", flush=True)
        if not ok:
            for r in reasons:
                print(f"       - {r}", flush=True)
        time.sleep(2)

    # ---- 汇总 ----
    n_total = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    critical = [r for r in results if r.get("critical")]
    critical_pass = sum(1 for r in critical if r["passed"])
    run_latencies = [r["latency_s"] for r in results if r["id"] not in prev]
    avg_latency = round(sum(run_latencies) / max(len(run_latencies), 1), 1)

    print("\n" + "=" * 60, flush=True)
    print(f"总体通过率: {n_pass}/{n_total} = {n_pass/n_total:.0%} (要求 >= {PASS_RATE:.0%})", flush=True)
    print(f"Critical 题通过率: {critical_pass}/{len(critical)} (要求 100%)", flush=True)
    print(f"平均延迟(本轮实跑{n_run}题): {avg_latency}s", flush=True)

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"明细已保存: {RESULT_PATH}", flush=True)

    overall_ok = (n_pass / n_total >= PASS_RATE) and (critical_pass == len(critical))
    print("\n=== Golden Set 评测通过 ===" if overall_ok else "\n=== Golden Set 评测未通过 ===", flush=True)
    sys.exit(0 if overall_ok else 1)


if __name__ == "__main__":
    main()
