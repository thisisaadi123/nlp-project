"""Blind comparison of translations by an independent judge (Gemini, a different model family from
the gpt-oss translator, so it doesn't favour its own style).

    python scripts/judge_translations.py            # every data/cache/dub/*.json with a before/ copy
    python scripts/judge_translations.py ta_01.ta-hi

Per line it sees the source, the scene, and up to three candidates in random order, labelled A/B/C:
the IndicTrans2 draft, the previous final translation (data/cache/dub/before/), and the current one.
It scores each 1-5 on meaning, naturalness (spoken film dialogue) and address (forms of address,
pronoun level, kinship words), and picks the best. Results: data/cache/judge/<name>.json and a
summary table. An LLM judge is a proxy; a native speaker's check is still the real test.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.llm.client import _gemini  # noqa: E402
from backend.settings import config  # noqa: E402

DUB = ROOT / "data/cache/dub"
OUT = ROOT / "data/cache/judge"
SYSTEMS = ("draft", "before", "after")
ASPECTS = ("meaning", "natural", "address")

PROMPT = """You are a strict bilingual reviewer of film dubbing from {src} into {tgt}.
Scene: {scene}

For each line below you get the {src} source and candidate {tgt} translations (A, B, C...), in
random order. Score every candidate from 1 (bad) to 5 (perfect) on:
- meaning: says what the source means in this scene; nothing lost or added
- natural: sounds like real spoken {tgt} film dialogue, not a textbook or a machine
- address: forms of address, kinship words and pronoun level fit who is speaking to whom
  (e.g. Tamil anni -> Hindi bhabhi). A {src} word written in {tgt} script instead of being
  translated (except names) is a serious error for meaning and address.
Then name the best candidate ("tie" if equal).

Lines:
{lines}

Return JSON: {{"lines": [{{"line_id": <id>, "scores": {{"A": {{"meaning": n, "natural": n, "address": n}}, ...}},
                         "best": "A|B|C|tie", "note": "<one short reason>"}}]}}"""


def judge(name: str, seed: int = 0) -> dict:
    after = json.loads((DUB / f"{name}.json").read_text(encoding="utf-8"))
    before_path = DUB / "before" / f"{name}.json"
    before = json.loads(before_path.read_text(encoding="utf-8")) if before_path.exists() else None
    langs = config()["languages"]
    rng = random.Random(seed)
    blocks, keys = [], {}
    for i, ln in enumerate(after["lines"]):
        cands = {"draft": ln.get("draft") or "", "after": ln["dub"]["translation"]}
        if before:
            cands["before"] = before["lines"][i]["dub"]["translation"]
        order = list(cands)
        rng.shuffle(order)
        labels = dict(zip("ABC", order))
        keys[ln["line_id"]] = labels
        body = "\n".join(f"  {k}: {cands[s]}" for k, s in labels.items())
        blocks.append(f"line_id {ln['line_id']} (speaker {ln.get('speaker')} to {ln.get('addressee')}):\n"
                      f"  source: {ln['text']}\n{body}")
    ctx = after.get("context") or {}
    scene = json.dumps({k: ctx.get(k) for k in ("summary", "speakers", "relationships") if ctx.get(k)}, ensure_ascii=False)
    prompt = PROMPT.format(src=langs[after["source_language"]]["name"], tgt=langs[after["target_language"]]["name"],
                           scene=scene, lines="\n\n".join(blocks))
    reply = _gemini([{"role": "user", "content": prompt}], 8192)
    rows = []
    for r in reply.get("lines", []):
        labels = keys.get(r.get("line_id"))
        if not labels:
            continue
        scores = {labels[k]: v for k, v in (r.get("scores") or {}).items() if k in labels}
        best = r.get("best")
        rows.append({"line_id": r["line_id"], "scores": scores, "best": labels.get(best, "tie"), "note": r.get("note")})
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"name": name, "judge": config()["llm"]["fallback_model"], "lines": rows}
    (OUT / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def summarize(results: list[dict]) -> None:
    rows = [r for res in results for r in res["lines"]]
    systems = [s for s in SYSTEMS if any(s in r["scores"] for r in rows)]
    print(f"\n{len(rows)} lines judged. Mean score 1-5 (higher is better):")
    print(f"  {'':8}" + "".join(f"{a:>9}" for a in ASPECTS) + f"{'overall':>9}{'best':>7}")
    for s in systems:
        sc = [r["scores"][s] for r in rows if s in r["scores"]]
        means = [sum(float(x.get(a, 0)) for x in sc) / len(sc) for a in ASPECTS]
        wins = sum(r["best"] == s for r in rows)
        print(f"  {s:8}" + "".join(f"{m:9.2f}" for m in means) + f"{sum(means) / 3:9.2f}{wins:7d}")
    print(f"  ties: {sum(r['best'] == 'tie' for r in rows)}")
    worse = [r for r in rows if "draft" in r["scores"] and "after" in r["scores"]
             and sum(r["scores"]["after"].values()) < sum(r["scores"]["draft"].values())]
    print(f"  lines where the final translation scored below the raw draft: {len(worse)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    args = ap.parse_args()
    names = args.names or sorted(p.stem for p in (DUB / "before").glob("*.json"))
    summarize([judge(n) for n in names])


if __name__ == "__main__":
    main()
