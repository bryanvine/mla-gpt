#!/usr/bin/env python3
"""Paper #3 follow-up: compare the cap-limited MQAR cells at 60k vs 120k budget.

Answers "capacity vs grok-speed": a cell that flips groked False->True given 2x budget
was BUDGET-limited (grok-speed); a cell that stays False is CAPACITY-limited. The 120k
multi-seed aggregate merges the re-run seeds with the early-stopped 60k seeds (which are
cap-invariant, so their 60k value already equals their 120k value).

Run AFTER the sweep + a dedicated eval of the _cap120k dirs:
  bash scripts/recall_cap120k.sh
  uv run python scripts/eval_recall.py --glob "runs/recall_mqar_*_cap120k" --out runs/recall_eval_cap120k
  uv run python scripts/recall_cap_compare.py
"""
import json, os, sys
from collections import defaultdict

CANON = "runs/recall_eval/recall.json"
CAP120 = "runs/recall_eval_cap120k/recall.json"


def load(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path}\n  -> run: uv run python scripts/eval_recall.py "
                 f'--glob "runs/recall_mqar_*_cap120k" --out runs/recall_eval_cap120k')
    d = json.load(open(path))
    by = {}
    for r in d["recall"]:
        if r["task"] != "mqar":
            continue
        by[(r["attn_type"], r["trained_difficulty"], r["train_seed"])] = r
    return by


def agg(cells):
    """mean accuracy + grok-fraction over a list of records."""
    n = len(cells)
    acc = sum(c["accuracy"] for c in cells) / n
    gk = sum(1 for c in cells if c.get("groked")) / n
    return acc, gk, n


def main():
    r60 = load(CANON)
    r120 = load(CAP120)

    # affected (attn, difficulty) groups = whatever was re-run at 120k
    groups = sorted({(a, p) for (a, p, s) in r120})
    seeds_of = defaultdict(set)
    for (a, p, s) in r60:
        seeds_of[(a, p)].add(s)

    print("=" * 78)
    print("PER-CELL: cap-limited cells, 60k -> 120k  (did the extra budget help?)")
    print("=" * 78)
    print(f"{'attn':4} {'p':>3} {'seed':>6} | {'acc60':>6} {'g60':>4} -> {'acc120':>6} {'g120':>4} "
          f"{'grok_iter120':>12}  verdict")
    print("-" * 78)
    flips = defaultdict(list)  # (a,p) -> list of "budget"/"capacity"
    for (a, p, s) in sorted(r120):
        c60 = r60.get((a, p, s))
        c120 = r120[(a, p, s)]
        a60 = c60["accuracy"] if c60 else float("nan")
        g60 = c60.get("groked") if c60 else None
        a120 = c120["accuracy"]
        g120 = c120.get("groked")
        gi = c120.get("iters_to_grok")
        if g120 and not g60:
            verdict = "BUDGET (flipped)"
            flips[(a, p)].append("budget")
        elif g120:
            verdict = "grok (was grok)"
        else:
            verdict = "capacity (still no grok)"
            flips[(a, p)].append("capacity")
        print(f"{a:4} {p:>3} {s:>6} | {a60:6.2f} {str(g60):>4} -> {a120:6.2f} {str(g120):>4} "
              f"{str(gi):>12}  {verdict}")

    print()
    print("=" * 78)
    print("AGGREGATE: 60k-budget vs 120k-budget (120k merges cap-invariant early-stopped seeds)")
    print("=" * 78)
    print(f"{'attn':4} {'KV':>5} {'p':>3} | {'acc60':>13}  {'acc120':>13}   delta")
    print("-" * 78)
    for (a, p) in groups:
        seeds = sorted(seeds_of[(a, p)])
        cells60 = [r60[(a, p, s)] for s in seeds if (a, p, s) in r60]
        # 120k budget: prefer the re-run cell, else the cap-invariant 60k cell
        cells120 = [r120.get((a, p, s)) or r60[(a, p, s)] for s in seeds if (a, p, s) in r60]
        m60, gk60, n60 = agg(cells60)
        m120, gk120, n120 = agg(cells120)
        kvb = cells60[0]["kv_bytes_per_token"]
        d = m120 - m60
        print(f"{a:4} {kvb:>5} {p:>3} | {m60:.2f} g{int(gk60*n60)}/{n60}    "
              f"{m120:.2f} g{int(gk120*n120)}/{n120}    {d:+.2f}")

    print()
    print("=" * 78)
    print("VERDICT per group")
    print("=" * 78)
    for (a, p) in groups:
        v = flips[(a, p)]
        nb = v.count("budget")
        nc = v.count("capacity")
        if nb and not nc:
            tag = f"BUDGET-limited ({nb}/{nb+nc} cap-limited seeds flipped with 2x budget)"
        elif nb:
            tag = f"MIXED ({nb} flipped, {nc} still capacity-bound)"
        else:
            tag = f"CAPACITY-limited (0/{nc} flipped even at 120k)"
        print(f"  {a} p{p}: {tag}")


if __name__ == "__main__":
    main()
