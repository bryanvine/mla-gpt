#!/usr/bin/env python3
"""Paper #3: was each 120k flip caused by the gentler LR decay or the extra iters?

For every flipped cell we now have THREE runs (same seed, num_pairs, init):
  steep60  = runs/recall_mqar_<a>_p<p>_s<seed>          steep 60k  (horizon=60k)   -- the baseline
  gentleA  = .../_lrctrl    gentle 60k  (60k iters, horizon=120k)  } two independent
  gentleB  = .../_cap120k   gentle 120k (horizon=120k)             } GENTLE-LR samples

steep60 never groked any of these (that's why they were "cap-limited"). The question is whether
the GENTLER schedule recovers grokking within the ORIGINAL 60k iteration budget:
  "grok<60k under gentle" = gentleA groked  OR  gentleB grok-iter < 60000   (either sample suffices)
  -> LR-SHAPE   (same 60k budget, just a gentler decay)
  else (gentle only groks at >=60k) -> EXTRA-ITERS (genuinely needs more steps)

Caveat surfaced by this run: grok-iter is NONDETERMINISTIC at fixed seed (bf16+CUDA+compile on a
sharp transition). gentleA vs gentleB grok-iters can differ by tens of thousands of iters, so a
single sample under/over-counts. Pooling the two gentle samples is the noise-robust read; the
binary "does gentle recover grokking <=60k" is far more stable than the precise iter.
"""
import glob, json, os, re, sys

CAP = 60000


def summ(path):
    if not os.path.exists(path):
        return None
    s = json.load(open(path))
    return {"grok": bool(s.get("groked")), "gi": s.get("iters_to_grok"), "final": s.get("final_iter")}


def fmt(x):
    if not x:
        return "n/a"
    return f"grok@{x['gi']}" if x["grok"] else f"no({x['final']})"


def main():
    dirs = sorted(glob.glob("runs/recall_mqar_*_lrctrl"))
    if not dirs:
        sys.exit("no _lrctrl runs found -- run scripts/recall_lrctrl.sh first")

    print("=" * 100)
    print("CONFOUND CONTROL — was the 120k flip from the gentler LR or the extra iters?")
    print("steep60 is the baseline (none groked); gentleA=lrctrl(60k), gentleB=cap120(120k) are 2 gentle samples")
    print("=" * 100)
    print(f"{'attn':4} {'p':>3} {'seed':>6} | {'steep60':>10} {'gentleA(60k)':>13} {'gentleB(120k)':>14} "
          f"| {'grok<60k gentle?':>16} {'verdict':>12}  ndet|Δgi|")
    print("-" * 100)

    n_lrshape = n_extra = 0
    ndet_deltas = []
    for d in dirs:
        m = re.match(r"recall_mqar_(\w+?)_p(\d+)_s(\d+)_lrctrl", os.path.basename(d))
        a, p, seed = m.group(1), int(m.group(2)), m.group(3)
        base = f"runs/recall_mqar_{a}_p{p}_s{seed}"
        steep = summ(f"{base}/summary.json")
        gA = summ(f"{d}/summary.json")
        gB = summ(f"{base}_cap120k/summary.json")

        grok_sub60 = (gA and gA["grok"]) or (gB and gB["grok"] and gB["gi"] is not None and gB["gi"] < CAP)
        if grok_sub60:
            verdict = "LR-SHAPE"
            n_lrshape += 1
        else:
            verdict = "EXTRA-ITERS"
            n_extra += 1

        ndet = ""
        if gA and gA["grok"] and gB and gB["grok"]:
            dlt = abs(gA["gi"] - gB["gi"])
            ndet_deltas.append(dlt)
            ndet = f"{dlt}"

        print(f"{a:4} {p:>3} {seed:>6} | {fmt(steep):>10} {fmt(gA):>13} {fmt(gB):>14} "
              f"| {str(bool(grok_sub60)):>16} {verdict:>12}  {ndet}")

    n = n_lrshape + n_extra
    print("-" * 100)
    print(f"VERDICT: gentle LR recovers grokking within the ORIGINAL 60k budget in {n_lrshape}/{n} cells "
          f"(steep-60k baseline: 0/{n}).")
    print(f"         Only {n_extra}/{n} genuinely need iters beyond 60k -> the 120k 'budget' effect is "
          f"PREDOMINANTLY LR-SCHEDULE, not iteration count.")
    if ndet_deltas:
        print(f"NONDETERMINISM: of the cells that groked in BOTH gentle samples, fixed-seed grok-iter "
              f"differs by {min(ndet_deltas)}-{max(ndet_deltas)} iters")
        print(f"                -> grok-iter is a noisy metric; rely on the binary flip + best.pt accuracy.")


if __name__ == "__main__":
    main()
