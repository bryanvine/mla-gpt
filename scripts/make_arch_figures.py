"""Emit the two architecture SVGs used in the paper (docs/figures/).

These are static schematics (not data-derived), kept reproducible here so the
diagrams stay in sync with the model:

    python3 scripts/make_arch_figures.py

  * attention_variants.svg : how MHA / GQA / MLA / MQA produce and cache K/V.
  * arch_backbone.svg      : the shared GPT backbone (embeddings, the pre-norm
                             block with SwiGLU + swappable attention, tied head).
"""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"

SANS = "ui-sans-serif,system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"
INK, MUT, RULE = "#1a1a1a", "#666", "#e2e2e2"
MHA, GQA, MQA, MLA = "#d62728", "#1f77b4", "#9467bd", "#2ca02c"
TINT = {MHA: "#fdecec", GQA: "#e9f1f9", MQA: "#f1ecf8", MLA: "#e9f5ea"}
NRM_F, NRM_S = "#eef0f2", "#c9ced6"
AMB_F, AMB_S = "#fff4df", "#e3b341"
EMB_F, EMB_S = "#eaf2ff", "#b9d4f1"
MLP_F, MLP_S = "#eef7ee", "#bcdcbb"
PILL = "#f1f1ee"


def rect(x, y, w, h, fill, stroke="none", rx=0, sw=1.0, op=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx}" fill="{fill}" fill-opacity="{op}" stroke="{stroke}" '
            f'stroke-width="{sw}"{d}/>')


def text(x, y, s, size=13.0, fill=INK, anchor="start", weight="400", style="normal", family=SANS):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
            f'font-weight="{weight}" font-style="{style}" fill="{fill}" '
            f'text-anchor="{anchor}">{s}</text>')


def line(x1, y1, x2, y2, stroke=MUT, sw=1.0, dash=None, cap="butt"):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{stroke}" stroke-width="{sw}" stroke-linecap="{cap}"{d}/>')


def path(d, fill="none", stroke="none", sw=1.0, op=1.0, dash=None):
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="{fill}" fill-opacity="{op}" stroke="{stroke}" '
            f'stroke-width="{sw}"{da}/>')


def dividers(x, y, w, h, n, stroke="#ffffff", sw=1.5):
    return "".join(line(x + i * w / n, y, x + i * w / n, y + h, stroke, sw)
                   for i in range(1, n))


def arrow(x1, y1, x2, y2, stroke="#9aa3ad", sw=2.0):
    """Vertical-ish arrow with a small triangular head at (x2,y2)."""
    return (line(x1, y1, x2, y2, stroke, sw)
            + path(f"M{x2-4},{y2-7} L{x2},{y2} L{x2+4},{y2-7} Z", fill=stroke))


def oplus(cx, cy, r=10):
    return ("".join([
        f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="#444" stroke-width="1.5"/>',
        line(cx - r * 0.5, cy, cx + r * 0.5, cy, "#444", 1.5),
        line(cx, cy - r * 0.5, cx, cy + r * 0.5, "#444", 1.5),
    ]))


# --------------------------------------------------------------------------- #
#  Figure A : attention variants                                              #
# --------------------------------------------------------------------------- #
def variants_svg() -> str:
    W, H = 1120, 600
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="{SANS}">']
    s.append(rect(0, 0, W, H, "#ffffff"))
    s.append(text(W / 2, 42, "Four ways to cache keys and values", 22, INK, "middle", "700"))
    s.append(text(W / 2, 64, "same per-head dim d_h = 64 and RoPE; they differ only in what is stored per token",
                  13, MUT, "middle"))

    cards = [
        dict(c=MHA, name="MHA", sub="standard — n_kv = 12", nkv=12,
             note=["every query head keeps", "its own key and value"], elems="1,536 elem · 1.0×"),
        dict(c=GQA, name="GQA", sub="grouped — g = 4", nkv=4,
             note=["3 query heads share", "each KV head"], elems="512 elem · 3.0×"),
        dict(c=MLA, name="MLA", sub="latent compression", nkv=None,
             note=None, elems="272 elem · 5.6×"),
        dict(c=MQA, name="MQA", sub="shared — n_kv = 1", nkv=1,
             note=["all 12 query heads", "share one KV head"], elems="128 elem · 12.0×"),
    ]
    pw, gap, top = 252.0, 18.0, 80.0
    x0 = (W - (4 * pw + 3 * gap)) / 2
    ph = 486.0
    for i, cd in enumerate(cards):
        px = x0 + i * (pw + gap)
        col = cd["c"]
        cw = pw - 36
        s.append(rect(px, top, pw, ph, "#ffffff", RULE, rx=12))
        s.append(text(px + 18, top + 36, cd["name"], 21, col, "start", "700"))
        s.append(text(px + 18, top + 58, cd["sub"], 12.5, MUT))
        # query bar (shared by all)
        s.append(text(px + 18, top + 92, "queries · 12 heads", 11.5, MUT))
        qb = top + 100
        s.append(rect(px + 18, qb, cw, 22, "#eceef1", NRM_S, rx=5))
        s.append(dividers(px + 18, qb, cw, 22, 12, "#ffffff", 1.4))

        if cd["nkv"] is not None:  # MHA / GQA / MQA
            nkv = cd["nkv"]
            kvw = cw * nkv / 12
            kvx = px + 18 + (cw - kvw) / 2
            kvy = top + 190
            s.append(path(f"M{px+18:.1f},{qb+24:.1f} L{px+18+cw:.1f},{qb+24:.1f} "
                          f"L{kvx+kvw:.1f},{kvy:.1f} L{kvx:.1f},{kvy:.1f} Z",
                          fill=col, op=0.13))
            s.append(rect(kvx, kvy, kvw, 26, col, rx=5))
            if nkv > 1:
                s.append(dividers(kvx, kvy, kvw, 26, nkv, "#ffffff", 1.5))
            s.append(text(px + pw / 2, kvy + 50, f"K and V · {nkv} head" + ("s" if nkv > 1 else ""),
                          11.5, "#444", "middle", "600"))
            for j, ln in enumerate(cd["note"]):
                s.append(text(px + pw / 2, top + 300 + j * 16, ln, 11.5, MUT, "middle"))
        else:  # MLA
            s.append(text(px + pw / 2, top + 150, "down-project  W_DKV", 10.5, MUT, "middle"))
            ring_y = top + 158
            s.append(rect(px + 20, ring_y, cw + 0, 86, MLA, "none", rx=10, op=0.06))
            s.append(rect(px + 20, ring_y, cw + 0, 86, "none", MLA, rx=10, sw=1.3, dash="5 4"))
            s.append(text(px + 28, ring_y - 6, "cached", 10.5, MLA, "start", "700"))
            s.append(rect(px + 38, ring_y + 14, cw - 36, 22, MLA, rx=5))
            s.append(text(px + pw / 2, ring_y + 29, "c_KV · 256", 11.5, "#ffffff", "middle", "700"))
            s.append(rect(px + 64, ring_y + 44, cw - 88, 18, "#5fa55f", rx=4))
            s.append(text(px + pw / 2, ring_y + 57, "k_rope · 16", 10.5, "#ffffff", "middle", "600"))
            s.append(text(px + pw / 2, ring_y + 104, "up-project  W_UK,UV", 10.5, MUT, "middle"))
            gy = ring_y + 116
            s.append(rect(px + 18, gy, cw, 22, MLA, "none", rx=5, op=0.16))
            s.append(rect(px + 18, gy, cw, 22, "none", MLA, rx=5, sw=1.0, dash="4 3"))
            s.append(dividers(px + 18, gy, cw, 22, 12, "#bfe0bf", 1.2))
            s.append(text(px + pw / 2, gy + 38, "per-head K, V — rebuilt at", 11, MUT, "middle"))
            s.append(text(px + pw / 2, gy + 53, "compute time, never cached", 11, MUT, "middle"))

        # cache pill
        py = top + ph - 66
        s.append(rect(px + 18, py, cw, 48, TINT[col], col, rx=8))
        s.append(text(px + pw / 2, py + 19, "cached / token · layer", 10.5, MUT, "middle"))
        s.append(text(px + pw / 2, py + 38, cd["elems"], 15, col, "middle", "700"))

    s.append("</svg>")
    return "\n".join(s)


# --------------------------------------------------------------------------- #
#  Figure B : backbone                                                        #
# --------------------------------------------------------------------------- #
def backbone_svg() -> str:
    W, H = 820, 700
    AX = 300.0            # vertical axis of the main column
    bw = 230.0
    bx = AX - bw / 2
    s = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" font-family="{SANS}">']
    s.append(rect(0, 0, W, H, "#ffffff"))
    s.append(text(AX, 34, "GPT backbone — shared across all four variants", 19, INK, "middle", "700"))

    def cbox(y, h, l1, l2, fill, stroke):
        out = [rect(bx, y, bw, h, fill, stroke, rx=8)]
        if l2:
            out.append(text(AX, y + h / 2 - 3, l1, 14, INK, "middle", "600"))
            out.append(text(AX, y + h / 2 + 15, l2, 11.5, MUT, "middle", family="ui-monospace,Menlo,monospace"))
        else:
            out.append(text(AX, y + h / 2 + 5, l1, 14, INK, "middle", "600"))
        return "".join(out)

    # input + embedding
    s.append(rect(bx, 50, bw, 32, PILL, RULE, rx=16))
    s.append(text(AX, 70, "input tokens  (B × T)", 13, INK, "middle", "600"))
    s.append(arrow(AX, 82, AX, 100))
    s.append(cbox(100, 48, "Token Embedding", "50,304 × 768", EMB_F, EMB_S))
    s.append(arrow(AX, 148, AX, 166))

    # ---- transformer block container (offset stack implies x12) ----
    cx0, cy0, cw, ch = 150.0, 166.0, 440.0, 286.0
    for off, fill in ((20, "#f1f2f0"), (10, "#f7f8f6")):
        s.append(rect(cx0 + off, cy0 + off, cw, ch, fill, "#d4d8dd", rx=14))
    s.append(rect(cx0, cy0, cw, ch, "#fcfcfb", "#cfd4da", rx=14))
    s.append(text(cx0 + 20, cy0 + 26, "Transformer Block", 16, INK, "start", "700"))
    s.append(rect(cx0 + cw - 70, cy0 + 10, 56, 26, INK, rx=13))
    s.append(text(cx0 + cw - 42, cy0 + 28, "× 12", 13, "#ffffff", "middle", "700"))
    s.append(line(cx0, cy0 + 42, cx0 + cw, cy0 + 42, RULE, 1))

    # residual stream
    s_top, s_bot = cy0 + 52, cy0 + ch - 16
    s.append(line(AX, s_top, AX, s_bot, "#9aa3ad", 3))
    s.append(text(AX - 14, s_top + 10, "x", 13, MUT, "end", "400", "italic"))

    def sublayer(y_row, op_l1, op_l2, op_fill, op_stroke, y_join):
        out = []
        out.append(line(AX, y_row, 345, y_row, "#9aa3ad", 1.6))
        out.append(rect(345, y_row - 14, 104, 28, NRM_F, NRM_S, rx=6))
        out.append(text(397, y_row + 4, "RMSNorm", 12, INK, "middle", "600"))
        out.append(arrow(449, y_row, 463, y_row, "#9aa3ad", 1.6))
        ow = 122
        out.append(rect(465, y_row - 18, ow, 36, op_fill, op_stroke, rx=6))
        out.append(text(465 + ow / 2, y_row - 1, op_l1, 12.5, INK, "middle", "700"))
        out.append(text(465 + ow / 2, y_row + 13, op_l2, 9.5, MUT, "middle",
                        family="ui-monospace,Menlo,monospace"))
        # return path op-output -> oplus on the stream
        out.append(path(f"M{465+ow/2:.1f},{y_row+18:.1f} V{y_join-12:.1f} H{AX:.1f}",
                        stroke="#9aa3ad", sw=1.6))
        out.append(path(f"M{AX-4},{y_join-12} L{AX},{y_join-4} L{AX+4},{y_join-12} Z", fill="#9aa3ad"))
        out.append(oplus(AX, y_join))
        return "".join(out)

    s.append(sublayer(cy0 + 60, "Attention", "MHA·MQA·GQA·MLA", AMB_F, AMB_S, cy0 + 134))
    s.append(sublayer(cy0 + 166, "SwiGLU MLP", "768→2048→768", MLP_F, MLP_S, cy0 + 240))

    # annotations to the right
    s.append(line(587, cy0 + 60, 612, cy0 + 60, RULE, 1))
    s.append(text(616, cy0 + 56, "RoPE θ = 10⁴", 11, MUT, "start"))
    s.append(text(616, cy0 + 70, "decoupled for MLA", 10, MUT, "start"))
    s.append(line(587, cy0 + 166, 612, cy0 + 166, RULE, 1))
    s.append(text(616, cy0 + 162, "SwiGLU gate", 11, MUT, "start"))
    s.append(text(616, cy0 + 176, "8/3× expansion", 10, MUT, "start"))

    # spec card (top-right)
    s.append(rect(612, 56, 192, 96, "#f7f7f5", RULE, rx=8))
    s.append(text(626, 76, "Held fixed", 12, INK, "start", "700"))
    for j, ln in enumerate(["d_model = 768", "n_heads = 12 · d_h = 64",
                            "layers = 12 · vocab = 50,304", "context = 1024 · RMSNorm"]):
        s.append(text(626, 94 + j * 15, ln, 10.5, MUT, "start",
                      family="ui-monospace,Menlo,monospace"))

    # ---- out of the block ----
    s.append(arrow(AX, cy0 + ch, AX, cy0 + ch + 20))
    s.append(cbox(cy0 + ch + 20, 38, "Final RMSNorm", "", NRM_F, NRM_S))
    yb = cy0 + ch + 20
    s.append(arrow(AX, yb + 38, AX, yb + 56))
    s.append(cbox(yb + 56, 46, "LM Head", "768 × 50,304", EMB_F, EMB_S))
    yh = yb + 56
    s.append(arrow(AX, yh + 46, AX, yh + 64))
    s.append(rect(bx, yh + 64, bw, 32, PILL, RULE, rx=16))
    s.append(text(AX, yh + 84, "logits  (B × T × 50,304)", 13, INK, "middle", "600"))

    # tied-weights link down the left margin (emb <-> head)
    s.append(path(f"M{bx:.1f},116 C 120,116 120,{yh+82:.1f} {bx:.1f},{yh+82:.1f}",
                  stroke="#7aa37a", sw=1.4, dash="5 4"))
    s.append(text(110, (116 + yh + 82) / 2, "weights tied", 11, "#3f7a3f", "middle", "600",
                  family=SANS) .replace("<text ", '<text transform="rotate(-90 110 %.0f)" ' % ((116 + yh + 82) / 2)))

    s.append(text(cx0, H - 14, "pre-norm residuals:  x ← x + Sublayer(RMSNorm(x))",
                  11.5, MUT, "start"))
    s.append("</svg>")
    return "\n".join(s)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attention_variants.svg").write_text(variants_svg())
    (OUT / "arch_backbone.svg").write_text(backbone_svg())
    print(f"wrote -> {OUT/'attention_variants.svg'}")
    print(f"wrote -> {OUT/'arch_backbone.svg'}")


if __name__ == "__main__":
    main()
