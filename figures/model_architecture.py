"""
Publication-ready model architecture figure for Fennec (TCNResNNWithMetaAndEmbeddings).
Generates a Nature-style multi-panel figure: main pipeline + two inset detail panels.
Run:  python figures/model_architecture.py
Output: figures/model_architecture.pdf  +  figures/model_architecture.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import to_rgba
import numpy as np
import os

# ── Nature-style global params — STIX for LaTeX-like typesetting ─────────────
plt.rcParams.update({
    "font.family":        "STIXGeneral",
    "mathtext.fontset":   "stix",
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Palette — muted, desaturated tones ───────────────────────────────────────
C = {
    "seq":   "#5B7BB5",   # dusty blue        (was vivid #4878CF)
    "meta":  "#6DA86A",   # sage green        (was bright #6ACC65)
    "fm":    "#B85C5C",   # terracotta        (was hot #D65F5F)
    "attn":  "#9B78B0",   # muted mauve       (was #B47CC7)
    "lora":  "#C97D4A",   # burnt sienna      (was orange #EE854A)
    "comb":  "#5B9EBF",   # slate blue        (was sky #56B4E9)
    "out":   "#3A3A3A",   # near-black
    "conv":  "#5B7BB5",   # matches seq
    "bn":    "#9A9A9A",   # mid-grey
    "elu":   "#C9A84C",   # warm straw        (was bright yellow #F8C471)
    "drop":  "#82B89A",   # muted sage        (was mint #ABEBC6)
    "res":   "#C4A882",   # sand              (was pale orange #FAD7A0)
    "arrow": "#555555",
}

FILL_ALPHA = 0.75   # TikZ-style: semi-transparent fill; border is same color, fully opaque

def _face(color):
    """Return an RGBA tuple at FILL_ALPHA for use as facecolor."""
    r, g, b, _ = to_rgba(color)
    return (r, g, b, FILL_ALPHA)

# Box heights: single-line vs two-line (label + sublabel)
H1 = 0.046   # single-line box
H2 = 0.068   # two-line box (label + sublabel)
RADIUS = 0.010

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def rbox(ax, cx, cy, w, color, label, sublabel=None,
         fontsize=6.8, sublabel_fs=5.5,
         text_color="white", bold=False, lw=0.7):
    """Rounded rectangle sized to content — TikZ style: translucent fill, opaque colored border."""
    h = H2 if sublabel else H1
    bp = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                        boxstyle=f"round,pad=0,rounding_size={RADIUS}",
                        facecolor=_face(color), edgecolor=color,
                        linewidth=lw, zorder=3)
    ax.add_patch(bp)
    fw = "bold" if bold else "normal"
    # Vertical offset: push label up when sublabel is present
    dy_main = h * 0.18 if sublabel else 0
    ax.text(cx, cy + dy_main, label,
            ha="center", va="center", fontsize=fontsize,
            color=text_color, fontweight=fw, zorder=4, clip_on=False)
    if sublabel:
        ax.text(cx, cy - h * 0.22, sublabel,
                ha="center", va="center", fontsize=sublabel_fs,
                color=text_color, alpha=0.88, zorder=4, clip_on=False)
    return h   # return actual height so callers can compute spacing


def box_h(sublabel):
    return H2 if sublabel else H1


def arr(ax, x, y_from, y_to, color="#444444", lw=0.85):
    """Vertical downward arrow from y_from to y_to (y_from > y_to)."""
    ax.annotate("", xy=(x, y_to), xytext=(x, y_from),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=4,
                                shrinkA=0, shrinkB=0),
                zorder=5)


GAP = 0.018   # gap between box bottom and next box top


def next_cy(cy, h_cur, h_next):
    """Centre of the next box below the current one."""
    return cy - h_cur/2 - GAP - h_next/2


# ─────────────────────────────────────────────────────────────────────────────
# Figure layout — three-column: main | TCN detail | Attn detail
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10.5, 5.8))
fig.patch.set_facecolor("white")

ax_main = fig.add_axes([0.01,  0.08, 0.50, 0.90])   # main pipeline — left 50%
ax_tcn  = fig.add_axes([0.535, 0.08, 0.225, 0.90])  # TCN detail — center
ax_attn = fig.add_axes([0.770, 0.08, 0.225, 0.90])  # Attn detail — right

for ax in (ax_main, ax_tcn, ax_attn):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("white")

# ─────────────────────────────────────────────────────────────────────────────
# (A) MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_main
ax.text(0.01, 0.99, "A", fontsize=9, fontweight="bold",
        va="top", ha="left", transform=ax.transAxes)

BW = 0.2475    # branch box width (−10% from 0.275)
Xseq  = 0.165
Xmeta = 0.490
Xfm   = 0.815

# ── Input boxes ──────────────────────────────────────────────────────────────
Y0 = 0.945
h_inp = H2
rbox(ax, Xseq,  Y0, BW, C["seq"],  "Sequence one-hot",  sublabel="(N, 25, L)", bold=True)
rbox(ax, Xmeta, Y0, BW, C["meta"], "Meta features",     sublabel="(N, 25)",    bold=True)
rbox(ax, Xfm,   Y0, BW, C["fm"],   "FM Embeddings", sublabel="RNA-FM / Orthrus / RiNALMo", bold=True)

# ── Helper: draw arrow between two box centres ────────────────────────────────
def connect(ax, x, cy_above, h_above, cy_below, h_below):
    y_from = cy_above - h_above/2
    y_to   = cy_below + h_below/2
    arr(ax, x, y_from, y_to)

# ─────────────────────────────────────────────────────────────────────────────
# TCN Branch
# ─────────────────────────────────────────────────────────────────────────────
tcn_defs = [
    ("TCN Residual Block 1", r"$k{=}3,\ d{=}1,\ 256\ \mathrm{ch}$"),
    ("TCN Residual Block 2", r"$k{=}5,\ d{=}2,\ 256\ \mathrm{ch}$"),
    ("TCN Residual Block 3", r"$k{=}7,\ d{=}4,\ 256\ \mathrm{ch}$"),
]
tcn_cy = []
cy = Y0
h_prev = h_inp
for lbl, sub in tcn_defs:
    h_cur = box_h(sub)
    cy = next_cy(cy, h_prev, h_cur)
    rbox(ax, Xseq, cy, BW, C["seq"], lbl, sublabel=sub)
    tcn_cy.append((cy, h_cur))
    h_prev = h_cur

# draw TCN arrows properly
connect(ax, Xseq, Y0, h_inp, tcn_cy[0][0], tcn_cy[0][1])
for i in range(len(tcn_cy)-1):
    cy_a, h_a = tcn_cy[i]
    cy_b, h_b = tcn_cy[i+1]
    connect(ax, Xseq, cy_a, h_a, cy_b, h_b)

# Concat + Permute
cy_cp = next_cy(tcn_cy[-1][0], tcn_cy[-1][1], H2)
rbox(ax, Xseq, cy_cp, BW, "#6E8FB5", "Concat + Permute", sublabel=r"$(N,\ L,\ 768)$")
connect(ax, Xseq, tcn_cy[-1][0], tcn_cy[-1][1], cy_cp, H2)

# ── Skip connections: out1 and out2 bypass directly to Concat+Permute ─────────
# Models:  combined_out = torch.cat([out1, out2, out3], dim=1)
bp_col = C["seq"]
bp_lw  = 0.65
bp_ls  = (0, (4, 2))   # small dashes
x_left = Xseq - BW/2   # left edge of all TCN column boxes (~0.041)
x_bp   = [0.005, 0.016] # two bypass columns, left of the dilation bracket
for i, xb in enumerate(x_bp):
    y_src = tcn_cy[i][0]    # centre height of Block i+1
    # horizontal left from box edge
    ax.plot([x_left, xb], [y_src, y_src], color=bp_col, lw=bp_lw, ls=bp_ls, zorder=2, clip_on=False)
    # vertical down to cy_cp level
    ax.plot([xb, xb], [y_src, cy_cp], color=bp_col, lw=bp_lw, ls=bp_ls, zorder=2, clip_on=False)
    # horizontal right back to box edge (dashed)
    ax.plot([xb, x_left], [cy_cp, cy_cp], color=bp_col, lw=bp_lw, ls=bp_ls, zorder=2, clip_on=False)
    # arrowhead at destination (tiny solid shaft blends into dashed line)
    ax.annotate("", xy=(x_left, cy_cp), xytext=(x_left - 0.002, cy_cp),
                arrowprops=dict(arrowstyle="-|>", color=bp_col, lw=bp_lw, mutation_scale=3.5),
                annotation_clip=False, zorder=4)

# CNN-MLP
cy_mlp = next_cy(cy_cp, H2, H2)
rbox(ax, Xseq, cy_mlp, BW, "#4A6A9E", r"CNN-MLP $768{\to}128{\to}128$",
     sublabel="ELU · Dropout")
connect(ax, Xseq, cy_cp, H2, cy_mlp, H2)

# Mean pool
cy_pool = next_cy(cy_mlp, H2, H2)
rbox(ax, Xseq, cy_pool, BW, "#4A6A9E", "Mean pool (dim=1)",
     sublabel=r"$\rightarrow (N,\ 128)$")
connect(ax, Xseq, cy_mlp, H2, cy_pool, H2)
y_seq_bot = cy_pool - H2/2   # bottom of last TCN-branch box

# ─────────────────────────────────────────────────────────────────────────────
# Meta Branch
# ─────────────────────────────────────────────────────────────────────────────
meta_steps = [
    r"Linear $25 \rightarrow 32$",
    "BatchNorm + ELU",
    r"Linear $32 \rightarrow 32$",
    "BatchNorm + ELU",
    r"Linear $32 \rightarrow 32$",
    "BatchNorm + ELU",
]
meta_colors = [C["meta"], "#72A06E"] * 3

cy = Y0
h_prev = h_inp
meta_boxes = []
for lbl, col in zip(meta_steps, meta_colors):
    h_cur = H1
    cy = next_cy(cy, h_prev, h_cur)
    meta_boxes.append((cy, h_cur, lbl, col))
    h_prev = h_cur

for i, (b_cy, b_h, b_lbl, b_col) in enumerate(meta_boxes):
    tc = "white"
    rbox(ax, Xmeta, b_cy, BW, b_col, b_lbl, text_color=tc)

connect(ax, Xmeta, Y0, h_inp, meta_boxes[0][0], meta_boxes[0][1])
for i in range(len(meta_boxes)-1):
    cy_a, h_a = meta_boxes[i][0], meta_boxes[i][1]
    cy_b, h_b = meta_boxes[i+1][0], meta_boxes[i+1][1]
    connect(ax, Xmeta, cy_a, h_a, cy_b, h_b)

# Meta output label
cy_meta_out = next_cy(meta_boxes[-1][0], meta_boxes[-1][1], H1)
rbox(ax, Xmeta, cy_meta_out, BW, "#579A54", r"$\rightarrow (N,\ 32)$")
connect(ax, Xmeta, meta_boxes[-1][0], meta_boxes[-1][1], cy_meta_out, H1)
y_meta_bot = cy_meta_out - H1/2

# ─────────────────────────────────────────────────────────────────────────────
# RNA-FM / Attention Branch
# ─────────────────────────────────────────────────────────────────────────────
fm_steps = [
    ("Multi-Head Attn Pooling", r"$4\ \mathrm{heads} \times 640 = 2560$", C["attn"],  "white"),
    (r"Linear $2560 \rightarrow 128$", None,                               "#607D8B",  "white"),
    (r"ELU + Dropout$(0.3)$",          None,                               C["elu"],   "white"),
    (r"Linear $128 \rightarrow 128$",  None,                               "#607D8B",  "white"),
    (r"ELU + Dropout$(0.3)$",          None,                               C["elu"],   "white"),
]

cy = Y0
h_prev = h_inp
fm_boxes = []
for lbl, sub, col, tc in fm_steps:
    h_cur = box_h(sub)
    cy = next_cy(cy, h_prev, h_cur)
    fm_boxes.append((cy, h_cur, lbl, sub, col, tc))
    h_prev = h_cur

for b_cy, b_h, b_lbl, b_sub, b_col, b_tc in fm_boxes:
    rbox(ax, Xfm, b_cy, BW, b_col, b_lbl, sublabel=b_sub, text_color=b_tc)

connect(ax, Xfm, Y0, h_inp, fm_boxes[0][0], fm_boxes[0][1])
for i in range(len(fm_boxes)-1):
    cy_a, h_a = fm_boxes[i][0], fm_boxes[i][1]
    cy_b, h_b = fm_boxes[i+1][0], fm_boxes[i+1][1]
    connect(ax, Xfm, cy_a, h_a, cy_b, h_b)

cy_fm_out = next_cy(fm_boxes[-1][0], fm_boxes[-1][1], H1)
rbox(ax, Xfm, cy_fm_out, BW, "#A85050", r"$\rightarrow (N,\ 128)$")
connect(ax, Xfm, fm_boxes[-1][0], fm_boxes[-1][1], cy_fm_out, H1)
y_fm_bot = cy_fm_out - H1/2

# ─────────────────────────────────────────────────────────────────────────────
# Convergence → Concat → Head
# ─────────────────────────────────────────────────────────────────────────────
y_merge = min(y_seq_bot, y_meta_bot, y_fm_bot) - 0.042
y_bar   = y_merge + 0.018

# Horizontal bar
ax.plot([Xseq, Xfm], [y_bar, y_bar], color=C["arrow"], lw=1.0, zorder=5)
# Drop lines from each branch to the bar
for xb, yb in [(Xseq, y_seq_bot), (Xmeta, y_meta_bot), (Xfm, y_fm_bot)]:
    ax.plot([xb, xb], [yb, y_bar], color=C["arrow"], lw=0.8, zorder=5)

Xctr = (Xseq + Xfm) / 2

# ⊕ merge symbol — sits on the horizontal bar at the centre
r_oplus = 0.013
circ = mpatches.Circle((Xctr, y_bar), radius=r_oplus,
                        facecolor="white", edgecolor=C["arrow"],
                        linewidth=1.0, zorder=7)
ax.add_patch(circ)
ax.text(Xctr, y_bar, "+", ha="center", va="center",
        fontsize=6.5, color=C["arrow"], zorder=8, clip_on=False)

# Concat box — lowered to clear the ⊕ circle
cy_cat = y_bar - r_oplus - 0.012 - H2/2
rbox(ax, Xctr, cy_cat, 0.45, "#5A7280",
     r"Concatenate $[$ TCN $\Vert$ Meta $\Vert$ FM $]$",
     sublabel=r"$(N,\ 288)$")
arr(ax, Xctr, y_bar - r_oplus, cy_cat + H2/2)

# Combined FC
cy_fc = next_cy(cy_cat, H2, H1)
rbox(ax, Xctr, cy_fc, 0.468, C["comb"],
     r"Linear $288{\to}64\ \cdot\ \mathrm{BN}\ \cdot\ \mathrm{ELU}\ \cdot\ \mathrm{Dropout}(0.3)$")
connect(ax, Xctr, cy_cat, H2, cy_fc, H1)

# Output
cy_out = next_cy(cy_fc, H1, H2)
rbox(ax, Xctr, cy_out, 0.342, C["out"],
     r"Linear $64 \rightarrow 1$",
     sublabel="Predicted silencing (%)", bold=True)
connect(ax, Xctr, cy_fc, H1, cy_out, H2)

# ── Branch labels (above each input box) ──────────────────────────────────────
y_label = Y0 + h_inp/2 + 0.012
ax.text(Xseq,  y_label, "Sequence Branch",
        fontsize=8.1, color=C["seq"],  ha="center", va="bottom", style="italic")
ax.text(Xmeta, y_label, "Meta Branch",
        fontsize=8.1, color=C["meta"], ha="center", va="bottom", style="italic")
ax.text(Xfm,   y_label, "Foundation Model Branch",
        fontsize=8.1, color=C["fm"],   ha="center", va="bottom", style="italic")

# ── Dilation bracket ──────────────────────────────────────────────────────────
cy0, h0 = tcn_cy[0]
cy2, h2 = tcn_cy[2]
# Push bracket well left of the skip-connection columns (x=0.005, 0.016)
xbr = Xseq - BW/2 - 0.055  # ≈ −0.014 — outside axes, rendered via clip_on=False
ax.annotate("", xy=(xbr, cy2 - h2/2), xytext=(xbr, cy0 + h0/2),
            arrowprops=dict(arrowstyle="<->", color=C["seq"], lw=0.7),
            annotation_clip=False)
ax.text(xbr - 0.006, (cy0 + cy2)/2, "Dilation\n×1→×4",
        ha="right", va="center", fontsize=4.8, color=C["seq"], clip_on=False)

# ─────────────────────────────────────────────────────────────────────────────
# (B) TCN RESIDUAL BLOCK DETAIL
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_tcn
ax.text(0.01, 0.99, "B", fontsize=9, fontweight="bold",
        va="top", ha="left", transform=ax.transAxes)
ax.text(0.5, 0.97, "TCN Residual Block (×3)", ha="center", va="top",
        fontsize=8.5, fontweight="bold")

bw_t = 0.78    # box width — wider to fill panel
bh_t = 0.052   # box height — squashed to match branch section
gap_t = 0.072  # total step (bh_t + ~0.020 gap)
Xm   = 0.47    # main column X
y0   = 0.83    # starting Y — lowered to give title breathing room

layers_tcn = [
    (r"ConstantPad1d  (causal, left)",           C["bn"],   "white"),
    (r"Conv1d  $k \in \{3,5,7\},\ d \in \{1,2,4\}$", C["conv"], "white"),
    (r"BatchNorm1d",                              C["bn"],   "white"),
    (r"ELU",                                      C["elu"],  "white"),
    (r"Dropout  $(p{=}0.3)$",                    C["drop"], "white"),
    (r"Conv1d  (same settings)",                  C["conv"], "white"),
    (r"BatchNorm1d",                              C["bn"],   "white"),
    (r"ELU",                                      C["elu"],  "white"),
    (r"Dropout  $(p{=}0.3)$",                    C["drop"], "white"),
]

ypos = []
y = y0
for lbl, col, tc in layers_tcn:
    bp = FancyBboxPatch((Xm - bw_t/2, y - bh_t/2), bw_t, bh_t,
                        boxstyle="round,pad=0,rounding_size=0.008",
                        facecolor=_face(col), edgecolor=col, linewidth=0.7,
                        zorder=3)
    ax.add_patch(bp)
    ax.text(Xm, y, lbl, ha="center", va="center", fontsize=7.2,
            color=tc, zorder=4, clip_on=False)
    ypos.append(y)
    if y - gap_t > 0.0:
        arr(ax, Xm, y - bh_t/2, y - gap_t + bh_t/2)
    y -= gap_t

# Residual addition box — y is already one gap_t below the last box centre
y_add = y
_add_col = "#607D8B"
bp = FancyBboxPatch((Xm - bw_t/2, y_add - bh_t/2), bw_t, bh_t,
                    boxstyle="round,pad=0,rounding_size=0.008",
                    facecolor=_face(_add_col), edgecolor=_add_col, linewidth=0.7,
                    zorder=3)
ax.add_patch(bp)
ax.text(Xm, y_add, r"ELU$\,($main $+$ shortcut$)$", ha="center", va="center",
        fontsize=7.2, color="white", zorder=4, clip_on=False)
# Arrow already drawn by the loop on the final iteration

# Skip-connection — LEFT side bypass, enters/exits at box centres
Xsh_L = 0.06
y_top_sh = ypos[0]   # centre of ConstantPad1d (first box)
y_bot_sh = y_add     # centre of ELU(main + shortcut)

_sc_ls = (0, (4, 2))   # dashed style matching panel-A skip connections
# Top horizontal line: box centre-left → skip column
ax.plot([Xm - bw_t/2, Xsh_L], [y_top_sh, y_top_sh], color=C["res"], lw=1.0, ls=_sc_ls)
# Vertical dashed segment (drawn with plot; annotate adds only the arrowhead via a tiny stub)
ax.plot([Xsh_L, Xsh_L], [y_top_sh, y_bot_sh], color=C["res"], lw=1.0, ls=_sc_ls)
ax.annotate("", xy=(Xsh_L, y_bot_sh), xytext=(Xsh_L, y_bot_sh + 0.002),
            arrowprops=dict(arrowstyle="-|>", color=C["res"], lw=1.0,
                            mutation_scale=5, shrinkA=0, shrinkB=0))
# Bottom horizontal line: skip column → box centre-left
ax.plot([Xsh_L, Xm - bw_t/2], [y_bot_sh, y_bot_sh], color=C["res"], lw=1.0, ls=_sc_ls)

# Label to the LEFT of the skip column
ax.text(Xsh_L - 0.025, (y_top_sh + y_bot_sh)/2,
        "Skip-connection\n(1×1 Conv\n+ BN if\nch differ)",
        ha="right", va="center", fontsize=5.0, color=C["res"], clip_on=False)

ax.text(Xm, ypos[0] + bh_t/2 + 0.022, r"$\mathbf{x}\ (N,\ C_\mathrm{in},\ L)$",
        ha="center", va="bottom", fontsize=6.5, color="#555", clip_on=False)
ax.text(Xm, y_add - bh_t/2 - 0.018, r"$\mathbf{out}\ (N,\ C_\mathrm{out},\ L)$",
        ha="center", va="top", fontsize=6.5, color="#555", clip_on=False)

# ─────────────────────────────────────────────────────────────────────────────
# (C) MULTI-HEAD ATTENTION POOLING DETAIL
# ─────────────────────────────────────────────────────────────────────────────
ax = ax_attn
ax.text(0.01, 0.99, "C", fontsize=9, fontweight="bold",
        va="top", ha="left", transform=ax.transAxes)
ax.text(0.5, 0.97, "Multi-Head Attention Pooling", ha="center", va="top",
        fontsize=8.5, fontweight="bold")

bw_a = 0.78    # wider to fill the panel
bh_a = 0.063   # box height — 30% shorter (was 0.090)
gap_a = 0.083  # total step — keeps ~0.020 inter-box gap
Xc   = 0.48    # center X
y    = 0.87    # starting Y

attn_steps = [
    ("Input $\\mathbf{x}$\n$(N,\\ 102,\\ 640)$",                               "#767676"),
    ("Linear $\\rightarrow$ tanh\n$(N,\\ 102,\\ 4{\\times}191)$",              C["attn"]),
    ("Reshape heads\n$(N,\\ 102,\\ 4,\\ 191)$",                                C["attn"]),
    ("Dot-product with context vectors\n$\\rightarrow$ scores $(N,\\ 102,\\ 4)$","#7E5C96"),
    ("Softmax over seq\\_len\n$\\rightarrow$ weights $(N,\\ 102,\\ 4,\\ 1)$",  "#7E5C96"),
    ("Weighted sum of $\\mathbf{x}$\n$\\rightarrow\\ (N,\\ 4,\\ 640)$",        C["fm"]),
    ("Flatten $\\rightarrow\\ (N,\\ 2560)$",                                    C["fm"]),
]

ay_list = []
for i, (lbl, col) in enumerate(attn_steps):
    bp = FancyBboxPatch((Xc - bw_a/2, y - bh_a/2), bw_a, bh_a,
                        boxstyle="round,pad=0,rounding_size=0.008",
                        facecolor=_face(col), edgecolor=col, linewidth=0.7,
                        zorder=3)
    ax.add_patch(bp)
    ax.text(Xc, y, lbl, ha="center", va="center", fontsize=7.0,
            color="white", zorder=4, clip_on=False)
    ay_list.append(y)
    if i < len(attn_steps) - 1:
        arr(ax, Xc, y - bh_a/2, y - gap_a + bh_a/2)
    y -= gap_a

# Learned context vector annotation — points horizontally to right edge of tanh box
ax.annotate("Learned context\nvectors $(4,\\ 191)$",
            xy    = (Xc + bw_a/2,        ay_list[1]),
            xytext= (Xc + bw_a/2 + 0.04, ay_list[1]),
            fontsize=5.0, ha="left", va="center", color="#6B4A85",
            arrowprops=dict(arrowstyle="->", color="#6B4A85", lw=0.7),
            annotation_clip=False, clip_on=False)

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C["seq"],   label="TCN / Sequence"),
    mpatches.Patch(color=C["meta"],  label="Meta features"),
    mpatches.Patch(color=C["fm"],    label="RNA-FM embedding"),
    mpatches.Patch(color=C["attn"],  label="Attention pooling"),
    mpatches.Patch(color="#607D8B", label="Linear (FM branch)"),
    mpatches.Patch(color=C["bn"],    label="BatchNorm / Pad"),
    mpatches.Patch(color=C["elu"],   label="ELU activation"),
    mpatches.Patch(color=C["drop"],  label="Dropout"),
    mpatches.Patch(color=C["comb"],  label="Combined head"),
]
ax_main.legend(handles=legend_items, loc="lower left",
               bbox_to_anchor=(-0.01, -0.06),
               frameon=True, framealpha=0.9, edgecolor="#ccc",
               fontsize=8.0, ncol=3,
               handlelength=1.6, handleheight=1.2,
               columnspacing=1.2, labelspacing=0.4)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/model_architecture.pdf", bbox_inches="tight", dpi=500)
fig.savefig("figures/model_architecture.png", bbox_inches="tight", dpi=500)
print("Saved → figures/model_architecture.pdf + .png")
