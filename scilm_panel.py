"""Generate an SVG sub-panel depicting the SciLM multi-modal encoder.

Output: scilm_panel.svg — drop into Inkscape via File > Import.
Layout: three modality icons (XRD, PL decay, BLDS) on the left ->
        transformer block in the middle ->
        embedding row + small property-prediction readout on the right.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(4.4, 4.4))
ax.set_xlim(0, 8.8)
ax.set_ylim(0, 8.8)
ax.set_aspect("equal")
ax.axis("off")

# ---------- Modality mini-plots (left column) ----------
# Each modality lives in a small framed inset.
def mini_frame(x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=0.7, ec="#444"))

mod_x, mod_w, mod_h = 0.2, 1.9, 1.55
gap = 0.40
y_xrd  = 6.55
y_pl   = y_xrd - mod_h - gap
y_blds = y_pl  - mod_h - gap

# XRD: a few sharp peaks
mini_frame(mod_x, y_xrd, mod_w, mod_h)
xs = np.linspace(0, 1, 200)
peaks = np.zeros_like(xs)
for c, a, w in [(0.18, 1.0, 0.012), (0.34, 0.7, 0.012),
                (0.52, 0.9, 0.012), (0.71, 0.55, 0.012), (0.86, 0.4, 0.012)]:
    peaks += a * np.exp(-((xs - c) ** 2) / (2 * w ** 2))
peaks = peaks / peaks.max() * (mod_h * 0.55)
ax.plot(mod_x + 0.08 + xs * (mod_w - 0.16), y_xrd + 0.12 + peaks,
        color="#1f77b4", lw=0.9)
ax.text(mod_x + mod_w - 0.08, y_xrd + mod_h - 0.08, "XRD",
        ha="right", va="top", fontsize=10)

# PL decay: exponential on log-y look
mini_frame(mod_x, y_pl, mod_w, mod_h)
xs = np.linspace(0, 1, 200)
decay = np.exp(-3.2 * xs) * (mod_h * 0.75)
ax.plot(mod_x + 0.08 + xs * (mod_w - 0.16), y_pl + 0.10 + decay,
        color="#d62728", lw=0.9)
# scatter to suggest noisy data
rng = np.random.default_rng(0)
sx = np.linspace(0.05, 0.95, 30)
sy = np.exp(-3.2 * sx) * (mod_h * 0.55) + rng.normal(0, 0.012, 30)
ax.scatter(mod_x + 0.08 + sx * (mod_w - 0.16), y_pl + 0.10 + sy,
           s=1.4, color="#d62728", alpha=0.6)
ax.text(mod_x + mod_w - 0.08, y_pl + mod_h - 0.08,
        "SRPL\nTRPL", ha="right", va="top", fontsize=10,
        linespacing=0.95)

# BLDS: a couple of sigmoidal frequency-response curves
mini_frame(mod_x, y_blds, mod_w, mod_h)
xs = np.linspace(0, 1, 200)
for shift, color in [(0.35, "#2ca02c"), (0.55, "#9467bd"), (0.72, "#8c564b")]:
    sig = 1 / (1 + np.exp(-12 * (xs - shift)))
    ax.plot(mod_x + 0.08 + xs * (mod_w - 0.16),
            y_blds + 0.12 + sig * (mod_h * 0.5), color=color, lw=0.8)
ax.text(mod_x + mod_w - 0.08, y_blds + mod_h - 0.08, "BLDS",
        ha="right", va="top", fontsize=10)

# ---------- Transformer block (middle) ----------
trf_x, trf_y, trf_w, trf_h = 3.0, 1.8, 2.9, 4.0
trf = FancyBboxPatch((trf_x, trf_y), trf_w, trf_h,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     fc="#f4f4f8", ec="#333", lw=0.9)
ax.add_patch(trf)
ax.text(trf_x + trf_w / 2, trf_y + trf_h + 0.25,
        "Multi-modal\ntransformer (SciLM)",
        ha="center", va="bottom", fontsize=11, weight="bold")

# Stacked encoder layers inside (3 boxes, larger labels)
n_layers = 3
lay_w = trf_w - 0.5
lay_h = 0.85
v_gap = 0.20
stack_h = n_layers * lay_h + (n_layers - 1) * v_gap
stack_bottom = trf_y + 0.65
for i in range(n_layers):
    ly = stack_bottom + i * (lay_h + v_gap)
    ax.add_patch(FancyBboxPatch(
        (trf_x + 0.25, ly), lay_w, lay_h,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        fc="#dbe7f5", ec="#3c6ea5", lw=0.7))
    ax.text(trf_x + trf_w / 2, ly + lay_h / 2,
            "self-attention", ha="center", va="center",
            fontsize=10, color="#234")
ax.text(trf_x + trf_w / 2, trf_y + 0.32, "× N", ha="center",
        va="center", fontsize=10, style="italic", color="#234")

# ---------- Embedding row (right) ----------
emb_n = 6
emb_w, emb_h = 0.36, 0.55
emb_total = emb_n * emb_w
emb_x = 8.7 - emb_total - 0.05
emb_y = 4.9
emb_vals = [0.6, 0.2, 0.8, 0.45, 0.1, 0.7]
for i, v in enumerate(emb_vals):
    ax.add_patch(Rectangle((emb_x + i * emb_w, emb_y), emb_w, emb_h,
                           fc=plt.cm.Blues(0.25 + 0.55 * v),
                           ec="#234", lw=0.6))
ax.text(emb_x + emb_total / 2, emb_y + emb_h + 0.25,
        "embedding  z", ha="center", va="bottom",
        fontsize=10, style="italic")

# ---------- Property prediction readout (right, below embedding) ----------
pred_w = emb_total + 0.50
pred_h = 2.20
pred_x = emb_x - 0.25
pred_y = 1.80
ax.add_patch(FancyBboxPatch((pred_x, pred_y), pred_w, pred_h,
                            boxstyle="round,pad=0.02,rounding_size=0.08",
                            fc="#fff8e6", ec="#a07a1f", lw=0.7))
ax.text(pred_x + pred_w / 2, pred_y + pred_h - 0.20,
        "property\nprediction",
        ha="center", va="top", fontsize=10, weight="bold", color="#5a4310")

# A tiny bar chart of predicted properties
labels = [r"$\sigma_{ion}$", r"$\tau_{PL}$", "rad"]
heights = [0.55, 0.78, 0.40]
bar_w = 0.34
bar_gap = 0.18
group_w = 3 * bar_w + 2 * bar_gap
base_x = pred_x + (pred_w - group_w) / 2
base_y = pred_y + 0.40
for i, (lab, h) in enumerate(zip(labels, heights)):
    bx = base_x + i * (bar_w + bar_gap)
    ax.add_patch(Rectangle((bx, base_y), bar_w, h * 0.55,
                           fc="#c79a2e", ec="#5a4310", lw=0.5))
    ax.text(bx + bar_w / 2, base_y - 0.06, lab,
            ha="center", va="top", fontsize=9)

# ---------- Arrows ----------
arrow_kw = dict(arrowstyle="-|>", mutation_scale=12, lw=1.0, color="#222")

# modality -> transformer (three arrows merging)
trf_left_mid = (trf_x, trf_y + trf_h / 2)
for ymid in [y_xrd + mod_h / 2, y_pl + mod_h / 2, y_blds + mod_h / 2]:
    ax.add_patch(FancyArrowPatch(
        (mod_x + mod_w + 0.05, ymid), trf_left_mid,
        connectionstyle="arc3,rad=0.0", **arrow_kw))

# transformer -> embedding
ax.add_patch(FancyArrowPatch(
    (trf_x + trf_w + 0.05, trf_y + trf_h * 0.78),
    (emb_x - 0.05, emb_y + emb_h / 2),
    connectionstyle="arc3,rad=0.0", **arrow_kw))

# transformer -> property prediction
ax.add_patch(FancyArrowPatch(
    (trf_x + trf_w + 0.05, trf_y + trf_h * 0.22),
    (pred_x - 0.05, pred_y + pred_h / 2),
    connectionstyle="arc3,rad=0.0", **arrow_kw))

plt.savefig("/Users/mlawler/Code/perovskites/scilm_panel.svg",
            bbox_inches="tight", pad_inches=0.05, transparent=True)
plt.savefig("/Users/mlawler/Code/perovskites/scilm_panel.pdf",
            bbox_inches="tight", pad_inches=0.05, transparent=True)
plt.savefig("/Users/mlawler/Code/perovskites/scilm_panel.png",
            bbox_inches="tight", pad_inches=0.05, dpi=300, transparent=True)
print("wrote scilm_panel.svg, scilm_panel.pdf, scilm_panel.png")
