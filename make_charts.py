"""Regenerate all charts for the Body_Rocking_Behaviour_Recognition README.

Data sources (all verified Nov 18, 2026 from the repo's own files):
- Session stats: time.txt + detection.txt of each session in "Training Data/"
  (rows, duration, rocking prevalence, bout counts — see SESSIONS below).
- graph1: prevalence per session (from the same files).
- graph2: wrist-accelerometer magnitude excerpt from Session01 armIMU/wristIMU
  with the detection annotation overlaid (data/ dir is the scratch copy of the
  repo's "Training Data/Session01/" files; not part of the draft).
- graph3: confusion matrix of verification baseline A (RandomForest, 150 trees,
  trained on the first 70% of Session01, tested on the last 30%),
  from docs/baseline-results.csv.

Run:  python3 make_charts.py   (writes docs/graph*.svg)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(HERE, "docs")
os.makedirs(DOCS, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#5f6368", "axes.labelcolor": "#222222",
    "text.color": "#222222", "xtick.color": "#333333", "ytick.color": "#333333",
    "axes.titlesize": 13, "axes.titleweight": "bold",
})

# ---------------------------------------------------------------- sessions
# rows, duration_min, rocking_pct, bouts — measured from time.txt/detection.txt
# (see docs/session-stats.csv); read from that file so the chart and the
# table can never drift apart.
import csv as _csv
with open(os.path.join(DOCS, "session-stats.csv")) as _fh:
    SESSIONS = [(r["session"], int(r["rows"]), float(r["duration_min"]),
                 float(r["rocking_pct"]), int(r["bouts"]))
                for r in _csv.DictReader(_fh)]

# ------------------------------------------------- graph1: prevalence bars
labels = [s[0] for s in SESSIONS]
prev = [s[3] for s in SESSIONS]
bouts = [s[4] for s in SESSIONS]
fig, ax = plt.subplots(figsize=(8.0, 3.6))
bars = ax.bar(labels, prev, color="#4285F4", width=0.62)
ax.set_ylabel("Samples labelled rocking (%)")
ax.set_title("Rocking prevalence varies widely across sessions")
ax.set_ylim(0, 100)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
ax.set_axisbelow(True)
for b, p, n in zip(bars, prev, bouts):
    ax.text(b.get_x() + b.get_width() / 2, p + 2, f"{p:.1f}%\n({n} bouts)",
            ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(DOCS, "graph1-prevalence.svg"), format="svg")
plt.close(fig)

# --------------------------------------- graph2: signal excerpt + annotation
DATA = os.environ.get("BRB_DATA", os.path.join(HERE, "Training Data"))
try:
    wrist = np.loadtxt(os.path.join(DATA, "Session01", "wristIMU.txt"))
    det = np.loadtxt(os.path.join(DATA, "Session01", "detection.txt"))
    t = np.loadtxt(os.path.join(DATA, "Session01", "time.txt"))
    # first rocking onset after 600 s, 120 s window around it
    on = np.where((det[1:] == 1) & (det[:-1] == 0))[0]
    on = on[t[on] > 600][0]
    i0, i1 = max(0, on - 1500), min(len(t), on + 4500)  # 30 s before, 90 s after
    mag = np.sqrt((wrist[i0:i1, 0:3] ** 2).sum(axis=1))
    tt = t[i0:i1] - t[i0]
    dd = det[i0:i1]
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    ax.plot(tt, mag, color="#4285F4", linewidth=0.9, label="Wrist accel magnitude (g)")
    ax.fill_between(tt, 0, mag.max() * 1.05, where=dd > 0,
                    color="#EA4335", alpha=0.25, label="Annotated rocking")
    ax.set_xlabel("Time (s, excerpt starts 30 s before a rocking onset)")
    ax.set_ylabel("Accel magnitude (g)")
    ax.set_title("Session01: wrist acceleration with rocking annotation")
    ax.set_ylim(0, mag.max() * 1.08)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(color="#e0e0e0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "graph2-signal-excerpt.svg"), format="svg")
    plt.close(fig)
    print("graph2: excerpt at t=%.0f s, %d samples" % (t[on], i1 - i0))
except Exception as exc:  # data dir absent (e.g. published copy): skip graph2
    print("graph2 skipped:", exc)

# ------------------------------------------- graph3: confusion matrix (A)
res_path = os.path.join(DOCS, "baseline-results.csv")
if os.path.exists(res_path):
    with open(res_path) as _fh:
        _rows = list(_csv.DictReader(_fh))
    a = _rows[0]
    cm = np.array([[int(a["tn"]), int(a["fp"])], [int(a["fn"]), int(a["tp"])]])
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nnot rocking", "Predicted\nrocking"])
    ax.set_yticklabels(["True\nnot rocking", "True\nrocking"])
    ax.set_title("Baseline A confusion matrix\n(RandomForest, Session01 time split)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}",
                    ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "#222222")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Samples")
    fig.tight_layout()
    fig.savefig(os.path.join(DOCS, "graph3-confusion-matrix.svg"), format="svg")
    plt.close(fig)
    print("graph3: accuracy %s, precision %s, recall %s"
          % (a["accuracy"], a["precision"], a["recall"]))
else:
    print("graph3 skipped: docs/baseline-results.csv not found")

print("charts written to", DOCS)
