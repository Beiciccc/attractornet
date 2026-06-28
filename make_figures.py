"""Submission-grade figures for AttractorNet (AAAI two-column).
Vector PDF + PNG preview. Numbers are the validated session results; per-net
transfer data (Fig 3) is regenerated from real_cache.pkl + a trained sign-aware GNN.
"""
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "font.family": "DejaVu Sans", "axes.linewidth": 0.8,
    "legend.frameon": False, "figure.constrained_layout.use": True,
})
# Okabe-Ito colourblind-safe
C = {"gnn": "#0072B2", "nosign": "#56B4E9", "mlp": "#999999", "gbdt": "#E69F00",
     "cana": "#D55E00", "topo": "#999999", "logic": "#009E73", "flat": "#CC79A7"}
W = 3.3   # column width inches


def save(fig, name):
    fig.savefig(f"fig_{name}.pdf", bbox_inches="tight")
    fig.savefig(f"fig_{name}.png", bbox_inches="tight")
    plt.close(fig); print(f"  wrote fig_{name}.pdf / .png", flush=True)


# ---------------- Fig 1: identifiability (feature groups) ----------------
def fig1():
    groups = ["topology\n(wiring only)", "signed\ngraph", "logic-aware\n(all bits)"]
    hascyc = [0.558, 0.860, 0.803]      # within-graph de-confounded (flat-TT for logic-aware)
    reach = [0.558, 0.566, 0.921]       # topology/signed at chance for reachability; logic predicts
    x = np.arange(len(groups)); w = 0.38
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.bar(x - w/2, hascyc, w, label="has-cyclic", color=C["logic"])
    ax.bar(x + w/2, reach, w, label="async reachability", color=C["gnn"])
    ax.axhline(0.5, ls="--", lw=0.8, color="k", alpha=.6)
    ax.text(2.35, 0.515, "chance", fontsize=6, color="k", alpha=.7)
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.set_ylim(0.45, 1.0)
    ax.set_ylabel("within-graph AUROC"); ax.legend(loc="upper left")
    ax.set_title("Topology is at chance; the logic decides")
    save(fig, "1_identifiability")


# ---------------- Fig 2: inductive-bias + CANA ablation (money figure) ----------------
def fig2():
    methods = ["GNN\n(sign)", "GNN\n(no-sign)", "GNN\n(MLP)", "feature\nGBDT", "CANA\nGBDT"]
    indist = [0.984, 0.963, 0.937, 0.929, 0.887]
    indist_e = [0.001, 0.002, 0.002, 0.003, 0.002]
    transfer = [0.911, 0.820, np.nan, 0.814, 0.465]
    transfer_e = [0.010, 0.022, 0, 0.003, 0.009]
    cols = [C["gnn"], C["nosign"], C["mlp"], C["gbdt"], C["cana"]]
    x = np.arange(len(methods)); w = 0.38
    fig, ax = plt.subplots(figsize=(W, 2.4))
    ax.bar(x - w/2, indist, w, yerr=indist_e, capsize=2, color=cols, alpha=.55, label="in-distribution")
    tv = [v if v == v else 0 for v in transfer]
    ax.bar(x + w/2, tv, w, yerr=transfer_e, capsize=2, color=cols, label="zero-shot transfer (real)",
           hatch="///", edgecolor="white", linewidth=0)
    ax.axhline(0.5, ls="--", lw=0.8, color="k", alpha=.6)
    ax.text(4.0, 0.515, "chance", fontsize=6, alpha=.7)
    ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("per-node reachability AUROC")
    ax.set_title("Sign-aware propagation, not canalization")
    # legend by shading
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#777", alpha=.55, label="in-distribution"),
                       Patch(facecolor="#777", hatch="///", label="zero-shot real GRN")],
              loc="lower left")
    save(fig, "2_ablation_cana")


# ---------------- Fig 3: per-net transfer (regenerated) ----------------
def fig3():
    try:
        data = pickle.load(open("pernet_transfer.pkl", "rb"))
    except FileNotFoundError:
        print("  pernet_transfer.pkl missing — run gen_pernet() first"); return
    n = np.array([d[0] for d in data]); auc = np.array([d[1] for d in data])
    fig, axes = plt.subplots(1, 2, figsize=(W*1.7, 2.2), gridspec_kw={"width_ratios": [2, 1]})
    axes[0].scatter(n, auc, s=14, color=C["gnn"], alpha=.7, edgecolor="white", linewidth=.3)
    axes[0].axhline(np.median(auc), ls="--", lw=0.9, color=C["cana"], label=f"median {np.median(auc):.2f}")
    axes[0].axhline(0.5, ls=":", lw=0.8, color="k", alpha=.5)
    axes[0].set_xlabel("network size $n$ (genes)"); axes[0].set_ylabel("per-net GNN AUROC")
    axes[0].set_ylim(0.3, 1.02); axes[0].legend(loc="lower right")
    axes[0].set_title(f"Transfer holds across sizes ({len(n)} real GRNs)")
    axes[1].hist(auc, bins=np.linspace(0.3, 1.0, 15), color=C["gnn"], alpha=.8, orientation="horizontal")
    axes[1].axhline(np.median(auc), ls="--", lw=0.9, color=C["cana"])
    axes[1].axhline(auc.mean(), ls="-", lw=0.9, color="k", label=f"mean {auc.mean():.2f}")
    axes[1].set_xlabel("# nets"); axes[1].set_ylim(0.3, 1.02); axes[1].legend(loc="lower right")
    axes[1].set_yticklabels([])
    save(fig, "3_pernet_transfer")


# ---------------- Fig 4: has-cyclic vs reachability (task-specific bias) ----------------
def fig4():
    tasks = ["has-cyclic\n(global)", "async reachability\n(propagation)"]
    gnn = [0.751, 0.962]; flat = [0.803, 0.921]
    x = np.arange(len(tasks)); w = 0.36
    fig, ax = plt.subplots(figsize=(W, 2.2))
    ax.bar(x - w/2, gnn, w, label="GNN (sign)", color=C["gnn"])
    ax.bar(x + w/2, flat, w, label="all-bits GBDT", color=C["gbdt"])
    for xi, (g, f) in enumerate(zip(gnn, flat)):
        ax.annotate(f"{g-f:+.3f}", (xi, max(g, f)+0.012), ha="center", fontsize=6.5,
                    color=(C["gnn"] if g > f else C["gbdt"]))
    ax.set_xticks(x); ax.set_xticklabels(tasks); ax.set_ylim(0.6, 1.0)
    ax.set_ylabel("within-graph AUROC"); ax.legend(loc="lower right")
    ax.set_title("GNN wins only where propagation matters")
    save(fig, "4_taskspecific")


def gen_pernet():
    """Regenerate per-net transfer AUROC + size from real_cache.pkl + a sign-aware GNN."""
    import torch
    from ablation_signaware import FlexGNN, train as train_gnn
    from day3_pernode_syn import build as build_pernode
    from day3_realtransfer import gnn_predict
    cache = pickle.load(open("real_cache.pkl", "rb"))
    both = [r for r in cache if r["both"] == 1]
    graphs, *_ = build_pernode([12, 14], [2, 3, 4], 40, 4)
    np.random.seed(0); torch.manual_seed(0)
    net = train_gnn(graphs, np.arange(len(graphs)), "sign", 60)
    out = []
    for r in both:
        p = gnn_predict(net, (r["X"], r["src"], r["dst"], r["sgn"], r["y"]))
        out.append((int(r["n"]), float(roc_auc_score(r["y"], p))))
    pickle.dump(out, open("pernet_transfer.pkl", "wb"))
    print(f"  per-net data: {len(out)} nets, median AUROC {np.median([o[1] for o in out]):.3f}", flush=True)


if __name__ == "__main__":
    import sys
    if "--gen" in sys.argv:
        gen_pernet()
    print("[figures]", flush=True)
    fig1(); fig2(); fig3(); fig4()
    print("done.", flush=True)
