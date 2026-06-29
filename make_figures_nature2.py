"""Nature-style COMPOSITE multi-panel figures (matplotlib only).
Two double-column composites with bold lowercase panel labels, a restrained
genomics/systems palette (neutral greys + one blue family [the GNN/ours] + one
warm family [baselines] + a muted green for 'good/non-degenerate'), direct labels
over legends, hero composition, white background, editable PDF text.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pickle

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.8,
    "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "legend.frameon": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
})
# restrained genomics/systems palette
N = {"gnn": "#2C6FA6", "gnn_lt": "#9DC3E0", "base": "#C99A4E", "drop": "#B5503C",
     "good": "#3F9C7E", "grey": "#AEB4BA", "grey_d": "#5C6066", "chance": "#A8AEB4",
     "ink": "#2A2D31"}


def panel(ax, lab, x=-0.18, y=1.02):
    ax.text(x, y, lab, transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=N["ink"], ha="left", va="bottom")


def chance(ax, x_text, y=0.5, ha="left"):
    ax.axhline(y, ls=(0, (5, 3)), lw=0.7, color=N["chance"], zorder=1)
    ax.annotate("chance", xy=(x_text, y), xytext=(0, 1.5), textcoords="offset points",
                fontsize=6, color=N["grey_d"], ha=ha, va="bottom")


def save(fig, name):
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{name}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


# ============ COMPOSITE 1: logic carries learnable structure (a,b,c) ============
def composite_learn():
    fig = plt.figure(figsize=(6.95, 2.25))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.15, 0.92, 1.25], wspace=0.42,
                           left=0.055, right=0.995, top=0.83, bottom=0.20)

    # (a) identifiability: has-cyclic (grey, secondary) vs reachability (blue, focus)
    ax = fig.add_subplot(gs[0])
    groups = ["topology", "signed\ngraph", "logic-\naware"]
    cyc = [0.558, 0.860, 0.803]; reach = [0.558, 0.566, 0.921]
    x = np.arange(3); w = 0.38
    chance(ax, 2.45, ha="right")
    ax.bar(x - w/2, cyc, w, color=N["grey"], label="has-cyclic")
    ax.bar(x + w/2, reach, w, color=N["gnn"], label="reachability")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0.45, 1.0); ax.set_yticks([0.5, 0.7, 0.9])
    ax.set_ylabel("within-graph AUROC")
    ax.set_title("topology at chance;\nthe logic decides", color=N["ink"])
    ax.text(0.30, 0.92, "reachability", color=N["gnn"], fontsize=6.5, transform=ax.transAxes, fontweight="bold")
    ax.text(0.30, 0.83, "has-cyclic", color=N["grey_d"], fontsize=6.5, transform=ax.transAxes)
    panel(ax, "a", x=-0.30)

    # (b) task-specific: GNN vs all-bits GBDT, two tasks, direct-labeled deltas
    ax = fig.add_subplot(gs[1])
    tasks = ["has-\ncyclic", "reach-\nability"]
    gnn = [0.751, 0.962]; gbdt = [0.803, 0.921]
    x = np.arange(2); w = 0.34
    ax.bar(x - w/2, gnn, w, color=N["gnn"])
    ax.bar(x + w/2, gbdt, w, color=N["base"])
    for xi, (g, f) in enumerate(zip(gnn, gbdt)):
        d = g - f
        ax.annotate(f"{d:+.3f}", xy=(xi, max(g, f)), xytext=(0, 2.5), textcoords="offset points",
                    ha="center", fontsize=6.8, fontweight="bold", color=(N["gnn"] if d > 0 else N["base"]))
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylim(0.6, 1.04); ax.set_yticks([0.6, 0.8, 1.0])
    ax.set_ylabel("within-graph AUROC")
    ax.set_title("GNN wins where\npropagation matters", color=N["ink"])
    ax.text(0.04, 0.93, "GNN", color=N["gnn"], fontsize=6.5, transform=ax.transAxes, fontweight="bold")
    ax.text(0.04, 0.84, "all-bits tree", color=N["base"], fontsize=6.5, transform=ax.transAxes)
    panel(ax, "b", x=-0.34)

    # (c) ablation + CANA: GNN-sign (blue hero) vs greys; in-dist light, transfer dark
    ax = fig.add_subplot(gs[2])
    meth = ["GNN\nsign", "GNN\nno-sgn", "GNN\nMLP", "feat.\ntree", "CANA\ntree"]
    indist = [0.984, 0.963, 0.937, 0.929, 0.887]
    transfer = [0.911, 0.820, np.nan, 0.814, 0.465]
    te = [0.010, 0.022, 0, 0.003, 0.009]
    base_c = [N["gnn"], N["grey"], N["grey"], N["base"], N["drop"]]
    x = np.arange(5); w = 0.40
    chance(ax, 4.45, ha="right")
    # in-distribution = light/open, transfer = filled
    ax.bar(x - w/2, indist, w, color=[N["gnn_lt"] if c == N["gnn"] else "#E2E5E8" for c in base_c],
           edgecolor=base_c, linewidth=0.7)
    m = ~np.isnan(transfer)
    ax.bar(x[m] + w/2, np.array(transfer)[m], w, yerr=np.array(te)[m], capsize=1.4,
           color=[base_c[i] for i in range(5) if m[i]], error_kw={"elinewidth": 0.6})
    ax.annotate("n/a", xy=(2 + w/2, 0.405), ha="center", va="bottom", fontsize=5.6, color=N["grey"])
    ax.set_xticks(x); ax.set_xticklabels(meth)
    ax.set_ylim(0.4, 1.04); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("per-node reach. AUROC")
    ax.set_title("sign-aware propagation,\nnot canalization", color=N["ink"])
    panel(ax, "c", x=-0.24)

    save(fig, "fig_composite_learn")


# ============ COMPOSITE 2: zero-shot transfer + principled task (a,b) ============
def composite_transfer():
    data = pickle.load(open("pernet_transfer.pkl", "rb"))
    nn = np.array([d[0] for d in data], float); auc = np.array([d[1] for d in data], float)
    med, mean = float(np.median(auc)), float(auc.mean())
    fig = plt.figure(figsize=(7.0, 2.4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[2.4, 0.72, 0.42, 2.30], wspace=0.10,
                           left=0.06, right=0.995, top=0.80, bottom=0.22)

    # (a) transfer: scatter + marginal hist
    axs = fig.add_subplot(gs[0])
    axs.scatter(nn, auc, s=11, color=N["gnn"], alpha=0.6, edgecolor="white", linewidth=0.25, zorder=3)
    axs.axhline(med, ls="--", lw=0.8, color=N["drop"], zorder=2)
    axs.axhline(mean, ls="-", lw=0.8, color=N["ink"], zorder=2)
    axs.axhline(0.5, ls=":", lw=0.7, color=N["chance"], zorder=2)
    # direct labels in the empty bottom-right corner (no points there at high n)
    axs.text(0.985, 0.115, f"median {med:.2f}", transform=axs.transAxes, ha="right",
             va="bottom", fontsize=6, color=N["drop"])
    axs.text(0.985, 0.03, f"mean {mean:.2f}", transform=axs.transAxes, ha="right",
             va="bottom", fontsize=6, color=N["ink"])
    axs.annotate("chance", xy=(nn.max(), 0.5), xytext=(0, 1.5), textcoords="offset points",
                 fontsize=6, color=N["grey_d"], ha="right", va="bottom")
    axs.set_xlabel("network size $n$ (genes)"); axs.set_ylabel("per-net GNN AUROC")
    axs.set_ylim(0.28, 1.03); axs.set_xlim(0, nn.max() * 1.05); axs.set_yticks([0.4, 0.6, 0.8, 1.0])
    axs.set_title("zero-shot transfer holds across sizes  (118 real GRNs)", color=N["ink"], loc="left")
    panel(axs, "a", x=-0.14)
    axh = fig.add_subplot(gs[1], sharey=axs)
    axh.hist(auc, bins=np.linspace(0.3, 1.0, 15), orientation="horizontal",
             color=N["gnn_lt"], edgecolor="white", linewidth=0.3)
    axh.axhline(med, ls="--", lw=0.8, color=N["drop"]); axh.axhline(mean, ls="-", lw=0.8, color=N["ink"])
    axh.set_xlabel("# nets"); axh.spines["left"].set_visible(False)
    plt.setp(axh.get_yticklabels(), visible=False); axh.tick_params(axis="y", length=0)

    # (b) driving conditions: only quiescence non-degenerate  (gs[3]; gs[2] is a spacer)
    axb = fig.add_subplot(gs[3])
    conds = ["quiescent\n[ours]", "stimulated", "free-input", "silenceable"]
    base = [0.34, 0.82, 0.88, 0.93]
    cols = [N["good"], N["grey"], N["grey"], N["grey"]]
    axb.axhspan(0.2, 0.7, color=N["good"], alpha=0.12, zorder=0)
    axb.text(-0.44, 0.605, "informative\n(non-degenerate)", fontsize=5.8, color=N["good"],
             ha="left", va="center", linespacing=1.0)
    axb.bar(range(4), base, 0.62, color=cols)
    for i, b in enumerate(base):
        axb.text(i, b + 0.02, f"{b:.2f}", ha="center", va="bottom", fontsize=6.6,
                 color=(N["good"] if i == 0 else N["grey_d"]))
    axb.set_xticks(range(4)); axb.set_xticklabels(conds)
    axb.set_ylim(0, 1.06); axb.set_yticks([0, 0.5, 1.0])
    axb.set_ylabel("mean fraction of genes reachable")
    axb.set_title("only quiescence yields a non-degenerate task", color=N["ink"])
    panel(axb, "b", x=-0.215)

    save(fig, "fig_composite_transfer")


if __name__ == "__main__":
    composite_learn(); composite_transfer()
    print("wrote fig_composite_learn, fig_composite_transfer (pdf+png)")
