"""Submission-grade reconstruction of the manuscript's 5 data figures
(nature-figure contract; matplotlib only). Colour-blind-safe Okabe-Ito palette,
editable PDF text (pdf.fonttype 42), restrained per-figure palette, statistics as
part of the figure, and explicit readability fixes (no label clipping, no legend
over data, no text-marker overlap). Outputs SAME filenames so the LaTeX picks them up.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "svg.fonttype": "none",
    "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 6.8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
    "legend.frameon": False, "figure.constrained_layout.use": True,
    "axes.titlepad": 4,
})
# Okabe-Ito colour-blind-safe
OK = {"blue": "#0072B2", "orange": "#E69F00", "verm": "#D55E00",
      "green": "#009E73", "sky": "#56B4E9", "grey": "#9A9A9A", "k": "#222222"}
CHANCE = "#555555"


def save(fig, name):
    fig.savefig(f"{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def chance_line(ax, x0, x1, y=0.5):
    ax.axhline(y, ls=(0, (4, 3)), lw=0.8, color=CHANCE, zorder=1)
    ax.annotate("chance", xy=(x0, y), xytext=(0, 2), textcoords="offset points",
                fontsize=6, color=CHANCE, ha="left", va="bottom")


# ---- Fig 1: identifiability (topology at chance; logic decides) ----
def fig_identifiability():
    groups = ["topology\n(wiring)", "signed\ngraph", "logic-aware\n(all bits)"]
    cyc = [0.558, 0.860, 0.803]; reach = [0.558, 0.566, 0.921]
    x = np.arange(3); w = 0.38
    fig, ax = plt.subplots(figsize=(3.15, 2.05))
    chance_line(ax, -0.5, 2.5)
    b1 = ax.bar(x - w/2, cyc, w, color=OK["green"], edgecolor="white", linewidth=0.5, label="has-cyclic")
    b2 = ax.bar(x + w/2, reach, w, color=OK["blue"], edgecolor="white", linewidth=0.5, label="async reachability")
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylim(0.45, 1.0); ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("within-graph AUROC")
    ax.set_title("Topology is at chance; the logic decides")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.1, borderaxespad=0.2)
    save(fig, "fig_1_identifiability")


# ---- Fig 4 (taskspecific): GNN wins only where propagation matters ----
def fig_taskspecific():
    tasks = ["has-cyclic\n(global)", "async reachability\n(propagation)"]
    gnn = [0.751, 0.962]; gbdt = [0.803, 0.921]
    x = np.arange(2); w = 0.34
    fig, ax = plt.subplots(figsize=(3.15, 2.1))
    ax.bar(x - w/2, gnn, w, color=OK["blue"], edgecolor="white", linewidth=0.5, label="GNN (sign-aware)")
    ax.bar(x + w/2, gbdt, w, color=OK["orange"], edgecolor="white", linewidth=0.5, label="all-bits GBDT")
    for xi, (g, f) in enumerate(zip(gnn, gbdt)):
        d = g - f
        ax.annotate(f"{d:+.3f}", xy=(xi, max(g, f)), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, fontweight="bold",
                    color=(OK["blue"] if d > 0 else OK["orange"]))
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylim(0.6, 1.02); ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("within-graph AUROC")
    ax.set_title("GNN wins only where propagation matters")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), handlelength=1.1,
              borderaxespad=0.2, labelspacing=0.3)
    save(fig, "fig_4_taskspecific")


# ---- Fig 3 (pernet transfer): scatter + marginal histogram ----
def fig_pernet():
    data = pickle.load(open("pernet_transfer.pkl", "rb"))
    n = np.array([d[0] for d in data], float); auc = np.array([d[1] for d in data], float)
    med = float(np.median(auc)); mean = float(auc.mean())
    fig, (axs, axh) = plt.subplots(1, 2, figsize=(3.32, 1.72),
                                   gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.06})
    # scatter
    axs.scatter(n, auc, s=9, color=OK["blue"], alpha=0.65, edgecolor="white", linewidth=0.25, zorder=3)
    axs.axhline(med, ls="--", lw=0.9, color=OK["verm"], zorder=2)
    axs.axhline(0.5, ls=":", lw=0.8, color=CHANCE, zorder=2)
    axs.annotate("chance", xy=(n.max(), 0.5), xytext=(0, 2), textcoords="offset points",
                 fontsize=6, color=CHANCE, ha="right", va="bottom")
    axs.set_xlabel("network size $n$ (genes)"); axs.set_ylabel("per-net GNN AUROC")
    axs.set_ylim(0.28, 1.03); axs.set_xlim(0, n.max() * 1.05)
    axs.set_title(f"Transfer holds across sizes ({len(n)} real GRNs)", fontsize=8)
    # histogram (shared y)
    axh.hist(auc, bins=np.linspace(0.3, 1.0, 15), orientation="horizontal",
             color=OK["sky"], edgecolor="white", linewidth=0.3)
    axh.axhline(med, ls="--", lw=0.9, color=OK["verm"])
    axh.axhline(mean, ls="-", lw=0.9, color=OK["k"])
    axh.set_ylim(0.28, 1.03); axh.set_yticks([]); axh.set_xlabel("# nets")
    axh.spines["left"].set_visible(False)
    axh.annotate(f"median {med:.2f}", xy=(axh.get_xlim()[1], med), xytext=(-1, 2),
                 textcoords="offset points", ha="right", va="bottom", fontsize=6, color=OK["verm"])
    axh.annotate(f"mean {mean:.2f}", xy=(axh.get_xlim()[1], mean), xytext=(-1, -2),
                 textcoords="offset points", ha="right", va="top", fontsize=6, color=OK["k"])
    save(fig, "fig_3_pernet_transfer")


# ---- Fig 2 (ablation + CANA): sign-aware propagation, not canalization ----
def fig_ablation():
    methods = ["GNN\n(sign)", "GNN\n(no-sign)", "GNN\n(MLP)", "feature\nGBDT", "CANA\nGBDT"]
    indist = [0.984, 0.963, 0.937, 0.929, 0.887]
    indist_e = [0.001, 0.002, 0.002, 0.003, 0.002]
    transfer = [0.911, 0.820, np.nan, 0.814, 0.465]
    transfer_e = [0.010, 0.022, 0, 0.003, 0.009]
    x = np.arange(5); w = 0.38
    fig, ax = plt.subplots(figsize=(3.32, 2.25))
    chance_line(ax, -0.5, 4.5)
    ax.bar(x - w/2, indist, w, yerr=indist_e, capsize=1.6, color=OK["blue"],
           edgecolor="white", linewidth=0.4, error_kw={"elinewidth": 0.7}, label="in-distribution")
    tv = np.array([v if v == v else 0 for v in transfer])
    mask = ~np.isnan(transfer)
    ax.bar(x[mask] - w/2 + w, tv[mask], w, yerr=np.array(transfer_e)[mask], capsize=1.6,
           color=OK["orange"], edgecolor="white", linewidth=0.4,
           error_kw={"elinewidth": 0.7}, label="zero-shot transfer (real GRN)")
    ax.annotate("n/a", xy=(2 + w/2, 0.41), ha="center", va="bottom", fontsize=6, color=OK["grey"])
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylim(0.4, 1.02); ax.set_yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel("per-node reachability AUROC")
    ax.set_title("Sign-aware propagation, not canalization")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2,
              handlelength=1.1, columnspacing=1.4, borderaxespad=0.0)
    save(fig, "fig_2_ablation_cana")


# ---- Fig driving: only quiescence is non-degenerate ----
def fig_driving():
    conds = ["quiescent\n(inputs OFF)\n[ours]", "stimulated\n(inputs ON)",
             "free-input\n(some signal)", "silenceable\n(from all-ON)"]
    base = [0.34, 0.82, 0.88, 0.93]
    cols = [OK["green"], OK["verm"], OK["verm"], OK["verm"]]
    fig, ax = plt.subplots(figsize=(3.2, 2.05))
    ax.axhspan(0.2, 0.7, color=OK["green"], alpha=0.10, zorder=0)
    # band label placed in the empty upper area over the (short) quiescent bar — no overlap
    ax.annotate("informative\n(non-degenerate)", xy=(0, 0.58), fontsize=6, color=OK["green"],
                ha="center", va="center", linespacing=1.0)
    ax.bar(range(4), base, 0.62, color=cols, edgecolor="white", linewidth=0.5)
    for i, b in enumerate(base):
        ax.text(i, b + 0.02, f"{b:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(4)); ax.set_xticklabels(conds)
    ax.set_ylim(0, 1.05); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("mean fraction of genes reachable")
    ax.set_title("Only quiescence yields a non-degenerate task")
    save(fig, "fig_driving")


if __name__ == "__main__":
    fig_identifiability(); fig_taskspecific(); fig_pernet(); fig_ablation(); fig_driving()
    print("regenerated: fig_1_identifiability, fig_4_taskspecific, fig_3_pernet_transfer, "
          "fig_2_ablation_cana, fig_driving (pdf+png)")
