import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi":200,"savefig.dpi":200,"font.size":8,"axes.titlesize":9,
 "axes.labelsize":8,"legend.fontsize":7,"xtick.labelsize":6.5,"ytick.labelsize":7,
 "axes.spines.top":False,"axes.spines.right":False,"font.family":"DejaVu Sans",
 "axes.linewidth":0.8,"legend.frameon":False,"figure.constrained_layout.use":True})
conds = ["quiescent\n(inputs OFF)\n[ours]","stimulated\n(inputs ON)","free-input\n(some signal)","silenceable\n(from all-ON)"]
base = [0.34, 0.82, 0.88, 0.93]
green="#009E73"; red="#D55E00"
cols=[green,red,red,red]
fig,ax=plt.subplots(figsize=(3.3,2.2))
ax.axhspan(0.2,0.7,color="#009E73",alpha=.10,zorder=0)
ax.text(3.4,0.45,"informative\n(non-degenerate)",fontsize=5.6,color=green,ha="right",va="center")
bars=ax.bar(range(4),base,0.62,color=cols)
for i,b in enumerate(base):
    ax.text(i,b+0.02,f"{b:.2f}",ha="center",fontsize=7)
ax.set_xticks(range(4)); ax.set_xticklabels(conds); ax.set_ylim(0,1.05)
ax.set_ylabel("mean fraction of genes reachable")
ax.set_title("Only quiescence yields a non-degenerate task")
fig.savefig("fig_driving.pdf",bbox_inches="tight"); fig.savefig("fig_driving.png",bbox_inches="tight")
print("wrote fig_driving.pdf/.png")
