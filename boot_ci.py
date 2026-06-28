"""P1 stats fixes: (a) honest failure-tail composition; (b) cluster bootstrap CI over
the 118 real networks for pooled GNN AUROC, feature-GBDT AUROC, and the paired
difference (accounts for within-network correlation, per Reviewer 1)."""
import pickle
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from ablation_signaware import FlexGNN, train as train_gnn
from day3_pernode_syn import build as build_pernode
from day3_realtransfer import gnn_predict, build_synth

cache = pickle.load(open("real_cache.pkl", "rb"))
both = [r for r in cache if r["both"] == 1]

# (a) failure tail from pernet_transfer.pkl
pn = pickle.load(open("pernet_transfer.pkl", "rb"))   # list of (n, auc)
tail = sorted([(n, a) for n, a in pn if a < 0.7])
ns = np.array([n for n, _ in tail])
print(f"(a) FAILURE TAIL (per-net AUROC < 0.7): {len(tail)}/{len(pn)} nets")
print(f"    sizes n: {sorted(ns.tolist())}")
print(f"    n>=15: {(ns>=15).sum()}/{len(ns)}; median n={int(np.median(ns))}; "
      f"worst: AUROC {min(a for _,a in tail):.3f} at n={[n for n,a in tail if a==min(x for _,x in tail)][0]}")
print(f"    Spearman(n, auc) over all {len(pn)} nets: ", end="")
alln = np.array([n for n,_ in pn]); alla=np.array([a for _,a in pn])
from scipy.stats import spearmanr
rho,p = spearmanr(alln, alla); print(f"rho={rho:+.3f} (p={p:.2f}) -> size {'does NOT' if p>0.05 else 'does'} explain the tail")

# (b) cluster bootstrap: train GNN + feature-GBDT, per-node preds, resample NETS
graphs, *_ = build_pernode([12,14],[2,3,4],40,4)
np.random.seed(0); torch.manual_seed(0)
net = train_gnn(graphs, np.arange(len(graphs)), "sign", 60)
gA = build_synth([12,14],[2,3,4],40,4,seed=7)
gb = HistGradientBoostingClassifier(max_iter=400).fit(gA[1], gA[2])

pg = [gnn_predict(net,(r["X"],r["src"],r["dst"],r["sgn"],r["y"])) for r in both]   # per-net GNN preds
pf = [gb.predict_proba(r["X"])[:,1] for r in both]                                  # per-net GBDT preds
ys = [r["y"] for r in both]
Yall=np.concatenate(ys); Pg=np.concatenate(pg); Pf=np.concatenate(pf)
print(f"\n(b) point estimates: pooled GNN {roc_auc_score(Yall,Pg):.3f}  GBDT {roc_auc_score(Yall,Pf):.3f}")

rs=np.random.RandomState(1); B=2000; gnn_b=[]; gb_b=[]; diff_b=[]
idx=np.arange(len(both))
for _ in range(B):
    samp=rs.choice(idx,len(idx),replace=True)
    Y=np.concatenate([ys[i] for i in samp]); G=np.concatenate([pg[i] for i in samp]); F=np.concatenate([pf[i] for i in samp])
    if len(np.unique(Y))<2: continue
    ag=roc_auc_score(Y,G); af=roc_auc_score(Y,F); gnn_b.append(ag); gb_b.append(af); diff_b.append(ag-af)
ci=lambda a:(np.percentile(a,2.5),np.percentile(a,97.5))
print(f"    GNN  pooled AUROC {np.mean(gnn_b):.3f}  95% CI [{ci(gnn_b)[0]:.3f}, {ci(gnn_b)[1]:.3f}]")
print(f"    GBDT pooled AUROC {np.mean(gb_b):.3f}  95% CI [{ci(gb_b)[0]:.3f}, {ci(gb_b)[1]:.3f}]")
print(f"    paired diff GNN-GBDT {np.mean(diff_b):+.3f}  95% CI [{ci(diff_b)[0]:+.3f}, {ci(diff_b)[1]:+.3f}]  "
      f"(excludes 0: {ci(diff_b)[0]>0})")
pernet=np.array([a for _,a in pn]); print(f"    per-net AUROC: median {np.median(pernet):.3f}, IQR [{np.percentile(pernet,25):.3f},{np.percentile(pernet,75):.3f}], mean {pernet.mean():.3f}")
