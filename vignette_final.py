import numpy as np, torch
import biodivine_aeon as ba
from sklearn.metrics import roc_auc_score
from ablation_signaware import FlexGNN, train as train_gnn
from day3_pernode_syn import build as build_pernode
from amortization import cheap_feats
from day3_realtransfer import gnn_predict

graphs, *_ = build_pernode([12,14],[2,3,4],40,4)
np.random.seed(0); torch.manual_seed(0)
net = train_gnn(graphs, np.arange(len(graphs)), "sign", 60)
m = ba.BiodivineBooleanModels.fetch_model('24'); bn = m.to_bn_inputs_false()
names=[nm[2:] if nm.startswith('v_') else nm for nm in bn.variable_names()]; n=bn.variable_count()
g=ba.AsynchronousGraph(bn); init=g.mk_subspace({nm:False for nm in bn.variable_names()})
reached=ba.Reachability.reach_fwd(g,init).vertices()
y=np.array([int(not reached.intersect(g.mk_subspace_vertices({nm:True})).is_empty()) for nm in bn.variable_names()],float)
X,src,dst,sgn=cheap_feats(bn); logit=gnn_predict(net,(X,src,dst,sgn,np.zeros(n,np.float32)))
prob=1/(1+np.exp(-logit)); auc=roc_auc_score(y,prob); acc=((logit>0).astype(int)==y).mean()
print(f"{m.name} n={n} base={y.mean():.2f} AUROC={auc:.3f} acc(logit>0)={acc:.2f} activatable={int(y.sum())}/{n}")
print(f"pub {m.url_publication}\n")
print(f"{'gene':<12}{'p(activatable)':>16}{'exact':>8}{'ok':>4}")
for j in np.argsort(-prob):
    ok = '+' if (prob[j]>0.5)==bool(y[j]) else 'x'
    print(f"{names[j]:<12}{prob[j]:>16.3f}{int(y[j]):>8}{ok:>4}")
