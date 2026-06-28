# AttractorNet

Code and data for **"Logic-Aware Graph Neural Networks Predict Asynchronous
Reachability in Boolean Gene-Regulatory Networks and Transfer Zero-Shot to Real
Networks"** (Kun Zhang, University of Leeds).

A sign-aware, size-agnostic graph neural network reads the *local Boolean logic* of a
Boolean network and predicts **per-node asynchronous activation reachability** — which
genes can switch on from a quiescent state. Trained only on small synthetic Boolean
networks, it transfers **zero-shot** to real curated gene-regulatory networks
(biodivine-boolean-models), predicting per-node activation reachability at pooled
AUROC ≈ 0.91 (95% CI [0.88, 0.93]), beating size-agnostic feature and CANA
effective-connectivity baselines. All labels come from exact symbolic computation
(state-transition-graph enumeration; `biodivine_aeon` BDD reachability) — no human
annotation, no external API — and everything runs on a single GPU.

## Setup

```bash
pip install -r requirements.txt   # numpy, scikit-learn, networkx, torch, scipy,
                                  # matplotlib, biobalm, biodivine_aeon
```

`biobalm`/`biodivine_aeon` install pre-built wheels (incl. `clingo`); no system deps
needed. Training/inference uses CPU or Apple MPS (`torch`); the GNN is tiny.

## Pipeline / key scripts

| script | what it does |
|---|---|
| `attractor_gonogo.py` | exact state-transition-graph oracle (sync attractors; structural features) |
| `decisive_gate.py` | de-confounded within-graph identifiability control; flattened-truth-table ceiling |
| `gnn_sparse.py` | sparse, size-agnostic, arity-invariant sign-aware GNN |
| `day3_pernode_syn.py` | per-node activation-reachability task on synthetic nets |
| `day3_realtransfer.py` | synthetic→real BBM extraction + zero-shot transfer |
| `build_real_cache2.py` | process-isolated symbolic labelling of real BBM nets → `real_cache.pkl` |
| `eval_transfer.py` | transfer evaluation (3 seeds) on the cached real nets |
| `ablation_signaware.py` | sign / no-sign / MLP inductive-bias ablation |
| `cana_baseline.py` | CANA effective-connectivity baseline |
| `boot_ci.py` | cluster bootstrap CIs + failure-tail analysis |
| `make_figures.py` | paper figures (`fig_1..4.pdf/png`) |
| `vignette_final.py` | budding-yeast cell-cycle application vignette |
| `probe_second_task.py` | driving-condition saturation probe (justifies the quiescent task) |

## Reproducing the headline results

```bash
python build_real_cache2.py        # label real BBM nets -> real_cache.pkl (provided)
python eval_transfer.py            # zero-shot transfer AUROC (3 seeds)
python boot_ci.py                  # cluster-bootstrap CI + failure-tail
python ablation_signaware.py       # sign-aware ablation (in-dist + transfer)
python cana_baseline.py            # CANA baseline (in-dist + transfer)
python make_figures.py --gen       # figures
```

`real_cache.pkl` (the symbolically-labelled 178 real GRNs) and `pernet_transfer.pkl`
are included so the transfer/bootstrap results reproduce without re-running the
(slower) symbolic labelling.

## Data

Real networks are the public **biodivine-boolean-models** collection, accessed via
`biodivine_aeon.BiodivineBooleanModels`. No data are redistributed here beyond the
derived label cache.

## Paper

`attractornet_tcbb.pdf` / `attractornet_tcbb.tex` (IEEE/ACM TCBB format).

## License

MIT (see `LICENSE`).
