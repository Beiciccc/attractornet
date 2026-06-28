"""Probe two candidate SECOND driving conditions for non-degeneracy on real BBM nets:
 (A) STIMULATED ACTIVATION: inputs fixed ON; from all-OFF, can gene v reach 1?
 (B) SILENCEABILITY: inputs fixed OFF; from all-ON, can gene v reach 0?
Report base-rate distribution; pick the one with spread (non-degenerate) for the full pipeline."""
import time, multiprocessing as mp
import numpy as np


def worker(i, q):
    try:
        import biodivine_aeon as ba
        m = ba.BiodivineBooleanModels.fetch_model(i)
        out = {}
        # (A) stimulated activation: inputs TRUE, reach from all-false, can v be 1
        bnA = m.to_bn_inputs_true(); gA = ba.AsynchronousGraph(bnA); nm = bnA.variable_names()
        rA = ba.Reachability.reach_fwd(gA, gA.mk_subspace({x: False for x in nm})).vertices()
        yA = np.array([int(not rA.intersect(gA.mk_subspace_vertices({x: True})).is_empty()) for x in nm])
        out['A_base'] = float(yA.mean())
        # (B) silenceability: inputs FALSE, reach from all-true, can v be 0
        bnB = m.to_bn_inputs_false(); gB = ba.AsynchronousGraph(bnB); nm2 = bnB.variable_names()
        rB = ba.Reachability.reach_fwd(gB, gB.mk_subspace({x: True for x in nm2})).vertices()
        yB = np.array([int(not rB.intersect(gB.mk_subspace_vertices({x: False})).is_empty()) for x in nm2])
        out['B_base'] = float(yB.mean())
        out['n'] = bnA.variable_count()
        q.put(("ok", i, out))
    except Exception as e:
        q.put(("err", i, repr(e)[:50]))


def main():
    import biodivine_aeon as ba
    ids = ba.BiodivineBooleanModels.fetch_ids()
    ctx = mp.get_context("spawn"); A, B = [], []
    done = 0
    for i in ids:
        if done >= 40:
            break
        q = ctx.Queue(); p = ctx.Process(target=worker, args=(i, q)); p.start(); p.join(20)
        if p.is_alive():
            p.terminate(); p.join(); continue
        try:
            r = q.get_nowait()
        except Exception:
            continue
        if r[0] == "ok" and r[2]['n'] <= 50:
            o = r[2]; A.append(o['A_base']); B.append(o['B_base']); done += 1
            print(f"  id={i:>3} n={o['n']:>3} | (A)stim-activation base={o['A_base']:.2f} | (B)silenceability base={o['B_base']:.2f}", flush=True)
    A, B = np.array(A), np.array(B)
    nd = lambda x: int(((x >= 0.15) & (x <= 0.85)).sum())
    print(f"\n=== {len(A)} nets ===")
    print(f"  (A) stimulated-activation : mean base {A.mean():.2f}, non-degenerate[0.15,0.85] {nd(A)}/{len(A)}, saturated(>0.9) {(A>0.9).sum()}")
    print(f"  (B) silenceability        : mean base {B.mean():.2f}, non-degenerate[0.15,0.85] {nd(B)}/{len(B)}, saturated(>0.9) {(B>0.9).sum()}")
    print(f"  -> pick the task with MORE non-degenerate nets and base near 0.3-0.6.")


if __name__ == "__main__":
    main()
