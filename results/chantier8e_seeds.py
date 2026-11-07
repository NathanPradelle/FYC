"""Chantier 8e : multi-graines du 6A (+ variance de graine sur la reserve 4bis/5).
Protocole : CLAUDE.md §8 Chantier 8 (commit f7e8489). Graine de foret 42 = chantier6a_eval.json
(deja faite) ; nouvelles graines 0, 1, 2, 3. Regle : un verdict est STABLE s'il est identique dans
au moins 4 graines sur 5.
"""
import json
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

OUT = HERE / "chantier8e_seeds.json"
SEEDS = [0, 1, 2, 3]
VARIANTS = {"pct_sans_churn": (True, False), "brut_sans_churn": (False, False),
            "pct_avec_churn": (True, True), "brut_avec_churn": (False, True)}


def run_one(data, target, pct, churn, seed):
    import chantier6a_eval as ev
    from fyc import evaluate
    def mk(n_estimators=300, random_state=seed):
        return RandomForestClassifier(n_estimators=n_estimators, random_state=seed,
                                      class_weight="balanced", n_jobs=-1)
    evaluate._make_model = mk
    return ev.run_target(data, target, pct, churn)


def verdict(r):
    g = [r[t]["G"]["auc"] for t in r]
    d100 = [r[t]["GL"]["100"]["auc"] - r[t]["L"]["100"]["auc"] for t in r if "100" in r[t]["L"]]
    nstar = {}
    for t in r:
        ok = [int(n) for n in r[t]["L"] if r[t]["L"][n]["auc"] >= r[t]["GL"][n]["auc"]]
        nstar[t] = min(ok) if ok else None
    vals = [x for x in nstar.values() if x is not None]
    return {"T1_mediane_auc_G": float(np.median(g)), "T1_ok": bool(np.median(g) >= 0.65),
            "T2_mediane_GL_moins_L_n100": float(np.median(d100)), "T2_ok": bool(np.median(d100) >= 0.05),
            "T3_mediane": float(np.median(vals)) if vals else None,
            "T3_depots_sans_croisement": [t for t, x in nstar.items() if x is None],
            "auc_G_par_depot": {t: r[t]["G"]["auc"] for t in r}}


def reserve_seeds():
    import chantier4bis_common as common
    data, fams = common.load_all()
    cut = int(len(data) * 0.8)
    dev, test = data.iloc[:cut], data.iloc[cut:]
    allc = [c for f in fams for c in fams[f]]
    out = {}
    for name, cols in (("BASE", fams["BASE"]), ("tout_empiler", allc)):
        a = []
        for s in range(20):
            m = RandomForestClassifier(n_estimators=300, random_state=s, class_weight="balanced", n_jobs=3)
            m.fit(dev[cols], dev["risky"])
            a.append(roc_auc_score(test["risky"], m.predict_proba(test[cols])[:, 1]))
        out[name] = {"moyenne": float(np.mean(a)), "ecart_type": float(np.std(a, ddof=1)),
                     "min": float(min(a)), "max": float(max(a))}
    d = np.array([x for x in []])
    return out


def main():
    t0 = time.time()
    data = pickle.loads((HERE / "dataset_6a.pkl").read_bytes())
    summ = json.loads((HERE / "chantier6a_dataset.json").read_text())["depots"]
    targets = [r for r, s in summ.items() if s["interpretable"]]
    ref = json.loads((HERE / "chantier6a_eval.json").read_text())["verdicts"]
    res = {}
    if OUT.exists():
        res = json.loads(OUT.read_text()).get("par_graine", {})
    for seed in SEEDS:
        if str(seed) in res:
            continue
        jobs = [(v, t) for v in VARIANTS for t in targets]
        r = Parallel(n_jobs=3)(delayed(run_one)(data, t, *VARIANTS[v], seed) for v, t in jobs)
        results = {v: {} for v in VARIANTS}
        for (v, t), x in zip(jobs, r):
            results[v][t] = x
        res[str(seed)] = {v: verdict(results[v]) for v in VARIANTS}
        OUT.write_text(json.dumps({"statut": "en cours", "par_graine": res}, indent=1))
        print(f"graine {seed} ok ({time.time() - t0:.0f}s)", {v: (round(res[str(seed)][v]['T1_mediane_auc_G'], 3), round(res[str(seed)][v]['T2_mediane_GL_moins_L_n100'], 3)) for v in VARIANTS}, flush=True)
    res["42"] = {v: {"T1_mediane_auc_G": ref[v]["T1_mediane_auc_G"], "T1_ok": ref[v]["T1_ok"],
                     "T2_mediane_GL_moins_L_n100": ref[v]["T2_mediane_GL_moins_L_n100"], "T2_ok": ref[v]["T2_ok"]}
                 for v in VARIANTS}
    synth = {}
    for v in VARIANTS:
        t1 = [res[s][v]["T1_mediane_auc_G"] for s in res]; t2 = [res[s][v]["T2_mediane_GL_moins_L_n100"] for s in res]
        t1ok = [res[s][v]["T1_ok"] for s in res]; t2ok = [res[s][v]["T2_ok"] for s in res]
        synth[v] = {"T1_auc_G_par_graine": t1, "T1_etendue": [min(t1), max(t1)], "T1_ok_sur_5": int(sum(t1ok)),
                    "T1_stable": bool(sum(t1ok) >= 4 or sum(t1ok) <= 1),
                    "T2_par_graine": t2, "T2_etendue": [min(t2), max(t2)], "T2_ok_sur_5": int(sum(t2ok)),
                    "T2_stable": bool(sum(t2ok) >= 4 or sum(t2ok) <= 1)}
    out = {"protocole": "CLAUDE.md §8 Chantier 8e (f7e8489)", "generated_at": datetime.now(timezone.utc).isoformat(),
           "graines": [42] + SEEDS, "par_graine": res, "synthese": synth}
    print(json.dumps(synth, indent=1))
    out["reserve_variance_graine_20"] = reserve_seeds()
    print(json.dumps(out["reserve_variance_graine_20"], indent=1))
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
