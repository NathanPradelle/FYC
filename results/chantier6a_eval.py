"""Extension 6A : transferabilite inter-depots et courbe d'apprentissage.
Protocole : CLAUDE.md §8, 6A (liste figee commit 97d7284 + precisions pre-enregistrees).

Pour chaque depot cible (leave-one-repo-out), queue de test = 30 % chronologiques recents.
Modeles : G (autres depots, restreints aux commits ANTERIEURS au debut de la queue de test),
L(n) (n premiers commits locaux), G+L(n) (union, sans ponderation).
Configuration principale = rang percentile SANS churn ; variantes : brut, +churn.
Un modele local a classe unique = predicteur constant, AUC = 0,5 (pre-enregistre).
Sortie : results/chantier6a_eval.json (toutes les valeurs, par depot/variante/n).
"""
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import brier_score_loss, roc_auc_score

import chantier6a_dataset as ds
from fyc import evaluate

NS = [50, 100, 200, 400, 800, 1600]
T1_AUC, T2_GAIN = 0.65, 0.05


def feature_cols(pct, churn):
    cols = []
    fams = ["BASE", "diff", "radon", "ruff"] + (["churn"] if churn else [])
    for f in fams:
        for c in ds.FAMILIES[f]:
            cols.append(f"{c}__pct" if pct and c not in ds.SCALE_FREE else c)
    return cols


def fit_predict(train, test, cols):
    y = train["risky"]
    if y.nunique() < 2:  # predicteur constant, AUC = 0,5 (pre-enregistre)
        return np.full(len(test), float(y.mean()) if len(y) else 0.0), True
    m = evaluate._make_model(n_estimators=300)
    m.set_params(n_jobs=1)
    m.fit(train[cols], y)
    return m.predict_proba(test[cols])[:, 1], False


def metrics(y, p, const, prev_train, seed=0):
    auc = 0.5 if const else float(roc_auc_score(y, p))
    rng, boots = np.random.default_rng(seed), []
    if not const:
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                boots.append(roc_auc_score(y[i], p[i]))
    top = np.argsort(-p, kind="stable")[: max(1, int(0.1 * len(y)))]
    return {
        "auc": round(auc, 4),
        "ic95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
        if boots else [0.5, 0.5],
        "brier": round(float(brier_score_loss(y, np.clip(p, 0, 1))), 4),
        "brier_const_prev_train": round(float(brier_score_loss(y, np.full(len(y), prev_train))), 4),
        "brier_const_prev_test": round(float(brier_score_loss(y, np.full(len(y), y.mean()))), 4),
        "precision_top10pct": round(float(y[top].mean()), 4),
        "model_constant": bool(const),
    }


def run_target(data, target, pct, churn):
    cols = feature_cols(pct, churn)
    t = data[data.repo == target].sort_values("ts", kind="stable").reset_index(drop=True)
    cut = int(len(t) * (1 - ds.TEST_TAIL))
    pool, test = t.iloc[:cut], t.iloc[cut:]
    y = test["risky"].values
    glob = data[(data.repo != target) & (data.ts < test["ts"].iloc[0])]
    out = {"n_test": len(test), "n_pos_test": int(y.sum()), "n_global_train": len(glob),
           "prevalence_test": round(float(y.mean()), 4), "G": None, "L": {}, "GL": {}}
    p, c = fit_predict(glob, test, cols)
    out["G"] = metrics(y, p, c, float(glob["risky"].mean()))
    for n in sorted(set([x for x in NS if x <= len(pool)] + [len(pool)])):
        loc = pool.iloc[:n]
        p, c = fit_predict(loc, test, cols)
        out["L"][str(n)] = metrics(y, p, c, float(loc["risky"].mean()))
        both = pd.concat([glob, loc], ignore_index=True)
        p, c = fit_predict(both, test, cols)
        out["GL"][str(n)] = metrics(y, p, c, float(both["risky"].mean()))
    return out


def main():
    data = pickle.loads((HERE / "dataset_6a.pkl").read_bytes())
    summ = json.loads((HERE / "chantier6a_dataset.json").read_text())["depots"]
    targets = [r for r, s in summ.items() if s["interpretable"]]
    excluded = [r for r in summ if r not in targets]
    print("cibles interpretables :", targets, "| exclus :", excluded, flush=True)
    variants = {"pct_sans_churn": (True, False), "brut_sans_churn": (False, False),
                "pct_avec_churn": (True, True), "brut_avec_churn": (False, True)}
    jobs = [(v, t) for v in variants for t in targets]
    res = Parallel(n_jobs=3)(delayed(run_target)(data, t, *variants[v]) for v, t in jobs)
    results = {v: {} for v in variants}
    for (v, t), r in zip(jobs, res):
        results[v][t] = r

    def verdict(v):
        r = results[v]
        g = [r[t]["G"]["auc"] for t in r]
        gl_minus_l_100 = [r[t]["GL"]["100"]["auc"] - r[t]["L"]["100"]["auc"] for t in r if "100" in r[t]["L"]]
        nstar = {}
        for t in r:
            ok = [int(n) for n in r[t]["L"] if r[t]["L"][n]["auc"] >= r[t]["GL"][n]["auc"]]
            nstar[t] = min(ok) if ok else None
        vals = [x for x in nstar.values() if x is not None]
        return {
            "T1_mediane_auc_G": round(float(np.median(g)), 4), "T1_ok": bool(np.median(g) >= T1_AUC),
            "T2_mediane_GL_moins_L_n100": round(float(np.median(gl_minus_l_100)), 4),
            "T2_ok": bool(np.median(gl_minus_l_100) >= T2_GAIN),
            "T3_n_etoile_par_depot": nstar,
            "T3_mediane_parmi_depots_ou_croisement_existe": float(np.median(vals)) if vals else None,
            "T3_depots_sans_croisement": [t for t, x in nstar.items() if x is None],
        }
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "protocole": "CLAUDE.md §8 6A",
           "depots_cibles": targets, "depots_exclus_garde_fou": excluded,
           "configuration_principale": "pct_sans_churn",
           "verdicts": {v: verdict(v) for v in variants}, "resultats": results}
    (HERE / "chantier6a_eval.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["verdicts"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
