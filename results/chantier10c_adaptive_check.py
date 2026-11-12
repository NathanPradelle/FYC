"""Chantier 10c : fenetre adaptative sous le protocole du Chantier 1 (M2bis + queues identiques).
Protocole : CLAUDE.md §8 Chantier 10 (75cb0ab). Reutilise results/cache_9d_<depot>.pkl (blame_par_commit, 7 depots du 6A).
M1 non refait : il ne depend pas de la fenetre (evenements de blame des fix, pas des positifs).
"""
import json
import pickle
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chantier6a_dataset as ds
import chantier9a_fix_strict as a9
import chantier9d_adaptive_window as w9

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fyc import evaluate, features, git_extract, labeling

OUT = HERE / "chantier10c_adaptive_check.json"
SEUIL = 0.9044
WIN = w9.WINDOWS


def cv_oof(df, cols, seed):
    m = evaluate._make_model(n_estimators=300); m.set_params(n_jobs=1)
    X, y = df[cols], df["risky"]
    oof = np.zeros(len(y)); aucs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m.fit(X.iloc[tr], y.iloc[tr]); oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], oof[te]))
    return float(np.mean(aucs)), oof


def one_seed(df, seed):
    b, ob = cv_oof(df, features.BASE_COLS, seed); c, oc = cv_oof(df, features.CHURN_COLS, seed)
    return b, c


def m2bis(commits, bpc, first, last, f):
    est = w9.make_est(bpc, first, last, f)
    df = pd.concat([features.build_base_features(commits, est), features.build_churn_features(commits, est)], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]
    res = Parallel(n_jobs=3)(delayed(one_seed)(df, s) for s in range(20))
    ratios = [c / b for b, c in res]
    _, ob = cv_oof(df, features.BASE_COLS, 0); _, oc = cv_oof(df, features.CHURN_COLS, 0)
    y = df["risky"].values; rng = np.random.default_rng(0); rat = []
    while len(rat) < 1000:
        j = rng.integers(0, len(y), len(y))
        if y[j].min() != y[j].max():
            rat.append(roc_auc_score(y[j], oc[j]) / roc_auc_score(y[j], ob[j]))
    ic = [float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5))]
    verdict = "satisfait" if ic[1] <= SEUIL else ("echoue" if ic[0] > SEUIL else "indetermine")
    return {"n": len(df), "ratio_moyen_20_graines": float(np.mean(ratios)), "ratio_std": float(np.std(ratios)),
            "ratio_ponctuel_OOF": float(roc_auc_score(y, oc) / roc_auc_score(y, ob)), "ic95_bootstrap_OOF": ic, "verdict": verdict}


def common_queue(name, path, data):
    commits = git_extract.extraire_commits(str(path))
    labeling.tag_fixes(commits, fix_predicate=ds.FIX_PREDICATE)
    bpc = pickle.loads((HERE / f"cache_9d_{name}.pkl").read_bytes())
    first, last = commits[0]["ts"], commits[-1]["ts"]
    ests = {wn: w9.make_est(bpc, first, last, f) for wn, f in WIN.items()}
    by_hash = {c["hash"]: c for c in commits}
    d = data[data.repo == name].sort_values("ts", kind="stable").reset_index(drop=True)
    labs = {wn: [ests[wn](by_hash[h]) for h in d["hash"]] for wn in WIN}
    keep = [i for i in range(len(d)) if all(labs[wn][i] is not None for wn in WIN)]
    d = d.iloc[keep].reset_index(drop=True)
    cut = int(len(d) * 0.7)
    out = {"n_intersection": len(d), "n_test": len(d) - cut}
    for wn in WIN:
        y = np.array([labs[wn][i] for i in keep], dtype=int)
        m = evaluate._make_model(n_estimators=300); m.set_params(n_jobs=3)
        m.fit(d[features.BASE_COLS].iloc[:cut], y[:cut])
        out[wn] = {"AUC_BASE": float(roc_auc_score(y[cut:], m.predict_proba(d[features.BASE_COLS].iloc[cut:])[:, 1])),
                   "n_pos_test": int(y[cut:].sum()), "part_risque": float(y.mean())}
    return out


def main():
    out = {"protocole": "CLAUDE.md §8 Chantier 10c (75cb0ab)", "generated_at": datetime.now(timezone.utc).isoformat()}
    commits, bpc = w9.load_repo("flask", ds.REPOS["flask"])
    first, last = commits[0]["ts"], commits[-1]["ts"]
    out["M2bis_flask"] = {}
    for wn, f in WIN.items():
        out["M2bis_flask"][wn] = m2bis(commits, bpc, first, last, f)
        print("M2bis", wn, json.dumps(out["M2bis_flask"][wn]), flush=True)
    data = pickle.loads((HERE / "dataset_6a.pkl").read_bytes())
    out["queues_identiques"] = {}
    for name, path in ds.REPOS.items():
        out["queues_identiques"][name] = common_queue(name, path, data)
        print(name, json.dumps(out["queues_identiques"][name]), flush=True)
    syn = {}
    rng = np.random.default_rng(0)
    for wn in ("adapt_f0.10", "adapt_f0.25"):
        delta = np.array([out["queues_identiques"][n][wn]["AUC_BASE"] - out["queues_identiques"][n]["fixe_90j"]["AUC_BASE"]
                          for n in ds.REPOS])
        med = [np.median(delta[rng.integers(0, len(delta), len(delta))]) for _ in range(10000)]
        m2 = out["M2bis_flask"][wn]["verdict"]
        syn[wn] = {"delta_par_depot": dict(zip(ds.REPOS, map(float, delta))), "mediane_delta": float(np.median(delta)),
                   "ic95_mediane": [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))],
                   "n_depots_delta_lt_moins_0_03": int((delta < -0.03).sum()), "M2bis_verdict": m2,
                   "acceptable_provisoirement": bool(m2 != "echoue" and np.median(delta) >= -0.03 and (delta < -0.03).sum() <= 2)}
    out["synthese"] = syn
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(syn, indent=1))


if __name__ == "__main__":
    main()
