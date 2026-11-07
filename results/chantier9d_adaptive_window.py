"""Chantier 9d : fenetre de label adaptative W(t) = max(90 j, f x age du depot), f in {0,1 ; 0,25}, contre 90 j fixes.
Protocole : CLAUDE.md §8 Chantier 9 (91990a2). 7 depots du 6A (label_v2 + detection B) ; Flask : AUC(BASE), M2bis.
Les commits dont ts + W(t) depasse le dernier commit sont exclus (etiquette non observable), y compris pour 90 j fixes.
"""
import bisect
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

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from fyc import evaluate, features, git_extract, labeling

DAY = 86400
WINDOWS = {"fixe_90j": 0.0, "adapt_f0.10": 0.10, "adapt_f0.25": 0.25}
OUT = HERE / "chantier9d_adaptive_window.json"


def make_est(bpc, first_ts, last_ts, f):
    def est(c):
        if c["is_fix"]:
            return None
        t = c["ts"]
        w = max(90 * DAY, f * (t - first_ts))
        if t + w > last_ts:
            return None
        dates = bpc.get(c["hash"])
        if not dates:
            return 0
        i = bisect.bisect_right(dates, t)
        return 1 if (i < len(dates) and dates[i] <= t + w) else 0
    return est


def load_repo(name, path):
    cache = HERE / f"cache_9d_{name}.pkl"
    commits = git_extract.extraire_commits(str(path))
    if cache.exists():
        bpc = pickle.loads(cache.read_bytes())
        labeling.tag_fixes(commits, fix_predicate=ds.FIX_PREDICATE)
        return commits, bpc
    bpc, _bf, _e = labeling.label_commits_v2(commits, str(path), fenetre_jours=90, fix_predicate=ds.FIX_PREDICATE)
    cache.write_bytes(pickle.dumps(dict(bpc)))
    return commits, bpc


def deciles(commits, est):
    lab = [(c["ts"], est(c)) for c in commits if est(c) is not None and c["files"]]
    y = np.array([l for _, l in lab], dtype=float)
    parts = np.array_split(np.arange(len(y)), 10)
    return [float(y[p].mean()) for p in parts], float(y.mean()), len(y)


def main():
    out = {"protocole": "CLAUDE.md §8 Chantier 9d (91990a2)", "generated_at": datetime.now(timezone.utc).isoformat(),
           "depots": {}}
    flask = None
    for name, path in ds.REPOS.items():
        commits, bpc = load_repo(name, path)
        first, last = commits[0]["ts"], commits[-1]["ts"]
        res = {}
        for wn, f in WINDOWS.items():
            est = make_est(bpc, first, last, f)
            dec, part, n = deciles(commits, est)
            ratio = dec[0] / dec[-1] if dec[-1] > 0 else None
            res[wn] = {"n_etiquetes": n, "part_risque": part, "deciles": dec, "ratio_1re_sur_derniere": ratio}
        out["depots"][name] = res
        print(name, {w: (round(r["part_risque"], 3), None if r["ratio_1re_sur_derniere"] is None else round(r["ratio_1re_sur_derniere"], 2))
                     for w, r in res.items()}, flush=True)
        if name == "flask":
            flask = (commits, bpc, first, last)
    syn = {}
    for wn in WINDOWS:
        rs = [out["depots"][n][wn]["ratio_1re_sur_derniere"] for n in out["depots"]]
        rs = [np.inf if r is None else r for r in rs]
        syn[wn] = {"mediane_ratio": float(np.median(rs)),
                   "n_depots_part_risque_ge_5pct": int(sum(out["depots"][n][wn]["part_risque"] >= 0.05 for n in out["depots"]))}
    out["synthese_derive"] = syn
    # --- Flask : AUC(BASE) CV, queue temporelle 30 %, M2bis ---
    commits, bpc, first, last = flask
    fl = {}
    for wn, f in WINDOWS.items():
        est = make_est(bpc, first, last, f)
        base = features.build_base_features(commits, est)
        churn = features.build_churn_features(commits, est)
        df = pd.concat([base, churn], axis=1).loc[:, lambda d: ~d.columns.duplicated()]
        bs = Parallel(n_jobs=3)(delayed(a9.cv_aucs)(df, features.BASE_COLS, s) for s in range(20))
        cs = Parallel(n_jobs=3)(delayed(a9.cv_aucs)(df, features.CHURN_COLS, s) for s in range(20))
        cut = int(len(df) * 0.7)
        m = evaluate._make_model(n_estimators=300); m.set_params(n_jobs=3)
        m.fit(df[features.BASE_COLS].iloc[:cut], df["risky"].iloc[:cut])
        tail = float(roc_auc_score(df["risky"].iloc[cut:], m.predict_proba(df[features.BASE_COLS].iloc[cut:])[:, 1]))
        fl[wn] = {"n": len(df), "AUC_BASE_cv": float(np.mean(bs)), "AUC_BASE_cv_std_graines": float(np.std(bs)),
                  "AUC_BASE_queue30": tail, "M2bis_ratio_moyen": float(np.mean([c / b for b, c in zip(bs, cs)])),
                  "n_pos_queue": int(df["risky"].iloc[cut:].sum())}
        print("flask", wn, json.dumps(fl[wn]), flush=True)
    out["flask"] = fl
    ref = syn["fixe_90j"]["mediane_ratio"]
    for wn in ("adapt_f0.10", "adapt_f0.25"):
        s = syn[wn]
        s["reduit_la_derive"] = bool(s["mediane_ratio"] <= 2.0 and s["n_depots_part_risque_ge_5pct"] >= 6)
        s["acceptable_pour_le_modele"] = bool(fl["fixe_90j"]["AUC_BASE_queue30"] - fl[wn]["AUC_BASE_queue30"] <= 0.03)
        s["mediane_ratio_90j_reference"] = ref
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(syn, indent=1))


if __name__ == "__main__":
    main()
