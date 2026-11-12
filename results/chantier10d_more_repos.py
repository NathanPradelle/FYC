"""Chantier 10d / extension 6E : 4 depots supplementaires (werkzeug, rich, marshmallow, alembic) -> 15 depots.
G15 = foret (config principale 6A) entrainee sur tous les AUTRES depots avant la queue de test de la cible.
Protocole : CLAUDE.md §8 Chantier 10 (75cb0ab). Usage : build <depot> | eval
"""
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chantier6a_dataset as ds
import chantier6a_eval as ev

import numpy as np
import pandas as pd

NEW = {n: ds.ROOT / "repos" / n for n in ("werkzeug", "rich", "marshmallow", "alembic")}
OLD_CACHES = (("datasets", "6c"), ("ignite", "6c"), ("poetry", "6d"), ("fastapi", "6d"))


def build(name):
    cache = HERE / f"cache_10d_{name}.pkl"
    if cache.exists():
        print(f"[{name}] cache"); return
    df, s = ds.build_repo(name, NEW[name])
    cache.write_bytes(pickle.dumps((df, s)))
    for f in HERE.glob(f"cache_6a_{name}__*.pkl"):
        f.unlink()
    print(json.dumps(s, ensure_ascii=False), flush=True)


def evaluate():
    parts, summ = [pickle.loads((HERE / "dataset_6a.pkl").read_bytes())], {}
    summ.update(json.loads((HERE / "chantier6a_dataset.json").read_text())["depots"])
    for n, pre in OLD_CACHES + tuple((n, "10d") for n in NEW):
        df, s = pickle.loads((HERE / f"cache_{pre}_{n}.pkl").read_bytes())
        parts.append(df); summ[n] = s
    data = pd.concat(parts, ignore_index=True)
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "protocole": "CLAUDE.md §8 Chantier 10d (75cb0ab)",
           "depots": {}}
    cols = ev.feature_cols(True, False)
    for name in data.repo.unique():
        t = data[data.repo == name].sort_values("ts", kind="stable").reset_index(drop=True)
        cut = int(len(t) * (1 - ds.TEST_TAIL)); test = t.iloc[cut:]; y = test["risky"].values
        glob = data[(data.repo != name) & (data.ts < test["ts"].iloc[0])]
        p, c = ev.fit_predict(glob, test, cols)
        g = ev.metrics(y, p, c, float(glob["risky"].mean()))
        out["depots"][name] = {"interpretable": bool(summ[name]["interpretable"]), "n_pos_test": int(y.sum()),
                               "prevalence_test": round(float(y.mean()), 4), "n_train": len(glob), "G15": g,
                               "lift_top10": round(g["precision_top10pct"] / float(y.mean()), 2),
                               "T1_ok": bool(g["auc"] >= ev.T1_AUC), "part_risque": summ[name]["part_risque"]}
        print(name, json.dumps(out["depots"][name]), flush=True)
    ok = {n: d for n, d in out["depots"].items() if d["interpretable"]}
    aucs = np.array([d["G15"]["auc"] for d in ok.values()])
    rng = np.random.default_rng(0)
    med = [np.median(aucs[rng.integers(0, len(aucs), len(aucs))]) for _ in range(10000)]
    out["synthese"] = {"depots_interpretables": list(ok), "exclus": [n for n in out["depots"] if n not in ok],
                       "mediane_AUC": float(np.median(aucs)),
                       "ic95_mediane_bootstrap_depots": [float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))],
                       "min": float(aucs.min()), "max": float(aucs.max()),
                       "n_depots_T1_ok": int(sum(d["T1_ok"] for d in ok.values())), "n_depots": len(ok),
                       "positifs_test_total": int(sum(d["n_pos_test"] for d in ok.values())),
                       "mediane_ok_T1": bool(np.median(aucs) >= ev.T1_AUC and np.percentile(med, 2.5) > 0.5)}
    (HERE / "chantier10d_more_repos.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out["synthese"], indent=1))


if __name__ == "__main__":
    {"build": lambda: build(sys.argv[2]), "eval": evaluate}[sys.argv[1]]()
