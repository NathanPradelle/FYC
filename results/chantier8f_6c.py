"""Chantier 8f / extension 6C : le modele global G (config principale 6A : rang percentile, sans churn,
entraine sur les 7 depots d'origine avant le debut de la queue de test) transfere-t-il a des depots data/ML ?
Protocole : CLAUDE.md §8 Chantier 8 (f7e8489). Depots figes : huggingface/datasets, pytorch/ignite.
Les 7 depots d'origine ne sont ni modifies ni re-entraines avec les nouveaux. Pas de T2/T3.
Usage : python results/chantier8f_6c.py build <depot> | eval
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

import pandas as pd

REPOS_6C = {"datasets": ds.ROOT / "repos" / "datasets", "ignite": ds.ROOT / "repos" / "ignite"}


def build(name):
    cache = HERE / f"cache_6c_{name}.pkl"
    if cache.exists():
        print(f"[{name}] cache"); return
    df, s = ds.build_repo(name, REPOS_6C[name])
    cache.write_bytes(pickle.dumps((df, s)))
    for f in HERE.glob(f"cache_6a_{name}__*.pkl"):
        f.unlink()
    print(json.dumps(s, ensure_ascii=False), flush=True)


def evaluate():
    orig = pickle.loads((HERE / "dataset_6a.pkl").read_bytes())
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "protocole": "CLAUDE.md §8 8f (f7e8489)",
           "depots": {}}
    for name in REPOS_6C:
        df, s = pickle.loads((HERE / f"cache_6c_{name}.pkl").read_bytes())
        data = pd.concat([orig, df], ignore_index=True)
        r = ev.run_target(data, name, True, False)  # G seul vaut r["G"] ; L/GL calcules mais non juges
        res = {"resume": s, "G": r["G"], "n_test": r["n_test"], "n_pos_test": r["n_pos_test"],
               "prevalence_test": r["prevalence_test"], "n_global_train": r["n_global_train"]}
        res["interpretable"] = bool(s["interpretable"])
        res["T1_auc_ge_0_65"] = bool(r["G"]["auc"] >= ev.T1_AUC)
        g = r["G"]
        res["lift_top10"] = round(g["precision_top10pct"] / r["prevalence_test"], 2)
        res["brier_G_pire_que_constante_test"] = bool(g["brier"] > g["brier_const_prev_test"])
        out["depots"][name] = res
        print(name, json.dumps({k: v for k, v in res.items() if k != "resume"}), flush=True)
    (HERE / "chantier8f_6c.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    {"build": lambda: build(sys.argv[2]), "eval": evaluate}[sys.argv[1]]()
