"""Chantier 11b : heterogeneite inter-depots du transfert (G15 du 10d). Protocole : CLAUDE.md §8 Chantier 11 (84e34fe)."""
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
import chantier6a_eval as ev
import chantier10d_more_repos as d10

import numpy as np
import pandas as pd
from scipy.stats import chi2, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

OUT = HERE / "chantier11b_heterogeneity.json"


def main():
    j = json.loads((HERE / "chantier10d_more_repos.json").read_text())
    ok = [n for n, d in j["depots"].items() if d["interpretable"]]
    parts, summ = [pickle.loads((HERE / "dataset_6a.pkl").read_bytes())], {}
    summ.update(json.loads((HERE / "chantier6a_dataset.json").read_text())["depots"])
    for n, pre in d10.OLD_CACHES + tuple((n, "10d") for n in d10.NEW):
        df, s = pickle.loads((HERE / f"cache_{pre}_{n}.pkl").read_bytes()); parts.append(df); summ[n] = s
    data = pd.concat(parts, ignore_index=True)
    cols = ev.feature_cols(True, False)
    auc = np.array([j["depots"][n]["G15"]["auc"] for n in ok])
    se = np.array([(j["depots"][n]["G15"]["ic95"][1] - j["depots"][n]["G15"]["ic95"][0]) / 3.92 for n in ok])
    w = 1 / se ** 2; mu = float((w * auc).sum() / w.sum())
    Q = float((w * (auc - mu) ** 2).sum()); df_ = len(ok) - 1
    out = {"protocole": "CLAUDE.md §8 Chantier 11b (84e34fe)", "generated_at": datetime.now(timezone.utc).isoformat(),
           "heterogeneite": {"moyenne_ponderee": mu, "Q": Q, "ddl": df_, "p": float(1 - chi2.cdf(Q, df_)),
                             "I2": float(max(0.0, (Q - df_) / Q)), "dans_le_bruit": bool(max(0.0, (Q - df_) / Q) < 0.25)},
           "depots": {}}
    rows = []
    for n, a, s_ in zip(ok, auc, se):
        t = data[data.repo == n].sort_values("ts", kind="stable").reset_index(drop=True)
        cut = int(len(t) * (1 - ds.TEST_TAIL)); pool, test = t.iloc[:cut], t.iloc[cut:]
        y = test["risky"].values
        glob = data[(data.repo != n) & (data.ts < test["ts"].iloc[0])]
        pl, _ = ev.fit_predict(pool, test, cols)
        pgl, _ = ev.fit_predict(pd.concat([glob, pool], ignore_index=True), test, cols)
        auc_l, auc_gl = float(roc_auc_score(y, pl)), float(roc_auc_score(y, pgl))
        # decalage de domaine : test du depot vs jeu d'entrainement de G (echantillon de 5000 max)
        g_s = glob.sample(min(5000, len(glob)), random_state=0)
        X = pd.concat([test[cols], g_s[cols]], ignore_index=True)
        z = np.r_[np.ones(len(test)), np.zeros(len(g_s))]
        rf = RandomForestClassifier(100, n_jobs=3, random_state=0, class_weight="balanced")
        shift = float(cross_val_score(rf, X, z, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean())
        d = {"AUC_G": float(a), "SE": float(s_), "z_vs_moyenne": float((a - mu) / s_), "AUC_L_pre_test": auc_l, "AUC_GL": auc_gl,
             "G_moins_L": float(a - auc_l), "decalage_domaine_AUC": shift,
             "prevalence_test": j["depots"][n]["prevalence_test"], "n_pos_test": j["depots"][n]["n_pos_test"],
             "part_touches_py": summ[n]["part_touches_py"]}
        out["depots"][n] = d; rows.append(d)
        print(n, json.dumps({k: round(v, 3) for k, v in d.items()}), flush=True)
    rho = {}
    for k in ("decalage_domaine_AUC", "prevalence_test", "n_pos_test", "part_touches_py"):
        r, p = spearmanr([d["AUC_G"] for d in rows], [d[k] for d in rows])
        rho[k] = {"rho": float(r), "p": float(p), "retenu_abs_ge_0_6": bool(abs(r) >= 0.6)}
    out["spearman_AUC_G_vs"] = rho
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"heterogeneite": out["heterogeneite"], "spearman": rho}, indent=1))


if __name__ == "__main__":
    main()
