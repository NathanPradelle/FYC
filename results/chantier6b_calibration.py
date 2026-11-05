"""Extension 6B : recalibration locale d'un modele global (protocole CLAUDE.md §8, 6B,
fige au commit 7ca5564 avant cette execution).

Pour chaque depot cible : G = modele de la configuration principale du 6A (rang percentile,
sans churn), entraine sur les autres depots (commits anterieurs au debut du test). Ses
probabilites brutes sont ensuite recalibrees avec n commits etiquetes locaux :
R0 brut ; R1 correction de prior (odds' = odds(s) * pi/(1-pi)) ; R2 Platt (repli R1 si < 5
positifs) ; R-oracle = R1 avec la vraie prevalence du test (borne haute, inutilisable).
Fenetres de calibration : 'recente' (n derniers commits anterieurs a test_start - 90 j) =
principale ; 'ancienne' (n premiers commits) = diagnostic de derive.
Sortie : results/chantier6b_calibration.json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

import chantier6a_dataset as ds
import chantier6a_eval as ev
from fyc import evaluate

NS = [25, 50, 100, 200, 400, 800]
GAP_S = 90 * 86400
MIN_POS_PLATT = 5


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def pi_hat(y):
    return (float(np.sum(y)) + 1) / (len(y) + 2)


def r1(s, pi):
    return sigmoid(logit(s) + np.log(pi / (1 - pi)))  # prior effectif de G = 0,5 (balanced)


def r2(s, y_cal, s_cal):
    if int(np.sum(y_cal)) < MIN_POS_PLATT or int(np.sum(1 - y_cal)) < MIN_POS_PLATT:
        return None
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s_cal).reshape(-1, 1), y_cal)
    return m.predict_proba(logit(s).reshape(-1, 1))[:, 1]


def ece(p, y, k=5):
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, k)
    return float(sum(len(b) / len(y) * abs(p[b].mean() - y[b].mean()) for b in bins if len(b)))


def scores(p, y, pi_fair, auc_ref):
    b = float(brier_score_loss(y, np.clip(p, 0, 1)))
    b_fair = float(brier_score_loss(y, np.full(len(y), pi_fair)))
    return {"brier": round(b, 4), "bss_vs_constante_loyale": round(1 - b / b_fair, 4),
            "ece5": round(ece(p, y), 4), "brier_constante_loyale": round(b_fair, 4),
            "auc": round(float(roc_auc_score(y, p)), 4), "auc_ecart_vs_brut": round(float(roc_auc_score(y, p)) - auc_ref, 5)}


def run_target(data, target):
    cols = ev.feature_cols(pct=True, churn=False)
    t = data[data.repo == target].sort_values("ts", kind="stable").reset_index(drop=True)
    cut = int(len(t) * (1 - ds.TEST_TAIL))
    pool, test = t.iloc[:cut], t.iloc[cut:]
    y = test["risky"].values.astype(int)
    t0 = float(test["ts"].iloc[0])
    glob = data[(data.repo != target) & (data.ts < t0)]
    s_all_pool, _ = ev.fit_predict(glob, pd.concat([pool, test]), cols)
    s_pool, s = s_all_pool[:len(pool)], s_all_pool[len(pool):]
    auc_ref = float(roc_auc_score(y, s))
    prev_test = float(y.mean())
    out = {"n_test": len(test), "n_pos_test": int(y.sum()), "prevalence_test": round(prev_test, 4),
           "prevalence_historique_pool": round(float(pool["risky"].mean()), 4),
           "auc_brut": round(auc_ref, 4),
           "R0_brut": {"brier": round(float(brier_score_loss(y, s)), 4), "ece5": round(ece(s, y), 4),
                       "brier_constante_oracle": round(float(brier_score_loss(y, np.full(len(y), prev_test))), 4)},
           "R_oracle": scores(r1(s, prev_test), y, prev_test, auc_ref), "fenetres": {}}
    ypool = pool["risky"].values.astype(int)
    eligible = np.where(pool["ts"].values <= t0 - GAP_S)[0]  # trou de 90 j (labels sans fuite du test)
    for nom in ("recente", "ancienne"):
        res = {}
        for n in NS:
            if nom == "recente":
                idx = eligible[-n:] if len(eligible) >= n else None
            else:
                idx = np.arange(n) if n <= len(pool) else None
            if idx is None or len(idx) < n:
                continue
            yc, sc = ypool[idx], s_pool[idx]
            pi = pi_hat(yc)
            e = {"n_pos_calibration": int(yc.sum()), "pi_hat": round(pi, 4),
                 "R1_prior": scores(r1(s, pi), y, pi, auc_ref)}
            p2 = r2(s, yc, sc)
            e["R2_platt"] = scores(p2, y, pi, auc_ref) if p2 is not None else None
            # Regle pre-enregistree : un ecart d'AUC > 0,001 invalide la mesure (Platt peut
            # inverser le classement si la pente estimee est negative, faute de positifs).
            if e["R2_platt"] is not None and abs(e["R2_platt"]["auc_ecart_vs_brut"]) > 0.001:
                e["R2_platt_INVALIDE_auc_modifiee"] = e["R2_platt"]
                e["R2_platt"] = None
            e["R2_repli_sur_R1"] = p2 is None
            res[str(n)] = e
        out["fenetres"][nom] = res
    return out


def main():
    data = pickle.loads((HERE / "dataset_6a.pkl").read_bytes())
    targets = list(ds.REPOS)
    from joblib import Parallel, delayed
    res = Parallel(n_jobs=3)(delayed(run_target)(data, t) for t in targets)
    R = dict(zip(targets, res))

    def med(f):
        v = [x for x in map(f, targets) if x is not None]
        return round(float(np.median(v)), 4) if v else None

    invalides = [(t, w, n) for t, r in R.items() for w, wd in r["fenetres"].items()
                 for n, e in wd.items() if "R2_platt_INVALIDE_auc_modifiee" in e]
    summ = {"cellules_platt_invalidees_auc_modifiee": invalides,
            "R1_auc_max_ecart": max(abs(e["R1_prior"]["auc_ecart_vs_brut"]) for r in R.values()
                                    for wd in r["fenetres"].values() for e in wd.values())}
    c1 = {"mediane_brier_R0_brut": med(lambda t: R[t]["R0_brut"]["brier"]),
          "mediane_brier_R1_n100_recente": med(lambda t: R[t]["fenetres"]["recente"]["100"]["R1_prior"]["brier"])}
    c1["C1_ok"] = c1["mediane_brier_R1_n100_recente"] < c1["mediane_brier_R0_brut"]
    c2 = {}
    for nom in ("recente", "ancienne"):
        for meth in ("R1_prior", "R2_platt"):
            c2[f"{nom}/{meth}"] = {}
            for n in NS:
                v = [R[t]["fenetres"][nom].get(str(n), {}).get(meth) for t in targets]
                v = [x["bss_vs_constante_loyale"] for x in v if x]
                if len(v) == len(targets):
                    c2[f"{nom}/{meth}"][str(n)] = {"mediane_BSS": round(float(np.median(v)), 4),
                                                   "n_depots_BSS_positif": int(sum(x > 0 for x in v))}
    # --- Syntheses DESCRIPTIVES ajoutees apres la premiere execution (non pre-enregistrees) ---
    desc = {"mediane_BSS_oracle_vs_constante_oracle": med(lambda t: R[t]["R_oracle"]["bss_vs_constante_loyale"]),
            "mediane_brier_R_oracle": med(lambda t: R[t]["R_oracle"]["brier"]),
            "mediane_brier_constante_oracle": med(lambda t: R[t]["R0_brut"]["brier_constante_oracle"]),
            "brier_mediane_par_fenetre_methode_n": {}, "erreur_abs_pi_hat_mediane_par_fenetre_n": {},
            "erreur_relative_pi_hat_mediane_par_fenetre_n": {}}
    for nom in ("recente", "ancienne"):
        for n in NS:
            ks = [R[t]["fenetres"][nom].get(str(n)) for t in targets]
            if any(k is None for k in ks):
                continue
            desc["brier_mediane_par_fenetre_methode_n"][f"{nom}/R1/{n}"] = round(float(np.median([k["R1_prior"]["brier"] for k in ks])), 4)
            desc["erreur_abs_pi_hat_mediane_par_fenetre_n"][f"{nom}/{n}"] = round(float(np.median(
                [abs(R[t]["fenetres"][nom][str(n)]["pi_hat"] - R[t]["prevalence_test"]) for t in targets])), 4)
            desc["erreur_relative_pi_hat_mediane_par_fenetre_n"][f"{nom}/{n}"] = round(float(np.median(
                [abs(R[t]["fenetres"][nom][str(n)]["pi_hat"] - R[t]["prevalence_test"]) / R[t]["prevalence_test"]
                 for t in targets])), 3)
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "protocole": "CLAUDE.md §8 6B (7ca5564)",
           "garde_fou": summ, "C1": c1, "C2_BSS_par_methode_fenetre_n": c2,
           "descriptif_non_preenregistre": desc, "par_depot": R}
    (HERE / "chantier6b_calibration.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({"garde_fou": summ, "C1": c1, "descriptif": desc}, indent=1))
    for k, v in c2.items():
        print(k, v)


if __name__ == "__main__":
    main()
