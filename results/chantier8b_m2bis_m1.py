"""Chantier 8b : (i) M2bis multi-graines, (ii) M2bis bootstrap, (iii) M1 contre modele nul de popularite.
Protocole : CLAUDE.md §8 Chantier 8 (commit f7e8489).

Ecart au protocole, declare AVANT l'execution : (ii) le bootstrap de commits AVANT entrainement
duplique des lignes entre plis de CV (fuite -> AUC gonflees, ratio biaise). On bootstrappe donc les
predictions hors-pli (OOF, CV 5 plis graine 0, RF 300 arbres), 1000 tirages : cela mesure la variance
d'echantillonnage de l'EVALUATION, pas celle de l'entrainement (la variance de graine est en (i)).
La regle de verdict est inchangee : satisfait si borne haute de l'IC <= 0,9044 ; echoue si borne basse >
0,9044 ; indetermine sinon.
"""
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chantier4bis_common as common
import chantier4bis_pyszz as pz

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fyc import evaluate, features, labeling

HERE = Path(__file__).resolve().parent
OUT = HERE / "chantier8b_m2bis_m1.json"
SEUIL = 0.9044
SEEDS = list(range(20))
N_NULL = 20


def cv_fold_aucs(X, y, seed, n_est=300):
    m = evaluate._make_model(n_estimators=n_est)
    m.set_params(n_jobs=1)
    aucs, oof = [], np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        oof[te] = p
        aucs.append(roc_auc_score(y.iloc[te], p))
    return float(np.mean(aucs)), oof


def ratio_for_seed(df, y, base_cols, churn_cols, seed):
    b, _ = cv_fold_aucs(df[base_cols], y, seed)
    c, _ = cv_fold_aucs(df[churn_cols], y, seed)
    return b, c, c / b


def stats(vals):
    v = np.array(vals)
    se = v.std(ddof=1) / np.sqrt(len(v))
    return {"moyenne": float(v.mean()), "ecart_type": float(v.std(ddof=1)),
            "ic95": [float(v.mean() - 1.96 * se), float(v.mean() + 1.96 * se)],
            "part_graines_sous_seuil": float((v <= SEUIL).mean()), "n": len(v)}


def main():
    t0 = time.time()
    commits, est_B, _ = common.load_context()
    _c, _e, full, idx = common.load_full()
    fams = common.FAMILIES
    base_cols, churn_cols = fams["BASE"], fams["churn"]
    out = {"protocole": "CLAUDE.md §8 Chantier 8b (f7e8489)", "generated_at": datetime.now(timezone.utc).isoformat(),
           "seuil_M2bis": SEUIL, "M2bis_multi_graines": {}}

    # ---------- variantes de label (population = commits non-fix sous cette variante) ----------
    par_fix_B = pz.v2_par_fix(commits, common.REPO_DIR)
    variants = {"v2_B": pz.to_est_risque(par_fix_B, commits)}
    variants["B_SZZ"] = pz.to_est_risque(pz.load_pyszz("b", commits), commits)
    variants["MA_SZZ"] = pz.to_est_risque(pz.load_pyszz("ma", commits), commits)
    # detection A (FIX_RE seul) : population plus petite (plus de fix exclus)
    cA = copy.deepcopy(commits)
    _bp, _bf, est_A = labeling.label_commits_v2(cA, common.REPO_DIR, fenetre_jours=common.FENETRE_JOURS)
    variants["v2_A"] = est_A
    commits_by_variant = {"v2_A": cA}

    for name, est in variants.items():
        cs = commits_by_variant.get(name, commits)
        rows, ys = [], []
        for pos, i in enumerate(idx):
            lab = est(cs[i])
            if lab is not None:
                rows.append(pos); ys.append(lab)
        df = full.iloc[rows].reset_index(drop=True)
        y = __import__("pandas").Series(ys, name="risky")
        res = Parallel(n_jobs=3)(delayed(ratio_for_seed)(df, y, base_cols, churn_cols, s) for s in SEEDS)
        bs, cs_, rs = zip(*res)
        out["M2bis_multi_graines"][name] = {
            "n": len(df), "part_risque": float(y.mean()),
            "AUC_BASE_moyenne": float(np.mean(bs)), "AUC_churn_moyenne": float(np.mean(cs_)),
            "ratio": stats(rs)}
        print(name, out["M2bis_multi_graines"][name]["ratio"], f"{time.time() - t0:.0f}s", flush=True)

    # ---------- (ii) bootstrap des predictions hors-pli, label v2_B ----------
    y = full["risky"].reset_index(drop=True)
    df = full.reset_index(drop=True)
    _, oof_b = cv_fold_aucs(df[base_cols], y, 0)
    _, oof_c = cv_fold_aucs(df[churn_cols], y, 0)
    rng = np.random.default_rng(0)
    rat = []
    while len(rat) < 1000:
        j = rng.integers(0, len(y), len(y))
        if y.values[j].min() == y.values[j].max():
            continue
        rat.append(roc_auc_score(y.values[j], oof_c[j]) / roc_auc_score(y.values[j], oof_b[j]))
    ic = [float(np.percentile(rat, 2.5)), float(np.percentile(rat, 97.5))]
    pt = float(roc_auc_score(y, oof_c) / roc_auc_score(y, oof_b))
    verdict = "satisfait" if ic[1] <= SEUIL else ("echoue" if ic[0] > SEUIL else "indetermine")
    out["M2bis_bootstrap_OOF"] = {"ratio_ponctuel_OOF_agrege": pt, "ic95": ic, "verdict": verdict}
    print("bootstrap", out["M2bis_bootstrap_OOF"], flush=True)

    # ---------- (iii) M1 reel et nuls de popularite ----------
    c_real = copy.deepcopy(commits)
    _bp, bfile_real = labeling.blame_line_fixes(c_real, common.REPO_DIR)
    real = features.churn_fix_correlation(c_real, bfile_real)
    from fyc import git_extract
    parents = git_extract.get_parents(common.REPO_DIR)
    pool = [i for i, c in enumerate(commits) if c["del"] > 0 and c["files"] and parents.get(c["hash"])]
    n_fix = sum(c["is_fix"] for c in commits)
    nulls = []
    rng = np.random.default_rng(1)
    for k in range(N_NULL):
        cn = copy.deepcopy(commits)
        for c in cn:
            c["is_fix"] = False
        for i in rng.choice(pool, n_fix, replace=False):
            cn[int(i)]["is_fix"] = True
        _bp, bfile = labeling.blame_line_fixes(cn, common.REPO_DIR)
        nulls.append(features.churn_fix_correlation(cn, bfile))
        print(f"  nul {k + 1}/{N_NULL}: {nulls[-1]}", flush=True)
    pn = np.array([n["pearson"] for n in nulls])
    lo, hi = float(np.percentile(pn, 2.5)), float(np.percentile(pn, 97.5))
    out["M1"] = {"reel_v2_B": real, "n_fix_reels": int(n_fix), "taille_pool_nul": len(pool),
                 "nuls_pearson": [float(x) for x in pn], "nuls_pearson_moyenne": float(pn.mean()),
                 "nuls_intervalle_2.5_97.5": [lo, hi],
                 "nuls_spearman": [n["spearman"] for n in nulls],
                 "conclusion": ("r reel DANS l'intervalle nul : M1 ne mesure que la popularite" if lo <= real["pearson"] <= hi
                                else ("r reel AU-DESSUS du nul : lien residuel au-dela de la popularite"
                                      if real["pearson"] > hi else "r reel EN DESSOUS du nul"))}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("M2bis_bootstrap_OOF",)}, indent=1))
    print("M1:", json.dumps({k: v for k, v in out["M1"].items() if k not in ("nuls_pearson", "nuls_spearman")}, indent=1))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
