"""Chantier 5 : stacking par famille + matrice de desaccord.
Protocole : CLAUDE.md §8, "PROTOCOLE PRE-ENREGISTRE -- Chantier 5, v1" (2026-11-05, commit 9e8e892).

- 12 familles (BASE incluse), une foret (300 arbres, balanced, non reglee) par famille, sur ses
  colonnes seules. Predictions OOF (CV interne 5 plis) sur le train.
- Meta-modeles : S1 = regression logistique sur les logits ; S2 = moyenne des rangs percentiles.
- References appariees : BASE, "tout empiler" (une foret sur les 12 familles concatenees).
- CV imbriquee : externe 5 plis x 3 graines (15 plis). Variante sans churn (sensibilite).
- Matrice de desaccord sur les predictions externes. Reserve 20 % recents evaluee une seule fois.
"""
import json
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chantier4bis_common as common

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
OUT = HERE / "chantier5_stacking.json"
N_TREES = 300
TEST_FRAC = 0.2
SEEDS = (0, 1, 2)
TOP = 0.10


def _rf():
    return RandomForestClassifier(n_estimators=N_TREES, random_state=42,
                                  class_weight="balanced", n_jobs=1)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _rank(p):
    return (rankdata(p) - 1) / max(len(p) - 1, 1)


def family_preds(tr, te, fams, names, seed):
    """Pour chaque famille : OOF sur tr (CV interne 5 plis) et prediction sur te
    (modele ajuste sur tout tr)."""
    ytr = tr["risky"]
    oof, pte = {}, {}
    inner = list(StratifiedKFold(5, shuffle=True, random_state=100 + seed).split(tr, ytr))
    for f in names:
        cols = list(fams[f])
        o = np.zeros(len(tr))
        for a, b in inner:
            m = _rf().fit(tr.iloc[a][cols], ytr.iloc[a])
            o[b] = m.predict_proba(tr.iloc[b][cols])[:, 1]
        oof[f] = o
        pte[f] = _rf().fit(tr[cols], ytr).predict_proba(te[cols])[:, 1]
    return oof, pte


def meta_scores(oof, pte, ytr, use):
    S1 = LogisticRegression(C=1.0, max_iter=1000).fit(
        np.column_stack([_logit(oof[f]) for f in use]), ytr)
    s1 = S1.predict_proba(np.column_stack([_logit(pte[f]) for f in use]))[:, 1]
    s2 = np.mean([_rank(pte[f]) for f in use], axis=0)
    return s1, s2


def concat_score(tr, te, cols):
    return _rf().fit(tr[cols], tr["risky"]).predict_proba(te[cols])[:, 1]


def one_fold(dev, fams, names, tr_idx, te_idx, seed):
    tr, te = dev.iloc[tr_idx].reset_index(drop=True), dev.iloc[te_idx].reset_index(drop=True)
    ytr, yte = tr["risky"], te["risky"]
    oof, pte = family_preds(tr, te, fams, names, seed)
    nochurn = [f for f in names if f != "churn"]
    allcols = [c for f in names for c in fams[f]]
    nochurncols = [c for f in nochurn for c in fams[f]]
    s1, s2 = meta_scores(oof, pte, ytr, names)
    s1n, s2n = meta_scores(oof, pte, ytr, nochurn)
    auc = lambda s: float(roc_auc_score(yte, s))
    out = {
        "seed": seed,
        "BASE": auc(pte["BASE"]),
        "concat": auc(concat_score(tr, te, allcols)),
        "concat_sans_churn": auc(concat_score(tr, te, nochurncols)),
        "S1": auc(s1), "S2": auc(s2), "S1_sans_churn": auc(s1n), "S2_sans_churn": auc(s2n),
        "familles": {f: auc(pte[f]) for f in names},
    }
    # --- matrice de desaccord (predictions de test de ce pli) ---
    k = max(int(round(TOP * len(te))), 1)
    top = {f: set(np.argsort(-pte[f])[:k]) for f in names}
    pos = set(np.where(yte.values == 1)[0])
    pairs = {}
    for a, b in combinations(names, 2):
        rho = spearmanr(pte[a], pte[b]).statistic
        jac = len(top[a] & top[b]) / len(top[a] | top[b])
        mix = auc(_rank(pte[a]) + _rank(pte[b]))
        pairs[f"{a}|{b}"] = {"spearman": float(rho), "jaccard_top": float(jac),
                             "gain_complementarite": float(mix - max(out["familles"][a], out["familles"][b]))}
    caught = {i: [f for f in names if i in top[f]] for i in pos}
    caught = {i: fs for i, fs in caught.items() if fs}
    only_one = (sum(len(fs) == 1 for fs in caught.values()) / len(caught)) if caught else float("nan")
    out["pairs"] = pairs
    out["part_positifs_vus_par_une_seule_famille"] = float(only_one)
    out["n_positifs_vus_top10"] = len(caught)
    out["n_positifs"] = len(pos)
    return out


def bootstrap_ci(y, s, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    while len(vals) < n:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def summarize(folds):
    keys = ["BASE", "concat", "S1", "S2", "concat_sans_churn", "S1_sans_churn", "S2_sans_churn"]
    res = {k: {"mean": float(np.mean([f[k] for f in folds])),
               "std": float(np.std([f[k] for f in folds]))} for k in keys}
    paired = {}
    for k in ("S1", "S2", "concat", "S1_sans_churn", "S2_sans_churn", "concat_sans_churn"):
        for ref in ("BASE", "concat"):
            if k == ref:
                continue
            d = np.array([f[k] - f[ref] for f in folds])
            paired[f"{k}-{ref}"] = {"mean": float(d.mean()), "std": float(d.std()),
                                    "plis_positifs": int((d > 0).sum()), "n_plis": len(d)}
    names = list(folds[0]["familles"])
    res["familles_seules"] = {f: float(np.mean([x["familles"][f] for x in folds])) for f in names}
    pair_keys = list(folds[0]["pairs"])
    agg = {}
    for pk in pair_keys:
        agg[pk] = {m: float(np.mean([x["pairs"][pk][m] for x in folds]))
                   for m in ("spearman", "jaccard_top", "gain_complementarite")}
        agg[pk]["spearman_mediane_plis"] = float(np.median([x["pairs"][pk]["spearman"] for x in folds]))
    hors_base = [pk for pk in pair_keys if "BASE" not in pk.split("|")]
    res["desaccord"] = {
        "paires": agg,
        "spearman_median_hors_BASE": float(np.median([agg[pk]["spearman_mediane_plis"] for pk in hors_base])),
        "jaccard_top10_median_hors_BASE": float(np.median([agg[pk]["jaccard_top"] for pk in hors_base])),
        "part_paires_gain_gt_0.01_hors_BASE": float(np.mean([agg[pk]["gain_complementarite"] > 0.01
                                                             for pk in hors_base])),
        "part_paires_gain_gt_0.01_toutes": float(np.mean([agg[pk]["gain_complementarite"] > 0.01
                                                          for pk in pair_keys])),
        "part_positifs_vus_par_une_seule_famille": float(
            np.nanmean([f["part_positifs_vus_par_une_seule_famille"] for f in folds])),
    }
    return res, paired


def verdicts(paired, desaccord):
    v = {}
    for k in ("S1", "S2"):
        for ref in ("BASE", "concat"):
            p = paired[f"{k}-{ref}"]
            v[f"{k}_bat_{ref}"] = bool(p["mean"] >= 0.02 and p["plis_positifs"] >= 12)
    v["complementarite"] = bool(desaccord["spearman_median_hors_BASE"] < 0.5
                                and desaccord["part_paires_gain_gt_0.01_hors_BASE"] >= 1 / 3)
    return v


def main():
    data, fams = common.load_all()
    names = list(fams)
    assert len(names) == 12, names
    cut = int(len(data) * (1 - TEST_FRAC))
    dev, test = data.iloc[:cut].reset_index(drop=True), data.iloc[cut:].reset_index(drop=True)
    print(f"dev={len(dev)} ({int(dev.risky.sum())} pos) | reserve={len(test)} ({int(test.risky.sum())} pos)",
          flush=True)
    jobs = []
    for seed in SEEDS:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(dev, dev["risky"]):
            jobs.append((tr, te, seed))
    t0 = time.time()
    folds = Parallel(n_jobs=3, verbose=5)(
        delayed(one_fold)(dev, fams, names, tr, te, seed) for tr, te, seed in jobs)
    print(f"CV imbriquee : {time.time() - t0:.0f}s", flush=True)
    res, paired = summarize(folds)
    verd = verdicts(paired, res["desaccord"])

    # --- reserve : evaluation UNIQUE ---
    print("Reserve (evaluation unique)...", flush=True)
    oof, pte = family_preds(dev, test, fams, names, seed=0)
    s1, s2 = meta_scores(oof, pte, dev["risky"], names)
    allcols = [c for f in names for c in fams[f]]
    yt = test["risky"]
    scores = {"BASE": pte["BASE"], "concat": concat_score(dev, test, allcols), "S1": s1, "S2": s2}
    reserve = {k: {"auc": float(roc_auc_score(yt, s)), "ic95": bootstrap_ci(yt, s)}
               for k, s in scores.items()}

    out = {
        "protocole": "CLAUDE.md §8 Chantier 5 v1 (commit 9e8e892)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_dev": len(dev), "n_reserve": len(test), "n_pos_dev": int(dev.risky.sum()),
        "n_pos_reserve": int(yt.sum()), "familles": names, "n_plis_externes": len(folds),
        "cv_imbriquee": res, "differences_appariees": paired, "verdicts": verd,
        "reserve": reserve, "plis": folds,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({"cv": {k: res[k] for k in ("BASE", "concat", "S1", "S2",
                                                   "S1_sans_churn", "S2_sans_churn",
                                                   "concat_sans_churn")},
                      "paired": paired, "verdicts": verd,
                      "desaccord_resume": {k: v for k, v in res["desaccord"].items() if k != "paires"},
                      "reserve": reserve}, indent=1))


if __name__ == "__main__":
    main()
