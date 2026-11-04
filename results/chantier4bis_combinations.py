"""Chantier 4bis : recherche de la meilleure combinaison de familles d'outils.
Protocole : CLAUDE.md §8, "PROTOCOLE PRE-ENREGISTRE" + AMENDEMENT n°1 (2026-11-04).

Etapes (chacune stockee dans results/chantier4bis_grid.json) :
  A. enumeration exhaustive des 2^11 = 2048 combinaisons (BASE toujours incluse), CV 5 plis,
     100 arbres, sur les 80 % chronologiques anciens UNIQUEMENT (reprenable : cache JSON) ;
  B. re-evaluation du top-20 sur 5 graines de CV neuves, 300 arbres (parade a la malediction
     du gagnant) ;
  C. selection pas-a-pas ascendante par famille, CV IMBRIQUEE (externe 5 plis x 3 graines,
     interne 3 plis, seuil de gain 0,01) = estimation non biaisee de la PROCEDURE ;
  D. evaluation UNIQUE sur les 20 % reserves : meilleur de la grille, selection pas-a-pas
     (refaite sur tout le 80 %), "tout empiler", BASE -- avec IC bootstrap.
Statistiques d'illustration : toutes les AUC, meilleure par taille, gain marginal par famille.
"""
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chantier4bis_common as common

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fyc import evaluate

HERE = Path(__file__).resolve().parent
GRID_CACHE = HERE / "cache_grid_v2.json"
OUT = HERE / "chantier4bis_grid.json"
SEUIL_SELECTION = 0.01
TEST_FRAC = 0.2


def _cv(X, y, n_est, seed, n_splits=5, n_jobs=1):
    m = evaluate._make_model(n_estimators=n_est)
    m.set_params(n_jobs=n_jobs)
    aucs = []
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        m.fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def _cols(combo, fams):
    return list(fams["BASE"]) + [c for f in combo for c in fams[f]]


def stage_a(dev, fams, names):
    cache = json.loads(GRID_CACHE.read_text()) if GRID_CACHE.exists() else {}
    todo = [c for r in range(len(names) + 1) for c in itertools.combinations(names, r)
            if "+".join(c) not in cache]
    print(f"[A] {len(cache)} deja en cache, {len(todo)} a evaluer", flush=True)
    y = dev["risky"]
    t0 = time.time()
    for k in range(0, len(todo), 64):
        chunk = todo[k:k + 64]
        res = Parallel(n_jobs=8)(delayed(_cv)(dev[_cols(c, fams)], y, 100, 0) for c in chunk)
        for c, (m, s) in zip(chunk, res):
            cache["+".join(c)] = [m, s]
        GRID_CACHE.write_text(json.dumps(cache))
        print(f"  {len(cache)}/2048  ({time.time() - t0:.0f}s)", flush=True)
    return cache


def forward_select(df, fams, names, seed, n_est=100, n_splits=3):
    y = df["risky"]
    chosen, best = [], _cv(df[_cols([], fams)], y, n_est, seed, n_splits)[0]
    trace = [("BASE", round(best, 4))]
    while True:
        cands = [(f, _cv(df[_cols(chosen + [f], fams)], y, n_est, seed, n_splits)[0])
                 for f in names if f not in chosen]
        if not cands:
            break
        f, sc = max(cands, key=lambda t: t[1])
        if sc - best <= SEUIL_SELECTION:
            break
        chosen.append(f)
        best = sc
        trace.append((f, round(sc, 4)))
    return chosen, trace


def _outer_task(dev, fams, names, seed, tr, te):
    y = dev["risky"]
    chosen, _ = forward_select(dev.iloc[tr].reset_index(drop=True), fams, names, seed)
    cols = _cols(chosen, fams)
    m = evaluate._make_model(n_estimators=300)
    m.set_params(n_jobs=1)
    m.fit(dev.iloc[tr][cols], y.iloc[tr])
    return chosen, roc_auc_score(y.iloc[te], m.predict_proba(dev.iloc[te][cols])[:, 1])


def stage_c(dev, fams, names):
    y = dev["risky"]
    tasks = [(seed, tr, te) for seed in (0, 1, 2)
             for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(dev, y)]
    res = Parallel(n_jobs=8)(delayed(_outer_task)(dev, fams, names, *t) for t in tasks)
    picks = [r[0] for r in res]
    outer = [r[1] for r in res]
    for (seed, _, _), (ch, a) in zip(tasks, res):
        print(f"[C] seed {seed}: {ch} -> {a:.4f}", flush=True)
    freq = {f: round(sum(f in p for p in picks) / len(picks), 3) for f in names}
    return {"auc_externe_moyenne": round(float(np.mean(outer)), 4),
            "auc_externe_std": round(float(np.std(outer)), 4),
            "frequence_selection_par_famille": freq, "selections": picks}


def holdout_auc(dev, test, cols, rng_seed=0):
    m = evaluate._make_model(n_estimators=300)
    m.fit(dev[cols], dev["risky"])
    p = m.predict_proba(test[cols])[:, 1]
    y = test["risky"].values
    auc = roc_auc_score(y, p)
    rng, boots = np.random.default_rng(rng_seed), []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], p[i]))
    return {"auc": round(float(auc), 4), "ic95": [round(float(np.percentile(boots, 2.5)), 4),
                                                   round(float(np.percentile(boots, 97.5)), 4)]}


def main():
    data, fams = common.load_all()
    names = [f for f in fams if f != "BASE"]
    assert len(names) == 11
    cut = int(len(data) * (1 - TEST_FRAC))
    dev, test = data.iloc[:cut].reset_index(drop=True), data.iloc[cut:].reset_index(drop=True)
    print(f"dev={len(dev)} (risque {dev.risky.mean():.3f}), reserve={len(test)} "
          f"(risque {test.risky.mean():.3f})", flush=True)

    grid = stage_a(dev, fams, names)
    rows = [{"combo": k.split("+") if k else [], "n_familles": len(k.split("+")) if k else 0,
             "auc": v[0], "std": v[1]} for k, v in grid.items()]
    base = next(r for r in rows if r["n_familles"] == 0)
    rows.sort(key=lambda r: -r["auc"])
    best_par_taille = {n: max((r for r in rows if r["n_familles"] == n), key=lambda r: r["auc"])
                       for n in range(len(names) + 1)}
    marginal = {}
    for f in names:
        gains = [grid["+".join(sorted_with(r["combo"], f, names))][0] - r["auc"]
                 for r in rows if f not in r["combo"]]
        marginal[f] = {"gain_marginal_moyen": round(float(np.mean(gains)), 4),
                       "gain_marginal_median": round(float(np.median(gains)), 4),
                       "part_gains_positifs": round(float(np.mean(np.array(gains) > 0)), 3)}
    seuil = base["std"]
    stats = {
        "auc_base": base["auc"], "std_base_plis": base["std"],
        "part_combinaisons_battant_base": round(float(np.mean([r["auc"] > base["auc"] for r in rows])), 4),
        "part_battant_base_de_plus_1_std": round(
            float(np.mean([r["auc"] > base["auc"] + seuil for r in rows])), 4),
        "auc_mediane_grille": round(float(np.median([r["auc"] for r in rows])), 4),
        "auc_min_max": [round(min(r["auc"] for r in rows), 4), round(max(r["auc"] for r in rows), 4)],
        "meilleure_par_nombre_de_familles": {str(n): {"combo": r["combo"], "auc": round(r["auc"], 4)}
                                             for n, r in best_par_taille.items()},
        "gain_marginal_par_famille": dict(sorted(marginal.items(),
                                                 key=lambda kv: -kv[1]["gain_marginal_moyen"])),
    }

    print("[B] re-evaluation du top-20 sur 5 graines neuves...", flush=True)
    top = []
    for r in rows[:20]:
        cols = _cols(r["combo"], fams)
        sc = [_cv(dev[cols], dev["risky"], 300, s, n_jobs=-1)[0] for s in (11, 12, 13, 14, 15)]
        top.append({"combo": r["combo"], "auc_grille": round(r["auc"], 4),
                    "auc_remesuree": round(float(np.mean(sc)), 4),
                    "std_graines": round(float(np.std(sc)), 4)})
    base_re = np.mean([_cv(dev[_cols([], fams)], dev["risky"], 300, s, n_jobs=-1)[0]
                       for s in (11, 12, 13, 14, 15)])

    print("[C] selection pas-a-pas, CV imbriquee...", flush=True)
    nested = stage_c(dev, fams, names)
    sel, trace = forward_select(dev, fams, names, seed=0, n_splits=5)

    print("[D] evaluation unique sur les 20 % reserves...", flush=True)
    best_grid = rows[0]["combo"]
    holdout = {
        "BASE": holdout_auc(dev, test, _cols([], fams)),
        "tout_empiler": holdout_auc(dev, test, _cols(names, fams)),
        "meilleur_de_la_grille": {"combo": best_grid, **holdout_auc(dev, test, _cols(best_grid, fams))},
        "selection_pas_a_pas": {"combo": sel, "trace": trace,
                                **holdout_auc(dev, test, _cols(sel, fams))},
    }
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flask_head": common.git_extract.get_head(common.REPO_DIR),
        "protocole": "CLAUDE.md §8 4bis + amendement n°1", "n_dev": len(dev), "n_reserve": len(test),
        "familles": {f: fams[f] for f in fams},
        "grille_stats": stats, "top20_remesure": top, "base_remesuree_300_arbres_5_graines": round(float(base_re), 4),
        "top20_grille": rows[:20], "bottom20_grille": rows[-20:],
        "selection_imbriquee": nested, "reserve_20pct": holdout,
        "grille_complete": rows,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print("ecrit", OUT)


def sorted_with(combo, f, names):
    return [n for n in names if n in set(combo) | {f}]


if __name__ == "__main__":
    main()
