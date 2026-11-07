"""Chantier 8d : cause de la derive de prevalence (6B, C3). Protocole : CLAUDE.md §8 Chantier 8 (f7e8489).

Par depot, 10 tranches chronologiques d'effectif egal (sur les commits etiquetes, hors 90 derniers
jours). Mesures : (a) part de commits is_fix (message) ; (b) prevalence label_v2 ; (c) prevalence
"retravail" independante des messages (tous les commits sont sources de blame, pas seulement les fix).
Regles de lecture fixees dans le protocole (H1 maturation / H3 conventions / H2 effets de bord).
"""
import bisect
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chantier6a_dataset as ds

import numpy as np

from fyc import git_extract, labeling

OUT = Path(__file__).resolve().parent / "chantier8d_drift.json"
W = 90 * 24 * 3600
NT = 10


def reedit_labels(commits, path):
    allc = [dict(c, is_fix=True) for c in commits]
    bpc, _ = labeling.blame_line_fixes(allc, path)
    def lab(c):
        dates = bpc.get(c["hash"])
        if not dates:
            return 0
        i = bisect.bisect_right(dates, c["ts"])
        return 1 if i < len(dates) and dates[i] <= c["ts"] + W else 0
    return lab


def analyse(name, path):
    t0 = time.time()
    path = str(path)
    commits = git_extract.extraire_commits(path)
    _b, _f, est = labeling.label_commits_v2(commits, path, fenetre_jours=90, fix_predicate=ds.FIX_PREDICATE)
    lab_re = reedit_labels(commits, path)
    P = sorted([c for c in commits if est(c) is not None and c["files"]], key=lambda c: c["ts"])
    allf = [c for c in commits if c["files"]]
    tmax = max(c["ts"] for c in commits)
    P = [c for c in P if c["ts"] <= tmax - W]  # hors 90 derniers jours (censure)
    y_v2 = np.array([est(c) for c in P]); y_re = np.array([lab_re(c) for c in P])
    fix_flag = {c["hash"]: c["is_fix"] for c in allf}
    def tranche_stats(sl):
        lo, hi = P[sl[0]]["ts"], P[sl[-1]]["ts"]
        inwin = [c for c in allf if lo <= c["ts"] <= hi]
        return {"debut": lo, "n": len(sl), "part_fix_message": float(np.mean([c["is_fix"] for c in inwin])),
                "prevalence_v2": float(y_v2[sl].mean()), "prevalence_retravail": float(y_re[sl].mean())}
    cuts = np.array_split(np.arange(len(P)), NT)
    tr = [tranche_stats(s) for s in cuts]
    def ratio(a, b, k):
        return tr[a][k] / tr[b][k] if tr[b][k] > 0 else float("nan")
    R = {k: ratio(0, NT - 1, k) for k in ("part_fix_message", "prevalence_v2", "prevalence_retravail")}
    R4 = {k: ratio(3, NT - 1, k) for k in ("part_fix_message", "prevalence_v2", "prevalence_retravail")}
    # variante : sans les 90 premiers jours de l'historique
    t_first = min(c["ts"] for c in commits)
    Pn = [i for i, c in enumerate(P) if c["ts"] >= t_first + W]
    cuts2 = np.array_split(np.array(Pn), NT)
    tr2 = [tranche_stats(s) for s in cuts2]
    R90 = {k: (tr2[0][k] / tr2[-1][k] if tr2[-1][k] > 0 else float("nan"))
           for k in ("part_fix_message", "prevalence_v2", "prevalence_retravail")}
    def verdict(r):
        a, c = r["part_fix_message"], r["prevalence_retravail"]
        if a >= 2 and c < 1.5:
            return "H3_conventions"
        if a >= 1.5 and c >= 1.5:
            return "H1_maturation"
        return "indetermine"
    out = {"n_commits_etiquetes_hors_90j": len(P), "tranches": tr,
           "ratio_premiere_sur_derniere": R, "verdict": verdict(R),
           "H2_sans_3_premieres_tranches_ratio_4eme_sur_derniere": R4, "H2_verdict_sans_3_premieres": verdict(R4),
           "H2_sans_90_premiers_jours": R90, "H2_verdict_sans_90_premiers_jours": verdict(R90),
           "secondes": time.time() - t0}
    print(name, "ratios 1/10:", {k: round(v, 2) for k, v in R.items()}, "->", out["verdict"],
          f"({out['secondes']:.0f}s)", flush=True)
    return out


def main():
    res = {}
    if OUT.exists():
        res = json.loads(OUT.read_text()).get("depots", {})
    for name, path in ds.REPOS.items():
        if name in res:
            continue
        res[name] = analyse(name, path)
        OUT.write_text(json.dumps({"protocole": "CLAUDE.md §8 Chantier 8d (f7e8489)",
                                   "generated_at": datetime.now(timezone.utc).isoformat(), "depots": res}, indent=1))
    verd = [r["verdict"] for r in res.values()]
    med = {k: float(np.nanmedian([r["ratio_premiere_sur_derniere"][k] for r in res.values()]))
           for k in ("part_fix_message", "prevalence_v2", "prevalence_retravail")}
    summ = {"verdicts_par_depot": {n: r["verdict"] for n, r in res.items()}, "comptes": {v: verd.count(v) for v in set(verd)},
            "mediane_ratios_premiere_sur_derniere": med,
            "verdict_sur_medianes": ("H3_conventions" if med["part_fix_message"] >= 2 and med["prevalence_retravail"] < 1.5 else
                                     "H1_maturation" if med["part_fix_message"] >= 1.5 and med["prevalence_retravail"] >= 1.5 else "indetermine")}
    d = json.loads(OUT.read_text()); d["synthese"] = summ
    OUT.write_text(json.dumps(d, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
