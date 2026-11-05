"""4bis, sous-volet "validation du label" : label_v2 vs PySZZ (B-SZZ, MA-SZZ).
Protocole : CLAUDE.md §8, "SOUS-VOLET 4bis -- PROTOCOLE PRE-ENREGISTRE PySZZ, v1" (commit 55b13fe).

Entrees : results/pyszz_out/bic_b.json et bic_ma.json (sorties brutes de PySZZ, commit 352c6bf,
produites par `main.py` sur les 732 fix de la detection B, config par defaut). Ce script ne lance
PAS PySZZ (venv separe, voir §7) : il convertit ses sorties en `blame_par_commit` et applique la
MEME `make_est_risque_v2` (fenetre 90 j) -- seule l'attribution change.
"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chantier4bis_common as common

import numpy as np
from sklearn.metrics import cohen_kappa_score

from fyc import evaluate, git_extract, labeling

HERE = Path(__file__).resolve().parent
OUT = HERE / "chantier4bis_pyszz.json"
SEUIL_KAPPA, SEUIL_AUC_BASE, SEUIL_M2BIS, PLANCHER = 0.6, 0.03, 0.9044, 0.05


def v2_par_fix(commits, repo_dir):
    """Meme boucle que labeling.blame_line_fixes, mais garde l'ensemble d'introducteurs PAR FIX
    (la fonction d'origine ne les expose pas) pour comparer fix par fix a PySZZ."""
    parents = git_extract.get_parents(repo_dir)
    by_hash = {c["hash"]: c for c in commits}
    res = {}
    for c in commits:
        if not c["is_fix"]:
            continue
        par = parents.get(c["hash"], [])
        if not par:
            res[c["hash"]] = set()
            continue
        intro = set()
        old_ranges = labeling._parse_diff_old_ranges(git_extract.diff_raw(repo_dir, par[0], c["hash"]))
        for path, ranges in old_ranges.items():
            porcelain = git_extract.blame_porcelain(repo_dir, par[0], path, ranges)
            if porcelain is None:
                continue
            for h in labeling._parse_blame_commits(porcelain):
                i = by_hash.get(h)
                if i is not None and i["ts"] <= c["ts"]:
                    intro.add(h)
        res[c["hash"]] = intro
    return res


def to_est_risque(par_fix, commits):
    by_hash = {c["hash"]: c for c in commits}
    bpc = defaultdict(list)
    for fix, intro in par_fix.items():
        for h in intro:
            if h in by_hash and by_hash[h]["ts"] <= by_hash[fix]["ts"]:
                bpc[h].append(by_hash[fix]["ts"])
    for k in bpc:
        bpc[k].sort()
    return labeling.make_est_risque_v2(bpc, fenetre_jours=common.FENETRE_JOURS)


def load_pyszz(name, commits):
    by_hash = {c["hash"] for c in commits}
    d = json.loads((HERE / "pyszz_out" / f"bic_{name}.json").read_text())
    return {x["fix_commit_hash"]: {h for h in x.get("inducing_commit_hash", []) if h in by_hash}
            for x in d}


def label_vector(est, commits, indices):
    return np.array([est(commits[i]) for i in indices])


def fix_level(a, b):
    jac = [len(a[f] & b[f]) / len(a[f] | b[f]) for f in a if a[f] | b[f]]
    return {
        "n_fix": len(a),
        "part_identique": float(np.mean([a[f] == b[f] for f in a])),
        "jaccard_moyen_hors_deux_vides": float(np.mean(jac)),
        "sans_introducteur_ref": float(np.mean([not a[f] for f in a])),
        "sans_introducteur_alt": float(np.mean([not b[f] for f in a])),
        "ref_vide_alt_non_vide": int(sum((not a[f]) and bool(b[f]) for f in a)),
        "ref_non_vide_alt_vide": int(sum(bool(a[f]) and (not b[f]) for f in a)),
        "alt_sous_ensemble_de_ref": int(sum(bool(b[f]) and b[f] < a[f] for f in a)),
    }


def agreement(y_ref, y_alt):
    tp = int(((y_ref == 1) & (y_alt == 1)).sum())
    return {
        "n": int(len(y_ref)), "part_risque_ref": float(y_ref.mean()), "part_risque_alt": float(y_alt.mean()),
        "n_pos_ref": int(y_ref.sum()), "n_pos_alt": int(y_alt.sum()), "n_pos_communs": tp,
        "accord_brut": float((y_ref == y_alt).mean()),
        "kappa": float(cohen_kappa_score(y_ref, y_alt)),
        "jaccard_positifs": float(tp / max(int(((y_ref == 1) | (y_alt == 1)).sum()), 1)),
        "positifs_ref_retrouves": float(tp / max(int(y_ref.sum()), 1)),
        "precision_alt_vs_ref": float(tp / max(int(y_alt.sum()), 1)),
    }


def aucs(full, y, fams):
    df = full.copy()
    df["risky"] = y
    base = evaluate.cv_auc(df, fams["BASE"])
    churn = evaluate.cv_auc(df, fams["churn"])
    allc = [c for f in fams for c in fams[f]]
    tout = evaluate.cv_auc(df, allc)
    return {"BASE": base, "churn_seul": churn, "all": tout, "M2bis_ratio": churn[0] / base[0]}


def main():
    commits, est_v2, _ = common.load_context()
    _c, _e, full, indices = common.load_full()
    fams = common.FAMILIES
    y_v2 = label_vector(est_v2, commits, indices)
    assert (y_v2 == full["risky"].values).all(), "label_v2 recalcule != colonne risky du cache"

    par_fix_v2 = v2_par_fix(commits, common.REPO_DIR)
    # controle de coherence : reconstruire le label v2 depuis les ensembles par fix
    assert (label_vector(to_est_risque(par_fix_v2, commits), commits, indices) == y_v2).all()

    out = {"protocole": "CLAUDE.md §8 sous-volet PySZZ v1 (commit 55b13fe)",
           "generated_at": datetime.now(timezone.utc).isoformat(),
           "pyszz_commit": "352c6bf4f35fca224bbfba02111f1e852a7aa4ad",
           "n_fix": len(par_fix_v2), "variantes": {}}
    out["variantes"]["v2"] = {"part_risque": float(y_v2.mean()), "auc": aucs(full, y_v2, fams)}
    print("v2", out["variantes"]["v2"], flush=True)
    for name, label in (("b", "B-SZZ"), ("ma", "MA-SZZ")):
        alt_fix = load_pyszz(name, commits)
        assert set(alt_fix) == set(par_fix_v2), "fix differents entre v2 et PySZZ"
        y_alt = label_vector(to_est_risque(alt_fix, commits), commits, indices)
        v = {"nom": label, "accord": agreement(y_v2, y_alt), "niveau_fix": fix_level(par_fix_v2, alt_fix)}
        v["interpretable"] = bool(v["accord"]["part_risque_alt"] >= PLANCHER)
        v["auc"] = aucs(full, y_alt, fams) if v["interpretable"] else None
        out["variantes"][name] = v
        print(label, json.dumps({k: v[k] for k in ("accord", "niveau_fix", "auc")}, indent=1), flush=True)

    ma = out["variantes"]["ma"]
    if ma["auc"] is not None:
        ref = out["variantes"]["v2"]["auc"]
        out["verdict"] = {
            "R1_kappa_ge_0.6": bool(ma["accord"]["kappa"] >= SEUIL_KAPPA),
            "R2_ecart_AUC_BASE_le_0.03": bool(abs(ma["auc"]["BASE"][0] - ref["BASE"][0]) <= SEUIL_AUC_BASE),
            "R3_M2bis_le_0.9044": bool(ma["auc"]["M2bis_ratio"] <= SEUIL_M2BIS),
        }
        out["verdict"]["label_v2_robuste"] = all(out["verdict"].values())
    OUT.write_text(json.dumps(out, indent=1))
    print("verdict:", out.get("verdict"))


if __name__ == "__main__":
    main()
