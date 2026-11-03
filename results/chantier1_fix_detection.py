"""Chantier 1, 2e volet : la detection de fix stricte (exclusion typo/docs/changelog,
exigence de reference d'issue #123) ameliore-t-elle le label, au-dela du changement
de granularite deja adopte (label_v2) ?

Methode : comparer 4 candidats de predicat de fix, tous sous label_v2 (deja adopte,
cf. CLAUDE.md §4/§6), sur le meme etat de donnees. Mesure, pas supposition : AUC CV
(BASE, BASE+churn, BASE+code, tout), part_risque, et la correlation churn/fix (M1)
pour verifier si une detection plus stricte reduit encore la circularite au niveau
fichier. Ecrit results/chantier1_fix_detection.json.

Usage : python results/chantier1_fix_detection.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from fyc import evaluate, features, git_extract, labeling

REPO_DIR = str(REPO_ROOT / "flask")
FENETRE_JOURS = 90

CANDIDATS = {
    "A_baseline": labeling.make_fix_predicate(),
    "B_exclude_typo_docs_changelog": labeling.make_fix_predicate(exclude_re=labeling.EXCLUDE_RE),
    "C_issue_ref_requise": labeling.make_fix_predicate(require_issue_ref=True),
    "D_exclude_et_issue_ref": labeling.make_fix_predicate(
        exclude_re=labeling.EXCLUDE_RE, require_issue_ref=True,
    ),
}


def evaluate_candidat(predicate, commits_src):
    commits = [dict(c) for c in commits_src]  # copie superficielle : is_fix independant par run
    n_fix = sum(1 for c in commits if predicate(c["msg"]))

    _blame_par_commit, blame_par_fichier, est_risque = labeling.label_commits_v2(
        commits, REPO_DIR, fenetre_jours=FENETRE_JOURS, fix_predicate=predicate,
    )

    base = features.build_base_features(commits, est_risque)
    churn = features.build_churn_features(commits, est_risque)
    code = features.build_code_features(commits, est_risque, REPO_DIR)
    full = pd.concat([base.drop(columns="risky"), churn, code, base["risky"]], axis=1)

    feature_sets = {
        "BASE": features.BASE_COLS,
        "BASE+churn": features.BASE_COLS + features.CHURN_COLS,
        "BASE+code": features.BASE_COLS + features.CODE_COLS,
        "all": features.BASE_COLS + features.CHURN_COLS + features.CODE_COLS,
    }
    auc = {}
    for name, cols in feature_sets.items():
        mean, std = evaluate.cv_auc(full, cols)
        auc[name] = {"mean": round(mean, 4), "std": round(std, 4)}

    corr = features.churn_fix_correlation(commits, blame_par_fichier)

    return {
        "n_fix_commits": n_fix,
        "n_commits_etiquetes": len(base),
        "part_risque": round(float(base["risky"].mean()), 4),
        "auc": auc,
        "churn_fix_correlation": corr,
    }


def main():
    print("Extraction des commits...")
    commits_src = git_extract.extraire_commits(REPO_DIR)
    flask_head = git_extract.get_head(REPO_DIR)

    resultats = {}
    for nom, predicate in CANDIDATS.items():
        print(f"\n=== Candidat {nom} ===")
        r = evaluate_candidat(predicate, commits_src)
        resultats[nom] = r
        print(f"  fix={r['n_fix_commits']} etiquetes={r['n_commits_etiquetes']} "
              f"part_risque={r['part_risque']}")
        for fs, v in r["auc"].items():
            print(f"  {fs:12s} AUC = {v['mean']:.3f} +- {v['std']:.3f}")
        print(f"  correlation churn/fix (Pearson) = {r['churn_fix_correlation']['pearson']:.3f}")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flask_head": flask_head,
        "label_version": "v2_ligne_szz_blame",
        "fenetre_jours": FENETRE_JOURS,
        "candidats": resultats,
    }
    out_path = Path(__file__).resolve().parent / "chantier1_fix_detection.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nEcrit dans {out_path}")


if __name__ == "__main__":
    main()
