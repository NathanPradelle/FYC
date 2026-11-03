"""Chantier 3 : l'analyse du diff (dispersion, ratio test en lignes, changement de
signature) ajoute-t-elle un signal reel, au-dela de l'etat statique du fichier
(radon/Ruff/Bandit, deja mesures aux Chantiers 2) ?

Meme methode que les chantiers precedents : comparer, mesurer la redondance et la
circularite avant de conclure. Ecrit results/chantier3_diff.json.

Usage : python results/chantier3_diff.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from fyc import diff_features, evaluate, features, git_extract, labeling

REPO_DIR = str(REPO_ROOT / "flask")
FENETRE_JOURS = 90
FIX_PREDICATE = labeling.make_fix_predicate(exclude_re=labeling.EXCLUDE_RE)


def main():
    print("Extraction des commits...")
    commits = git_extract.extraire_commits(REPO_DIR)
    flask_head = git_extract.get_head(REPO_DIR)

    _blame, _bpf, est_risque = labeling.label_commits_v2(
        commits, REPO_DIR, fenetre_jours=FENETRE_JOURS, fix_predicate=FIX_PREDICATE,
    )

    print("Construction des features BASE/churn/code...")
    base = features.build_base_features(commits, est_risque)
    churn = features.build_churn_features(commits, est_risque)
    code = features.build_code_features(commits, est_risque, REPO_DIR)

    print("Construction des features de diff (Chantier 3)...")
    diff_df = diff_features.build_diff_features(commits, est_risque, REPO_DIR)

    for name, df in [("churn", churn), ("code", code), ("diff", diff_df)]:
        assert len(df) == len(base), f"desynchronisation {name}"

    full = pd.concat(
        [base.drop(columns="risky"), churn, code, diff_df, base["risky"]], axis=1,
    )

    DIFF = diff_features.DIFF_COLS  # jeu adopte (curate, cf. fyc/diff_features.py)
    DIFF_ALL = diff_features.ALL_COLS  # tout ce qui a ete mesure, y compris le rejete
    feature_sets = {
        "BASE": features.BASE_COLS,
        "BASE+diff (adopte)": features.BASE_COLS + DIFF,
        "BASE+diff (tout mesure)": features.BASE_COLS + DIFF_ALL,
        "BASE+code(radon)": features.BASE_COLS + features.CODE_COLS,
        "BASE+code+diff": features.BASE_COLS + features.CODE_COLS + DIFF,
        "diff_only (adopte)": DIFF,
        "n_hunks_seul": ["n_hunks"],
        "avg_hunk_size_seul (rejete)": ["avg_hunk_size"],
        "touches_tests_seul": ["touches_tests"],
        "test_line_ratio_seul (rejete)": ["test_line_ratio"],
        "touches_signature_seul": ["touches_signature"],
        "tout_sans_diff": features.BASE_COLS + features.CHURN_COLS + features.CODE_COLS,
        "tout_avec_diff": features.BASE_COLS + features.CHURN_COLS + features.CODE_COLS + DIFF,
    }

    auc = {}
    for name, cols in feature_sets.items():
        mean, std = evaluate.cv_auc(full, cols)
        auc[name] = {"mean": round(mean, 4), "std": round(std, 4)}
        print(f"  {name:38s} AUC = {mean:.3f} +- {std:.3f}")

    # Redondance avec le churn existant (total_churn est deja dans BASE) : les hunks
    # et la taille de hunk sont-ils juste une reformulation du volume de changement ?
    redondance = {
        "n_hunks_vs_total_churn": round(float(full["n_hunks"].corr(full["total_churn"])), 4),
        "avg_hunk_size_vs_total_churn": round(
            float(full["avg_hunk_size"].corr(full["total_churn"])), 4,
        ),
        "test_line_ratio_vs_touches_tests": round(
            float(full["test_line_ratio"].corr(full["touches_tests"])), 4,
        ),
    }
    # Circularite avec le churn fichier (meme diagnostic qu'aux chantiers precedents)
    circularite = {
        "n_hunks_vs_file_commits_max": round(
            float(full["n_hunks"].corr(full["file_commits_max"])), 4,
        ),
        "touches_signature_vs_file_commits_max": round(
            float(full["touches_signature"].corr(full["file_commits_max"])), 4,
        ),
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flask_head": flask_head,
        "label_version": "v2_ligne_szz_blame",
        "fix_detection": "candidat_B_exclude_typo_docs_changelog",
        "n_commits_etiquetes": len(base),
        "auc": auc,
        "redondance": redondance,
        "circularite_avec_churn": circularite,
        "couverture_signature": {
            "n_non_nan": int(full["touches_signature"].notna().sum()),
            "n_total": len(full),
        },
    }

    out_path = Path(__file__).resolve().parent / "chantier3_diff.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nRedondance : {redondance}")
    print(f"Circularite : {circularite}")
    print(f"\nEcrit dans {out_path}")


if __name__ == "__main__":
    main()
