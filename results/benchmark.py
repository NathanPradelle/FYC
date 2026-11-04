"""Recalcule tous les chiffres de la Partie 4 du notebook (comparaison des jeux de
features) et les ecrit dans results/results.json (label_v1) ou results/results_v2.json
(label_v2), avec la date, les versions des librairies et le hash du HEAD du depot
Flask analyse.

But : plus aucun chiffre en dur dans le markdown du notebook. Chaque resultat est
date et rattache a un etat de donnees identifiable, pour que l'ecart constate entre
le narratif du notebook et une execution reelle (2026-09-09, cf. CLAUDE.md §5/§7) ne
puisse plus se reproduire silencieusement.

Usage :
  python results/benchmark.py             # label_v1 (fichier), ecrit results.json
  python results/benchmark.py --label v2  # label_v2 (ligne, git blame), ecrit results_v2.json
"""
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import radon
import sklearn

from fyc import code_lint, diff_features, evaluate, features, git_extract, labeling, process_features

REPO_DIR = str(REPO_ROOT / "flask")
FENETRE_JOURS = 90

# Detection de fix assainie (Chantier 1, 2e volet, mesuree le 2026-09-09 via
# results/chantier1_fix_detection.py) : exclut typo/docs/changelog (candidat B).
# L'exigence de reference d'issue (#123) fait tomber part_risque sous le plancher
# de 5% deja etabli pour la granularite -- resultats non fiables a cette taille de
# classe, donc PAS adoptee malgre une AUC brute plus haute (cf. CLAUDE.md §6).
FIX_PREDICATE = labeling.make_fix_predicate(exclude_re=labeling.EXCLUDE_RE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=["v1", "v2"], default="v2",
                         help="v1 = SZZ fichier ; v2 = SZZ ligne via git blame (defaut, adopte"
                              " le 2026-09-09, cf. CLAUDE.md §4/§6)")
    args = parser.parse_args()

    print("Extraction des commits...")
    commits = git_extract.extraire_commits(REPO_DIR)
    flask_head = git_extract.get_head(REPO_DIR)
    print(f"{len(commits)} commits, HEAD={flask_head}")

    if args.label == "v1":
        label_version = "v1_fichier_szz_simplifie"
        fix_index, est_risque = labeling.label_commits(
            commits, fenetre_jours=FENETRE_JOURS, fix_predicate=FIX_PREDICATE,
        )
    else:
        label_version = "v2_ligne_szz_blame"
        print("Calcul du SZZ ligne (git blame -w -M -C sur les commits de fix)...")
        _blame_par_commit, fix_index, est_risque = labeling.label_commits_v2(
            commits, REPO_DIR, fenetre_jours=FENETRE_JOURS, fix_predicate=FIX_PREDICATE,
        )

    print("Construction des features BASE...")
    base = features.build_base_features(commits, est_risque)
    is_night = features.build_is_night(base)
    print("Construction des features churn...")
    churn = features.build_churn_features(commits, est_risque)
    print("Construction des features de code (radon, via git cat-file)...")
    code = features.build_code_features(commits, est_risque, REPO_DIR)
    print("Lancement de Ruff + Bandit (batch, Chantier 2)...")
    ruff_df, bandit_df = code_lint.lint_features(commits, est_risque, REPO_DIR)
    print("Construction des features de diff (Chantier 3, jeu curate)...")
    diff_df = diff_features.build_diff_features(commits, est_risque, REPO_DIR)
    print("Signaux de process GitHub (Chantier 4, cache disque resumable)...")
    process_df, (n_mesures, n_total), mask = process_features.build_process_features(
        commits, est_risque,
    )
    couverture = n_mesures / n_total
    print(f"Couverture cache GitHub : {n_mesures}/{n_total} ({couverture:.1%})")
    if couverture < 1.0:
        print(f"ATTENTION : cache incomplet ({n_total - n_mesures} commits non mesures) "
              f"-- lignes exclues ci-dessous, pas remplies par defaut.")

    for name, df in [("churn", churn), ("code", code), ("ruff", ruff_df), ("bandit", bandit_df),
                      ("diff", diff_df)]:
        assert len(df) == len(base), f"desynchronisation {name} ({len(df)} != {len(base)})"

    mask = pd.Series(mask).values
    base, is_night, churn, code = (
        base[mask].reset_index(drop=True), is_night[mask].reset_index(drop=True),
        churn[mask].reset_index(drop=True), code[mask].reset_index(drop=True),
    )
    ruff_df, bandit_df, diff_df = (
        ruff_df[mask].reset_index(drop=True), bandit_df[mask].reset_index(drop=True),
        diff_df[mask].reset_index(drop=True),
    )
    assert len(process_df) == len(base), "desynchronisation process (couverture cache GitHub)"

    full = pd.concat(
        [base.drop(columns="risky"), is_night, churn, code, ruff_df, bandit_df, diff_df,
         process_df, base["risky"]],
        axis=1,
    )

    LINT = code_lint.RUFF_COLS + code_lint.BANDIT_COLS
    DIFF = diff_features.DIFF_COLS  # jeu curate -- avg_hunk_size/test_line_ratio rejetes (§6)
    PROCESS = process_features.PROCESS_COLS
    feature_sets = {
        "BASE": features.BASE_COLS,
        "BASE+is_night": features.BASE_COLS + ["is_night"],
        "BASE+churn": features.BASE_COLS + features.CHURN_COLS,
        "BASE+code": features.BASE_COLS + features.CODE_COLS,
        "BASE+lint": features.BASE_COLS + LINT,
        "BASE+diff": features.BASE_COLS + DIFF,
        "BASE+process": features.BASE_COLS + PROCESS,
        "code_only": features.CODE_COLS,
        "churn_only": features.CHURN_COLS,
        "lint_only": LINT,
        "diff_only": DIFF,
        "process_only": PROCESS,
        "all_sans_lint_diff_process": features.BASE_COLS + features.CHURN_COLS + features.CODE_COLS,
        "all": (
            features.BASE_COLS + features.CHURN_COLS + features.CODE_COLS + LINT + DIFF + PROCESS
        ),
    }

    results = {}
    for name, cols in feature_sets.items():
        print(f"Evaluation {name}...")
        cv_mean, cv_std = evaluate.cv_auc(full, cols)
        temporal = evaluate.temporal_split_auc(full, cols)
        results[name] = {
            "cols": cols,
            "cv_auc_mean": round(cv_mean, 4),
            "cv_auc_std": round(cv_std, 4),
            "temporal_split_auc": round(temporal, 4),
        }
        print(f"  CV AUC = {cv_mean:.3f} +- {cv_std:.3f} | temporel = {temporal:.3f}")

    diagnostics = {
        "churn_fix_correlation": features.churn_fix_correlation(commits, fix_index),
        "baseline_majority_accuracy": evaluate.baseline_majority_accuracy(base),
        "n_commits_total": len(commits),
        "n_commits_etiquetes": len(base),
        "part_risque": round(float(base["risky"].mean()), 4),
        "ruff_redondance_vs_loc_avg": round(float(full["ruff_total_avg"].corr(full["loc_avg"])), 4),
        "bandit_redondance_vs_loc_avg": round(
            float(full["bandit_total_avg"].corr(full["loc_avg"])), 4,
        ),
        "ruff_circularite_vs_churn": round(
            float(full["ruff_total_avg"].corr(full["file_commits_max"])), 4,
        ),
        "bandit_circularite_vs_churn": round(
            float(full["bandit_total_avg"].corr(full["file_commits_max"])), 4,
        ),
        "n_hunks_redondance_vs_total_churn": round(
            float(full["n_hunks"].corr(full["total_churn"])), 4,
        ),
        "n_hunks_circularite_vs_churn": round(
            float(full["n_hunks"].corr(full["file_commits_max"])), 4,
        ),
        "touches_signature_circularite_vs_churn": round(
            float(full["touches_signature"].corr(full["file_commits_max"])), 4,
        ),
        "couverture_cache_github": {"n_mesures": n_mesures, "n_total": n_total,
                                     "part": round(couverture, 4)},
        "part_has_pr": round(float(full["has_pr"].mean()), 4),
        "n_reviewers_circularite_vs_churn": round(
            float(full["n_reviewers"].corr(full["file_commits_max"])), 4,
        ),
        "merge_delay_circularite_vs_churn": round(
            float(full["merge_delay_hours"].corr(full["file_commits_max"])), 4,
        ),
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flask_head": flask_head,
        "flask_commit_count": len(commits),
        "label_version": label_version,
        "fenetre_jours": FENETRE_JOURS,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "radon": radon.__version__,
        },
        "feature_sets": results,
        "diagnostics": diagnostics,
    }

    out_name = "results.json" if args.label == "v1" else "results_v2.json"
    out_path = Path(__file__).resolve().parent / out_name
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nEcrit dans {out_path}")


if __name__ == "__main__":
    main()
