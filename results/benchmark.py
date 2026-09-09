"""Recalcule tous les chiffres de la Partie 4 du notebook (comparaison des jeux de
features) et les ecrit dans results/results.json, avec la date, les versions des
librairies et le hash du HEAD du depot Flask analyse.

But : plus aucun chiffre en dur dans le markdown du notebook. Chaque resultat est
date et rattache a un etat de donnees identifiable, pour que l'ecart constate entre
le narratif du notebook et une execution reelle (2026-09-09, cf. CLAUDE.md §5/§7) ne
puisse plus se reproduire silencieusement.

Usage : python results/benchmark.py   (depuis la racine du repo, venv active)
"""
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

from fyc import evaluate, features, git_extract, labeling

REPO_DIR = str(REPO_ROOT / "flask")
FENETRE_JOURS = 90


def main():
    print("Extraction des commits...")
    commits = git_extract.extraire_commits(REPO_DIR)
    flask_head = git_extract.get_head(REPO_DIR)
    print(f"{len(commits)} commits, HEAD={flask_head}")

    fix_par_fichier, est_risque = labeling.label_commits(commits, fenetre_jours=FENETRE_JOURS)

    print("Construction des features BASE...")
    base = features.build_base_features(commits, est_risque)
    is_night = features.build_is_night(base)
    print("Construction des features churn...")
    churn = features.build_churn_features(commits, est_risque)
    print("Construction des features de code (radon, via git cat-file)...")
    code = features.build_code_features(commits, est_risque, REPO_DIR)

    for name, df in [("churn", churn), ("code", code)]:
        assert len(df) == len(base), f"desynchronisation {name} ({len(df)} != {len(base)})"

    full = pd.concat(
        [base.drop(columns="risky"), is_night, churn, code, base["risky"]], axis=1,
    )

    feature_sets = {
        "BASE": features.BASE_COLS,
        "BASE+is_night": features.BASE_COLS + ["is_night"],
        "BASE+churn": features.BASE_COLS + features.CHURN_COLS,
        "BASE+code": features.BASE_COLS + features.CODE_COLS,
        "code_only": features.CODE_COLS,
        "churn_only": features.CHURN_COLS,
        "all": features.BASE_COLS + features.CHURN_COLS + features.CODE_COLS,
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
        "churn_fix_correlation": features.churn_fix_correlation(commits, fix_par_fichier),
        "baseline_majority_accuracy": evaluate.baseline_majority_accuracy(base),
        "n_commits_total": len(commits),
        "n_commits_etiquetes": len(base),
        "part_risque": round(float(base["risky"].mean()), 4),
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flask_head": flask_head,
        "flask_commit_count": len(commits),
        "label_version": "v1_fichier_szz_simplifie",
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

    out_path = Path(__file__).resolve().parent / "results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nEcrit dans {out_path}")


if __name__ == "__main__":
    main()
