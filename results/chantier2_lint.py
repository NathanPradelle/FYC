"""Chantier 2 : Ruff et Bandit apportent-ils un signal reel, non redondant avec
radon et non circulaire avec le churn ?

Methode identique au reste du projet : comparer, mesurer la correlation avant de
conclure, ne pas empiler sans verifier. Ecrit results/chantier2_lint.json.

Usage : python results/chantier2_lint.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from fyc import code_lint, evaluate, features, git_extract, labeling

REPO_DIR = str(REPO_ROOT / "flask")
FENETRE_JOURS = 90
FIX_PREDICATE = labeling.make_fix_predicate(exclude_re=labeling.EXCLUDE_RE)


def main():
    print("Extraction des commits...")
    commits = git_extract.extraire_commits(REPO_DIR)
    flask_head = git_extract.get_head(REPO_DIR)

    _blame, blame_par_fichier, est_risque = labeling.label_commits_v2(
        commits, REPO_DIR, fenetre_jours=FENETRE_JOURS, fix_predicate=FIX_PREDICATE,
    )

    print("Construction des features BASE/churn/code (radon)...")
    base = features.build_base_features(commits, est_risque)
    churn = features.build_churn_features(commits, est_risque)
    code = features.build_code_features(commits, est_risque, REPO_DIR)

    print("Lancement de Ruff + Bandit (batch)...")
    ruff_df, bandit_df = code_lint.lint_features(commits, est_risque, REPO_DIR)

    for name, df in [("churn", churn), ("code", code), ("ruff", ruff_df), ("bandit", bandit_df)]:
        assert len(df) == len(base), f"desynchronisation {name}"

    full = pd.concat(
        [base.drop(columns="risky"), churn, code, ruff_df, bandit_df, base["risky"]], axis=1,
    )

    feature_sets = {
        "BASE": features.BASE_COLS,
        "BASE+code(radon)": features.BASE_COLS + features.CODE_COLS,
        "BASE+ruff": features.BASE_COLS + code_lint.RUFF_COLS,
        "BASE+bandit": features.BASE_COLS + code_lint.BANDIT_COLS,
        "BASE+ruff+bandit": features.BASE_COLS + code_lint.RUFF_COLS + code_lint.BANDIT_COLS,
        "BASE+radon+ruff+bandit": (
            features.BASE_COLS + features.CODE_COLS + code_lint.RUFF_COLS + code_lint.BANDIT_COLS
        ),
        "ruff+bandit_seuls": code_lint.RUFF_COLS + code_lint.BANDIT_COLS,
        "tout": (
            features.BASE_COLS + features.CHURN_COLS + features.CODE_COLS
            + code_lint.RUFF_COLS + code_lint.BANDIT_COLS
        ),
    }

    auc = {}
    for name, cols in feature_sets.items():
        mean, std = evaluate.cv_auc(full, cols)
        auc[name] = {"mean": round(mean, 4), "std": round(std, 4)}
        print(f"  {name:28s} AUC = {mean:.3f} +- {std:.3f}")

    # Redondance avec radon (les features de code existantes) : corrélation entre le
    # total de violations ruff et loc_avg (deja identifie comme proxy de "gros fichier")
    redondance_radon = {
        "ruff_total_vs_loc_avg": round(float(full["ruff_total_avg"].corr(full["loc_avg"])), 4),
        "bandit_total_vs_loc_avg": round(float(full["bandit_total_avg"].corr(full["loc_avg"])), 4),
    }

    # Circularite avec le churn : le total de violations ruff/bandit est-il correle au
    # churn du fichier (meme diagnostic que pour radon/code, cf. CLAUDE.md §6) ?
    circularite_churn = {
        "ruff_total_vs_file_commits_max": round(
            float(full["ruff_total_avg"].corr(full["file_commits_max"])), 4,
        ),
        "bandit_total_vs_file_commits_max": round(
            float(full["bandit_total_avg"].corr(full["file_commits_max"])), 4,
        ),
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flask_head": flask_head,
        "label_version": "v2_ligne_szz_blame",
        "fix_detection": "candidat_B_exclude_typo_docs_changelog",
        "n_commits_etiquetes": len(base),
        "auc": auc,
        "redondance_avec_radon": redondance_radon,
        "circularite_avec_churn": circularite_churn,
        "ruff_couverture": {
            "n_non_nan": int(full["ruff_total_avg"].notna().sum()),
            "n_total": len(full),
        },
        "bandit_couverture": {
            "n_non_nan": int(full["bandit_total_avg"].notna().sum()),
            "n_total": len(full),
        },
    }

    out_path = Path(__file__).resolve().parent / "chantier2_lint.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nRedondance avec radon : {redondance_radon}")
    print(f"Circularite avec churn : {circularite_churn}")
    print(f"\nEcrit dans {out_path}")


if __name__ == "__main__":
    main()
