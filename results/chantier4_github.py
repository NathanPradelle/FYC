"""Chantier 4 : les signaux de processus GitHub (reviewers, delai de merge, etat
CI) ajoutent-ils un signal reel, au-dela du contexte Git/code deja mesure ?

Meme methode que les chantiers precedents : mesurer la couverture du cache avant
toute conclusion (un cache incomplet biaiserait silencieusement les chiffres),
comparer a BASE, verifier la redondance et la circularite. Ecrit
results/chantier4_github.json.

Usage : python results/chantier4_github.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from fyc import evaluate, features, git_extract, labeling, process_features

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

    print("Signaux de process GitHub (cache disque, resumable)...")
    process_df, (n_mesures, n_total), mask = process_features.build_process_features(
        commits, est_risque,
    )
    couverture = n_mesures / n_total
    print(f"Couverture cache GitHub : {n_mesures}/{n_total} ({couverture:.1%})")

    if couverture < 1.0:
        print(f"ATTENTION : cache incomplet ({n_total - n_mesures} commits non "
              f"mesures) -- les lignes correspondantes sont exclues ci-dessous, "
              f"pas remplies par defaut.")

    mask = pd.Series(mask)
    base_m = base[mask.values].reset_index(drop=True)
    churn_m = churn[mask.values].reset_index(drop=True)
    code_m = code[mask.values].reset_index(drop=True)

    for name, df in [("churn", churn_m), ("code", code_m), ("process", process_df)]:
        assert len(df) == len(base_m), f"desynchronisation {name} ({len(df)} != {len(base_m)})"

    full = pd.concat(
        [base_m.drop(columns="risky"), churn_m, code_m, process_df, base_m["risky"]], axis=1,
    )

    PROCESS = process_features.PROCESS_COLS
    feature_sets = {
        "BASE": features.BASE_COLS,
        "BASE+process": features.BASE_COLS + PROCESS,
        "BASE+code(radon)": features.BASE_COLS + features.CODE_COLS,
        "BASE+code+process": features.BASE_COLS + features.CODE_COLS + PROCESS,
        "process_only": PROCESS,
        "has_pr_seul": ["has_pr"],
        "n_reviewers_seul": ["n_reviewers"],
        "merge_delay_seul": ["merge_delay_hours"],
        "ci_failed_seul": ["ci_failed"],
        "tout_sans_process": features.BASE_COLS + features.CHURN_COLS + features.CODE_COLS,
        "tout_avec_process": (
            features.BASE_COLS + features.CHURN_COLS + features.CODE_COLS + PROCESS
        ),
    }

    auc = {}
    for name, cols in feature_sets.items():
        mean, std = evaluate.cv_auc(full, cols)
        auc[name] = {"mean": round(mean, 4), "std": round(std, 4)}
        print(f"  {name:22s} AUC = {mean:.3f} +- {std:.3f}")

    # Circularite : le delai de merge/nb de reviewers sont-ils juste une autre
    # facon de mesurer le churn du fichier (meme diagnostic qu'aux chantiers precedents) ?
    circularite = {
        "n_reviewers_vs_file_commits_max": round(
            float(full["n_reviewers"].corr(full["file_commits_max"])), 4,
        ),
        "merge_delay_vs_file_commits_max": round(
            float(full["merge_delay_hours"].corr(full["file_commits_max"])), 4,
        ),
        "n_reviewers_vs_total_churn": round(
            float(full["n_reviewers"].corr(full["total_churn"])), 4,
        ),
    }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flask_head": flask_head,
        "label_version": "v2_ligne_szz_blame",
        "fix_detection": "candidat_B_exclude_typo_docs_changelog",
        "n_commits_etiquetes": n_total,
        "couverture_cache_github": {"n_mesures": n_mesures, "n_total": n_total,
                                     "part": round(couverture, 4)},
        "part_has_pr": round(float(process_df["has_pr"].mean()), 4),
        "auc": auc,
        "circularite_avec_churn": circularite,
    }

    out_path = Path(__file__).resolve().parent / "chantier4_github.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nCircularite : {circularite}")
    print(f"\nEcrit dans {out_path}")


if __name__ == "__main__":
    main()
