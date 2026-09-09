"""Protocole de mesure pre-enregistre (CLAUDE.md §8, Chantier 1) : label_v1 (fichier)
vs label_v2 (ligne, git blame) -- la circularite churn/label a-t-elle baisse ?

Mesures : M1 (correlation churn/fix), M2 (AUC churn seul absolue), M2bis (ratio
churn/BASE, condition de controle), M3 (taux de bascule des labels). Seuils fixes
AVANT d'avoir vu les resultats (cf. CLAUDE.md). Ecrit results/chantier1_diagnostic.json.

Usage : python results/chantier1_protocol.py   (depuis la racine du repo, venv active)
"""
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from fyc import evaluate, features, git_extract, labeling

REPO_DIR = str(REPO_ROOT / "flask")
FENETRE_JOURS = 90

# Seuils pre-enregistres (CLAUDE.md §8, avant implementation de label_v2)
SEUIL_M1_PEARSON = 0.5
SEUIL_M2_AUC_CHURN_ABS = 0.644
SEUIL_M2BIS_RATIO_FACTEUR = 0.90
SEUIL_M2BIS_GARDE_FOU_BASE_DELTA = 0.03
SEUIL_GARDE_FOU_PART_RISQUE = 0.05


def main():
    print("Extraction des commits...")
    commits = git_extract.extraire_commits(REPO_DIR)
    flask_head = git_extract.get_head(REPO_DIR)
    labeling.tag_fixes(commits)

    fix_par_fichier = labeling.build_fix_index(commits)
    est_risque_v1 = labeling.make_est_risque(fix_par_fichier, fenetre_jours=FENETRE_JOURS)

    print("Calcul du SZZ ligne (git blame -w -M -C)...")
    blame_par_commit, blame_par_fichier, est_risque_v2 = labeling.label_commits_v2(
        commits, REPO_DIR, fenetre_jours=FENETRE_JOURS,
    )

    base_v1 = features.build_base_features(commits, est_risque_v1)
    base_v2 = features.build_base_features(commits, est_risque_v2)
    churn = features.build_churn_features(commits, est_risque_v1)
    assert len(base_v1) == len(base_v2) == len(churn)

    full_v1 = pd.concat([base_v1.drop(columns="risky"), churn, base_v1["risky"]], axis=1)
    full_v2 = pd.concat([base_v2.drop(columns="risky"), churn, base_v2["risky"]], axis=1)

    # Garde-fou (verifie avant tout le reste : si echec, les mesures suivantes ne
    # doivent pas etre interpretees)
    part_risque_v2 = float(base_v2["risky"].mean())
    garde_fou_ok = part_risque_v2 >= SEUIL_GARDE_FOU_PART_RISQUE

    # M1
    corr_v1 = features.churn_fix_correlation(commits, fix_par_fichier)
    corr_v2 = features.churn_fix_correlation(commits, blame_par_fichier)
    m1_pass = corr_v2["pearson"] < SEUIL_M1_PEARSON

    # M2 / M2bis
    auc_churn_v1, std_churn_v1 = evaluate.cv_auc(full_v1, features.CHURN_COLS)
    auc_base_v1, std_base_v1 = evaluate.cv_auc(full_v1, features.BASE_COLS)
    auc_churn_v2, std_churn_v2 = evaluate.cv_auc(full_v2, features.CHURN_COLS)
    auc_base_v2, std_base_v2 = evaluate.cv_auc(full_v2, features.BASE_COLS)

    ratio_v1 = auc_churn_v1 / auc_base_v1
    ratio_v2 = auc_churn_v2 / auc_base_v2
    m2_pass = auc_churn_v2 < SEUIL_M2_AUC_CHURN_ABS
    base_delta = auc_base_v1 - auc_base_v2
    m2bis_ratio_pass = ratio_v2 <= SEUIL_M2BIS_RATIO_FACTEUR * ratio_v1
    m2bis_gardefou_pass = base_delta <= SEUIL_M2BIS_GARDE_FOU_BASE_DELTA
    m2bis_pass = m2bis_ratio_pass and m2bis_gardefou_pass

    # M3
    flip = (base_v1["risky"] == 1) & (base_v2["risky"] == 0)
    pos_v1 = base_v1["risky"] == 1
    q4 = churn["file_commits_max"] >= churn["file_commits_max"].quantile(0.75)
    flip_rate_global = float(flip.mean())
    flip_rate_among_pos_v1 = float(flip.sum() / pos_v1.sum())
    flip_rate_among_pos_v1_q4_churn = (
        float(flip[q4 & pos_v1].sum() / pos_v1[q4].sum()) if pos_v1[q4].sum() else None
    )

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flask_head": flask_head,
        "python_version": platform.python_version(),
        "fenetre_jours": FENETRE_JOURS,
        "n_commits_etiquetes": len(base_v1),
        "seuils_pre_enregistres": {
            "M1_pearson_max": SEUIL_M1_PEARSON,
            "M2_auc_churn_max_abs": SEUIL_M2_AUC_CHURN_ABS,
            "M2bis_ratio_facteur_max": SEUIL_M2BIS_RATIO_FACTEUR,
            "M2bis_garde_fou_base_delta_max": SEUIL_M2BIS_GARDE_FOU_BASE_DELTA,
            "garde_fou_part_risque_min": SEUIL_GARDE_FOU_PART_RISQUE,
        },
        "garde_fou": {
            "part_risque_v2": round(part_risque_v2, 4),
            "pass": garde_fou_ok,
        },
        "M1_correlation_churn_fix": {
            "v1": corr_v1,
            "v2": corr_v2,
            "pass": m1_pass,
        },
        "M2_auc_churn_seul": {
            "v1": {"mean": round(auc_churn_v1, 4), "std": round(std_churn_v1, 4)},
            "v2": {"mean": round(auc_churn_v2, 4), "std": round(std_churn_v2, 4)},
            "pass": m2_pass,
        },
        "M2bis_ratio_churn_base": {
            "auc_base_v1": {"mean": round(auc_base_v1, 4), "std": round(std_base_v1, 4)},
            "auc_base_v2": {"mean": round(auc_base_v2, 4), "std": round(std_base_v2, 4)},
            "ratio_v1": round(ratio_v1, 4),
            "ratio_v2": round(ratio_v2, 4),
            "base_delta_v1_moins_v2": round(base_delta, 4),
            "ratio_pass": m2bis_ratio_pass,
            "garde_fou_base_pass": m2bis_gardefou_pass,
            "pass": m2bis_pass,
        },
        "M3_taux_bascule_diagnostic": {
            "flip_rate_global": round(flip_rate_global, 4),
            "flip_rate_parmi_positifs_v1": round(flip_rate_among_pos_v1, 4),
            "flip_rate_parmi_positifs_v1_fort_churn_q4": (
                round(flip_rate_among_pos_v1_q4_churn, 4)
                if flip_rate_among_pos_v1_q4_churn is not None else None
            ),
        },
    }

    out_path = Path(__file__).resolve().parent / "chantier1_diagnostic.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nEcrit dans {out_path}")


if __name__ == "__main__":
    main()
