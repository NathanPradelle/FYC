"""Chantier 4 : features derivees des signaux de process GitHub (fyc.github_signals) --
nombre de reviewers distincts, delai de merge, etat de la CI.

Meme filtre et ordre que les autres familles (cf. fyc.features) pour l'alignement
ligne a ligne dans results/benchmark.py.

Discipline NaN : `has_pr` distingue explicitement "aucune PR associee" (0, mesure
confirmee) de "pas encore mesure" (budget GraphQL epuise avant ce commit -- cf.
fyc.github_signals) -- un commit non mesure est exclu du DataFrame plutot que
recevoir une valeur par defaut, meme piege que has_py/Bandit (CLAUDE.md §7).
"""
from datetime import datetime

import numpy as np
import pandas as pd

from . import github_signals

PROCESS_COLS = ["has_pr", "n_reviewers", "n_review_comments", "merge_delay_hours", "ci_failed"]


def build_process_features(commits, est_risque, min_remaining=50):
    """Retourne (df, couverture, mask) : couverture = (n_mesures, n_total) pour
    les commits etiquetables ; mask = booleens alignes sur le MEME filtre/ordre
    que fyc.features (est_risque(c) is not None and c["files"]), pour sous-
    echantillonner base/churn/code aux memes lignes avant de les concatener
    (cf. results/chantier4_github.py)."""
    indices = [i for i, c in enumerate(commits) if est_risque(c) is not None and c["files"]]
    shas = [commits[i]["hash"] for i in indices]

    cache = github_signals.fetch_commit_pr_signals(shas, min_remaining=min_remaining)

    rows, mesures = [], []
    for i in indices:
        sha = commits[i]["hash"]
        mesure = sha in cache
        mesures.append(mesure)
        if not mesure:
            continue

        s = cache[sha]
        if s is None:
            rows.append({"has_pr": 0, "n_reviewers": np.nan, "n_review_comments": np.nan,
                         "merge_delay_hours": np.nan, "ci_failed": np.nan})
            continue

        merge_delay = np.nan
        if s.get("created_at") and s.get("merged_at"):
            created = datetime.fromisoformat(s["created_at"].replace("Z", "+00:00"))
            merged = datetime.fromisoformat(s["merged_at"].replace("Z", "+00:00"))
            merge_delay = (merged - created).total_seconds() / 3600

        ci_state = s.get("ci_state")
        ci_failed = np.nan
        if ci_state in ("SUCCESS", "FAILURE", "ERROR"):
            ci_failed = int(ci_state in ("FAILURE", "ERROR"))

        n_comments = s.get("n_comments")
        rows.append({
            "has_pr": 1,
            "n_reviewers": float(s.get("n_reviewers") or 0),
            "n_review_comments": float(n_comments) if n_comments is not None else np.nan,
            "merge_delay_hours": merge_delay,
            "ci_failed": ci_failed,
        })

    df = pd.DataFrame(rows)
    couverture = (sum(mesures), len(indices))
    return df, couverture, mesures
