"""Chantier 4bis, vague 1 : indicateurs de Kamei et al. (2013) absents du pipeline.

Deja couverts (pas recalcules) : NF=n_files, LA/LD=lines_added/deleted (BASE), EXP=
author_experience (BASE), AGE/NUC ~ churn (file_age_days_avg, file_commits_*).
Nouveaux : NS (sous-systemes touches), ND (repertoires touches), entropie du changement,
NDEV (auteurs distincts ayant deja touche ces fichiers), REXP (experience recente
ponderee), SEXP (experience de l'auteur sur les sous-systemes touches).
FIX et LT sont ecartes : FIX est la definition meme du label ; LT (taille du fichier
avant changement) est ce que `loc_avg` (radon) mesure deja.

Entropie : H = -sum p_f log2 p_f, p_f = part des lignes modifiees (add+del) du fichier f.
`entropy_norm` = H / log2(n_files) (0 si un seul fichier). NaN si aucune ligne comptee
(commit binaire/renommage pur) -- pas 0, qui signifierait 'modification parfaitement
concentree'.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

KAMEI_COLS = ["ns", "nd", "entropy", "entropy_norm", "ndev", "rexp", "sexp"]
SECONDS_PER_YEAR = 365.25 * 86400


def _subsystem(path):
    parts = path.split("/")
    if parts[0] == "src" and len(parts) > 2:
        parts = parts[1:]
    return parts[0] if len(parts) > 1 else "."


def _directory(path):
    return path.rsplit("/", 1)[0] if "/" in path else "."


def _entropy(file_stats):
    lines = np.array([a + d for _f, a, d in file_stats], dtype=float)
    total = lines.sum()
    if total <= 0:
        return np.nan, np.nan
    p = lines[lines > 0] / total
    h = float(-(p * np.log2(p)).sum()) + 0.0  # +0.0 : evite -0.0
    n = len(file_stats)
    return h, (h / np.log2(n) if n > 1 else 0.0)


def build_kamei_features(commits, est_risque):
    """Meme filtre/ordre que fyc.features.build_base_features ; l'etat historique
    (auteurs par fichier, experience) est mis a jour pour TOUS les commits."""
    file_authors = defaultdict(set)
    author_times = defaultdict(list)
    author_sub = defaultdict(int)  # (auteur, sous-systeme) -> nb de commits
    rows = []

    for c in commits:
        files = c["files"]
        subs = sorted({_subsystem(f) for f in files})
        if est_risque(c) is not None and files:
            h, hn = _entropy(c.get("file_stats", []))
            ndev = len(set().union(*(file_authors[f] for f in files)))
            rexp = sum(1.0 / ((c["ts"] - t) / SECONDS_PER_YEAR + 1.0)
                       for t in author_times[c["author"]])
            sexp = float(np.mean([author_sub[(c["author"], s)] for s in subs]))
            rows.append({"ns": len(subs), "nd": len({_directory(f) for f in files}),
                         "entropy": h, "entropy_norm": hn, "ndev": ndev,
                         "rexp": rexp, "sexp": sexp})
        author_times[c["author"]].append(c["ts"])
        for s in subs:
            author_sub[(c["author"], s)] += 1
        for f in files:
            file_authors[f].add(c["author"])
    return pd.DataFrame(rows)
