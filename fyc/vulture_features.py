"""Chantier 4bis, vague 1 : Vulture (code mort) -- en process, un objet Vulture PAR commit
(les noms utilises dans une version d'un autre commit ne doivent pas 'sauver' un
symbole). Pas de subprocess : la bibliotheque est utilisee directement.

Biais connu et assume : sur un fichier isole, toute fonction/classe publique non
appelee dans CE fichier est 'inutilisee' -- Vulture mesure donc en partie 'nombre de
definitions publiques' (redondance attendue avec loc/radon, a quantifier). Pour limiter
ce biais, min_confidence=60 (defaut) et comptes par fichier, pas absolus.
Fichiers non parsables : exclus, NaN si aucun fichier analyse (jamais 0).
"""
import ast
from collections import defaultdict

import numpy as np
import pandas as pd
import vulture

from .code_lint import _collect_after_sources

VULTURE_COLS = ["vulture_total_avg", "vulture_funcs_avg", "vulture_imports_avg"]


def vulture_features(commits, est_risque, repo_dir, max_py_par_commit=5):
    indices, par_commit = _collect_after_sources(commits, est_risque, repo_dir, max_py_par_commit)
    rows = []
    for i in indices:
        v = vulture.Vulture()
        ok = 0
        for f, src in par_commit.get(i, []):
            try:
                ast.parse(src)
            except (SyntaxError, ValueError, RecursionError):
                continue
            v.scan(src, filename=f)
            ok += 1
        if not ok:
            rows.append({c: np.nan for c in VULTURE_COLS})
            continue
        kinds = defaultdict(int)
        for item in v.get_unused_code(min_confidence=60):
            kinds[item.typ] += 1
        rows.append({
            "vulture_total_avg": sum(kinds.values()) / ok,
            "vulture_funcs_avg": (kinds["function"] + kinds["method"] + kinds["class"]) / ok,
            "vulture_imports_avg": kinds["import"] / ok,
        })
    return pd.DataFrame(rows)
