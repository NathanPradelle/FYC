"""Chantier 3 : analyse du diff lui-meme, pas seulement l'etat du fichier.

Trois familles, independantes les unes des autres :
- dispersion du diff (nombre de hunks, taille moyenne de hunk) ;
- ratio code/test en lignes (amelioration du `touches_tests` binaire de BASE) ;
- changement de signature de fonction via un diff d'AST avant/apres (ajout,
  suppression, modification de signature).

Meme discipline que fyc.features/fyc.code_lint : NaN sur echec de mesure (parsing
AST rate, pas de parent a diffar...), jamais une valeur par defaut qui ressemble a
"rien a signaler".
"""
import ast
from collections import defaultdict

import numpy as np
import pandas as pd

from . import git_extract

DISPERSION_COLS = ["n_hunks", "avg_hunk_size"]
TEST_RATIO_COLS = ["test_line_ratio"]
SIGNATURE_COLS = ["sig_added", "sig_removed", "sig_modified", "touches_signature"]
# Toutes les colonnes calculees -- utile pour re-mesurer/reproduire (cf.
# results/chantier3_diff.py), mais PAS ce qui est adopte par defaut (voir DIFF_COLS).
ALL_COLS = DISPERSION_COLS + TEST_RATIO_COLS + SIGNATURE_COLS

# Jeu adopte (mesure le 2026-09-09, cf. CLAUDE.md §6) : `avg_hunk_size` (AUC seul
# 0.550, trop faible) et `test_line_ratio` (AUC seul 0.499, PIRE que le `touches_tests`
# binaire existant -- resultat negatif, pas adopte) sont exclus. `n_hunks` et les
# features de signature sont gardes (signal reel, non circulaire avec le churn).
DIFF_COLS = ["n_hunks"] + SIGNATURE_COLS


def _function_signatures(src):
    """{nom_qualifie: tuple(args)} pour toutes les def (sync/async) du fichier,
    imbrication incluse. None si le parsing echoue -- un echec AST ne doit pas
    ressembler a "ce fichier n'a aucune fonction"."""
    if src is None:
        return None
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return None

    sigs = {}
    stack = []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join(stack + [node.name])
            sigs[qualname] = tuple(a.arg for a in node.args.args)
            stack.append(node.name)
            for child in ast.iter_child_nodes(node):
                visit(child)
            stack.pop()
        elif isinstance(node, ast.ClassDef):
            stack.append(node.name)
            for child in ast.iter_child_nodes(node):
                visit(child)
            stack.pop()
        else:
            for child in ast.iter_child_nodes(node):
                visit(child)

    visit(tree)
    return sigs


def _signature_diff(before_sigs, after_sigs):
    """None si l'un des deux cotes n'a pas pu etre parse."""
    if before_sigs is None or after_sigs is None:
        return None
    before_keys, after_keys = set(before_sigs), set(after_sigs)
    common = before_keys & after_keys
    return {
        "added": len(after_keys - before_keys),
        "removed": len(before_keys - after_keys),
        "modified": sum(1 for k in common if before_sigs[k] != after_sigs[k]),
    }


def _count_hunks(diff_text):
    """Nombre de hunks dans un diff unifie (lignes '@@ -x,y +a,b @@')."""
    return sum(1 for line in diff_text.splitlines() if line.startswith("@@ "))


def build_diff_features(commits, est_risque, repo_dir, max_py_par_commit=5):
    """Memes filtre et ordre que fyc.features.build_base_features : un echec de
    recuperation/parsing produit NaN, jamais une valeur qui imite un cas normal."""
    parents = git_extract.get_parents(repo_dir)

    indices = [i for i, c in enumerate(commits)
               if est_risque(c) is not None and c["files"]]

    # Signatures AST : meme principe de recuperation batch que radon (git cat-file).
    besoins, specs = [], []
    for i in indices:
        c = commits[i]
        pyfiles = [f for f in c["files"] if f.endswith(".py")][:max_py_par_commit]
        par = parents.get(c["hash"], [])
        for f in pyfiles:
            besoins.append((i, f, "after"))
            specs.append(f'{c["hash"]}:{f}')
            if par:
                besoins.append((i, f, "before"))
                specs.append(f'{par[0]}:{f}')

    raw = git_extract.cat_file_batch(repo_dir, specs)
    contents = {besoins[k]: raw[k] for k in range(len(besoins))}

    sig_agg = defaultdict(lambda: {"added": 0, "removed": 0, "modified": 0, "n_ok": 0})
    for (i, f, quand), src in contents.items():
        if quand != "after":
            continue
        diff = _signature_diff(_function_signatures(contents.get((i, f, "before"))),
                                _function_signatures(src))
        if diff is None:
            continue
        agg = sig_agg[i]
        agg["added"] += diff["added"]
        agg["removed"] += diff["removed"]
        agg["modified"] += diff["modified"]
        agg["n_ok"] += 1

    rows = []
    for i in indices:
        c = commits[i]
        par = parents.get(c["hash"], [])

        # Dispersion : un appel `git diff` par commit (pas de cat-file batch possible
        # ici, le besoin -- compter des hunks -- est par construction par-commit).
        if par:
            diff_text = git_extract.diff_raw(repo_dir, par[0], c["hash"])
            n_hunks = float(_count_hunks(diff_text))
            total_churn = c["add"] + c["del"]
            avg_hunk_size = total_churn / n_hunks if n_hunks > 0 else np.nan
        else:
            n_hunks, avg_hunk_size = 0.0, np.nan  # racine : pas de parent, pas d'echec

        # Ratio test en lignes (necessite file_stats, ajoute a extraire_commits).
        total_churn = c["add"] + c["del"]
        test_churn = sum(a + d for f, a, d in c.get("file_stats", []) if "test" in f.lower())
        test_line_ratio = test_churn / total_churn if total_churn > 0 else np.nan

        s = sig_agg.get(i)
        if s and s["n_ok"] > 0:
            sig_row = {
                "sig_added": float(s["added"]),
                "sig_removed": float(s["removed"]),
                "sig_modified": float(s["modified"]),
                "touches_signature": int(bool(s["added"] or s["removed"] or s["modified"])),
            }
        else:
            sig_row = {k: np.nan for k in SIGNATURE_COLS}

        rows.append({
            "n_hunks": n_hunks,
            "avg_hunk_size": avg_hunk_size,
            "test_line_ratio": test_line_ratio,
            **sig_row,
        })

    return pd.DataFrame(rows)
