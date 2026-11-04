"""Chantier 4bis, vague 1 : couplage entre modules par AST (imports), a l'etat de CHAQUE
commit (arbre complet via `git ls-tree`, contenu des blobs en un seul cat-file --batch,
analyse de chaque blob distinct une seule fois).

fan_out = nb de modules INTERNES au depot importes par un fichier modifie ;
fan_in  = nb de fichiers du depot (a ce commit) qui importent ce module.
Agreges sur les .py modifies par le commit (moyenne, et max pour fan_in).

Pieges (meme famille que CLAUDE.md §7 pts 8/12) : un blob non parsable (vieux Python 2)
n'est pas 'sans imports' -- il est exclu de l'index de fan-in (sous-estimation possible,
mesuree par `coupling_tree_unparsable` dans le script de mesure) ; un commit dont aucun
.py modifie n'a pu etre analyse recoit NaN.
"""
import ast
import subprocess
from collections import Counter

import numpy as np
import pandas as pd

from . import git_extract

COUPLING_COLS = ["fan_out_avg", "fan_in_avg", "fan_in_max"]


def _module_name(path):
    p = path[:-3] if path.endswith(".py") else path
    parts = p.split("/")
    if parts[0] == "src" and len(parts) > 1:
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _raw_imports(src):
    """[(level, module, names)] ou None si non parsable."""
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError):
        return None
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [(0, a.name, ()) for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out.append((n.level, n.module or "", tuple(a.name for a in n.names)))
    return out


def _resolve(path, raw):
    """Cibles absolues (str) d'un fichier : module importe + sous-modules potentiels."""
    pkg = _module_name(path).split(".")
    if not path.endswith("__init__.py"):
        pkg = pkg[:-1]
    targets = set()
    for level, mod, names in raw:
        if level:
            base = pkg[:len(pkg) - (level - 1)] if level > 1 else pkg
            full = ".".join(base + ([mod] if mod else []))
        else:
            full = mod
        if full:
            targets.add(full)
        for nm in names:
            targets.add(f"{full}.{nm}" if full else nm)
    return targets


def _ls_py(repo_dir, rev):
    out = subprocess.run(["git", "-C", repo_dir, "ls-tree", "-r", rev],
                         capture_output=True, text=True, check=True).stdout
    res = []
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        if path.endswith(".py"):
            res.append((meta.split()[2], path))
    return res


def build_coupling_features(commits, est_risque, repo_dir):
    indices = [i for i, c in enumerate(commits) if est_risque(c) is not None and c["files"]]
    trees = {i: _ls_py(repo_dir, commits[i]["hash"]) for i in indices}

    shas = sorted({sha for t in trees.values() for sha, _ in t})
    raw = git_extract.cat_file_batch(repo_dir, shas)
    imports = {}
    for k, sha in enumerate(shas):
        src = raw[k]
        imports[sha] = _raw_imports(src) if src is not None else None

    rows, unparsable = [], []
    for i in indices:
        tree = trees[i]
        modules = {_module_name(p) for _s, p in tree}
        fan_in = Counter()
        out_targets = {}
        n_bad = 0
        for sha, path in tree:
            r = imports[sha]
            if r is None:
                n_bad += 1
                continue
            internal = _resolve(path, r) & modules
            internal.discard(_module_name(path))
            out_targets[path] = internal
            fan_in.update(internal)
        unparsable.append(n_bad / len(tree) if tree else np.nan)

        changed = [f for f in commits[i]["files"] if f in out_targets]
        if not changed:
            rows.append({c: np.nan for c in COUPLING_COLS})
            continue
        fi = [fan_in[_module_name(f)] for f in changed]
        rows.append({"fan_out_avg": float(np.mean([len(out_targets[f]) for f in changed])),
                     "fan_in_avg": float(np.mean(fi)), "fan_in_max": float(max(fi))})
    df = pd.DataFrame(rows)
    df.attrs["tree_unparsable"] = unparsable
    return df
