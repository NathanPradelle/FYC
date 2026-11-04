"""Chantier 4bis, vague 1 : mypy (types) sur les versions 'apres' des .py touches.

Risque identifie dans le protocole : sur des fichiers isoles, la plupart des erreurs
seront du bruit d'import ou de typage de bibliotheques absentes. Mitigations :
--ignore-missing-imports, --follow-imports=skip (pas de propagation entre fichiers),
codes d'erreur retenus limites a ceux qui parlent du code lui-meme (voir CODES_RETENUS).
Un SyntaxError dans UN fichier fait avorter TOUTE l'analyse mypy ('errors prevented
further checking') : on pre-filtre donc avec ast.parse et on n'envoie que les fichiers
parsables ; les autres comptent comme 'non analyses' (NaN si aucun), jamais 0.
Resultat du test de bruit (100 commits) : voir results/chantier4bis_wave1.py / CLAUDE.md §6.
"""
import ast
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .code_lint import _collect_after_sources

MYPY_COLS = ["mypy_total_avg", "mypy_core_avg"]
# Codes qui concernent la logique du fichier (pas l'environnement) :
CODES_RETENUS = {"attr-defined", "arg-type", "call-arg", "index", "operator", "return-value",
                 "assignment", "union-attr", "name-defined", "override", "misc", "call-overload",
                 "has-type", "var-annotated", "valid-type", "return", "no-redef", "str-format"}
_LINE = re.compile(r"^(?P<path>.+?):\d+(?::\d+)?: error: .*\[(?P<code>[\w-]+)\]\s*$")


def mypy_features(commits, est_risque, repo_dir, max_py_par_commit=5, only_indices=None):
    indices, par_commit = _collect_after_sources(commits, est_risque, repo_dir, max_py_par_commit)
    if only_indices is not None:
        keep = set(only_indices)
        indices = [i for i in indices if i in keep]
        par_commit = {i: v for i, v in par_commit.items() if i in keep}
    parsable = {}
    for i, files in par_commit.items():
        ok = []
        for f, src in files:
            try:
                ast.parse(src)
                ok.append((f, src))
            except (SyntaxError, ValueError, RecursionError):
                pass
        parsable[i] = ok

    total, core = defaultdict(int), defaultdict(int)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        # Un PACKAGE par commit (pk<i>/__init__.py) : sans parent de package, un simple
        # 'from . import x' est une erreur BLOQUANTE de mypy qui avorte toute l'analyse.
        name_to_commit = {}
        for i, files in parsable.items():
            if not files:
                continue
            # 3 niveaux : 'from ..x' / 'from ...x' (sous-paquets Flask) doivent aussi resoudre.
            pkg = tmpdir / f"pk{i}"
            deep = pkg / "a" / "b"
            deep.mkdir(parents=True)
            for d in (pkg, pkg / "a", deep):
                (d / "__init__.py").write_text("")
            for f, src in files:
                name = f"{f.replace('/', '_')[:-3]}_{len(name_to_commit)}.py"
                (deep / name).write_text(src, encoding="utf-8", errors="replace")
                name_to_commit[f"pk{i}/{name}"] = i
        r = subprocess.run(
            ["mypy", "--ignore-missing-imports", "--follow-imports=skip", "--no-incremental", "--check-untyped-defs",
             "--no-error-summary", "--show-error-codes", "--no-color-output", "--python-version=3.14",
             str(tmpdir)],
            capture_output=True, text=True, cwd=tmp,
        )
        if r.returncode not in (0, 1):
            raise RuntimeError(f"mypy code {r.returncode} : {(r.stdout + r.stderr)[:400]!r}")
        for line in r.stdout.splitlines():
            m = _LINE.match(line)
            if not m:
                continue
            i = name_to_commit.get(Path(m["path"]).parts[-4] + "/" + Path(m["path"]).name)
            if i is None:
                continue
            total[i] += 1
            if m["code"] in CODES_RETENUS:
                core[i] += 1

    rows = []
    for i in indices:
        n_ok = len(parsable.get(i, []))
        if not n_ok:
            rows.append({c: np.nan for c in MYPY_COLS})
        else:
            rows.append({"mypy_total_avg": total[i] / n_ok, "mypy_core_avg": core[i] / n_ok})
    return pd.DataFrame(rows)
