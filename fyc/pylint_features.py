"""Chantier 4bis, vague 1 : Pylint sur les versions 'apres' des .py touches (meme filtre
et meme materialisation disque que fyc.code_lint : un seul appel pylint pour tout).

Choix documentes (CLAUDE.md §8, 4bis) :
- duplicate-code (R0801) desactive : les fichiers de commits differents sont melanges
  dans un meme dossier, donc deux versions du meme fichier se 'dupliqueraient' -- mesure
  artificielle. La duplication est le role de jscpd (vague 2).
- import-error / no-name-in-module / relative-beyond-top-level desactives : fichiers
  isoles, imports non resolvables par construction (bruit pur, pas un defaut du code).
- invalid-name desactive : les noms de fichiers sont aplatis par la materialisation.
- Piege §7 pts 8/12 : un fichier non parsable ressort comme message 'syntax-error' (E0001),
  pas comme silence. On l'exclut du denominateur et du numerateur ; NaN si aucun fichier
  du commit n'a ete analyse -- jamais 0.
"""
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .code_lint import _collect_after_sources, _materialize

PYLINT_COLS = ["pylint_total_avg", "pylint_error_avg", "pylint_warning_avg",
               "pylint_refactor_avg", "pylint_convention_avg"]
DISABLED = ["duplicate-code", "import-error", "no-name-in-module", "relative-beyond-top-level",
            "invalid-name"]
SYNTAX_IDS = {"E0001", "E0011", "F0001", "F0002", "F0010"}  # echec de parsing / fatal


def _run_pylint(tmpdir):
    r = subprocess.run(
        ["pylint", "--output-format=json2", "--jobs=2", "--exit-zero",
         f"--disable={','.join(DISABLED)}", "--score=n", str(tmpdir)],
        capture_output=True, text=True,
    )
    if not r.stdout.strip():
        raise RuntimeError(f"pylint sans sortie JSON (stderr : {r.stderr[:300]!r})")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"sortie pylint non-JSON : {r.stdout[:200]!r}") from e


def pylint_features(commits, est_risque, repo_dir, max_py_par_commit=5, only_indices=None):
    """DataFrame PYLINT_COLS aligne sur les commits etiquetes (meme ordre que fyc.features).
    only_indices : sous-ensemble d'indices de `commits` (chronometrage sur echantillon) ;
    les lignes retournees suivent alors cet ordre filtre."""
    indices, par_commit = _collect_after_sources(commits, est_risque, repo_dir, max_py_par_commit)
    if only_indices is not None:
        keep = set(only_indices)
        indices = [i for i in indices if i in keep]
        par_commit = {i: v for i, v in par_commit.items() if i in keep}

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        name_to_commit = _materialize(tmpdir, par_commit)
        report = _run_pylint(tmpdir)

    failed = set()
    counts = defaultdict(lambda: defaultdict(int))
    for m in report["messages"]:
        name = Path(m["path"]).name
        if name not in name_to_commit:
            continue
        if m["messageId"] in SYNTAX_IDS:
            failed.add(name)
            continue
        counts[name][m["type"]] += 1  # convention / refactor / warning / error / fatal
    # Un fichier en echec de syntaxe peut aussi avoir d'autres messages : on l'ecarte entierement.
    ok_by_commit = defaultdict(list)
    for name, i in name_to_commit.items():
        if name not in failed:
            ok_by_commit[i].append(name)

    rows = []
    for i in indices:
        names = ok_by_commit.get(i, [])
        if not names:
            rows.append({c: np.nan for c in PYLINT_COLS})
            continue
        agg = defaultdict(int)
        for n in names:
            for t, k in counts[n].items():
                agg[t] += k
        n_ok = len(names)
        rows.append({
            "pylint_total_avg": sum(agg.values()) / n_ok,
            "pylint_error_avg": agg["error"] / n_ok,
            "pylint_warning_avg": agg["warning"] / n_ok,
            "pylint_refactor_avg": agg["refactor"] / n_ok,
            "pylint_convention_avg": agg["convention"] / n_ok,
        })
    return pd.DataFrame(rows)
