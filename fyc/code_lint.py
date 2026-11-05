"""Chantier 2 : Ruff et Bandit sur les versions 'apres' des fichiers .py touches
par chaque commit etiquete (meme filtre que fyc.features.build_code_features).

Batch par dossier temporaire : un seul subprocess ruff + un seul subprocess
bandit sur l'ensemble des fichiers, pas un appel par fichier -- inutilisable a
cette echelle autrement (cf. notebook, cellule 55/56, et fyc.git_extract pour
le meme principe applique a git cat-file).
"""
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import git_extract

RUFF_COLS = ["ruff_total_avg", "ruff_security_avg"]
BANDIT_COLS = ["bandit_total_avg", "bandit_high_avg"]


def _collect_after_sources(commits, est_risque, repo_dir, max_py_par_commit=5):
    """Meme filtre que build_code_features. Retourne (indices, par_commit) avec
    par_commit : i -> liste de (chemin, contenu) pour les .py recuperes avec succes."""
    besoins, specs = [], []
    indices = []
    for i, c in enumerate(commits):
        if est_risque(c) is None or not c["files"]:
            continue
        indices.append(i)
        pyfiles = [f for f in c["files"] if f.endswith(".py")][:max_py_par_commit]
        for f in pyfiles:
            besoins.append((i, f))
            specs.append(f'{c["hash"]}:{f}')

    raw = git_extract.cat_file_batch(repo_dir, specs)
    par_commit = defaultdict(list)
    for k, (i, f) in enumerate(besoins):
        src = raw[k]
        if src is not None:
            par_commit[i].append((f, src))
    return indices, par_commit


def _materialize(tmpdir, par_commit):
    """Ecrit chaque source sur disque sous <tmpdir>/<i>___<chemin aplati>.py.
    Retourne {nom_de_fichier: i} pour retrouver le commit d'origine."""
    name_to_commit = {}
    for i, files in par_commit.items():
        for f, src in files:
            name = f"{i}___{f.replace('/', '_')}"
            name_to_commit[name] = i
            (tmpdir / name).write_text(src, encoding="utf-8", errors="replace")
    return name_to_commit


# Lots de fichiers (6A, 2026-11-05) : avec --select=ALL, la sortie JSON de Ruff sur tous les
# fichiers d'un depot (millions de violations) puis son parsing en objets Python prenaient
# ~1,5 Go pour 1800 commits et ~5,5 Go pour 4200 -- ce qui faisait planter la machine quand
# plusieurs depots etaient traites en parallele. Chaque fichier est analyse independamment
# par Ruff et par Bandit : traiter par lots et agreger au fil de l'eau donne le MEME resultat
# (verifie sur click contre le cache d'origine) avec une memoire bornee.
LOT_FICHIERS = 400


def _lots(noms, taille=LOT_FICHIERS):
    for k in range(0, len(noms), taille):
        yield noms[k:k + taille]


def _run_ruff(targets):
    r = subprocess.run(
        ["ruff", "check", "--output-format=json", "--select=ALL", "--no-cache", *map(str, targets)],
        capture_output=True, text=True,
    )
    if not r.stdout.strip():
        return []
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"sortie ruff non-JSON (premiers 200 car.) : {r.stdout[:200]!r}") from e


def _run_bandit(targets):
    # -q : sans elle, bandit ecrit une barre de progression sur STDOUT (pas stderr),
    # avant le JSON -- casse le parsing. Trouve en debogant un bandit_total_avg a 0
    # partout (cf. CLAUDE.md §7) : toujours verifier les sorties "silencieuses" d'un
    # outil externe plutot que de faire confiance a un except qui avale l'erreur.
    r = subprocess.run(
        ["bandit", "-r", "-q", "-f", "json", *map(str, targets)],
        capture_output=True, text=True,
    )
    if not r.stdout.strip():
        return {}
    try:
        report = json.loads(r.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"sortie bandit non-JSON (premiers 200 car.) : {r.stdout[:200]!r}") from e
    return report.get("metrics", {})


def lint_features(commits, est_risque, repo_dir, max_py_par_commit=5):
    """Lance Ruff ET Bandit en un seul passage de materialisation disque.
    Retourne (ruff_df, bandit_df), alignes ligne a ligne sur les commits etiquetes
    (meme ordre/filtre que fyc.features.build_*), NaN quand aucun .py n'a pu etre
    recupere pour ce commit (meme logique que build_code_features : un echec de
    recuperation ne doit pas ressembler a "aucune violation")."""
    indices, par_commit = _collect_after_sources(commits, est_risque, repo_dir, max_py_par_commit)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        name_to_commit = _materialize(tmpdir, par_commit)
        noms = sorted(name_to_commit)
        ruff_total = defaultdict(int)
        ruff_security = defaultdict(int)
        # Bandit, comme radon (cf. CLAUDE.md §7 point 8), echoue silencieusement sur
        # certains fichiers (vieux code Python 2) : l'entree "metrics" existe mais ne
        # contient alors QUE loc/nosec/skipped_tests, sans les cles SEVERITY.*. Un
        # .get(cle, 0) naif compterait ca comme "0 violation" -- exactement le piege
        # deja rencontre avec has_py/mi_avg=100.0. On ne compte que les fichiers ou
        # bandit a reellement produit des metriques de severite, et on normalise par
        # ce compte-la, pas par le nombre de fichiers recuperes depuis git.
        bandit_total = defaultdict(int)
        bandit_high = defaultdict(int)
        bandit_ok_files = defaultdict(int)
        for lot in _lots(noms):
            chemins = [tmpdir / n for n in lot]
            for v in _run_ruff(chemins):
                i = name_to_commit.get(Path(v["filename"]).name)
                if i is None:
                    continue
                ruff_total[i] += 1
                if v["code"].startswith("S"):
                    ruff_security[i] += 1
            for path, m in _run_bandit(chemins).items():
                if path == "_totals" or "SEVERITY.HIGH" not in m:
                    continue
                i = name_to_commit.get(Path(path).name)
                if i is None:
                    continue
                bandit_total[i] += m["SEVERITY.LOW"] + m["SEVERITY.MEDIUM"] + m["SEVERITY.HIGH"]
                bandit_high[i] += m["SEVERITY.HIGH"]
                bandit_ok_files[i] += 1

    ruff_rows, bandit_rows = [], []
    for i in indices:
        n_files = len(par_commit.get(i, []))
        if n_files:
            ruff_rows.append({
                "ruff_total_avg": ruff_total[i] / n_files,
                "ruff_security_avg": ruff_security[i] / n_files,
            })
        else:
            ruff_rows.append({"ruff_total_avg": np.nan, "ruff_security_avg": np.nan})

        n_ok = bandit_ok_files.get(i, 0)
        if n_ok:
            bandit_rows.append({
                "bandit_total_avg": bandit_total[i] / n_ok,
                "bandit_high_avg": bandit_high[i] / n_ok,
            })
        else:
            bandit_rows.append({"bandit_total_avg": np.nan, "bandit_high_avg": np.nan})

    return pd.DataFrame(ruff_rows), pd.DataFrame(bandit_rows)
