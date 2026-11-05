"""Construction des jeux de features. Porte les cellules 22, 40, 43, 49-51 du notebook.

Chaque fonction filtre les commits sur (est_risque(c) is not None and len(c["files"]) > 0)
et retourne un DataFrame dans cet ordre : les quatre familles (BASE, is_night, churn, code)
produisent donc des DataFrames alignes ligne a ligne, pret a etre concatenes.
"""
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import git_extract

BASE_COLS = ["hour", "weekend", "n_files", "lines_added", "lines_deleted",
             "author_experience", "avg_authors_per_file", "total_churn", "msg_len",
             "touches_tests"]
CHURN_COLS = ["file_commits_max", "file_commits_avg", "file_age_days_avg"]
CODE_COLS = ["cc_tot_avg", "cc_max_avg", "mi_avg", "loc_avg", "comment_ratio", "delta_cc",
             "touches_py", "code_parse_ok"]


def _commits_etiquetables(commits, est_risque):
    """Filtre + label, dans l'ordre chronologique (meme filtre partout pour l'alignement)."""
    for c in commits:
        label = est_risque(c)
        if label is not None and len(c["files"]) > 0:
            yield c, label


def build_base_features(commits, est_risque):
    author_xp = defaultdict(int)
    file_authors = defaultdict(set)
    lignes = []

    for commit in commits:
        label = est_risque(commit)
        n_files = len(commit["files"])
        if label is not None and n_files > 0:
            dt = datetime.fromtimestamp(commit["ts"], tz=timezone.utc)
            ownership = sum(len(file_authors[f]) for f in commit["files"]) / n_files
            lignes.append({
                "hour": dt.hour,
                "weekend": int(dt.weekday() >= 5),
                "n_files": n_files,
                "lines_added": commit["add"],
                "lines_deleted": commit["del"],
                "author_experience": author_xp[commit["author"]],
                "avg_authors_per_file": ownership,
                "total_churn": commit["add"] + commit["del"],
                "msg_len": len(commit["msg"]),
                "touches_tests": int(any("test" in f.lower() for f in commit["files"])),
                "risky": label,
            })
        author_xp[commit["author"]] += 1
        for f in commit["files"]:
            file_authors[f].add(commit["author"])

    return pd.DataFrame(lignes)


def build_is_night(base_df):
    """Derive de la colonne `hour` de BASE : pas besoin d'une seconde boucle sur `commits`."""
    return ((base_df["hour"] >= 22) | (base_df["hour"] < 6)).astype(int).rename("is_night")


def build_churn_features(commits, est_risque):
    file_commits = defaultdict(int)
    file_first_ts = {}
    rows = []

    for commit in commits:
        label = est_risque(commit)
        n = len(commit["files"])
        if label is not None and n > 0:
            hist = [file_commits[f] for f in commit["files"]]
            ages = [(commit["ts"] - file_first_ts[f]) / 86400
                    for f in commit["files"] if f in file_first_ts]
            rows.append({
                "file_commits_max": max(hist),
                "file_commits_avg": float(np.mean(hist)),
                "file_age_days_avg": float(np.mean(ages)) if ages else 0.0,
            })
        for f in commit["files"]:
            file_commits[f] += 1
            file_first_ts.setdefault(f, commit["ts"])

    return pd.DataFrame(rows)


def churn_fix_correlation(commits, fix_par_fichier):
    """Diagnostic de circularite (cellule 46) : correlation commits/fix par fichier."""
    nb_commits_f = defaultdict(int)
    for c in commits:
        for f in c["files"]:
            nb_commits_f[f] += 1

    diag = pd.DataFrame([
        {"n_commits": nb_commits_f[f], "n_fix": len(fix_par_fichier.get(f, []))}
        for f in nb_commits_f
    ])
    return {
        "pearson": float(diag["n_commits"].corr(diag["n_fix"])),
        "spearman": float(diag["n_commits"].corr(diag["n_fix"], method="spearman")),
    }


_RADON_CACHE = {}
# Plafond de temps par fichier (6A, 2026-11-04) : certains depots versionnent des fichiers
# GENERES gigantesques (ex. black : profiling/mix_huge.py, 1,4 Mo, une seule expression) sur
# lesquels radon met des minutes -- un seul de ces fichiers bloquait tout le calcul. Depasser
# le plafond = echec d'analyse = NaN (meme traitement que le code illisible, CLAUDE.md §7 pt 8),
# jamais une valeur par defaut. Les vrais fichiers source (le plus gros de black : 187 Ko)
# prennent ~1 s. RADON_STATS compte les depassements, rapportes par depot.
RADON_TIMEOUT_S = 10
RADON_STATS = {"timeouts": 0}


class _RadonTimeout(Exception):
    pass


def _metriques_radon(src):
    if src is None:
        return None
    key = hash(src)
    if key in _RADON_CACHE:
        return _RADON_CACHE[key]
    import signal
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
    from radon.raw import analyze

    def _on_alarm(_sig, _frm):
        raise _RadonTimeout()

    try:
        old = signal.signal(signal.SIGALRM, _on_alarm)
        signal.setitimer(signal.ITIMER_REAL, RADON_TIMEOUT_S)
        armed = True
    except ValueError:  # pas le thread principal : pas de plafond, comportement d'origine
        armed = False
    try:
        blocs = cc_visit(src)
        res = {"cc_tot": sum(b.complexity for b in blocs),
               "cc_max": max((b.complexity for b in blocs), default=0),
               "mi": mi_visit(src, multi=True),
               "loc": analyze(src).loc,
               "comments": analyze(src).comments}
    except _RadonTimeout:
        RADON_STATS["timeouts"] += 1
        res = None
    except Exception:
        res = None
    finally:
        if armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    _RADON_CACHE[key] = res
    return res


def build_code_features(commits, est_risque, repo_dir, max_py_par_commit=5):
    """Metriques radon sur les versions avant/apres de chaque fichier .py touche.

    Recupere les versions via `git cat-file --batch` en un seul processus
    (cf. fyc.git_extract.cat_file_batch) : un `git show` par fichier serait
    inutilisable a cette echelle (dizaines de minutes).

    Un echec de parsing radon (courant sur du code d'epoque Python 2 dans
    l'historique Flask, cf. CLAUDE.md §7) ne doit jamais ressembler a du code
    sain : les metriques numeriques sont NaN quand aucun fichier n'a pu etre
    mesure, pas 0.0/100.0. `touches_py` et `code_parse_ok` distinguent
    explicitement "aucun .py touche" de ".py touche mais illisible par radon" --
    les deux etaient confondus dans l'ancien `has_py`.
    """
    parents = git_extract.get_parents(repo_dir)

    besoins, specs = [], []
    indices = []
    touches_py = {}
    for i, c in enumerate(commits):
        if est_risque(c) is None or not c["files"]:
            continue
        indices.append(i)
        pyfiles = [f for f in c["files"] if f.endswith(".py")][:max_py_par_commit]
        touches_py[i] = bool(pyfiles)
        par = parents.get(c["hash"], [])
        for f in pyfiles:
            besoins.append((i, f, "after"))
            specs.append(f'{c["hash"]}:{f}')
            if par:
                besoins.append((i, f, "before"))
                specs.append(f'{par[0]}:{f}')

    raw_contents = git_extract.cat_file_batch(repo_dir, specs)
    contents = {besoins[k]: raw_contents[k] for k in range(len(besoins))}

    agg = defaultdict(lambda: defaultdict(list))
    for (i, f, quand), src in contents.items():
        if quand != "after":
            continue
        ma = _metriques_radon(src)
        if not ma:
            continue
        for k_ in ("cc_tot", "cc_max", "mi", "loc", "comments"):
            agg[i][k_].append(ma[k_])
        mb = _metriques_radon(contents.get((i, f, "before")))
        if mb:
            agg[i]["d_cc"].append(ma["cc_tot"] - mb["cc_tot"])

    rows = []
    for i in indices:
        m = agg.get(i, {})
        parse_ok = bool(m.get("cc_tot"))

        def moy(k, m=m):
            v = m.get(k) or []
            return float(np.mean(v)) if v else np.nan

        if parse_ok:
            loc = moy("loc")
            comments = moy("comments")
            row = {
                "cc_tot_avg": moy("cc_tot"),
                "cc_max_avg": moy("cc_max"),
                "mi_avg": moy("mi"),
                "loc_avg": loc,
                "comment_ratio": (comments / loc) if loc and loc > 0 else np.nan,
                "delta_cc": moy("d_cc"),
            }
        else:
            row = {
                "cc_tot_avg": np.nan, "cc_max_avg": np.nan, "mi_avg": np.nan,
                "loc_avg": np.nan, "comment_ratio": np.nan, "delta_cc": np.nan,
            }
        row["touches_py"] = int(touches_py[i])
        row["code_parse_ok"] = int(parse_ok)
        rows.append(row)

    return pd.DataFrame(rows)
