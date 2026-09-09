"""Etiquetage SZZ simplifie (granularite fichier, fenetre temporelle).

Porte a l'identique la logique des cellules 16 et 19 du notebook.

C'est le module que le Chantier 1 va faire evoluer (detection de fix plus stricte,
granularite ligne via git blame). L'interface `label_commits` -> (fix_par_fichier, est_risque)
est concue pour rester stable : une nouvelle implementation de SZZ doit pouvoir
la respecter pour que `fyc.features` et `results/benchmark.py` n'aient pas a changer.
"""
import bisect
import re
from collections import defaultdict

from . import git_extract

FIX_RE = re.compile(r"\b(fix|bug|regression|crash|error|fail)\w*\b", re.I)

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))?\s\+\d+(?:,\d+)?\s@@")
_OLD_PATH_RE = re.compile(r"^--- (?:a/(.+)|/dev/null)$")
_BLAME_HEADER_RE = re.compile(r"^([0-9a-f]{40}) \d+ \d+(?: \d+)?$")


def tag_fixes(commits, fix_re=FIX_RE):
    """Ajoute `is_fix` (bool) a chaque commit, en place. Retourne `commits`."""
    for c in commits:
        c["is_fix"] = bool(fix_re.search(c["msg"]))
    return commits


def build_fix_index(commits):
    """fichier -> liste triee des timestamps des commits de fix qui l'ont touche."""
    fix_par_fichier = defaultdict(list)
    for c in commits:
        if c["is_fix"]:
            for f in c["files"]:
                fix_par_fichier[f].append(c["ts"])
    for f in fix_par_fichier:
        fix_par_fichier[f].sort()
    return fix_par_fichier


def make_est_risque(fix_par_fichier, fenetre_jours=90):
    """Retourne une fonction commit -> label (1/0/None), fenetre en jours parametrable."""
    fenetre = fenetre_jours * 24 * 3600

    def est_risque(commit):
        if commit["is_fix"]:
            return None
        t = commit["ts"]
        for f in commit["files"]:
            dates = fix_par_fichier.get(f)
            if not dates:
                continue
            i = bisect.bisect_right(dates, t)
            if i < len(dates) and dates[i] <= t + fenetre:
                return 1
        return 0

    return est_risque


def label_commits(commits, fenetre_jours=90, fix_re=FIX_RE):
    """Orchestration complete : tag_fixes + build_fix_index + make_est_risque.

    Mutate `commits` (ajoute is_fix) et retourne (fix_par_fichier, est_risque).
    """
    tag_fixes(commits, fix_re=fix_re)
    fix_par_fichier = build_fix_index(commits)
    est_risque = make_est_risque(fix_par_fichier, fenetre_jours=fenetre_jours)
    return fix_par_fichier, est_risque


# ---------------------------------------------------------------------------
# label_v2 : SZZ vrai, granularite ligne, via git blame.
#
# Protocole de mesure (CLAUDE.md §8, Chantier 1) : meme detection de fix (FIX_RE),
# meme fenetre, pour isoler l'effet de la seule granularite par rapport a label_v1.
# ---------------------------------------------------------------------------

def _parse_diff_old_ranges(diff_text):
    """{chemin_dans_le_parent: [(debut, fin), ...]} des lignes ANCIENNES supprimees
    ou modifiees par le diff (les lignes "coupables" a blamer dans le parent)."""
    ranges = defaultdict(list)
    current_path = None
    for line in diff_text.splitlines():
        if line.startswith("--- "):
            m = _OLD_PATH_RE.match(line)
            current_path = m.group(1) if (m and m.group(1)) else None
        elif line.startswith("@@ ") and current_path:
            m = _HUNK_RE.match(line)
            if m:
                start = int(m.group(1))
                count = int(m.group(2)) if m.group(2) is not None else 1
                if count > 0:
                    ranges[current_path].append((start, start + count - 1))
    return ranges


def _parse_blame_commits(porcelain_text):
    """Ensemble des hash de commit distincts trouves dans une sortie --porcelain."""
    hashes = set()
    for line in porcelain_text.splitlines():
        m = _BLAME_HEADER_RE.match(line)
        if m:
            hashes.add(m.group(1))
    return hashes


def blame_line_fixes(commits, repo_dir, fix_re=FIX_RE):
    """SZZ vrai (granularite ligne) : pour chaque commit de fix, identifie via
    `git blame -w -M -C` les commits qui ont introduit les lignes corrigees.

    `commits` doit deja avoir `is_fix` tague (cf. tag_fixes). Retourne :
    - blame_par_commit : hash -> liste triee des timestamps des fix qui blament
      une ligne de CE commit precis. Definit label_v2 (make_est_risque_v2).
    - blame_par_fichier : fichier -> liste triee des timestamps des evenements de
      blame touchant ce fichier (un timestamp par (fix, fichier), independamment
      du nombre de lignes/commits blames) -- meme forme que fix_par_fichier (v1),
      pour que le diagnostic de circularite (fyc.features.churn_fix_correlation)
      s'applique sans modification. C'est la mesure M1 du protocole.
    """
    parents = git_extract.get_parents(repo_dir)
    by_hash = {c["hash"]: c for c in commits}

    blame_par_commit = defaultdict(list)
    blame_par_fichier = defaultdict(list)

    for c in commits:
        if not c["is_fix"]:
            continue
        par = parents.get(c["hash"], [])
        if not par:
            continue  # commit racine, pas de parent a blamer
        p = par[0]

        diff_text = git_extract.diff_raw(repo_dir, p, c["hash"])
        old_ranges = _parse_diff_old_ranges(diff_text)

        for path, ranges in old_ranges.items():
            porcelain = git_extract.blame_porcelain(repo_dir, p, path, ranges)
            if porcelain is None:
                continue
            blamed = _parse_blame_commits(porcelain)
            if not blamed:
                continue
            blame_par_fichier[path].append(c["ts"])
            for h in blamed:
                intro = by_hash.get(h)
                if intro is None or intro["ts"] > c["ts"]:
                    continue  # hors de notre jeu de commits, ou incoherent (ne doit pas arriver)
                blame_par_commit[h].append(c["ts"])

    for d in (blame_par_commit, blame_par_fichier):
        for k in d:
            d[k].sort()

    return blame_par_commit, blame_par_fichier


def make_est_risque_v2(blame_par_commit, fenetre_jours=90):
    """Meme semantique temporelle que make_est_risque (v1), mais l'attribution
    est directe (le commit a-t-il ete blame par un fix ?) au lieu de passer par
    "un de ses fichiers a-t-il ete touche par un fix" -- c'est le changement de
    granularite : fichier -> ligne."""
    fenetre = fenetre_jours * 24 * 3600

    def est_risque(commit):
        if commit["is_fix"]:
            return None
        dates = blame_par_commit.get(commit["hash"])
        if not dates:
            return 0
        t = commit["ts"]
        i = bisect.bisect_right(dates, t)
        if i < len(dates) and dates[i] <= t + fenetre:
            return 1
        return 0

    return est_risque


def label_commits_v2(commits, repo_dir, fenetre_jours=90, fix_re=FIX_RE):
    """Orchestration complete du label_v2 (SZZ ligne). Mutate `commits` (ajoute
    is_fix, partage avec v1) et retourne (blame_par_commit, blame_par_fichier,
    est_risque_v2)."""
    tag_fixes(commits, fix_re=fix_re)
    blame_par_commit, blame_par_fichier = blame_line_fixes(commits, repo_dir, fix_re=fix_re)
    est_risque = make_est_risque_v2(blame_par_commit, fenetre_jours=fenetre_jours)
    return blame_par_commit, blame_par_fichier, est_risque
