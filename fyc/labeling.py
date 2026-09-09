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

FIX_RE = re.compile(r"\b(fix|bug|regression|crash|error|fail)\w*\b", re.I)


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
