"""Chargement partage du jeu de donnees complet (label_v2 + detection de fix B + les six
familles existantes) pour les scripts du Chantier 4bis. Meme logique que
results/benchmark.py (memes appels, meme masque de couverture GitHub), mise en cache
disque (results/cache_full_v2.pkl, gitignore) car Ruff+Bandit+diff+blame prennent ~3 min.

Le cache est invalide si le HEAD Flask change.
"""
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from fyc import code_lint, diff_features, features, git_extract, labeling, process_features

REPO_DIR = str(REPO_ROOT / "flask")
FENETRE_JOURS = 90
FIX_PREDICATE = labeling.make_fix_predicate(exclude_re=labeling.EXCLUDE_RE)
CACHE = Path(__file__).resolve().parent / "cache_full_v2.pkl"

LINT = code_lint.RUFF_COLS + code_lint.BANDIT_COLS
DIFF = diff_features.DIFF_COLS
PROCESS = process_features.PROCESS_COLS
FAMILIES = {
    "BASE": features.BASE_COLS,
    "churn": features.CHURN_COLS,
    "radon": features.CODE_COLS,
    "ruff": code_lint.RUFF_COLS,
    "bandit": code_lint.BANDIT_COLS,
    "diff": DIFF,
    "process": PROCESS,
}


def load_context():
    """(commits, est_risque, fix_index) -- rapide hors blame (~15 s)."""
    commits = git_extract.extraire_commits(REPO_DIR)
    _b, fix_index, est_risque = labeling.label_commits_v2(
        commits, REPO_DIR, fenetre_jours=FENETRE_JOURS, fix_predicate=FIX_PREDICATE,
    )
    return commits, est_risque, fix_index


def load_full():
    """(commits, est_risque, full_df, mask) ; full_df = BASE+churn+radon+ruff+bandit+diff
    +process + 'risky', lignes filtrees par le masque de couverture GitHub (100 %)."""
    head = git_extract.get_head(REPO_DIR)
    commits, est_risque, _ = load_context()
    if CACHE.exists():
        c = pickle.loads(CACHE.read_bytes())
        if c["head"] == head:
            return commits, est_risque, c["full"], c["indices"]
    base = features.build_base_features(commits, est_risque)
    churn = features.build_churn_features(commits, est_risque)
    code = features.build_code_features(commits, est_risque, REPO_DIR)
    ruff_df, bandit_df = code_lint.lint_features(commits, est_risque, REPO_DIR)
    diff_df = diff_features.build_diff_features(commits, est_risque, REPO_DIR)
    process_df, (nm, nt), mask = process_features.build_process_features(commits, est_risque)
    assert nm == nt, f"cache GitHub incomplet ({nm}/{nt})"
    full = pd.concat([base.drop(columns="risky"), churn, code, ruff_df, bandit_df, diff_df,
                      process_df, base["risky"]], axis=1)
    indices = [i for i, c in enumerate(commits) if est_risque(c) is not None and c["files"]]
    assert len(full) == len(indices)
    CACHE.write_bytes(pickle.dumps({"head": head, "full": full, "indices": indices}))
    return commits, est_risque, full, indices


def load_all():
    """full_df + les 5 outils de la vague 1 (lit results/cache_wave1_v2.pkl, produit par
    chantier4bis_wave1.py). Retourne (data, families) ; BASE est toujours incluse."""
    from fyc import coupling, jit_kamei, mypy_features, pylint_features, vulture_features
    _c, _e, full, _i = load_full()
    w = pickle.loads((Path(__file__).resolve().parent / "cache_wave1_v2.pkl").read_bytes())
    data = pd.concat([full] + [w["dfs"][n] for n in ("pylint", "vulture", "mypy", "kamei", "coupling")],
                     axis=1)
    families = dict(FAMILIES)
    families.update({
        "pylint": pylint_features.PYLINT_COLS, "vulture": vulture_features.VULTURE_COLS,
        "mypy": mypy_features.MYPY_COLS, "kamei": jit_kamei.KAMEI_COLS,
        "coupling": coupling.COUPLING_COLS,
    })
    assert len(data) == len(full)
    return data, families
