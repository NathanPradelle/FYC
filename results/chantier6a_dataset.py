"""Extension 6A : construit le jeu de donnees multi-depots (protocole CLAUDE.md §8, 6A).

Pour chaque depot de la liste FIGEE (commit 97d7284) : label_v2 + detection de fix B + fenetre
90 j (identiques a Flask, aucun reglage par depot), puis familles du jeu fige : BASE, diff,
radon, Ruff, + churn (ablation). Chaque feature dependante de l'echelle du depot existe en
deux versions : brute et `<col>__pct` = rang percentile dans le depot, calcule en fenetre
croissante sur le PASSE seul (aucune fuite du futur).

Sorties : results/cache_6a_<depot>.pkl (non versionne), results/dataset_6a.pkl (jeu exporte,
tous depots, pour le cours), results/chantier6a_dataset.json (resume par depot : volumes,
part de fix, part de risque, garde-fous).
Usage : python results/chantier6a_dataset.py [depot ...]   (defaut : les 7)
"""
import bisect
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from fyc import code_lint, diff_features, features, git_extract, labeling

REPOS = {  # liste FIGEE le 2026-11-04 (CLAUDE.md §8, 6A)
    "flask": ROOT / "flask",
    "requests": ROOT / "repos" / "requests",
    "click": ROOT / "repos" / "click",
    "black": ROOT / "repos" / "black",
    "tornado": ROOT / "repos" / "tornado",
    "pydantic": ROOT / "repos" / "pydantic",
    "django-rest-framework": ROOT / "repos" / "django-rest-framework",
}
FENETRE_JOURS = 90
FIX_PREDICATE = labeling.make_fix_predicate(exclude_re=labeling.EXCLUDE_RE)
FAMILIES = {
    "BASE": features.BASE_COLS,
    "diff": diff_features.DIFF_COLS,
    "radon": features.CODE_COLS,
    "ruff": code_lint.RUFF_COLS,
    "churn": features.CHURN_COLS,  # ablation
}
# Features independantes de l'echelle du depot : pas de version percentile.
SCALE_FREE = {"hour", "weekend", "touches_tests", "touches_py", "code_parse_ok", "touches_signature"}
TEST_TAIL, MIN_POS_TAIL, MIN_PART_RISQUE = 0.30, 40, 0.05


def expanding_percentile(values):
    """Rang percentile de chaque valeur parmi les valeurs STRICTEMENT passees (ties = moitie).
    NaN -> NaN (et n'entre pas dans l'historique). Premiere valeur : 0,5 (pas d'information)."""
    past, out = [], []
    for v in values:
        if v != v:
            out.append(np.nan)
            continue
        n = len(past)
        if n == 0:
            out.append(0.5)
        else:
            lo, hi = bisect.bisect_left(past, v), bisect.bisect_right(past, v)
            out.append((lo + 0.5 * (hi - lo)) / n)
        bisect.insort(past, v)
    return out


def build_repo(name, path):
    t0 = time.time()
    features.RADON_STATS["timeouts"] = 0
    path = str(path)
    commits = git_extract.extraire_commits(path)
    _b, fix_index, est_risque = labeling.label_commits_v2(
        commits, path, fenetre_jours=FENETRE_JOURS, fix_predicate=FIX_PREDICATE)
    idx = [i for i, c in enumerate(commits) if est_risque(c) is not None and c["files"]]
    base = features.build_base_features(commits, est_risque)
    churn = features.build_churn_features(commits, est_risque)
    # Points de reprise par etape (les calculs longs ont ete tues deux fois par un redemarrage
    # de WSL le 2026-11-04/05) : un fichier par depot et par etape, supprimes une fois le cache
    # final ecrit. radon_timeouts est sauvegarde avec l'etape radon.
    def stage(nom, fn):
        f = HERE / f"cache_6a_{Path(path).name}__{nom}.pkl"
        if f.exists():
            return pickle.loads(f.read_bytes())
        r = fn()
        f.write_bytes(pickle.dumps(r))
        print(f"[{name}] etape {nom} ok ({time.time() - t0:.0f}s)", flush=True)
        return r

    code, n_to = stage("radon", lambda: (features.build_code_features(commits, est_risque, path),
                                         features.RADON_STATS["timeouts"]))
    features.RADON_STATS["timeouts"] = n_to
    ruff_df = stage("ruff", lambda: code_lint.lint_features(commits, est_risque, path)[0])
    diff_df = stage("diff", lambda: diff_features.build_diff_features(commits, est_risque, path))
    df = pd.concat([base.drop(columns="risky"), churn, code, ruff_df, diff_df, base["risky"]], axis=1)
    assert len(df) == len(idx)
    df.insert(0, "repo", name)
    df.insert(1, "ts", [commits[i]["ts"] for i in idx])
    df.insert(2, "hash", [commits[i]["hash"] for i in idx])
    for col in [c for fam in FAMILIES.values() for c in fam if c not in SCALE_FREE]:
        df[f"{col}__pct"] = expanding_percentile(df[col].tolist())

    tail = df.iloc[int(len(df) * (1 - TEST_TAIL)):]
    summary = {
        "repo": name, "n_commits": len(commits), "n_etiquetes": len(df),
        "premier_commit": datetime.fromtimestamp(commits[0]["ts"], timezone.utc).date().isoformat(),
        "dernier_commit": datetime.fromtimestamp(commits[-1]["ts"], timezone.utc).date().isoformat(),
        "part_risque": round(float(df["risky"].mean()), 4),
        "n_positifs": int(df["risky"].sum()),
        "n_positifs_queue_test_30pct": int(tail["risky"].sum()),
        "part_touches_py": round(float((df["touches_py"] == 1).mean()), 4),
        "part_code_parse_ok_parmi_py": round(float(df.loc[df["touches_py"] == 1, "code_parse_ok"].mean()), 4),
        "garde_fou_part_risque_ok": bool(df["risky"].mean() >= MIN_PART_RISQUE),
        "garde_fou_positifs_queue_ok": bool(tail["risky"].sum() >= MIN_POS_TAIL),
        "radon_timeouts_fichiers": features.RADON_STATS["timeouts"],
        "temps_s": round(time.time() - t0, 1),
    }
    summary["interpretable"] = summary["garde_fou_part_risque_ok"] and summary["garde_fou_positifs_queue_ok"]
    return df, summary


def main():
    names = sys.argv[1:] or list(REPOS)
    summaries = {}
    for name in names:
        cache = HERE / f"cache_6a_{name}.pkl"
        if cache.exists():
            df, s = pickle.loads(cache.read_bytes())
            print(f"[{name}] cache", flush=True)
        else:
            print(f"[{name}] calcul...", flush=True)
            df, s = build_repo(name, REPOS[name])
            cache.write_bytes(pickle.dumps((df, s)))
            for f in HERE.glob(f"cache_6a_{Path(REPOS[name]).name}__*.pkl"):
                f.unlink()
        summaries[name] = s
        print(f"[{name}] {json.dumps(s, ensure_ascii=False)}", flush=True)
    if set(names) == set(REPOS):
        full = pd.concat([pickle.loads((HERE / f"cache_6a_{n}.pkl").read_bytes())[0] for n in REPOS],
                         ignore_index=True)
        (HERE / "dataset_6a.pkl").write_bytes(pickle.dumps(full))
        (HERE / "chantier6a_dataset.json").write_text(json.dumps(
            {"generated_at": datetime.now(timezone.utc).isoformat(), "label": "v2 + fix B, 90 j",
             "liste_figee_commit": "97d7284", "depots": summaries, "n_total": len(full)},
            indent=2, ensure_ascii=False))
        print("ecrit dataset_6a.pkl et chantier6a_dataset.json", flush=True)


if __name__ == "__main__":
    main()
