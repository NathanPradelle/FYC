"""Chantier 8d, analyse DESCRIPTIVE NON PRE-ENREGISTREE (ajoutee apres avoir vu les ratios de
chantier8d_drift.json, qui ne tranchaient pas entre H1 et H3) : l'AGE du code corrige.
Pour chaque couple (fix, commit introducteur) de label_v2, age = ts(fix) - ts(introducteur).
Par tranche de dates de fix (10 tranches d'effectif egal), part des liens dont l'age <= 90 jours
(ceux qui produisent un positif a la fenetre de 90 j) et age median.
"""
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chantier6a_dataset as ds

import numpy as np

from fyc import git_extract, labeling

OUT = Path(__file__).resolve().parent / "chantier8d_posthoc.json"
W = 90 * 24 * 3600


def main():
    res = {}
    for name, path in ds.REPOS.items():
        path = str(path)
        commits = git_extract.extraire_commits(path)
        bpc, _bf, _e = labeling.label_commits_v2(commits, path, fenetre_jours=90, fix_predicate=ds.FIX_PREDICATE)
        ts = {c["hash"]: c["ts"] for c in commits}
        links = sorted([(tf, tf - ts[h]) for h, fl in bpc.items() for tf in fl])  # (date du fix, age)
        arr = np.array(links, dtype=float)
        parts = np.array_split(np.arange(len(arr)), 10)
        tr = [{"n_liens": int(len(s)), "part_age_le_90j": float((arr[s, 1] <= W).mean()),
               "age_median_jours": float(np.median(arr[s, 1]) / 86400)} for s in parts]
        n_fix = sum(c["is_fix"] for c in commits)
        res[name] = {"n_liens": len(arr), "n_fix": n_fix, "tranches": tr,
                     "ratio_part_le_90j_premiere_sur_derniere": tr[0]["part_age_le_90j"] / tr[-1]["part_age_le_90j"],
                     "ratio_age_median_derniere_sur_premiere": tr[-1]["age_median_jours"] / max(tr[0]["age_median_jours"], 1e-9)}
        print(name, {k: round(v, 2) for k, v in res[name].items() if k.startswith("ratio")}, flush=True)
    med = {k: float(np.median([r[k] for r in res.values()])) for k in
           ("ratio_part_le_90j_premiere_sur_derniere", "ratio_age_median_derniere_sur_premiere")}
    OUT.write_text(json.dumps({"statut": "descriptif, NON pre-enregistre, ajoute apres chantier8d_drift.json",
                               "generated_at": datetime.now(timezone.utc).isoformat(),
                               "mediane": med, "depots": res}, indent=1))
    print("mediane", med)


if __name__ == "__main__":
    main()
