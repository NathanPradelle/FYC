"""Chantier 4bis, vague 1 : mesure de chaque outil SEUL puis BASE+outil (etape 1 du
protocole pre-enregistre, CLAUDE.md §8). Aucune recherche de combinaison ici.

Par outil : couverture (NaN parmi les commits touchant des .py pour les outils sur .py,
parmi tous pour Kamei), AUC de chaque feature seule, BASE+outil, redondance (Pearson vs
loc_avg et vs la feature existante la plus correlee), circularite (vs file_commits_max,
plancher de vigilance 0,3), correlations entre outils. Verifie les seuils d'abandon
(a) temps, (b) >50 % NaN, (c) |AUC-0,5|<0,02 ET corr>0,7. Sortie :
results/chantier4bis_wave1.json.

Usage : python results/chantier4bis_wave1.py   (cache : results/cache_wave1_v2.pkl)
"""
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chantier4bis_common as common

import numpy as np
import pandas as pd

from fyc import coupling, evaluate, features, jit_kamei, mypy_features, pylint_features, vulture_features

CACHE = Path(__file__).resolve().parent / "cache_wave1_v2.pkl"
TOOLS = {  # nom -> (colonnes, necessite des .py ?)
    "pylint": (pylint_features.PYLINT_COLS, True),
    "vulture": (vulture_features.VULTURE_COLS, True),
    "mypy": (mypy_features.MYPY_COLS, True),
    "kamei": (jit_kamei.KAMEI_COLS, False),
    "coupling": (coupling.COUPLING_COLS, True),
}
TEMPS_MAX_S = 15 * 60
NAN_MAX, AUC_MIN_ECART, CORR_MAX, CIRC_VIGILANCE = 0.50, 0.02, 0.7, 0.3


def compute_all(commits, er):
    t, out = {}, {}
    for name, fn in [
        ("pylint", lambda: pylint_features.pylint_features(commits, er, common.REPO_DIR)),
        ("vulture", lambda: vulture_features.vulture_features(commits, er, common.REPO_DIR)),
        ("mypy", lambda: mypy_features.mypy_features(commits, er, common.REPO_DIR)),
        ("kamei", lambda: jit_kamei.build_kamei_features(commits, er)),
        ("coupling", lambda: coupling.build_coupling_features(commits, er, common.REPO_DIR)),
    ]:
        print(f"calcul {name}...", flush=True)
        t0 = time.time()
        out[name] = fn()
        t[name] = round(time.time() - t0, 1)
        print(f"  {t[name]} s", flush=True)
    return out, t


def main():
    commits, er, full, _idx = common.load_full()
    if CACHE.exists():
        c = pickle.loads(CACHE.read_bytes())
        dfs, timings = c["dfs"], c["timings"]
    else:
        dfs, timings = compute_all(commits, er)
        CACHE.write_bytes(pickle.dumps({"dfs": dfs, "timings": timings}))
    for name, df in dfs.items():
        assert len(df) == len(full), f"desynchronisation {name}"
    data = pd.concat([full] + [dfs[n] for n in TOOLS], axis=1)

    base = features.BASE_COLS
    existing = sorted({c for fam in common.FAMILIES.values() for c in fam} - {"touches_py", "code_parse_ok"})
    existing = [c for c in existing if data[c].dtype != object]
    touches_py = data["touches_py"] == 1
    ref_base, _ = evaluate.cv_auc(data, base)
    print(f"BASE = {ref_base:.4f}", flush=True)

    report = {}
    for name, (cols, needs_py) in TOOLS.items():
        scope = data[touches_py] if needs_py else data
        nan_part = {c: round(float(scope[c].isna().mean()), 4) for c in cols}
        single = {}
        for c in cols:
            m, s = evaluate.cv_auc(data, [c])
            single[c] = {"auc": round(m, 4), "std": round(s, 4)}
        m, s = evaluate.cv_auc(data, base + cols)
        tmp = round(m - ref_base, 4)
        redund = {}
        for c in cols:
            others = [e for e in existing if e != c]
            cor = data[others].corrwith(data[c]).abs().sort_values(ascending=False)
            redund[c] = {
                "corr_loc_avg": round(float(data[c].corr(data["loc_avg"])), 4),
                "max_corr_existante": [cor.index[0], round(float(cor.iloc[0]), 4)],
                "corr_churn_file_commits_max": round(float(data[c].corr(data["file_commits_max"])), 4),
            }
        best = max(cols, key=lambda c: abs(single[c]["auc"] - 0.5))
        abandon = []
        if timings[name] > TEMPS_MAX_S:
            abandon.append("(a) cout > 15 min")
        if min(nan_part.values()) > NAN_MAX:
            abandon.append("(b) > 50 % NaN parmi les commits applicables")
        if (abs(single[best]["auc"] - 0.5) < AUC_MIN_ECART
                and redund[best]["max_corr_existante"][1] > CORR_MAX):
            abandon.append("(c) quasi-hasard ET redondant")
        report[name] = {
            "cols": cols, "temps_s": timings[name], "nan_part_parmi_applicables": nan_part,
            "auc_seule": single,
            "base_plus_outil": {"auc": round(m, 4), "std": round(s, 4), "gain_vs_base": tmp},
            "redondance_circularite": redund, "meilleure_feature_seule": best,
            "circularite_au_dessus_vigilance": [
                c for c in cols if abs(redund[c]["corr_churn_file_commits_max"]) > CIRC_VIGILANCE],
            "seuils_abandon_declenches": abandon,
        }
        print(f"{name}: BASE+outil {m:.4f} ({tmp:+.4f}) | abandon : {abandon or 'aucun'}", flush=True)

    inter = data[[c for cols, _ in TOOLS.values() for c in cols]].corr().round(3)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "flask_head": common.git_extract.get_head(common.REPO_DIR),
        "label": "v2 + detection de fix B, fenetre 90 j", "n": len(data),
        "auc_base": round(ref_base, 4), "outils": report,
        "correlations_entre_outils": inter.to_dict(),
        "tree_unparsable_coupling_moyen": round(float(np.nanmean(
            dfs["coupling"].attrs.get("tree_unparsable", [np.nan]))), 4),
    }
    Path(__file__).with_name("chantier4bis_wave1.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False))
    print("ecrit chantier4bis_wave1.json")


if __name__ == "__main__":
    main()
