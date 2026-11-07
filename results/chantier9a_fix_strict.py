"""Chantier 9a : detection de fix plus stricte (B, B1 = sans merges, B2 = + vrai code .py), Flask.
Protocole : CLAUDE.md §8 Chantier 9 (91990a2). Precision mesuree sur l'audit 8g (biais declare : variantes
concues en regardant cet audit -> estimation optimiste, adoption provisoire)."""
import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chantier4bis_common as common

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fyc import evaluate, features, git_extract, labeling

OUT = HERE / "chantier9a_fix_strict.json"


def real_code(c):
    for f in c["files"]:
        if f.endswith(".py") and not any(k in f for k in ("test", "docs/", "examples/")):
            return True
    return False


def variant_fix(name, c):
    if not c["is_fix_B"]:
        return False
    if name == "B":
        return True
    if not c["files"]:
        return False
    return True if name == "B1" else real_code(c)


def cv_aucs(df, cols, seed):
    m = evaluate._make_model(n_estimators=300); m.set_params(n_jobs=1)
    X, y = df[cols], df["risky"]
    return float(np.mean([roc_auc_score(y.iloc[te], m.fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1])
                          for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)]))


def main():
    audit = [r for r in csv.DictReader(open(HERE / "audit_fix_labels.csv", encoding="utf-8")) if r["groupe"] == "detecte"]
    out = {"protocole": "CLAUDE.md §8 Chantier 9a (91990a2)", "generated_at": datetime.now(timezone.utc).isoformat(),
           "variantes": {}}
    for name in ("B", "B1", "B2"):
        commits = git_extract.extraire_commits(common.REPO_DIR)
        labeling.tag_fixes(commits, fix_predicate=common.FIX_PREDICATE)
        for c in commits:
            c["is_fix_B"] = c["is_fix"]
        for c in commits:
            c["is_fix"] = variant_fix(name, c)
        bpc, _bf = labeling.blame_line_fixes(commits, common.REPO_DIR)
        est = labeling.make_est_risque_v2(bpc, fenetre_jours=common.FENETRE_JOURS)
        idx = [i for i, c in enumerate(commits) if est(c) is not None and c["files"]]
        base = features.build_base_features(commits, est)
        churn = features.build_churn_features(commits, est)
        df = pd.concat([base, churn], axis=1).loc[:, lambda d: ~d.columns.duplicated()]
        assert len(df) == len(idx)
        bcols, ccols = features.BASE_COLS, features.CHURN_COLS
        by_hash = {c["hash"][:10]: c for c in commits}
        kept = [r for r in audit if by_hash[r["hash"][:10]]["is_fix"]]
        bug = lambda r: r["label_assistant"] == "bug"
        real_bugs = [r for r in audit if bug(r) and not r["sujet"].startswith("Merge")]
        kept_real = [r for r in real_bugs if by_hash[r["hash"][:10]]["is_fix"]]
        bs = Parallel(n_jobs=3)(delayed(cv_aucs)(df, bcols, s) for s in range(20))
        cs = Parallel(n_jobs=3)(delayed(cv_aucs)(df, ccols, s) for s in range(20))
        bs5 = bs[:5]
        ratios = [c / b for b, c in zip(bs, cs)]
        n_fix = sum(c["is_fix"] for c in commits)
        cut = int(len(df) * 0.7)
        m = evaluate._make_model(n_estimators=300); m.set_params(n_jobs=3)
        m.fit(df[bcols].iloc[:cut], df["risky"].iloc[:cut])
        tail_auc = float(roc_auc_score(df["risky"].iloc[cut:], m.predict_proba(df[bcols].iloc[cut:])[:, 1]))
        out["variantes"][name] = {
            "n_fix": int(n_fix), "n_etiquetes": len(df), "part_risque": float(df["risky"].mean()),
            "audit_fix_retenus": len(kept), "audit_precision_parmi_retenus": (sum(bug(r) for r in kept) / len(kept)) if kept else None,
            "audit_vrais_bugs_hors_merge_total": len(real_bugs), "audit_vrais_bugs_conserves": len(kept_real),
            "part_vrais_bugs_conserves": len(kept_real) / len(real_bugs),
            "AUC_BASE_cv_moy": float(np.mean(bs)), "AUC_BASE_cv_std_graines": float(np.std(bs)),
            "AUC_BASE_queue30": tail_auc, "M2bis_ratio_moyen": float(np.mean(ratios)),
            "M2bis_ratio_std": float(np.std(ratios))}
        print(name, json.dumps(out["variantes"][name]), flush=True)
    B = out["variantes"]["B"]
    for n in ("B1", "B2"):
        v = out["variantes"][n]
        out["variantes"][n]["regle_adoption"] = {
            "i_part_risque_ge_5pct": v["part_risque"] >= 0.05,
            "ii_vrais_bugs_ge_80pct": v["part_vrais_bugs_conserves"] >= 0.8,
            "ii_precision_gain_ge_0_10": (v["audit_precision_parmi_retenus"] or 0) - B["audit_precision_parmi_retenus"] >= 0.10,
            "iii_AUC_BASE_pas_de_chute_gt_0_03": B["AUC_BASE_cv_moy"] - v["AUC_BASE_cv_moy"] <= 0.03}
        out["variantes"][n]["adoptable"] = bool(all(v["regle_adoption"].values()))
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({n: out["variantes"][n].get("regle_adoption") for n in ("B1", "B2")}, indent=1))


if __name__ == "__main__":
    main()
