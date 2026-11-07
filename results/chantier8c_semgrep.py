"""Chantier 8c : effet de l'echec silencieux de Semgrep (fichiers non parsables) sur la vague 2.
Protocole : CLAUDE.md §8 Chantier 8 (commit f7e8489). Meme echantillon (graine 0) que la vague 2.
"""
import ast
import json
import subprocess
import sys
import tempfile
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chantier4bis_common as common
import chantier4bis_wave2 as w2

import numpy as np
import pandas as pd

from fyc import code_lint

OUT = Path(__file__).resolve().parent / "chantier8c_semgrep.json"


def main():
    commits, est, full, idx = common.load_full()
    pos_of = {i: k for k, i in enumerate(idx)}
    _ids, par_commit = code_lint._collect_after_sources(commits, est, common.REPO_DIR)
    sample, _, _ = w2.sample_commits(commits, est, par_commit)
    parsable = {}
    for i in sample:
        for f, src in par_commit[i]:
            try:
                ast.parse(src); ok = True
            except Exception:
                ok = False
            parsable[f"{i}___{f.replace('/', '_')}"] = ok
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        n2c = code_lint._materialize(td, {i: par_commit[i] for i in sample})
        t0 = time.time()
        p = subprocess.run([str(w2.SEMGREP_BIN), "scan", "--config", "p/python", "--config", "p/security-audit",
                            "--metrics=off", "--json", "--quiet", "--timeout", "30", str(td)],
                           capture_output=True, text=True)
        dt = time.time() - t0
        d = json.loads(p.stdout)
    errs = {Path(e["path"]).name for e in d["errors"] if e.get("path")}
    per = {n: {"total": 0, "error": 0, "warning": 0} for n in n2c}
    for r in d["results"]:
        n = Path(r["path"]).name
        if n in per:
            per[n]["total"] += 1
            sev = r["extra"].get("severity", "").lower()
            if sev in ("error", "warning"):
                per[n][sev] += 1
    ok_files = [n for n in per if parsable[n]]
    bad_files = [n for n in per if not parsable[n]]
    def agg(names):
        return {"n_fichiers": len(names),
                "violations_moyenne_par_fichier": float(np.mean([per[n]["total"] for n in names])) if names else None,
                "part_fichiers_zero_violation": float(np.mean([per[n]["total"] == 0 for n in names])) if names else None}
    out = {"protocole": "CLAUDE.md §8 Chantier 8c (f7e8489)", "generated_at": datetime.now(timezone.utc).isoformat(),
           "secondes_semgrep": dt, "n_erreurs_json_semgrep": len(d["errors"]),
           "parsables_ast": agg(ok_files), "non_parsables_ast": agg(bad_files)}
    # features avec les fichiers non parsables EXCLUS
    rows = {i: [] for i in sample}
    for n, c in n2c.items():
        if parsable[n]:
            rows[c].append(per[n])
    cols = w2.SEM_COLS
    sem = pd.DataFrame(index=sample, columns=cols, dtype=float)
    for i, lst in rows.items():
        if lst:
            sem.loc[i] = [np.mean([x["total"] for x in lst]), np.mean([x["error"] for x in lst]),
                          np.mean([x["warning"] for x in lst])]
    sub = full.iloc[[pos_of[i] for i in sample]].reset_index(drop=True)
    data = pd.concat([sub, sem.reset_index(drop=True)], axis=1)
    out["commits_sans_aucun_fichier_parsable"] = int(sem["semgrep_total_avg"].isna().sum())
    base = common.FAMILIES["BASE"]
    r_base = w2.repeated_full(data, base)
    r_sem = w2.repeated_full(data, base + cols)
    dlt = np.array(r_sem) - np.array(r_base)
    out["delta_BASE_plus_semgrep_corrige"] = {"moyenne": float(dlt.mean()), "ecart_type": float(dlt.std()),
                                              "repetitions_positives": int((dlt > 0).sum()), "n": len(dlt),
                                              "AUC_BASE": float(np.mean(r_base)), "AUC_BASE_semgrep": float(np.mean(r_sem))}
    out["delta_reference_vague2_non_corrige"] = {"moyenne": 0.012546, "repetitions_positives": 16, "n": 25}
    out["conclusion_inchangee"] = bool(not (dlt.mean() >= 0.05 and (dlt > 0).mean() >= 0.8))
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
