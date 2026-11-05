"""4bis, vague 2 : Semgrep (p/python + p/security-audit) et jscpd sur un ECHANTILLON de 300 commits.
Protocole : CLAUDE.md §8, "VAGUE 2 du 4bis -- PROTOCOLE PRE-ENREGISTRE v1" (commit 7350811).
Resultats rapportes a part (jamais dans benchmark.py ni le tableau pleine echelle).

Outils installes hors du venv du projet (dossiers isoles) :
  SEMGREP_BIN  (venv separe)   JSCPD_BIN  (npm local)
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chantier4bis_common as common

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from fyc import code_lint, evaluate

HERE = Path(__file__).resolve().parent
OUT = HERE / "chantier4bis_wave2.json"
SCRATCH = Path(os.environ.get("WAVE2_TOOLS", "/tmp/claude-1000/-home-natha-dev-FYC/"
                              "3246c96e-8d7d-43e2-9017-0dd9af457751/scratchpad"))
SEMGREP_BIN = SCRATCH / "sgvenv" / "bin" / "semgrep"
JSCPD_BIN = SCRATCH / "jsc" / "node_modules" / ".bin" / "jscpd"
N_POS, N_NEG, SEED = 120, 180, 0
SEM_COLS = ["semgrep_total_avg", "semgrep_error_avg", "semgrep_warning_avg"]
JSC_COLS = ["jscpd_dup_pct", "jscpd_clones"]


def sample_commits(commits, est_risque, par_commit):
    cand = [i for i in sorted(par_commit) if est_risque(commits[i]) is not None]
    pos = [i for i in cand if est_risque(commits[i]) == 1]
    neg = [i for i in cand if est_risque(commits[i]) == 0]
    rng = np.random.default_rng(SEED)
    chosen = sorted(list(rng.choice(pos, N_POS, replace=False)) + list(rng.choice(neg, N_NEG, replace=False)))
    return [int(i) for i in chosen], len(pos), len(neg)


def run_semgrep(sample, par_commit):
    rows = {i: {"n_ok": 0, "total": 0, "error": 0, "warning": 0} for i in sample}
    rules = Counter()
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        name_to_commit = code_lint._materialize(td, {i: par_commit[i] for i in sample})
        t0 = time.time()
        p = subprocess.run(
            [str(SEMGREP_BIN), "scan", "--config", "p/python", "--config", "p/security-audit",
             "--metrics=off", "--json", "--quiet", "--timeout", "30", str(td)],
            capture_output=True, text=True)
        dt = time.time() - t0
        d = json.loads(p.stdout)
        failed = {Path(e["path"]).name for e in d["errors"] if e.get("path")}
        scanned = {Path(x).name for x in d["paths"]["scanned"]}
        for name in scanned - failed:
            rows[name_to_commit[name]]["n_ok"] += 1
        for r in d["results"]:
            name = Path(r["path"]).name
            if name in failed or name not in name_to_commit:
                continue
            row = rows[name_to_commit[name]]
            row["total"] += 1
            sev = r["extra"].get("severity", "")
            if sev in ("ERROR", "WARNING"):
                row[sev.lower()] += 1
            rules[r["check_id"].split(".")[-1]] += 1
        n_files, n_fail = len(name_to_commit), len(failed & set(name_to_commit))
    df = pd.DataFrame(index=sample, columns=SEM_COLS, dtype=float)
    for i, r in rows.items():
        if r["n_ok"] > 0:
            df.loc[i] = [r["total"] / r["n_ok"], r["error"] / r["n_ok"], r["warning"] / r["n_ok"]]
    meta = {"secondes": dt, "fichiers": n_files, "fichiers_en_erreur_ou_non_scannes": n_files - len(scanned - failed),
            "top10_regles": rules.most_common(10), "n_erreurs_json": len(d["errors"]),
            "version": subprocess.run([str(SEMGREP_BIN), "--version"], capture_output=True, text=True).stdout.strip()}
    return df, meta


def run_jscpd(sample, par_commit):
    df = pd.DataFrame(index=sample, columns=JSC_COLS, dtype=float)
    t0, n_fail = time.time(), 0
    for i in sample:
        with tempfile.TemporaryDirectory() as td:
            src, rep = Path(td) / "src", Path(td) / "rep"
            src.mkdir()
            code_lint._materialize(src, {i: par_commit[i]})
            p = subprocess.run([str(JSCPD_BIN), str(src), "--reporters", "json", "--output", str(rep), "--silent"],
                               capture_output=True, text=True)
            f = rep / "jscpd-report.json"
            if p.returncode != 0 or not f.exists():
                n_fail += 1
                continue
            t = json.loads(f.read_text())["statistics"]["total"]
            df.loc[i] = [t["percentageTokens"], t["clones"]]
    return df, {"secondes": time.time() - t0, "echecs": n_fail}


def boot_auc(y, x, n=1000, seed=0):
    ok = ~np.isnan(x)
    y, x = np.asarray(y)[ok], np.asarray(x)[ok]
    if len(set(y)) < 2 or len(set(x)) < 2:
        return {"n": int(ok.sum()), "auc": None}
    rng = np.random.default_rng(seed)
    v = []
    while len(v) < n:
        j = rng.integers(0, len(y), len(y))
        if y[j].min() != y[j].max():
            v.append(roc_auc_score(y[j], x[j]))
    return {"n": int(ok.sum()), "auc": float(roc_auc_score(y, x)),
            "ic95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}


def main():
    commits, est_risque, full, idx = common.load_full()
    pos_of = {i: k for k, i in enumerate(idx)}
    _ids, par_commit = code_lint._collect_after_sources(commits, est_risque, common.REPO_DIR)
    sample, n_pos_pool, n_neg_pool = sample_commits(commits, est_risque, par_commit)
    print(f"echantillon: {len(sample)} commits (pool: {n_pos_pool} pos, {n_neg_pool} neg)", flush=True)

    sem, sem_meta = run_semgrep(sample, par_commit)
    print("semgrep", sem_meta["secondes"], "s", flush=True)
    jsc, jsc_meta = run_jscpd(sample, par_commit)
    print("jscpd", jsc_meta, flush=True)

    sub = full.iloc[[pos_of[i] for i in sample]].reset_index(drop=True)
    sem, jsc = sem.reset_index(drop=True), jsc.reset_index(drop=True)
    data = pd.concat([sub, sem, jsc], axis=1)
    y = data["risky"]
    fams = common.FAMILIES
    out = {"protocole": "CLAUDE.md §8 vague 2 v1 (commit 7350811)",
           "generated_at": datetime.now(timezone.utc).isoformat(),
           "echantillon": {"n": len(data), "n_pos": int(y.sum()), "graine": SEED,
                           "pool_pos": n_pos_pool, "pool_neg": n_neg_pool},
           "semgrep_meta": sem_meta, "jscpd_meta": jsc_meta,
           "extrapolation_3200_commits_min": {
               "semgrep": sem_meta["secondes"] / len(sample) * 3200 / 60,
               "jscpd": jsc_meta["secondes"] / len(sample) * 3200 / 60}}
    out["nan_part"] = {c: float(data[c].isna().mean()) for c in SEM_COLS + JSC_COLS}
    out["auc_seule"] = {c: boot_auc(y, data[c].values.astype(float)) for c in SEM_COLS + JSC_COLS}
    ref = ["loc_avg", "ruff_total_avg", "file_commits_max"]
    out["correlations_spearman"] = {
        c: {r: (None if data[[c, r]].dropna().shape[0] < 10 else
                float(spearmanr(data[c], data[r], nan_policy="omit").statistic)) for r in ref}
        for c in SEM_COLS + JSC_COLS}
    out["correlation_semgrep_jscpd"] = float(spearmanr(data["semgrep_total_avg"], data["jscpd_dup_pct"],
                                                      nan_policy="omit").statistic)
    base = fams["BASE"]
    sets = {"BASE": base, "BASE+semgrep": base + SEM_COLS, "BASE+jscpd": base + JSC_COLS,
            "BASE+semgrep+jscpd": base + SEM_COLS + JSC_COLS}
    res = {k: repeated_full(data, c) for k, c in sets.items()}
    out["cv_repetee_5x5"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in res.items()}
    out["difference_appariee_vs_BASE"] = {}
    for k in sets:
        if k == "BASE":
            continue
        d = np.array(res[k]) - np.array(res["BASE"])
        out["difference_appariee_vs_BASE"][k] = {"mean": float(d.mean()), "std": float(d.std()),
                                                  "repetitions_positives": int((d > 0).sum()), "n": len(d)}
    OUT.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("echantillon", "extrapolation_3200_commits_min", "nan_part",
                                           "auc_seule", "correlations_spearman", "correlation_semgrep_jscpd",
                                           "cv_repetee_5x5", "difference_appariee_vs_BASE")}, indent=1, default=str))
    print("top regles:", sem_meta["top10_regles"])


def repeated_full(data, cols):
    """5 plis x 5 graines de CV -> 25 AUC (une par repetition de pli), pour appariement."""
    def one(seed):
        m = evaluate._make_model(n_estimators=300)
        m.set_params(n_jobs=1)
        X, y = data[cols], data["risky"]
        return [roc_auc_score(y.iloc[te], m.fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[te])[:, 1])
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y)]
    nested = Parallel(n_jobs=3)(delayed(one)(s) for s in range(5))
    return [a for n in nested for a in n]


if __name__ == "__main__":
    main()
