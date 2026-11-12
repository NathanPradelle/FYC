"""Chantier 10a : signaux de fix hors message (S_test, S_issue, S_bug, S_bug_seul) sur Flask.
Protocole : CLAUDE.md §8 Chantier 10 (75cb0ab). Annotations d'audit (8g) faites sur le message seul.
Cache des issues fermees par PR : results/github_issues_cache.json (gitignore, derive de l'API).
"""
import csv
import json
import math
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chantier4bis_common as common
import chantier9a_fix_strict as a9

import numpy as np
import pandas as pd
import requests
from joblib import Parallel, delayed

from fyc import features, git_extract, labeling
from fyc.github_signals import GITHUB_GRAPHQL_URL, _load_token, fetch_commit_pr_signals

OUT = HERE / "chantier10a_fix_signals.json"
ISSUES_CACHE = HERE / "github_issues_cache.json"
BUG_RE = re.compile(r"bug|defect|regression|crash", re.I)


def fetch_issues(pr_numbers, batch=25):
    cache = json.loads(ISSUES_CACHE.read_text()) if ISSUES_CACHE.exists() else {}
    todo = [n for n in sorted(set(pr_numbers)) if str(n) not in cache]
    s = requests.Session(); s.headers["Authorization"] = f"Bearer {_load_token()}"
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        body = " ".join(
            f'p{n}: pullRequest(number: {n}) {{ closingIssuesReferences(first: 10) {{ nodes {{ number '
            f'labels(first: 10) {{ nodes {{ name }} }} }} }} }}' for n in chunk)
        r = s.post(GITHUB_GRAPHQL_URL, json={"query": 'query { repository(owner: "pallets", name: "flask") { %s } }' % body})
        r.raise_for_status()
        data = r.json()
        if "errors" in data and "data" not in data:
            raise RuntimeError(data["errors"])
        repo = (data.get("data") or {}).get("repository") or {}
        for n in chunk:
            node = repo.get(f"p{n}")
            if node is None:   # PR introuvable : mesure faite, pas d'information -> None (distinct de "pas mesure")
                cache[str(n)] = None
            else:
                cache[str(n)] = [{"number": x["number"], "labels": [l["name"] for l in x["labels"]["nodes"]]}
                                 for x in node["closingIssuesReferences"]["nodes"]]
        ISSUES_CACHE.write_text(json.dumps(cache))
        print(f"  issues {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    return cache


def wil(k, n, z=1.96):
    if n == 0:
        return [None, None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - h, c + h]


def main():
    commits = git_extract.extraire_commits(common.REPO_DIR)
    labeling.tag_fixes(commits, fix_predicate=common.FIX_PREDICATE)
    for c in commits:
        c["is_fix_B"] = c["is_fix"]
    # Le cache GitHub du Chantier 4 ne couvre que les commits NON-fix etiquetes (9 des 732 fix) :
    # on complete pour les fix detectes et les commits audites (erreur de couverture trouvee au 1er essai).
    need = [c["hash"] for c in commits if c["is_fix_B"] or c["hash"][:10] in {r["hash"][:10] for r in
            csv.DictReader(open(HERE / "audit_fix_labels.csv", encoding="utf-8"))}]
    fetch_commit_pr_signals(need)
    gh = json.loads((HERE / "github_cache.json").read_text())
    by10 = {c["hash"][:10]: c for c in commits}
    audit = list(csv.DictReader(open(HERE / "audit_fix_labels.csv", encoding="utf-8")))
    prs = {c["hash"]: (gh.get(c["hash"]) or {}).get("pr_number") for c in commits
           if c["is_fix_B"] or c["hash"][:10] in {r["hash"][:10] for r in audit}}
    cache = fetch_issues([p for p in prs.values() if p])
    for c in commits:
        pr = prs.get(c["hash"]) if c["hash"] in prs else None
        iss = cache.get(str(pr)) if pr else None
        c["s_test"] = any("test" in f for f in c["files"])
        c["s_issue"] = bool(iss)
        c["s_bug"] = bool(iss) and any(BUG_RE.search(l) for i in iss for l in i["labels"])
    N_det = sum(c["is_fix_B"] for c in commits)
    N_non = sum((not c["is_fix_B"]) and any(f.endswith(".py") for f in c["files"]) for c in commits)
    sig = {"S_test": lambda c: c["is_fix_B"] and c["s_test"], "S_issue": lambda c: c["is_fix_B"] and c["s_issue"],
           "S_bug": lambda c: c["is_fix_B"] and c["s_bug"], "S_bug_seul": lambda c: c["s_bug"]}
    isbug = lambda r: r["label_assistant"] == "bug"
    merge = lambda r: r["sujet"].startswith("Merge")
    det = [r for r in audit if r["groupe"] == "detecte"]
    real = [r for r in det if isbug(r) and not merge(r)]
    out = {"protocole": "CLAUDE.md §8 Chantier 10a (75cb0ab)", "generated_at": datetime.now(timezone.utc).isoformat(),
           "N_detectes": N_det, "N_non_detectes_py": N_non,
           "couverture_PR_detectes": float(np.mean([bool(prs.get(c["hash"])) for c in commits if c["is_fix_B"]])),
           "reference_B": {"precision_hors_merge": len(real) / 100, "vrais_bugs": len(real)}, "variantes": {}}
    for name, f in sig.items():
        v = {}
        if name != "S_bug_seul":
            kept = [r for r in det if f(by10[r["hash"][:10]])]
            kr = [r for r in kept if isbug(r) and not merge(r)]
            v.update({"retenus_audit": len(kept), "vrais_bugs_conserves": len(kr),
                      "part_vrais_bugs_conserves": len(kr) / len(real),
                      "precision_ic95": wil(len(kr), len(kept))})
        else:
            sel = [r for r in audit if f(by10[r["hash"][:10]])]
            sd = [r for r in sel if r["groupe"] == "detecte"]; sn = [r for r in sel if r["groupe"] != "detecte"]
            tp_d = sum(isbug(r) and not merge(r) for r in sd) / 100 * N_det   # vrais bugs retrouves parmi detectes
            tp_n = sum(isbug(r) for r in sn) / 200 * N_non                      # parmi non detectes
            n_sel = len(sd) / 100 * N_det + len(sn) / 200 * N_non
            real_n = sum(isbug(r) for r in audit if r["groupe"] != "detecte") / 200 * N_non
            real_all = len(real) / 100 * N_det + real_n
            v.update({"selectionnes_audit": len(sel), "precision_estimee": (tp_d + tp_n) / n_sel if n_sel else None,
                      "rappel_estime": (tp_d + tp_n) / real_all if real_all else None,
                      "n_selectionnes_estime": n_sel,
                      "rappel_B_reference": (len(real) / 100 * N_det) / real_all})
        out["variantes"][name] = v
        print(name, json.dumps(v), flush=True)
    # --- label recalcule + AUC(BASE) par variante ---
    for name, f in sig.items():
        for c in commits:
            c["is_fix"] = bool(f(c))
        bpc, _ = labeling.blame_line_fixes(commits, common.REPO_DIR)
        est = labeling.make_est_risque_v2(bpc, fenetre_jours=common.FENETRE_JOURS)
        base = features.build_base_features(commits, est)
        npos = int(base["risky"].sum())
        # < 25 positifs : AUC en CV non evaluable (une classe unique dans un pli) -> None, rapporte tel quel
        bs = Parallel(n_jobs=3)(delayed(a9.cv_aucs)(base, features.BASE_COLS, s) for s in range(5)) if npos >= 25 else [float("nan")]
        out["variantes"][name].update({"n_positifs": npos, "n_fix": int(sum(c["is_fix"] for c in commits)), "n_etiquetes": len(base),
                                       "part_risque": float(base["risky"].mean()), "AUC_BASE_cv": (None if npos < 25 else float(np.mean(bs)))})
        print(name, "label", json.dumps({k: out["variantes"][name][k] for k in ("n_fix", "part_risque", "AUC_BASE_cv")}), flush=True)
    for c in commits:
        c["is_fix"] = c["is_fix_B"]
    bpc, _ = labeling.blame_line_fixes(commits, common.REPO_DIR)
    est = labeling.make_est_risque_v2(bpc, fenetre_jours=common.FENETRE_JOURS)
    base = features.build_base_features(commits, est)
    bref = float(np.mean(Parallel(n_jobs=3)(delayed(a9.cv_aucs)(base, features.BASE_COLS, s) for s in range(5))))
    out["reference_B"]["AUC_BASE_cv"] = bref
    for n in ("S_test", "S_issue", "S_bug"):
        v = out["variantes"][n]
        r = {"precision_ge_B_plus_0_15": v["precision_ic95"][0] is not None and v["precision_ic95"][0] - len(real) / 100 >= 0.15,
             "vrais_bugs_ge_70pct": v["part_vrais_bugs_conserves"] >= 0.7, "part_risque_ge_5pct": v["part_risque"] >= 0.05,
             "AUC_BASE_pas_de_chute_gt_0_03": v["AUC_BASE_cv"] is not None and bref - v["AUC_BASE_cv"] <= 0.03}
        v["regle_adoption"] = r; v["adoptable"] = bool(all(r.values()))
    v = out["variantes"]["S_bug_seul"]
    v["regle_remplacement"] = {"precision_ge_0_5": (v["precision_estimee"] or 0) >= 0.5,
                               "rappel_ge_celui_de_B": (v["rappel_estime"] or 0) >= v["rappel_B_reference"],
                               "part_risque_ge_5pct": v["part_risque"] >= 0.05}
    v["remplacement_ok"] = bool(all(v["regle_remplacement"].values()))
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out["variantes"][k].get("adoptable", out["variantes"][k].get("remplacement_ok")) for k in out["variantes"]}))


if __name__ == "__main__":
    main()
