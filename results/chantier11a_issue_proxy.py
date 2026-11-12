"""Chantier 11a : proxy 'la PR du commit ferme une issue labellisee bug' sur 2 depots retenus par une sonde.
Protocole : CLAUDE.md §8 Chantier 11 (84e34fe). Caches API : results/github_proxy_cache_<depot>.json (gitignore).
Usage : python results/chantier11a_issue_proxy.py probe | measure
"""
import json
import math
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chantier6a_dataset as ds
import chantier9a_fix_strict as a9

import numpy as np
import requests
from joblib import Parallel, delayed

from fyc import features, git_extract, labeling
from fyc.github_signals import GITHUB_GRAPHQL_URL, _load_token

BUG_RE = re.compile(r"bug|defect|regression|crash", re.I)
GH = {"flask": "pallets/flask", "requests": "psf/requests", "click": "pallets/click", "black": "psf/black",
      "tornado": "tornadoweb/tornado", "pydantic": "pydantic/pydantic", "django-rest-framework": "encode/django-rest-framework",
      "datasets": "huggingface/datasets", "ignite": "pytorch/ignite", "poetry": "python-poetry/poetry",
      "werkzeug": "pallets/werkzeug", "rich": "Textualize/rich"}
PATHS = {**{k: v for k, v in ds.REPOS.items()}, "datasets": ds.ROOT / "repos" / "datasets", "ignite": ds.ROOT / "repos" / "ignite",
         "poetry": ds.ROOT / "repos" / "poetry", "werkzeug": ds.ROOT / "repos" / "werkzeug", "rich": ds.ROOT / "repos" / "rich"}
SESS = requests.Session(); SESS.headers["Authorization"] = f"Bearer {_load_token()}"


def gql(query):
    while True:
        r = SESS.post(GITHUB_GRAPHQL_URL, json={"query": query}, timeout=120)
        if r.status_code in (502, 503, 504):
            time.sleep(5); continue
        r.raise_for_status()
        d = r.json()
        rl = (d.get("data") or {}).get("rateLimit")
        if rl and rl["remaining"] < 150:
            wait = max(0, (datetime.fromisoformat(rl["resetAt"].replace("Z", "+00:00")) - datetime.now(timezone.utc)).total_seconds()) + 5
            print(f"  budget bas ({rl['remaining']}), pause {wait:.0f}s", flush=True); time.sleep(wait)
        if "data" not in d or d["data"] is None:
            raise RuntimeError(d.get("errors"))
        return d["data"]


def is_bug(issues):
    return bool(issues) and any(BUG_RE.search(l) for i in issues for l in i["labels"])


def probe():
    out = {"protocole": "CLAUDE.md §8 Chantier 11a (84e34fe), sonde", "generated_at": datetime.now(timezone.utc).isoformat(), "depots": {}}
    for name, full in GH.items():
        owner, rep = full.split("/"); cursor, prs = None, []
        for _ in range(2):
            after = f', after: "{cursor}"' if cursor else ""
            d = gql('query { rateLimit { remaining resetAt } repository(owner: "%s", name: "%s") { pullRequests(first: 100, states: MERGED, '
                    'orderBy: {field: CREATED_AT, direction: DESC}%s) { nodes { number closingIssuesReferences(first: 5) { nodes { number '
                    'labels(first: 5) { nodes { name } } } } } pageInfo { endCursor } } } }' % (owner, rep, after))
            pr = d["repository"]["pullRequests"]; prs += pr["nodes"]; cursor = pr["pageInfo"]["endCursor"]
        iss = [[{"number": i["number"], "labels": [l["name"] for l in i["labels"]["nodes"]]} for i in p["closingIssuesReferences"]["nodes"]] for p in prs]
        out["depots"][name] = {"n_pr": len(prs), "part_ferme_issue": float(np.mean([bool(x) for x in iss])),
                               "part_ferme_issue_bug": float(np.mean([is_bug(x) for x in iss]))}
        print(name, out["depots"][name], flush=True)
    ranked = sorted(out["depots"], key=lambda n: -out["depots"][n]["part_ferme_issue_bug"])
    out["retenus"] = ranked[:2]
    (HERE / "chantier11a_probe.json").write_text(json.dumps(out, indent=1))
    print("retenus :", out["retenus"])


def fetch_proxy(name, commits, batch=25):
    cache_f = HERE / f"github_proxy_cache_{name}.json"
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    owner, rep = GH[name].split("/")
    todo = [c["hash"] for c in commits if c["hash"] not in cache]
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        al = " ".join('c%d: object(oid: "%s") { ... on Commit { associatedPullRequests(first: 2) { nodes { number merged '
                      'closingIssuesReferences(first: 5) { nodes { number labels(first: 5) { nodes { name } } } } } } } }' % (k, h)
                      for k, h in enumerate(chunk))
        d = gql('query { rateLimit { remaining resetAt } repository(owner: "%s", name: "%s") { %s } }' % (owner, rep, al))
        for k, h in enumerate(chunk):
            node = d["repository"].get(f"c{k}")
            if node is None or "associatedPullRequests" not in node:
                cache[h] = {"pr": None, "issues": None}; continue   # mesure faite : introuvable / pas de PR
            prs = [p for p in node["associatedPullRequests"]["nodes"] if p["merged"]] or []
            if not prs:
                cache[h] = {"pr": None, "issues": None}; continue
            p = prs[0]
            cache[h] = {"pr": p["number"], "issues": [{"number": x["number"], "labels": [l["name"] for l in x["labels"]["nodes"]]}
                                                      for x in p["closingIssuesReferences"]["nodes"]]}
        if (i // batch) % 10 == 0:
            cache_f.write_text(json.dumps(cache)); print(f"  [{name}] {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    cache_f.write_text(json.dumps(cache))
    return cache


def wil(k, n, z=1.96):
    if n == 0:
        return [None, None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - h, c + h]


def label_variant(commits, path, fixflag):
    for c in commits:
        c["is_fix"] = bool(fixflag(c))
    bpc, _ = labeling.blame_line_fixes(commits, str(path))
    est = labeling.make_est_risque_v2(bpc, fenetre_jours=90)
    labs = {c["hash"]: est(c) for c in commits}
    base = features.build_base_features(commits, est)
    ch = features.build_churn_features(commits, est)
    import pandas as pd
    df = pd.concat([base, ch], axis=1).loc[:, lambda d: ~d.columns.duplicated()]
    return labs, df


def measure():
    sel = json.loads((HERE / "chantier11a_probe.json").read_text())["retenus"]
    out = {"protocole": "CLAUDE.md §8 Chantier 11a (84e34fe)", "generated_at": datetime.now(timezone.utc).isoformat(), "depots": {}}
    for name in sel:
        commits = git_extract.extraire_commits(str(PATHS[name]))
        labeling.tag_fixes(commits, fix_predicate=ds.FIX_PREDICATE)
        for c in commits:
            c["B"] = c["is_fix"]
        cache = fetch_proxy(name, commits)
        for c in commits:
            e = cache.get(c["hash"], {})
            c["pr"], c["P"] = e.get("pr"), is_bug(e.get("issues"))
        wp = [c for c in commits if c["pr"]]
        nB = sum(c["B"] for c in wp); nBP = sum(c["B"] and c["P"] for c in wp); nP = sum(c["P"] for c in wp)
        a, b = np.array([c["B"] for c in wp]), np.array([c["P"] for c in wp])
        po = float((a == b).mean()); pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
        r = {"n_commits": len(commits), "part_avec_PR": len(wp) / len(commits), "n_B_avec_PR": int(nB), "n_P": int(nP), "n_B_et_P": int(nBP),
             "precision_B_vs_P": wil(nBP, nB), "rappel_B_vs_P": wil(nBP, nP), "kappa_B_P": (po - pe) / (1 - pe) if pe < 1 else None}
        print(name, json.dumps(r), flush=True)
        for vn, f in (("B", lambda c: c["B"]), ("P_seul", lambda c: c["P"]), ("B_union_P", lambda c: c["B"] or c["P"])):
            labs, df = label_variant(commits, PATHS[name], f)
            npos = int(df["risky"].sum())
            ok = npos >= 25
            bs = Parallel(n_jobs=3)(delayed(a9.cv_aucs)(df, features.BASE_COLS, s) for s in range(5)) if ok else None
            cs = Parallel(n_jobs=3)(delayed(a9.cv_aucs)(df, features.CHURN_COLS, s) for s in range(20)) if ok else None
            b20 = Parallel(n_jobs=3)(delayed(a9.cv_aucs)(df, features.BASE_COLS, s) for s in range(20)) if ok else None
            r[vn] = {"n_fix": int(sum(c["is_fix"] for c in commits)), "n_etiquetes": len(df), "n_positifs": npos,
                     "part_risque": float(df["risky"].mean()),
                     "AUC_BASE_cv": float(np.mean(bs)) if ok else None,
                     "M2bis_ratio": float(np.mean([c / b for b, c in zip(b20, cs)])) if ok else None, "_labs": labs}
            print(name, vn, {k: v for k, v in r[vn].items() if k != "_labs"}, flush=True)
        # kappa des etiquettes sur les commits communs
        for vn in ("P_seul", "B_union_P"):
            com = [h for h, l in r["B"]["_labs"].items() if l is not None and r[vn]["_labs"][h] is not None]
            x = np.array([r["B"]["_labs"][h] for h in com]); y = np.array([r[vn]["_labs"][h] for h in com])
            po = (x == y).mean(); pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
            r[vn]["kappa_etiquettes_vs_B"] = float((po - pe) / (1 - pe)) if pe < 1 else None
        for vn in ("B", "P_seul", "B_union_P"):
            r[vn].pop("_labs")
        pv = r["P_seul"]
        r["robuste_P_seul"] = bool(pv["AUC_BASE_cv"] is not None and pv["part_risque"] >= 0.05
                                   and abs(pv["AUC_BASE_cv"] - r["B"]["AUC_BASE_cv"]) <= 0.03 and pv["M2bis_ratio"] <= 0.9044)
        out["depots"][name] = r
    (HERE / "chantier11a_issue_proxy.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({n: {"precision": d["precision_B_vs_P"][0], "rappel": d["rappel_B_vs_P"][0], "robuste": d["robuste_P_seul"]}
                      for n, d in out["depots"].items()}, indent=1))


if __name__ == "__main__":
    {"probe": probe, "measure": measure}[sys.argv[1]]()
