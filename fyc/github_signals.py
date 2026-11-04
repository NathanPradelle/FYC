"""Chantier 4 : signaux de processus via l'API GitHub (reviewers, delai de merge,
etat CI), rattaches a chaque commit etiquete via la PR associee.

Approche batchee en GraphQL (meme principe que `git cat-file --batch` et le
batching Ruff+Bandit, cf. CLAUDE.md §4) : une requete GraphQL peut interroger
plusieurs commits a la fois (champ "object(oid: ...)" aliase), plutot qu'un GET
REST par commit juste pour trouver sa PR (3198 commits etiquetes -> 3198+ appels
en serie). Mesure sur un petit lot avant generalisation, comme pour les chantiers
precedents : voir results/chantier4_github.py.

Cache disque resumable (results/github_cache.json, gitignore) : un rerun ne
refait jamais une requete deja en cache, et peut reprendre proprement si le
budget de requetes est epuise avant la fin.

Discipline NaN/distinction explicite (meme piege que has_py/Bandit, §7 CLAUDE.md) :
un sha ABSENT du cache signifie "pas encore mesure" (budget epuise avant d'y
arriver) -- different de "mesure, confirme sans PR associee" (cache[sha] = None).
Ne jamais confondre les deux : fyc.process_features doit pouvoir les distinguer.
"""
import json
import time
from pathlib import Path

import requests

GITHUB_GRAPHQL_URL = "https://api.github.com/graphql"
CACHE_PATH = Path(__file__).resolve().parent.parent / "results" / "github_cache.json"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

QUERY_TMPL = """
query {{
  rateLimit {{ cost remaining resetAt }}
  repository(owner: "{owner}", name: "{repo}") {{
{aliases}
  }}
}}
"""

COMMIT_FRAGMENT = """
    c{i}: object(oid: "{sha}") {{
      ... on Commit {{
        associatedPullRequests(first: 1) {{
          nodes {{
            number
            merged
            createdAt
            mergedAt
            comments {{ totalCount }}
            reviews(first: 50) {{
              totalCount
              nodes {{ author {{ login }} }}
            }}
            commits(last: 1) {{
              nodes {{ commit {{ statusCheckRollup {{ state }} }} }}
            }}
          }}
        }}
      }}
    }}
"""


def _load_token():
    for line in ENV_PATH.read_text().splitlines():
        if line.strip().startswith("token"):
            return line.split("=", 1)[1].strip()
    raise RuntimeError(f"variable 'token' introuvable dans {ENV_PATH}")


def _load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache):
    CACHE_PATH.write_text(json.dumps(cache))


def _parse_commit_node(node):
    """None si la PR associee a ete confirmee absente -- PAS si la mesure n'a
    simplement pas encore eu lieu (cette fonction n'est appelee que sur un noeud
    recu, donc la mesure a bien eu lieu ici)."""
    if not node:
        return None
    prs = node.get("associatedPullRequests", {}).get("nodes", [])
    if not prs:
        return None
    pr = prs[0]
    reviews = pr.get("reviews", {})
    reviewer_logins = {
        n["author"]["login"] for n in reviews.get("nodes", []) if n.get("author")
    }
    ci_state = None
    commit_nodes = pr.get("commits", {}).get("nodes", [])
    if commit_nodes:
        rollup = commit_nodes[0]["commit"].get("statusCheckRollup")
        ci_state = rollup["state"] if rollup else None

    return {
        "pr_number": pr["number"],
        "merged": pr.get("merged"),
        "created_at": pr.get("createdAt"),
        "merged_at": pr.get("mergedAt"),
        "n_comments": pr.get("comments", {}).get("totalCount"),
        "n_reviews": reviews.get("totalCount"),
        "n_reviewers": len(reviewer_logins),
        "ci_state": ci_state,
    }


def fetch_commit_pr_signals(shas, owner="pallets", repo="flask", batch_size=25,
                             min_remaining=100, sleep_between=0.3):
    """Renseigne le cache disque pour tous les sha de `shas` qui n'y sont pas
    deja (quelle que soit la valeur en cache, y compris None = confirme sans PR).
    S'arrete proprement (sans lever) si le budget GraphQL descend sous
    `min_remaining` -- un sha non couvert reste simplement absent du cache, un
    futur appel le reprendra sans tout refaire.

    Retourne le cache complet (dict sha -> signaux_ou_None pour les sha mesures,
    cle absente pour les sha pas encore mesures)."""
    cache = _load_cache()
    token = _load_token()
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    })

    todo = [s for s in shas if s not in cache]
    print(f"{len(shas) - len(todo)} deja en cache, {len(todo)} a recuperer")

    for start in range(0, len(todo), batch_size):
        batch = todo[start:start + batch_size]
        aliases = "\n".join(
            COMMIT_FRAGMENT.format(i=i, sha=sha) for i, sha in enumerate(batch)
        )
        query = QUERY_TMPL.format(owner=owner, repo=repo, aliases=aliases)
        r = session.post(GITHUB_GRAPHQL_URL, json={"query": query})
        r.raise_for_status()
        data = r.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")

        remaining = data["data"]["rateLimit"]["remaining"]
        cost = data["data"]["rateLimit"]["cost"]
        repo_data = data["data"]["repository"]
        for i, sha in enumerate(batch):
            cache[sha] = _parse_commit_node(repo_data.get(f"c{i}"))

        _save_cache(cache)
        done = start + len(batch)
        print(f"  {done}/{len(todo)} -- cout requete: {cost}, budget restant: {remaining}")

        if remaining < min_remaining:
            print(f"Budget sous le seuil ({min_remaining}) -- arret propre, "
                  f"{len(todo) - done} commits restants pour un futur run.")
            break

        time.sleep(sleep_between)

    return cache
