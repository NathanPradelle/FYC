"""Extraction de l'historique Git et des versions historiques de fichiers.

Porte a l'identique la logique des cellules 6 et 49-50 du notebook.
"""
import subprocess


def extraire_commits(repo_dir):
    """Retourne une liste de dicts, un par commit, en ordre chronologique."""
    fmt = "COMMIT|%H|%an|%at|%s"
    out = subprocess.run(
        ["git", "-C", repo_dir, "log", f"--pretty=format:{fmt}", "--numstat"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    ).stdout

    commits, cur = [], None
    for line in out.splitlines():
        if line.startswith("COMMIT|"):
            if cur:
                commits.append(cur)
            _, h, author, ts, msg = line.split("|", 4)
            cur = {"hash": h, "author": author, "ts": int(ts),
                   "msg": msg, "files": [], "add": 0, "del": 0}
        elif line.strip() and cur is not None:
            parts = line.split("\t")
            if len(parts) == 3:
                a, d, f = parts
                cur["files"].append(f)
                cur["add"] += int(a) if a.isdigit() else 0
                cur["del"] += int(d) if d.isdigit() else 0
    if cur:
        commits.append(cur)
    commits.reverse()
    return commits


def get_head(repo_dir):
    return subprocess.run(
        ["git", "-C", repo_dir, "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def get_parents(repo_dir):
    """hash -> liste des hash parents, pour tous les commits du depot."""
    parents = {}
    out = subprocess.run(
        ["git", "-C", repo_dir, "log", "--pretty=format:%H %P"],
        capture_output=True, text=True, check=True,
    ).stdout
    for line in out.splitlines():
        p = line.split()
        parents[p[0]] = p[1:] if len(p) > 1 else []
    return parents


def diff_raw(repo_dir, old_rev, new_rev):
    """Diff unifie, zero contexte, espaces ignores -- pour reperer les lignes
    supprimees/modifiees par un commit par rapport a son parent (SZZ ligne)."""
    return subprocess.run(
        ["git", "-C", repo_dir, "diff", "--no-color", "--ignore-all-space", "-U0",
         old_rev, new_rev],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    ).stdout


def blame_porcelain(repo_dir, rev, path, ranges):
    """git blame --porcelain sur `path` a l'etat `rev`, restreint a `ranges`
    (liste de (start, end), 1-indexe inclusif). -w -M -C : ignore les espaces,
    detecte les deplacements/copies de code -- evite d'attribuer une ligne a un
    commit de pur reformatage/renommage (cf. CLAUDE.md, protocole Chantier 1).
    Retourne None si le fichier n'existe pas a `rev` (renommage non suivi, etc.)."""
    cmd = ["git", "-C", repo_dir, "blame", "-w", "-M", "-C", "--porcelain"]
    for start, end in ranges:
        cmd += ["-L", f"{start},{end}"]
    cmd += [rev, "--", path]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return None
    return r.stdout


def cat_file_batch(repo_dir, specs):
    """Recupere en un seul processus le contenu de plusieurs objets Git.

    specs : liste de chaines "sha:chemin" (ou "sha" seul).
    Retourne un dict {spec_index: contenu_str_ou_None}, aligne sur `specs`.
    """
    raw = subprocess.run(
        ["git", "-C", repo_dir, "cat-file", "--batch"],
        input=("\n".join(specs) + "\n").encode(),
        capture_output=True, check=True,
    ).stdout

    contents = {}
    pos, k = 0, 0
    while pos < len(raw) and k < len(specs):
        nl = raw.index(b"\n", pos)
        header = raw[pos:nl].decode("utf-8", "replace")
        pos = nl + 1
        if header.endswith("missing"):
            contents[k] = None
            k += 1
            continue
        size = int(header.rsplit(" ", 2)[2])
        contents[k] = raw[pos:pos + size].decode("utf-8", "replace")
        pos += size + 1
        k += 1
    return contents
