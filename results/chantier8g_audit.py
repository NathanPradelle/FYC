"""Chantier 8g : audit preliminaire des etiquettes de fix (precision / rappel de la detection par mots-cles).
Annotation faite par l'assistant (aveugle au groupe), colonne label_assistant de results/audit_fix_labels.csv ;
label_humain laisse vide pour l'equipe. Protocole : CLAUDE.md §8 Chantier 8 (f7e8489). Non independant."""
import csv, json, math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import chantier4bis_common as common

HERE = Path(__file__).resolve().parent
commits, _est, _ = common.load_context()

def py(c):
    return any(n.endswith(".py") for n, *_ in c["file_stats"])

Nd = sum(c["is_fix"] for c in commits)
Nn = sum((not c["is_fix"]) and py(c) for c in commits)
rows = list(csv.DictReader(open(HERE / "audit_fix_labels.csv", encoding="utf-8")))

def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [p, c - h, c + h]

def cnt(g, ls):
    return sum(r["groupe"] == g and r["label_assistant"] in ls for r in rows)

out = {"N_detectes": Nd, "N_non_detectes_touchant_py": Nn, "echantillons": {"detectes": 100, "non_detectes": 200}}
for nm, ls in (("incertain_compte_non_bug", ("bug",)), ("incertain_compte_bug", ("bug", "incertain"))):
    kd, kn = cnt("detecte", ls), cnt("non_detecte", ls)
    pd_, dl, dh = wil(kd, 100); pn, nl, nh = wil(kn, 200)
    tp, fn = pd_ * Nd, pn * Nn
    out[nm] = {"bugs_parmi_detectes": kd, "precision_ic95": wil(kd, 100),
               "bugs_parmi_non_detectes": kn, "taux_non_detectes_ic95": wil(kn, 200),
               "rappel": tp / (tp + fn),
               "rappel_ic95_approx": [dl * Nd / (dl * Nd + nh * Nn), dh * Nd / (dh * Nd + nl * Nn)]}
merges = [r for r in rows if r["groupe"] == "detecte" and r["sujet"].startswith("Merge")]
out["merges_detectes"] = {"n": len(merges), "labels": [r["label_assistant"] for r in merges]}
(HERE / "chantier8g_audit.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
