"""Chantier 9b/10b : kappa de Cohen entre label_assistant (results/audit_fix_labels.csv) et l'annotation humaine,
lue dans la feuille AVEUGLE results/audit_annotation_humaine.csv (4e colonne) ou, a defaut, dans la colonne
label_humain de audit_fix_labels.csv. Binaire bug / non-bug ('incertain' = non-bug des deux cotes).
Echoue proprement si rien n'est annote. Usage : python results/audit_kappa.py
"""
import csv
import sys
from pathlib import Path

D = Path(__file__).parent
master = list(csv.DictReader(open(D / "audit_fix_labels.csv", encoding="utf-8")))
human = {}
blind = D / "audit_annotation_humaine.csv"
if blind.exists():
    for r in csv.reader(open(blind, encoding="utf-8")):
        if r and r[0] != "id" and len(r) >= 4 and r[3].strip():
            human[r[0]] = r[3].strip().lower()
for r in master:
    if r["label_humain"].strip():
        human.setdefault(r["id"], r["label_humain"].strip().lower())
done = [r for r in master if r["id"] in human]
if not done:
    sys.exit("Aucune annotation humaine : remplir la 4e colonne de results/audit_annotation_humaine.csv (bug / non_bug / incertain).")
a = [r["label_assistant"] == "bug" for r in done]
h = [human[r["id"]] == "bug" for r in done]
n = len(done)
po = sum(x == y for x, y in zip(a, h)) / n
pa, ph = sum(a) / n, sum(h) / n
pe = pa * ph + (1 - pa) * (1 - ph)
kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
print(f"annotes par l'humain : {n}/{len(master)} ; accord brut {po:.3f} ; kappa de Cohen {kappa:.3f}")
print(f"bug assistant {sum(a)}, bug humain {sum(h)}")
