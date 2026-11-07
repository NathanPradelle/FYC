"""Chantier 9b : kappa de Cohen entre label_assistant et label_humain (results/audit_fix_labels.csv).
Binaire bug / non-bug ('incertain' compte non-bug cote assistant ; cote humain, ecrire bug, non_bug ou incertain).
Echoue proprement si label_humain est vide. Usage : python results/audit_kappa.py
"""
import csv
import sys
from pathlib import Path

rows = list(csv.DictReader(open(Path(__file__).parent / "audit_fix_labels.csv", encoding="utf-8")))
done = [r for r in rows if r["label_humain"].strip()]
if not done:
    sys.exit("label_humain est vide : l'equipe doit annoter results/audit_fix_labels.csv (bug / non_bug / incertain).")
a = [r["label_assistant"] == "bug" for r in done]
h = [r["label_humain"].strip().lower() == "bug" for r in done]
n = len(done)
po = sum(x == y for x, y in zip(a, h)) / n
pa, ph = sum(a) / n, sum(h) / n
pe = pa * ph + (1 - pa) * (1 - ph)
kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
print(f"n annotes humain : {n}/{len(rows)} ; accord brut {po:.3f} ; kappa de Cohen {kappa:.3f}")
print(f"bug assistant {sum(a)}, bug humain {sum(h)}")
