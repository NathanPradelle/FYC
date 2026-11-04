"""Figures d'illustration du Chantier 4bis, regenerees depuis results/chantier4bis_grid.json
(les PNG ne sont pas versionnes : *.png est dans .gitignore ; ce script est la source).
Sorties : results/fig_4bis_grille.png, results/fig_4bis_gain_marginal.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
d = json.loads((HERE / "chantier4bis_grid.json").read_text())
rows, st = d["grille_complete"], d["grille_stats"]
INK, MUTED, GRID, ACC, NEG = "#1f2933", "#6b7785", "#e4e8ec", "#2a6fb0", "#b8860b"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK})

# 1) toutes les combinaisons, par nombre de familles
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
rng = np.random.default_rng(0)
x = np.array([r["n_familles"] for r in rows]) + rng.uniform(-0.28, 0.28, len(rows))
ax.scatter(x, [r["auc"] for r in rows], s=6, color=ACC, alpha=0.28, linewidths=0)
best = st["meilleure_par_nombre_de_familles"]
ax.plot(range(12), [best[str(n)]["auc"] for n in range(12)], color=INK, lw=1.6, marker="o", ms=4)
ax.axhline(st["auc_base"], color=MUTED, lw=1, ls="--")
ax.text(0.9, st["auc_base"] + 0.0008, "BASE seule", ha="left", va="bottom", color=MUTED, fontsize=9)
ax.set_ylim(0.790, 0.8265)
ax.text(7.6, 0.8247, "meilleure combinaison par taille", ha="center", fontsize=9)
ax.set_xlabel("Nombre de familles d'outils ajoutées à BASE")
ax.set_ylabel("AUC (CV 5 plis, 80 % anciens)")
ax.set_xticks(range(12)); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("2048 combinaisons : le gain plafonne dès 4-5 familles",
             loc="left", fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(HERE / "fig_4bis_grille.png"); plt.close(fig)

# 2) gain marginal moyen par famille
g = st["gain_marginal_par_famille"]
names = list(g)[::-1]; vals = [g[n]["gain_marginal_moyen"] for n in names]
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
ax.barh(names, vals, color=[ACC if v >= 0 else NEG for v in vals], height=0.55)
for n, v in zip(names, vals):
    ax.text(v + (0.0001 if v >= 0 else -0.0001), n, f"{v:+.4f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=9)
ax.axvline(0, color=MUTED, lw=1)
ax.set_xlabel("Gain d'AUC moyen quand on ajoute la famille (sur toutes les combinaisons sans elle)")
ax.set_xlim(-0.0022, 0.0092); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("Seul le churn apporte un gain régulier ; Vulture dégrade (en jaune)", loc="left",
             fontsize=11, color=INK)
fig.tight_layout(); fig.savefig(HERE / "fig_4bis_gain_marginal.png"); plt.close(fig)
print("ok")
