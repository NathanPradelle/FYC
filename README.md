# FYC — prédiction des anomalies logicielles (just-in-time)

Projet étudiant ESGI 4A (Yanis, Lior, Nathan, Thomas). Question : **peut-on prédire quels commits
d'un dépôt introduiront un défaut**, en croisant plusieurs familles de signaux (historique Git,
structure du code, analyse statique, diff, processus GitHub) ? Dépôt principal étudié :
[pallets/flask](https://github.com/pallets/flask). L'évaluation porte sur la **rigueur de la
démarche** : chaque choix est mesuré, les résultats négatifs sont gardés et documentés.

> Mémoire détaillée du projet (décisions, protocoles pré-enregistrés, résultats chiffrés, pièges,
> journal) : [`CLAUDE.md`](CLAUDE.md). Ce README n'en est que l'entrée.

## Installation

WSL2/Linux, **Python 3.14.4** (version figée volontairement : les métriques radon dépendent du taux
d'échec de parsing sur du code Python 2 de l'historique, qui varie avec l'interpréteur).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git clone https://github.com/pallets/flask flask      # non versionné
```

Facultatif : un token GitHub dans `.env` (`token=...`, non versionné) pour les signaux de processus
(API GraphQL). Les autres dépôts du chantier 6A se clonent dans `repos/` (non versionné).

## Organisation

| Chemin | Rôle |
|---|---|
| `fyc/` | Modules réutilisables : extraction Git, étiquetage SZZ (fichier v1, **ligne v2 par défaut**), features (BASE, churn, radon), analyse statique (Ruff, Bandit, Pylint, Vulture, mypy), diff, couplage, indicateurs de Kamei, signaux GitHub, évaluation |
| `results/benchmark.py` | Recalcule tous les jeux de features et écrit `results/results_v2.json` (source unique du tableau comparatif) |
| `results/chantier*.py` | Un script de mesure par chantier, chacun écrit son JSON daté (`results/*.json`) |
| `results/dataset_6a.csv.gz` | Jeu multi-dépôts exporté (7 dépôts, 25 179 lignes) pour le cours |
| `cas_pratique_FYC_complete_v2.ipynb` | Notebook qui orchestre et raconte (voir « Réserve » ci-dessous) |
| `CLAUDE.md` | Protocoles, résultats, pièges, journal |

## Reproduire

```bash
python results/benchmark.py --label v2        # ~6 min avec les caches (bien plus à froid) -> results/results_v2.json
python results/chantier5_stacking.py          # stacking par famille (~6 min)
python results/chantier6a_dataset.py          # construit le jeu multi-dépôts (long, un dépôt à la fois)
python results/chantier6a_eval.py             # transfert inter-dépôts (lit results/dataset_6a.pkl, non versionné)
```

Les calculs lourds sont **mis en cache** (`results/cache_*.pkl`, non versionnés, recalculables).
⚠️ Lancer les traitements **un à la fois** : plusieurs processus en parallèle ont saturé la RAM
(Ruff/Pylint). Les scripts longs ont des points de reprise et un garde-fou mémoire.

## Résultats en bref (chiffres complets et réserves dans `CLAUDE.md` §5-§6)

- **Étiquetage.** Le SZZ « fichier » d'origine rend le churn circulaire avec le label (r = 0,91).
  Le SZZ ligne (`git blame -w -M -C`) réduit nettement cette circularité au niveau prédictif ; il
  est confirmé robuste face à PySZZ (kappa 0,92 avec B-SZZ, 0,83 avec MA-SZZ).
- **Familles de signaux.** Individuellement, elles ajoutent un signal réel mais modeste à BASE
  (≈ 0,787 d'AUC) : Ruff +0,011, process GitHub +0,012, diff +0,005, radon +0,004. **Les combiner
  n'améliore pas de façon démontrée** (grille de 2048 combinaisons, sélection imbriquée, stacking) :
  le seul gain régulier vient du churn, la famille la plus suspecte.
- **Résultats négatifs conservés :** heure de nuit, ratio code/test, entropie du changement
  (indicateur phare de la littérature), Semgrep et jscpd à notre échelle.
- **Transfert entre dépôts.** Un modèle entraîné sur 6 dépôts classe correctement les commits à
  risque d'un 7ᵉ jamais vu (AUC médiane 0,745, lift ≈ 3× sur le top 10 %), mais ses probabilités
  brutes sont mal calibrées : **afficher un classement, pas un pourcentage**.

## Limites assumées

Dépôts Python uniquement ; peu de positifs (≈ 300 sur Flask, ≈ 33 dans la réserve temporelle,
donc des intervalles de confiance d'environ ±0,08) ; une seule graine pour plusieurs mesures ;
étiquettes dérivées de mots-clés de messages de commit ; SonarQube et CodeQL non testés ; pas de
vérité terrain pour les étiquettes. Voir `CLAUDE.md` §7 pour la liste complète des pièges rencontrés
(échecs silencieux des outils d'analyse sur le vieux code Python 2, mémoire, etc.).

## Réserve sur le notebook

Les cellules 49-53 (Partie 4.3) conservent le code inline d'origine du TP (ancien remplissage par
défaut des échecs radon, détection de fix brute) : elles ne reproduisent plus exactement
`results/results_v2.json`. Seul le tableau dynamique de la cellule 52 reflète l'état courant.
