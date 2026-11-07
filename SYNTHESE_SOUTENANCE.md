# Synthèse de soutenance — FYC

Projet ESGI 4A (Yanis, Lior, Nathan, Thomas). Tous les chiffres viennent de `CLAUDE.md` §5-§6 et
des JSON de `results/` (source de vérité). Ce document dit **ce qu'on peut affirmer, avec quelle
réserve**, et ce qu'on ne peut pas.

## 1. Question et démarche

**Question :** peut-on prédire quels commits d'un dépôt introduiront un défaut, en croisant
plusieurs familles de signaux (historique Git, code, analyse statique, diff, processus) ?

**Démarche (le point fort du projet) :** pour chaque piste, un protocole avec seuils écrit et
commité **avant** la mesure, puis la mesure, puis la lecture, y compris quand elle est négative.
Les protocoles sont datés dans l'historique Git (ex. `9e8e892` stacking, `55b13fe` PySZZ,
`7350811` vague 2). Aucun résultat n'a été retouché après coup.

## 2. Les cinq messages à retenir

1. **L'étiquette est le maillon faible, pas le modèle.** Avec le SZZ « fichier » d'origine, le
   churn est corrélé à 0,91 avec le label : le gain annoncé (0,784 → 0,854) était en grande partie
   circulaire. Le SZZ « ligne » (`git blame -w -M -C`) réduit nettement cette circularité au niveau
   prédictif, et reste **robuste face à PySZZ** (kappa 0,92 avec B-SZZ, 0,83 avec MA-SZZ).
2. **Chaque famille de signaux apporte peu, aucune n'est inutile.** Gains sur BASE (0,787) :
   process GitHub +0,012, Ruff +0,011, Pylint +0,009, mypy +0,007, diff +0,005, radon +0,004,
   couplage +0,003. Tous sous le bruit de validation croisée (≈ ±0,03).
3. **Combiner n'améliore pas de façon démontrée.** Grille de 2048 combinaisons : plateau dès 4-5
   familles, « malédiction du gagnant » mesurée (−0,010 en re-mesure), sélection instable.
   Stacking par famille : ne bat ni BASE ni l'empilement en une forêt. Sur la réserve temporelle,
   BASE, l'empilement et la meilleure combinaison sont indiscernables (≈ 0,74, IC ±0,08).
4. **Le seul gain régulier vient du churn, la famille la plus suspecte.** « Tout empiler » bat BASE
   de +0,026 (15 plis sur 15 en comparaison appariée), mais +0,017 seulement sans churn.
5. **Un modèle pré-entraîné ailleurs transfère, pour classer et pas pour chiffrer.** Entraîné sur
   6 dépôts, il classe les commits d'un 7ᵉ jamais vu (AUC médiane 0,745, lift ≈ 3× sur le top
   10 %), mais ses probabilités brutes sont pires qu'un taux de base constant.
   Conclusion produit : **afficher un rang (« top 10 % à risque »), pas un pourcentage.**

## 3. Résultats négatifs établis (à présenter comme des résultats)

| Piste | Résultat |
|---|---|
| Commits nocturnes (`is_night`) | AUC seule 0,522, gain −0,001 ; hypothèse horaires UTC |
| Ratio code/test en lignes | AUC 0,499, pire que le binaire `touches_tests` (0,621) |
| Entropie du changement (indicateur phare de la littérature JIT) | AUC seule 0,478-0,510, rien |
| Semgrep, jscpd | pas de gain démontré (+0,0125, +0,0004) ; ≈ 35-38 min sur 3200 commits |
| Stacking par famille | S1 +0,010, S2 −0,027 vs BASE ; réserve : 0,680 et 0,635 contre 0,736 |
| Recalibration locale des probabilités | corrige le Brier mais n'apporte presque rien au-delà du taux de base (plafond oracle +0,012) |
| Pré-entraînement au démarrage à froid | échec sur la configuration principale (+0,009 contre +0,05 visé) ; dépend de la variante |

## 4. Reproductibilité : ce qu'on a trouvé en route

- **Version de Python = résultat.** `has_py == 0` sur 49,8 % des commits sous Python 3.14.4 contre
  54,5 % sous 3.12.3 : le taux d'échec de parsing du vieux code Python 2 change les features radon
  (d'où Python 3.14.4 figé).
- **Échecs silencieux des outils d'analyse**, même famille d'erreur, trouvée 5 fois : radon
  (`mi_avg=100.0` = « parfait »), Bandit, mypy (0 erreur sur du code non vérifié), Semgrep (aucune
  erreur sur du Python 2), jscpd (aucun rapport, code retour 0). Règle adoptée : **NaN, jamais 0**,
  et tester chaque outil sur un fichier volontairement cassé avant de lui faire confiance.
- **Mémoire :** 7 dépôts en parallèle ont fait planter la machine (JSON de Ruff) → traitement par
  lots, exécution séquentielle, garde-fou RAM (§7 point 15).

## 5. Limites à dire soi-même avant qu'on nous les pose

- **Peu de positifs :** ≈ 310 sur Flask, ≈ 33 dans la réserve temporelle → IC d'environ ±0,08 ; la
  plupart des mesures sont à une graine.
- **Pas de vérité terrain** pour les étiquettes : les mots-clés de messages de commit et SZZ sont
  des approximations ; PySZZ mesure un accord entre algorithmes, pas la justesse.
- **Une réserve non levée sur le Chantier 1 :** le ratio M2bis de `label_v2` + détection B vaut
  0,910, au-dessus du seuil 0,9044 (le 0,846 cité datait d'avant la détection B) ; écart dans le
  bruit, mais le critère n'est pas satisfait stricto sensu. La corrélation agrégée par fichier (M1)
  reste à 0,81.
- **Sous MA-SZZ**, le gain des familles d'outils disparaît (« tout » 0,772 contre 0,811).
- **Python uniquement**, 7 dépôts, aucun dépôt data/ML. **SonarQube et CodeQL non testés** : c'est
  un non-test (RAM), pas un résultat sur ces outils.
- **Notebook :** les Parties 2-3 gardent le label d'origine du TP ; ses chiffres (ex. `BASE+churn`
  0,864) ne sont pas ceux du projet (voir `README.md`).

## 6. Questions probables et réponses

- *« Pourquoi pas plus de features ? »* — On en a testé 11 familles ; la grille montre un plateau
  et un gain régulier uniquement pour le churn. Empiler ne marche pas, c'est mesuré.
- *« Votre AUC de 0,81 est-elle fiable ? »* — C'est une validation croisée mélangée ; sur la réserve
  temporelle on est à ≈ 0,74 avec un IC de ±0,08. On ne sait pas dire si l'écart est de la dérive
  ou du bruit (hypothèse non démontrée).
- *« Pourquoi garder `label_v2` ? »* — Trois mesures sur quatre du protocole préenregistré, plus la
  robustesse à PySZZ ; avec la réserve M1/M2bis ci-dessus.
- *« À quoi sert le modèle ? »* — À prioriser la relecture : lift ≈ 3× sur le top 10 %, pas à
  prédire une probabilité de bug.
- *« Qu'auriez-vous fait avec plus de temps ? »* — Sonar/CodeQL sur une machine plus grosse ;
  plusieurs graines et un dépôt data/ML ; comprendre la dérive de prévalence (22-46 % au début de
  l'historique contre 3-13 % en queue de test).
