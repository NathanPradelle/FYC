# CLAUDE.md — mémoire du projet FYC

## 1. Objectif et périmètre

Projet étudiant ESGI 4A (équipe : Yanis, Lior, Nathan, Thomas). Construire un système de
**prédiction des anomalies logicielles** croisant plusieurs familles de signaux (historique Git,
structure du code, outils d'analyse statique, signaux de processus) et plusieurs outils, pour
obtenir la prédiction la plus fiable et la mieux argumentée possible.

Direction donnée par l'encadrant : le périmètre initial (contexte Git seul, sans lire le code) est
**trop étroit**. L'analyse de code devient un pilier à part entière. On cherche à **combiner**
intelligemment les sources de signal, pas à les empiler pour faire monter un chiffre.

Évaluation : rigueur de la démarche et capacité à expliquer chaque choix, pas le score brut. Un
résultat négatif bien établi vaut mieux qu'un bon chiffre non compris.

Dépôt étudié : [pallets/flask](https://github.com/pallets/flask) (Python, ~5 500 commits).

## 2. État de l'environnement

- WSL2 (Ubuntu). Travail dans le filesystem Linux (`~/dev/FYC`), jamais `/mnt/c/`.
- venv Python **3.14.4** dans `.venv/` (déjà présent au démarrage de ce projet).
- Dépendances figées dans `requirements.txt` (créé le 2026-09-09). Cœur : pandas 3.0.5, numpy
  2.5.3, scikit-learn 1.9.0, matplotlib 3.11.1, seaborn 0.13.2. Analyse de code : radon 6.0.1,
  ruff 0.16.6 (**installés le 2026-09-09** — absents du venv initial, nécessaires aux cellules 49
  et 56 de la Partie 4).
- Dépôt Flask cloné en local dans `flask/` (5557 commits, non versionné, dans `.gitignore`).
- `.gitignore` couvre `flask/`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.csv`, `*.png`,
  `*Zone.Identifier`.
- Pas encore de `README.md`.

## 3. Architecture du code

- **`fyc/`** — modules réutilisables, extraits du notebook le 2026-09-09 (squelette minimal,
  seulement ce que le Chantier 1 va réellement utiliser) :
  - `git_extract.py` : `extraire_commits`, `get_head`, `get_parents`, `cat_file_batch`.
  - `labeling.py` : SZZ simplifié — `FIX_RE`, `tag_fixes`, `build_fix_index`, `make_est_risque`,
    `label_commits`. **Interface pensée pour rester stable** quand le Chantier 1 changera
    l'implémentation (détection de fix plus stricte, granularité ligne).
  - `features.py` : `build_base_features`, `build_is_night`, `build_churn_features`,
    `build_code_features` (radon via `git cat-file`), `churn_fix_correlation` (diagnostic de
    circularité).
  - `evaluate.py` : `cv_auc`, `temporal_split_auc`, `baseline_majority_accuracy`.
- **`results/benchmark.py`** — recalcule tous les jeux de features de la Partie 4 et écrit
  **`results/results.json`** (AUC CV ± std, AUC split temporel, date, hash HEAD Flask, versions des
  libs). Source unique de vérité pour le tableau comparatif — plus aucun chiffre à recopier à la
  main dans le notebook pour cette table précise. `python results/benchmark.py` depuis la racine
  (venv actif) pour régénérer.
- **`cas_pratique_FYC_complete_v2.ipynb`** (58 cellules) — orchestre et raconte. Structure :

- **Partie 1** (cellules 0–13) : clone Flask, extraction `git log --numstat`, DataFrame, exploration.
- **Partie 2** (14–24) : étiquetage SZZ simplifié (`is_fix` par regex, propagation par fenêtre de
  90 jours), construction des features de contexte (BASE), premier `RandomForestClassifier`.
- **Partie 3** (25–37) : évaluation (AUC, matrice de confusion), importance impureté vs permutation,
  analyses complémentaires (sensibilité fenêtre, split temporel, comparaison à la régression
  logistique).
- **Partie 4** (38–57) : trois pistes d'enrichissement mesurées une à une —
  **A.** `is_night` (heure UTC), **B.** churn/âge fichier, **C.** métriques radon (`cc`, `mi`, `loc`,
  `comment_ratio`, `delta_cc`) récupérées via `git cat-file --batch` sur les versions avant/après
  de chaque commit. Ouverture sur Ruff/Bandit/Lizard/Semgrep/SonarQube et sur les signaux de
  process (API GitHub).

Aucun module `.py` réutilisable pour l'instant (todo : Chantier 7, extraire la plomberie du
notebook au fur et à mesure des chantiers, cf. §8).

## 4. Décisions de conception

- **Étiquetage SZZ simplifié, granularité fichier, fenêtre 90 jours.** *Pourquoi :* fenêtre courte
  pour éviter que le label dégénère en « popularité du fichier » (voir §6, analyse de sensibilité) ;
  90 j retenu comme compromis donnant ~51 % de positifs (classe équilibrée). *Limite assumée :*
  granularité fichier plutôt que ligne, ce qui est la cause directe de la circularité churn/label
  mesurée en §5.
- **RandomForest plutôt que régression logistique.** *Pourquoi :* gain mesuré et non supposé —
  AUC 0.775 (forêt) vs 0.671 (logistique) sur le même split, l'écart justifie le choix (relations
  non linéaires, ex. risque vs nombre de fichiers).
- **Importance par permutation en plus de l'importance par impureté.** *Pourquoi :* l'impureté
  surévalue les variables à forte cardinalité (`msg_len`, `hour`) ; nécessaire pour ne pas tirer de
  fausses conclusions sur ce qui est réellement prédictif.
- **`git cat-file --batch` plutôt que `git show` par fichier** pour récupérer les versions
  historiques. *Pourquoi :* mesuré comme indispensable — 5539 versions récupérées en 0.6 s dans un
  seul process, un `git show` par fichier aurait pris des dizaines de minutes.
- **radon plutôt que SonarQube pour l'historique complet.** *Pourquoi :* radon tourne en pur Python
  sans serveur, donc utilisable sur des milliers de versions ; SonarQube est un serveur, quelques
  dizaines de secondes par analyse, donc réservé à un échantillon (200–300 commits) ou au flux
  courant — décision **pas encore mise en œuvre**, c'est une piste (§8, Chantier 2).

## 5. Résultats mesurés

**Exécution de référence : 2026-09-09**, notebook exécuté de bout en bout
(`jupyter nbconvert --execute`), Python 3.14.4 / pandas 3.0.5 / scikit-learn 1.9.0 / radon 6.0.1,
sans erreur. `random_state` fixé partout, CV = `StratifiedKFold(5, shuffle=True, random_state=0)`.

| Jeu de features | AUC (CV 5 plis) | AUC split temporel | Réserves |
|---|---|---|---|
| BASE (contexte Git, 10 features) | 0.784 ± 0.020 | 0.795 | référence |
| BASE + `is_night` | 0.783 ± 0.019 | — | **réfutée** — aucun gain |
| BASE + churn fichier | 0.854 ± 0.016 | — | **circulaire** — corrélation 0.91 avec le label |
| BASE + métriques radon | **0.836 ± 0.013** | — | gain réel, non circulaire (voir écart ci-dessous) |
| Métriques radon seules | **0.737 ± 0.007** | — | signal réel sans aucune donnée de process |
| Tout combiné | **0.874 ± 0.014** | — | à annoncer avec réserves (gain en partie circulaire) |
| Régression logistique (BASE) | — | — | AUC 0.671 sur split random, contre 0.775 pour la forêt |
| Baseline classe majoritaire | — | — | accuracy 0.511 (contre 0.70 pour le modèle BASE) |

**⚠️ Écart avec les chiffres annoncés dans les cellules markdown du notebook**, à signaler en
soutenance : le notebook contient déjà une narration (Partie 4.3/4.4) écrite par l'encadrant avec
des chiffres de référence (`BASE+code` = 0.820, `code seul` = 0.691, `TOUT` = 0.870). Le run réel
du 2026-09-09 mesure des valeurs **systématiquement plus hautes** pour tout ce qui touche aux
métriques de code : `BASE+code` **0.836** (vs 0.820 annoncé, écart +0.016), `code seul` **0.737**
(vs 0.691 annoncé, écart +0.046 — le plus gros écart du tableau). Les chiffres hors-code (BASE,
`is_night`, churn) coïncident presque exactement avec la narration (écarts ≤ 0.002).

**Cause : non expliquée à ce stade.** L'hypothèse « version de radon différente » a été **vérifiée
et écartée** (radon 6.0.1 confirmé comme la version utilisée pour produire les chiffres de
référence — ce n'était qu'une hypothèse non testée avant vérification, cf. §7). L'hypothèse « état
du clone Flask différent » a aussi été vérifiée et écartée le 2026-09-09 (HEAD et nombre de commits
identiques à la référence, cf. §7). Candidats restants non vérifiés : versions scikit-learn/numpy,
divergence entre le code actuel et celui ayant produit les chiffres d'origine, cap de 5 fichiers
`.py`/commit. **Décision : on n'investigue pas plus loin maintenant** — le Chantier 1 va changer le
label et invalider ces chiffres de toute façon, creuser l'écart serait du travail perdu. **Ce qui
compte : le signal code-only mesuré aujourd'hui (0.737) est réel et non circulaire, cite les
chiffres mesurés aujourd'hui, pas ceux du markdown.**

Détail complémentaire (modèle final, tout combiné, importance par permutation, run du
2026-09-09) : la feature la plus importante n'est **pas** une feature de code mais
`file_age_days_avg` (perte d'AUC 0.0635), suivie de `file_commits_max` (0.0391, toutes deux des
features de churn/piste B). Parmi les features de code, `loc_avg` domine (0.0199), `comment_ratio`
suit (0.0133) ; `cc_max_avg` et `delta_cc` restent faibles (0.0060 et 0.0016) — cohérent avec la
lecture du notebook : *ce n'est pas le commit qui complexifie le code qui est risqué, c'est le
commit qui touche du code déjà complexe/volumineux*.

## 6. Hypothèses testées

- **✅ Validée — Ownership diffus = risque.** `avg_authors_per_file` reste parmi les features les
  plus importantes en permutation dans tous les modèles testés.
- **✅ Validée — Taille du changement = risque.** `n_files`, `total_churn` pèsent significativement
  plus que `hour`/`weekend` en importance par permutation.
- **❌ Réfutée — Commits nocturnes plus risqués.** AUC d'une feature `is_night` seule ≈ 0.522
  (hasard = 0.500). Ajoutée aux autres features, gain en CV = **−0.001** (0.784 → 0.783). Cause
  la plus probable : timestamps `%at` en UTC, pas en heure locale de l'auteur — confondu avec la
  répartition géographique des contributeurs plutôt qu'avec la fatigue. Piste de correction non
  explorée : `%ai` (expose le fuseau).
- **⚠️ Circulaire — Churn fichier prédit le risque.** Gain brut fort (0.784 → 0.854, +0.070) mais
  corrélation de Pearson mesurée à **0.91** (Spearman 0.727) entre nombre de commits d'un fichier
  et nombre de fix reçus par ce fichier — qui définit le label. Le modèle glisse de « commit à
  risque » vers « fichier à risque ». Gain à ne pas vendre tel quel.
- **✅ Validée, gain réel — Métriques de code (radon).** Gain non circulaire mesuré (le churn ne
  contamine pas ces features, calculées sur le contenu du fichier, pas sur son historique de
  commits). Ampleur du gain réévaluée à la hausse par rapport au narratif du notebook (voir §5).
- **✅ Validée — La forêt aléatoire bat la régression logistique** sur ce problème (AUC 0.775 vs
  0.671, split identique). Les relations captées sont non linéaires.
- **Non concluant à ce stade — Sensibilité de l'étiquette à la fenêtre temporelle.** Mesuré :
  7 j → 23,0 % ; 30 j → 38,7 % ; 90 j → 51,2 % ; 180 j → 59,8 % ; 365 j → 68,1 % ; 3650 j → 86,9 %
  de commits étiquetés risqués. Confirme que 90 j est un choix arbitraire qui pèse lourd sur tout
  le reste — **c'est le maillon faible identifié comme priorité n°1** (Chantier 1).

## 7. Pièges connus et limites assumées

1. **Circularité churn/label (r = 0.91).** Toute feature au niveau fichier doit être testée contre
   cette corrélation avant d'être ajoutée sans réserve.
2. **Horaires en UTC**, pas en heure locale de l'auteur (`%at`). Invalide l'hypothèse `is_night`
   telle que testée ; `%ai` exposerait le fuseau si on veut la retester correctement.
3. **Biais d'importance par impureté** (`feature_importances_` surévalue `msg_len`, `hour`) —
   toujours contre-vérifier avec `permutation_importance`.
4. **Étiquetage par mots-clés lexical** (`FIX_RE`) : capture « fix typo » (pas un bug), rate les
   fix bien rédigés sans le mot « fix ». Cause probable de bruit dans le label, pas encore quantifiée.
5. **Granularité fichier, pas ligne** pour le label SZZ — cause structurelle de la circularité du
   piège n°1. Un vrai SZZ (git blame) la réduirait mais demande un chantier dédié (Chantier 1).
6. **Biais de fin d'historique** : les commits des ~90 derniers jours ne peuvent presque pas être
   étiquetés risqués (pas assez de futur pour observer un fix). Non corrigé.
7. **Écart de reproductibilité radon** (voir §5) : les chiffres de la narration Partie 4.3/4.4 du
   notebook ne sont pas reproduits à l'identique dans cet environnement. **Cause non expliquée** —
   vérifié et éliminé le 2026-09-09 (timebox 15 min, cf. journal session 2) : l'état du clone Flask
   n'est pas en cause (5557 commits, HEAD `d73fa1cdcbd8b1465c151db8924ba58b1dd14e35`, identique à la
   référence). **Candidats restants, non vérifiés :** versions scikit-learn/numpy au moment de la
   rédaction du narratif (inconnues, non documentées à l'époque) ; divergence possible entre le
   code actuellement dans les cellules 49-53 et celui qui a produit les chiffres cités dans le
   markdown (le TP a pu être édité après la mesure de référence) ; le cap arbitraire de 5 fichiers
   `.py` par commit dans la cellule radon (jugé peu probable — déterministe sur un état de dépôt
   identique, sauf si le code différait). **Décision : on n'investigue pas plus loin maintenant**
   (le Chantier 1 va changer le label et invalider ces chiffres de toute façon). Le vrai correctif
   structurel est `results/benchmark.py` (§3/§8) : plus aucun chiffre en dur dans le markdown, tout
   résultat daté et rattaché à un état de données identifiable (hash HEAD Flask, versions des libs).
8. **Un seul dépôt (Flask)** étudié à ce stade — aucune validation externe (Chantier 6).
9. **Cellules markdown encore à chiffres en dur, en dehors du tableau bilan** (traité le
   2026-09-09) : cellule 27 (accuracy/précision/rappel/AUC Partie 3), cellule 31 (importances par
   permutation du modèle BASE), cellule 41 (répartition horaire, AUC `is_night` seule, gain en
   permutation), cellule 44/45 (tableau intermédiaire Piste B — `churn seul`, avant l'entrée des
   features de code). **Décision assumée** : pas de correction maintenant — mêmes arguments qu'au
   point 7 (le Chantier 1 va changer le label et invalider ces chiffres). Zéro exception a été
   faite : cette liste doit être reprise avant la restitution finale (Chantier 7), pas oubliée.

## 8. Prochaines étapes priorisées

Ordre de priorité du prompt de démarrage (à traiter un par un, avec validation avant chaque
implémentation) :

1. **Chantier 1 — assainir l'étiquetage** (priorité absolue, cf. §6/§7 point 5). Pistes : détection
   de fix plus stricte (exiger `#\d+`, exclure `typo|docs|changelog`), granularité ligne via
   `git blame` (vrai SZZ).

   **Protocole de mesure pré-enregistré (2026-09-09, avant toute implémentation)** pour juger si
   le passage à la granularité ligne réduit effectivement la circularité churn/label :
   - Comparer `label_v1` (fichier, actuel) à `label_v2` (ligne, `git blame`) sur le même état de
     données (même HEAD Flask, même fenêtre 90 j, pour isoler l'effet de la seule granularité).
   - **M1** — reproduire `churn_fix_correlation` sous `label_v2` (n_fix redéfini comme « nb de fois
     où une ligne du fichier est blâmée par un fix », pas « nb de fix qui ont un jour touché le
     fichier »). Seuil pré-enregistré : circularité réduite seulement si r_v2 < 0,5 (r_v1 = 0,91).
   - **M2** (décisif) — `cv_auc(CHURN seul)` sous `label_v2`, comparé à 0,788 (`label_v1`, cellule
     44 du run initial — à recalculer avec `fyc.evaluate.cv_auc` pour un chiffre certifié). Lift
     actuel au-dessus du hasard : 0,288. Seuil pré-enregistré : circularité réduite si le lift sous
     `label_v2` est inférieur à sa moitié, soit AUC(churn seul, v2) < 0,644.
   - **M3** (diagnostic, pas une preuve) — taux de bascule des labels `1→0` entre `label_v1` et
     `label_v2`, en particulier sur les commits touchant des fichiers à fort churn.
   - **Garde-fou** : si `part_risque` sous `label_v2` tombe sous ~5 %, les comparaisons AUC
     deviennent bruitées — le dire explicitement plutôt que conclure sur un chiffre instable.
   - Toutes ces mesures doivent être ajoutées à `results/benchmark.py` (paramétrer par
     `label_version`), pas calculées à la main.
2. **Chantier 2 — industrialiser l'analyse de code** au-delà de radon (Ruff, Bandit, Lizard,
   Semgrep). Une démo Ruff existe déjà (cellule 56, 5 fichiers) mais n'est pas intégrée au pipeline
   de features.
3. **Chantier 3 — analyse du diff** (hunks, AST, ratio code/test en lignes).
4. **Chantier 4 — signaux de processus** via l'API GitHub (reviewers, délai merge, échecs CI).
5. **Chantier 5 — croisement et combinaison** (stacking, matrice de désaccord entre modèles).
6. **Chantier 6 — validation externe** sur un deuxième dépôt.
7. **Chantier 7 — restitution** (modules `.py`, `README.md`, synthèse soutenance).

## 9. Journal des sessions

### 2026-09-09 — Session 1 (Étape 0)
- Lu intégralement le notebook (58 cellules), pas de README préexistant.
- Installé `radon` et `ruff` dans le venv (absents initialement, requis Partie 4).
- Exécuté le notebook de bout en bout sans erreur (`jupyter nbconvert --execute --inplace`).
- Comparé les sorties réelles aux chiffres annoncés dans les cellules markdown : concordance
  quasi-parfaite pour Parties 1–3 et pistes A/B de la Partie 4 ; **écart notable sur la piste C**
  (métriques radon) — voir §5 pour le détail et l'hypothèse retenue (version d'outil non figée
  au moment de la rédaction du narratif).
- Créé `requirements.txt` (freeze du venv, cœur du pipeline) et ce `CLAUDE.md`.
- **Reste à faire** : validation du résumé de compréhension par l'utilisateur, puis démarrage du
  Chantier 1 après accord explicite.

### 2026-09-09 — Session 2 (avant Chantier 1)
- Corrections reçues sur la session 1 : chiffres radon incomplets dans le résumé (corrigé),
  hypothèse "version radon" présentée comme cause probable sans vérification (fautif — la version
  radon 6.0.1 est confirmée comme celle ayant produit les chiffres de référence, donc **pas** la
  cause), et nécessité de vérifier l'état du clone Flask avant toute autre piste.
- Vérifié (15 min) : clone Flask conforme à la référence (5557 commits, HEAD `d73fa1cd...`).
  N'explique pas l'écart. Cause de l'écart radon toujours **non expliquée** — candidats restants
  listés en §7 point 7, décision de ne pas creuser davantage (le Chantier 1 va invalider ces
  chiffres de toute façon).
- Créé `fyc/` (git_extract, labeling, features, evaluate) et `results/benchmark.py` +
  `results/results.json`. Validation croisée : les chiffres recalculés via les modules
  reproduisent ceux du notebook au millième près (bonne confiance dans le refactor).
- Modifié les cellules 52 et 54 du notebook pour charger le tableau bilan depuis
  `results/results.json` au lieu de valeurs figées. Bug introduit puis corrigé en cours de route :
  la suppression de la définition de `CODE` dans l'ancienne cellule 52 cassait la cellule 53
  (`NameError`) — réintroduite en tête de la nouvelle cellule 52.
- Ré-exécuté le notebook de bout en bout deux fois (une avec le bug, une propre) : 0 erreur sur
  58 cellules dans la version finale.
- Rédigé le protocole de mesure pré-enregistré (M1/M2/M3 + garde-fou) pour juger si le futur SZZ
  ligne réduit la circularité — voir §8, Chantier 1. Écrit **avant** toute implémentation, comme
  demandé.
- **Décision explicite prise par l'utilisateur** : pas de validation externe (Chantier 6) avant
  Chantier 1 — valider un pipeline dont le label est mauvais n'apprend rien.
- **Reste à faire** : implémenter le Chantier 1 (assainir l'étiquetage) en suivant le protocole
  M1/M2/M3 ci-dessus.
