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
- venv Python **3.14.4** dans `.venv/` (déjà présent au démarrage de ce projet). **Version figée
  volontairement, pas un détail** : les features issues de radon dépendent du taux d'échec de
  parsing sur du code d'époque Python 2 dans l'historique Flask, qui varie avec la version de
  l'interpréteur (voir §7 points 7-8). `results/results.json` enregistre automatiquement la
  version de Python utilisée à chaque run (`versions.python`).
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

- **Étiquetage SZZ simplifié, granularité fichier, fenêtre 90 jours — SUPERSÉDÉ (2026-09-09).**
  *Pourquoi à l'origine :* fenêtre courte pour éviter que le label dégénère en « popularité du
  fichier » (voir §6, analyse de sensibilité) ; 90 j retenu comme compromis donnant ~51 % de
  positifs (classe équilibrée). *Limite assumée :* granularité fichier plutôt que ligne, cause
  directe de la circularité churn/label mesurée en §5. **Remplacé par `label_v2` (SZZ ligne via
  `git blame -w -M -C`) comme label par défaut** — voir décision ci-dessous et §6. `label_v1` reste
  dans le code (`fyc.labeling.label_commits`) pour comparaison, mais n'est plus le défaut.
- **`label_v2` (SZZ ligne, `git blame -w -M -C`) adopté comme label par défaut.** *Pourquoi :*
  protocole de mesure pré-enregistré (§8, Chantier 1) — 3 mesures sur 4 (dont la mesure décisive
  contrôlée pour la difficulté de la tâche) montrent une réduction nette de la circularité
  prédictive, avec marge. *Réserve assumée, pas résolue :* la corrélation agrégée par fichier
  (M1) ne baisse pas sous le seuil visé — propriété différente de la circularité qui comptait pour
  le modèle, mais à garder en tête en soutenance (§6).
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
- **⚠️ MIXTE — SZZ ligne (`label_v2`, `git blame -w -M -C`) réduit-il la circularité churn/label ?**
  Mesuré le 2026-09-09 via `results/chantier1_protocol.py` (protocole M1/M2/M2bis/M3 pré-enregistré,
  seuils fixés avant implémentation — voir §8 Chantier 1). Résultat brut dans
  `results/chantier1_diagnostic.json`. **Les 4 mesures ne s'accordent pas — à dire tel quel, pas
  à trancher artificiellement :**
  - Garde-fou : `part_risque(v2)` = 12,7 % (> seuil 5 %) — mesures interprétables, pas d'abandon.
  - **M1 (corrélation fichier) — ÉCHEC** du seuil pré-enregistré (r_v2 < 0,5) : r_v2 = **0,810**
    Pearson (baisse depuis 0,910, mais insuffisante), Spearman monte même à 0,795 (depuis 0,727).
  - **M2 (AUC absolue churn seul) — ÉCHEC** du seuil (< 0,644) : AUC(churn seul, v2) = **0,679**
    (depuis 0,788) — proche du seuil sans le franchir.
  - **M2bis (ratio churn/BASE, la mesure décisive contrôlée) — RÉUSSITE** claire : ratio
    1,005 → **0,846** (seuil ≤ 0,904), et le garde-fou associé passe aussi (AUC(BASE) **monte**
    légèrement sous v2, 0,784 → 0,802 — la tâche n'est pas devenue plus dure globalement, donc la
    chute relative de churn n'est pas un artefact de difficulté).
  - **M3 (diagnostic) — corrobore fortement** : parmi les commits étiquetés `1` sous `label_v1`,
    **75,6 %** basculent à `0` sous `label_v2` (82,2 % sur le dernier quartile de churn) — beaucoup
    de l'ancien positif était bien de la contamination par popularité de fichier.
  - **Corroboration indépendante, tableau complet des features sous `label_v2`**
    (`results/results_v2.json`) : `BASE` seul = 0,802 ; `BASE+churn` = 0,807 (**+0,005**, contre
    +0,070 sous v1) ; `code seul` = **0,599** (depuis 0,746) ; `BASE+code` = 0,809 (+0,007, contre
    +0,056 sous v1) ; `TOUT` = 0,807 (ne bat même pas `BASE+code` seul). Sous label_v2, churn et
    code s'effondrent en absolu et n'ajoutent presque rien à BASE — signal net et cohérent avec
    M2bis/M3, indépendant du désaccord M1/M2.
  - **Lecture honnête** : la circularité baisse nettement **au niveau prédictif** (ce qui compte
    pour le modèle — M2bis, M3, tableau complet), mais **pas au niveau de la statistique agrégée
    par fichier** (M1) — plausible car M1 mesure une propriété différente (densité totale
    d'événements de blame par fichier, qui reste liée à la taille/centralité du fichier) plutôt que
    la contamination commit-par-commit que le modèle exploitait sous v1. Ne pas annoncer "la
    circularité a disparu" sans cette nuance — les deux affirmations sont vraies et se complètent :
    circularité prédictive fortement réduite, circularité structurelle du fichier persistante.
  - **Décision arbitrée (2026-09-09, par l'utilisateur) : `label_v2` ADOPTÉ comme label par défaut**
    des chantiers suivants. Raisonnement : 3 mesures sur 4 (M2bis — la mesure décisive, contrôlée
    pour la difficulté de la tâche — M3, et la corroboration indépendante du tableau complet)
    convergent avec une marge confortable, pas de justesse. Le seul signal dissonant (M1) mesure
    une propriété différente de ce qui compte pour le modèle (densité de blame agrégée par fichier,
    mécaniquement liée à la taille/activité du fichier, plutôt que la contamination commit-par-
    commit que le modèle exploitait sous v1) — creuser davantage ce point précis aurait un
    rendement décroissant face aux chantiers restants. **La réserve M1 reste documentée ci-dessus,
    pas effacée** : la circularité baisse au niveau prédictif, pas au niveau de la statistique de
    fichier agrégée, et ces deux affirmations coexistent en soutenance.
- **✅ Validée, partiellement — Une détection de fix plus stricte réduit encore le bruit du label.**
  Mesuré le 2026-09-09 (`results/chantier1_fix_detection.py`), 4 candidats sous `label_v2` adopté,
  même fenêtre 90j :

  | Candidat | n fix | part_risque | AUC(BASE) | AUC(tout) | corr. churn/fix |
  |---|---|---|---|---|---|
  | A. `FIX_RE` seul (ancien défaut) | 1069 | 12,7 % | 0,802 | 0,807 | 0,810 |
  | **B. + exclusion `typo\|docs\|changelog` (ADOPTÉ)** | **732** | **9,7 %** | **0,787** | **0,805** | **0,789** |
  | C. + référence d'issue `#\d+` exigée | 320 | 3,9 % | 0,824 | 0,851 | 0,751 |
  | D. exclusion + référence d'issue | 237 | 3,1 % | 0,822 | 0,861 | 0,705 |

  **Décision : candidat B adopté** comme détection de fix par défaut (`results/benchmark.py`,
  `fyc.labeling.make_fix_predicate(exclude_re=EXCLUDE_RE)`). **C et D rejetés malgré une AUC brute
  plus haute** — `part_risque` tombe sous le plancher de 5 % déjà établi comme garde-fou
  d'interprétabilité pour le protocole de granularité (§8) : même règle appliquée ici par
  cohérence, pas une exception. Avec ~3-4 % de positifs, les écarts-types d'AUC explosent
  (jusqu'à ± 0,042) et le gain apparent n'est pas distinguable du bruit d'échantillonnage sur une
  classe aussi réduite. **Ce n'est pas une preuve que C/D sont de mauvaises pistes** — seulement
  qu'elles ne sont pas validables avec cette méthode de validation croisée à cet effectif ; à
  reconsidérer si le projet élargit son jeu de données (Chantier 6) ou change de métrique
  d'évaluation (precision/recall à seuil fixe, moins sensible au déséquilibre que l'AUC).

## 7. Pièges connus et limites assumées

1. **Circularité churn/label (r = 0.91).** Toute feature au niveau fichier doit être testée contre
   cette corrélation avant d'être ajoutée sans réserve.
2. **Horaires en UTC**, pas en heure locale de l'auteur (`%at`). Invalide l'hypothèse `is_night`
   telle que testée ; `%ai` exposerait le fuseau si on veut la retester correctement.
3. **Biais d'importance par impureté** (`feature_importances_` surévalue `msg_len`, `hour`) —
   toujours contre-vérifier avec `permutation_importance`.
4. **Étiquetage par mots-clés lexical** (`FIX_RE`) : capture « fix typo » (pas un bug), rate les
   fix bien rédigés sans le mot « fix ». **Quantifié et partiellement corrigé le 2026-09-09** —
   voir §6 (hypothèse sur la détection de fix stricte) et §8 Chantier 1 : exclusion
   `typo|docs|changelog` adoptée (candidat B), exigence de référence d'issue mesurée mais rejetée
   (classe positive trop petite). La seconde moitié du piège (fix bien rédigés sans mot-clé, donc
   des faux négatifs) reste non quantifiée et non corrigée — limite assumée.
5. **Granularité fichier, pas ligne** pour le label SZZ — cause structurelle de la circularité du
   piège n°1. Un vrai SZZ (git blame) la réduirait mais demande un chantier dédié (Chantier 1).
6. **Biais de fin d'historique** : les commits des ~90 derniers jours ne peuvent presque pas être
   étiquetés risqués (pas assez de futur pour observer un fix). Non corrigé.
7. **Écart de reproductibilité radon — RÉSOLU, cause identifiée (2026-09-09). C'est un résultat
   du projet, pas un incident.** Les features issues de radon dépendent de la version de
   l'interpréteur Python, via le taux d'échec de parsing sur du code d'époque Python 2 présent
   dans l'historique Flask. **Mesure comparative** (même clone Flask, même code, deux environnements) :
   `has_py == 0` sur 49,8 % des commits étiquetés sous Python 3.14.4, contre **54,5 % sous Python
   3.12.3** — soit 137 commits de couverture code en plus sous 3.14.4, largement suffisant pour
   expliquer l'écart mesuré sur `BASE+code` (+0.016) et `code seul` (+0.046).
   **Conséquence : les chiffres mesurés dans cet environnement (Python 3.14.4) sont mieux fondés
   que le narratif d'origine (moindre couverture) et servent désormais de référence.** Python
   3.14.4 est figé dans `requirements.txt` pour cette raison précise. `results/results.json`
   enregistre automatiquement la version de Python utilisée (`versions.python`) pour que ce genre
   d'écart soit traçable pour toute reproduction future — **c'est un vrai point de reproductibilité
   à présenter en soutenance**, pas une anecdote de debug.

   **Leçon méthodologique sur l'instrumentation** (piège à ne pas refaire) : ne jamais comparer un
   taux d'échec brut instrumenté derrière un cache entre deux mesures — mon compteur et celui de
   l'encadrant n'avaient pas le même dénominateur (le mien comptait before+after, le sien after
   seul ; le cache de `_metriques_radon` rend le nombre d'appels bruts ininterprétable d'une
   instrumentation à l'autre). La bonne mesure, insensible au cache, est **`has_py == 0`** (même
   dénominateur des deux côtés : nombre de commits étiquetés).

   **Ancien texte (session 2, pour mémoire — ne pas effacer l'historique) :** l'écart avait été
   documenté comme "non expliqué, candidats non vérifiés (version Python, divergence de code, cap
   5 fichiers)" après avoir éliminé l'aléa, sklearn/numpy et l'état du clone. C'était la bonne
   décision d'investigation (l'écart était bien localisé sur radon), mais la clôture était
   prématurée : la localisation de l'écart sur une seule famille de features aurait dû être lue
   comme un signal exploitable, pas comme une raison d'arrêter.

8. **Défaut de conception révélé pendant le diagnostic — CORRIGÉ (2026-09-09).** Les échecs de
   parsing radon recevaient des valeurs par défaut qui ressemblaient à du code sain (`mi_avg=100.0`
   = « parfaitement maintenable », complexités à 0) : un échec d'analyse était encodé comme le
   meilleur code possible du dataset. `has_py` conflatait en plus deux situations différentes
   (« pas de `.py` touché », 1358 commits, 46,8 % — normal) et (« `.py` touché mais illisible par
   radon », 88 commits, 3,0 % — le vrai problème).
   **Correctif appliqué** dans `fyc/features.py::build_code_features` : les 6 métriques numériques
   (`cc_tot_avg`, `cc_max_avg`, `mi_avg`, `loc_avg`, `comment_ratio`, `delta_cc`) sont `NaN` quand
   aucun fichier du commit n'a pu être mesuré, plus deux features booléennes explicites
   `touches_py` et `code_parse_ok` remplaçant l'ancien `has_py` ambigu. `RandomForestClassifier`
   (scikit-learn 1.9.0) gère nativement les NaN — vérifié directement avant d'appliquer le
   correctif à l'échelle du pipeline.
   **Effet mesuré, isolé** (avant → après correctif, même run sinon, CV 5 plis) :
   - `BASE + code` : 0.836 ± 0.013 → **0.840 ± 0.016** (CV), 0.824 → 0.818 (temporel)
   - `code seul` : 0.737 ± 0.007 → **0.746 ± 0.010** (CV), 0.774 → 0.778 (temporel)
   - `TOUT` : 0.874 ± 0.014 → 0.873 ± 0.014 (CV), 0.803 → 0.796 (temporel)
   **Lecture** : gain net petit mais positif sur les deux jeux contenant les features de code en
   CV (+0.004 et +0.009), signe que les 88 commits en échec silencieux portaient un vrai signal
   dilué par le remplissage trompeur — pas juste du bruit. Effet mixte sur le split temporel
   (léger recul), à ne pas sur-interpréter vu la faible ampleur. **Ces nouveaux chiffres (0.840 /
   0.746) sont désormais la référence courante, dans `results/results.json`.**

9. **Un seul dépôt (Flask)** étudié à ce stade — aucune validation externe (Chantier 6).
10. **Cellules markdown encore à chiffres en dur, en dehors du tableau bilan** (traité le
   2026-09-09) : cellule 27 (accuracy/précision/rappel/AUC Partie 3), cellule 31 (importances par
   permutation du modèle BASE), cellule 41 (répartition horaire, AUC `is_night` seule, gain en
   permutation), cellule 44/45 (tableau intermédiaire Piste B — `churn seul`, avant l'entrée des
   features de code). **Décision assumée** : pas de correction maintenant — mêmes arguments qu'aux
   points 7-8 (le Chantier 1 va changer le label et invalider ces chiffres). Zéro exception a été
   faite : cette liste doit être reprise avant la restitution finale (Chantier 7), pas oubliée.
11. **Divergence notebook/`fyc` introduite le 2026-09-09, assumée.** Les cellules 49-53 (Partie 4.3,
   code inline du TP original) utilisent encore l'ancien remplissage par défaut sur échec de
   parsing radon (`mi_avg=100.0`, `has_py`) et le `FIX_RE` brut — elles n'ont **pas** été mises à
   jour avec le correctif NaN (§7 point 8) ni la détection de fix assainie (§6, candidat B). Seul
   le tableau dynamique de la cellule 52 (chargé depuis `results/benchmark.py` via `fyc/`) reflète
   l'état courant du pipeline. Les cellules 49-53 restent la démonstration pédagogique **originale**
   du TP, explicitement signalée comme telle dans une note ajoutée à la cellule 52. **Risque** : un
   lecteur qui ne lit pas cette note peut croire que cellule 53 (« Modèle final ») produit les
   mêmes chiffres que `results/results_v2.json` — ce n'est plus le cas. À corriger au Chantier 7
   (porter les correctifs dans les cellules inline, ou les faire appeler `fyc` directement).

## 8. Prochaines étapes priorisées

Ordre de priorité du prompt de démarrage (à traiter un par un, avec validation avant chaque
implémentation) :

1. **Chantier 1 — assainir l'étiquetage** (priorité absolue, cf. §6/§7 point 5). Pistes : détection
   de fix plus stricte (exiger `#\d+`, exclure `typo|docs|changelog`), granularité ligne via
   `git blame` (vrai SZZ).

   **Protocole de mesure pré-enregistré — v2, amendé le 2026-09-09 suite revue (3 corrections
   bloquantes intégrées), avant toute implémentation** pour juger si le passage à la granularité
   ligne réduit effectivement la circularité churn/label :
   - Comparer `label_v1` (fichier, actuel) à `label_v2` (ligne, `git blame`) sur le même état de
     données (même HEAD Flask, même fenêtre 90 j, pour isoler l'effet de la seule granularité).
   - **Chiffres de référence certifiés AVANT implémentation de `label_v2`** (recalculés le
     2026-09-09 via `fyc.evaluate.cv_auc`, post-correctif NaN §7 point 8, dans `results.json`) :
     AUC(BASE, v1) = 0,7843 ± 0,0205 ; AUC(churn seul, v1) = 0,7881 ± 0,0102 ; corrélation
     churn/fix (v1) = 0,910 (Pearson) / 0,727 (Spearman).
   - **M1** — reproduire `churn_fix_correlation` sous `label_v2` (n_fix redéfini comme « nb de fois
     où une ligne du fichier est blâmée par un fix », pas « nb de fix qui ont un jour touché le
     fichier »). Seuil pré-enregistré : circularité réduite seulement si r_v2 < 0,5 (r_v1 = 0,910).
   - **M2** (décisif, fonctionnel) — `cv_auc(CHURN seul)` sous `label_v2`, comparé à 0,7881
     (`label_v1`, chiffre certifié ci-dessus). Seuil absolu pré-enregistré : AUC(churn seul, v2) <
     0,644 (lift sous label_v2 inférieur à la moitié du lift actuel au-dessus du hasard, 0,2881).
   - **M2bis — condition de contrôle, ajoutée suite revue (bloquant sans elle).** L'AUC absolue de
     churn seul ne suffit pas : si `label_v2` rend la tâche plus dure pour *tout le monde*, une
     baisse de l'AUC de churn ne prouve rien sur la circularité. Mesurer donc **aussi**
     `cv_auc(BASE seul)` sous `label_v2`, et suivre le **ratio** r = AUC(churn seul) / AUC(BASE
     seul), pas l'AUC absolue de churn isolée.
     - r_v1 = 0,7881 / 0,7843 = **1,0048** (certifié — churn seul égale à peu près tout le pouvoir
       prédictif des 10 features BASE combinées : c'est le symptôme de circularité déjà mesuré).
     - **Seuil pré-enregistré** : circularité considérée comme réduite seulement si **r_v2 ≤ 0,90
       × r_v1 = 0,9044** ET **AUC(BASE, v2) ne chute pas de plus de 0,03 par rapport à AUC(BASE,
       v1) = 0,7843** (si BASE s'effondre autant que churn, on ne peut rien conclure — c'est un
       signe que `label_v2` rend juste la tâche plus dure, pas que la circularité a baissé).
   - **M3** (diagnostic, pas une preuve) — taux de bascule des labels `1→0` entre `label_v1` et
     `label_v2`, en particulier sur les commits touchant des fichiers à fort churn.
   - **Garde-fou et RÈGLE D'ABANDON pré-enregistrée, ajoutée suite revue (bloquant sans elle)** :
     si `part_risque` sous `label_v2` tombe sous ~5 %, les comparaisons AUC deviennent bruitées.
     Décision fixée maintenant, pas après avoir vu le chiffre : (a) ne pas conclure sur M1/M2/M2bis
     — chiffres trop instables ; (b) documenter comme résultat négatif en §6, daté ; (c) tester en
     **expérience séparée**, non mélangée à la comparaison principale, un élargissement de fenêtre
     (ex. 180 j) *uniquement* pour vérifier si le problème vient de la fenêtre ou de la granularité
     — mais ne jamais adopter ce nouveau réglage comme label par défaut sans repasser par
     M1/M2/M2bis sous cette fenêtre ; (d) `label_v1` reste le label par défaut des chantiers
     suivants tant que `label_v2` n'a pas franchi ce plancher.
   - **Hygiène `git blame`, ajoutée suite revue (bloquant sans elle)** : utiliser au minimum
     `git blame -w` (ignorer les changements d'espaces) et `-M -C` (détecter déplacements et
     copies). Sans ces options, `git blame` attribue une ligne à son dernier modificateur — un
     commit de reformatage, renommage ou déplacement de code volerait la responsabilité au vrai
     auteur du défaut, remplaçant la circularité fichier/label par un biais d'attribution tout
     aussi trompeur. Choix documenté ici pour être challengé en soutenance.
   - Toutes ces mesures (M1, M2, M2bis, M3, garde-fou) doivent être ajoutées à
     `results/benchmark.py` (paramétrer par `label_version`), pas calculées à la main.

   **RÉSULTAT (2026-09-09)** — implémenté (`fyc/labeling.py::label_commits_v2`, SZZ ligne via
   `git blame -w -M -C`, même détection de fix et même fenêtre que v1 pour isoler la granularité) et
   mesuré via `results/chantier1_protocol.py` (reproductible, seuils pré-enregistrés, sortie dans
   `results/chantier1_diagnostic.json`). **Verdict détaillé en §6** (mixte, pas de clôture
   artificielle) : garde-fou passé (part_risque=12,7 %), M2bis (mesure décisive) et M3 passent
   nettement, M1 et M2 (absolus) échouent de peu. Tableau complet des features sous label_v2 dans
   `results/results_v2.json` — corrobore M2bis indépendamment (churn et code s'effondrent en
   absolu, BASE tient). **Détection de fix plus stricte (`#\d+`, exclusion `typo|docs|changelog`)
   PAS encore traitée** — seule la granularité a été isolée et testée, comme le protocole
   l'exigeait. **Arbitrage rendu (2026-09-09) : `label_v2` ADOPTÉ comme label par défaut** (voir §4
   et §6 pour le raisonnement et la réserve M1 conservée). **Détection de fix stricte traitée
   (2026-09-09)** — 4 candidats mesurés (§6), candidat B (exclusion `typo|docs|changelog`) adopté
   dans `results/benchmark.py` ; candidats avec référence d'issue exigée mesurés mais rejetés
   (classe positive sous le plancher de 5 %, §6). **Chantier 1 CLOS** sur son périmètre défini
   (granularité + détection de fix), avec deux réserves documentées et non résolues : M1 (§6) et la
   divergence notebook/`fyc` introduite en cours de route (§7 point 11) — les deux renvoient à un
   chantier ultérieur (respectivement : plus d'investigation si utile, Chantier 7 pour la
   divergence).
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

### 2026-09-09 — Session 2 (suite) — correction : clôture prématurée
- Erreur signalée : j'avais classé l'écart radon comme "non expliqué, on passe" alors que mes
  propres mesures montraient un signal exploitable — l'écart est **localisé** sur les jeux de
  features contenant radon (BASE, is_night, churn, corrélation 0.91 coïncident tous à ±0.002 près
  avec la référence). Ça élimine les causes générales (aléa, sklearn/numpy, clone) et pointe vers
  quelque chose de spécifique au calcul radon. Je n'avais pas tiré cette conclusion. Corrigé.
- Confirmé par l'encadrant : la référence (0.820/0.691) a été produite avec radon 6.0.1 — la
  version n'est donc **pas** la cause (l'hypothèse de session 1 était non seulement non testée,
  mais fausse une fois testée).
- Diagnostic instrumenté (hypothèse version Python, 3.14.4 ici vs 3.12.3 référence) : voir §7
  point 7 pour les 4 chiffres et le verdict (consistant mais pas prouvé).
- Défaut de conception documenté (échecs de parsing radon masqués par des valeurs par défaut) —
  voir §7 point 7. Correctif proposé, pas implémenté, en attente d'accord.
- Commit `c7dd597` : refactor complet (fyc/, results/benchmark.py, CLAUDE.md, requirements.txt) —
  point de retour explicite avant le Chantier 1, comme demandé.
- Correction de process reçue : ne pas écrire "question ouverte : aucune" quand un écart reste
  non expliqué — un écart non expliqué EST une question ouverte, même documenté.
- **Reste à faire** : décider si on installe Python 3.12.3 pour trancher l'hypothèse (pas fait,
  pas dans le timebox accordé) ; décider du correctif `touches_py`/`code_parse_ok` ; puis
  Chantier 1 après validation du protocole M1/M2/M3 (recollé intégralement en réponse, pas encore
  validé par l'utilisateur).

### 2026-09-09 — Session 2 (suite) — diagnostic tranché, écart clos, protocole amendé
- L'encadrant a mesuré côté référence (Python 3.12.3, même clone, même code) : `has_py == 0` sur
  1583/2902 (54,5 %), contre 1446/2902 (49,8 %) ici — 137 commits de couverture code en plus sous
  3.14.4. **Hypothèse Python confirmée, suffisante pour expliquer l'écart.** Version de radon déjà
  écartée précédemment (identique des deux côtés). §7 fermé (points 7-8), plus "non résolu".
- **Correction méthodologique reçue et retenue** : ne pas comparer des taux d'échec bruts
  instrumentés derrière un cache entre deux environnements (dénominateurs différents, non
  comparables) — comparer `has_py == 0` à la place (même dénominateur, insensible au cache).
- **Conséquence actée** : mes chiffres (Python 3.14.4, meilleure couverture) deviennent la
  référence du projet, pas ceux du narratif d'origine. Cellule 54 du notebook corrigée en
  conséquence (ne dit plus "candidats non tranchés").
- Python 3.14.4 explicitement figé et justifié dans `requirements.txt` et §2 (pas un détail
  d'environnement — un point de reproductibilité qui affecte les résultats).
- **Correctif `touches_py`/`code_parse_ok` implémenté** (validé, arbitrage reçu) : `fyc/features.py`
  remplace les défauts trompeurs (`mi_avg=100.0`, complexités à 0 sur échec de parsing) par `NaN`
  sur les 6 métriques numériques, plus deux booléens explicites remplaçant l'ancien `has_py`
  ambigu. `RandomForestClassifier` (sklearn 1.9.0) vérifié pour gérer nativement les NaN avant
  d'appliquer à l'échelle. Effet isolé mesuré : `BASE+code` 0.836→0.840 (CV), `code seul`
  0.737→0.746 (CV) — petit gain positif net, signe que les 88 commits en échec silencieux
  portaient un vrai signal. Détail en §7 point 8.
- **Protocole M1/M2/M3 amendé** avec les 3 corrections bloquantes reçues : (1) condition de
  contrôle M2bis — ratio churn/BASE plutôt que l'AUC absolue de churn seul, pour ne pas confondre
  "tâche plus dure" et "circularité réduite" ; chiffres de référence certifiés maintenant
  (r_v1 = 1,0048, seuil r_v2 ≤ 0,9044) ; (2) hygiène `git blame -w -M -C` documentée pour éviter de
  remplacer la circularité fichier/label par un biais d'attribution (reformatage/renommage/déplacement) ;
  (3) règle d'abandon pré-enregistrée si `part_risque(v2) < 5%` (ne pas conclure, documenter comme
  résultat négatif, tester l'élargissement de fenêtre en expérience séparée, garder `label_v1` par
  défaut). Recopié intégralement dans la réponse à l'utilisateur, comme demandé.
- **Feu vert reçu pour démarrer le Chantier 1** sans validation supplémentaire, une fois ces trois
  points corrigés et le protocole recollé.
- **Reste à faire** : implémenter le Chantier 1 (SZZ ligne via `git blame -w -M -C`, détection de
  fix plus stricte) en suivant le protocole M1/M2/M2bis/M3 amendé.

### 2026-09-09 — Session 2 (fin) — Chantier 1 : SZZ ligne implémenté et mesuré
- `fyc/labeling.py::label_commits_v2` (+ `blame_line_fixes`, `make_est_risque_v2`) : SZZ vrai,
  granularité ligne, via `git blame -w -M -C` sur le parent de chaque commit de fix. Testé sur un
  sous-échantillon (20 fix) avant l'échelle complète, un cas vérifié à la main (ligne exacte, hash
  de commit exact). Coût mesuré : 15,3 s pour les 1069 commits de fix (14,3 ms/fix) — largement
  praticable, pas besoin d'échantillonnage.
- `fyc/git_extract.py` : ajout `diff_raw` et `blame_porcelain` (plomberie git, cohérent avec le
  rôle du module).
- `results/benchmark.py` : paramétré par `--label {v1,v2}`, écrit `results.json` ou
  `results_v2.json`. Non-régression vérifiée sur v1 (chiffres identiques au run précédent).
- `results/chantier1_protocol.py` (nouveau) : calcule M1/M2/M2bis/M3 + garde-fou avec les seuils
  pré-enregistrés en dur dans le script (pas dans le markdown), écrit
  `results/chantier1_diagnostic.json`.
- **Résultat mesuré, verdict mixte, détaillé en §6** : garde-fou passé (part_risque=12,7 %) ; M2bis
  (mesure de contrôle, décisive) et M3 passent nettement ; M1 (corrélation fichier) et M2 (AUC
  absolue) échouent de peu. Tableau complet des features sous label_v2 corrobore M2bis
  indépendamment (churn seul 0,788→0,679 et n'ajoute que +0,005 à BASE, contre +0,070 sous v1 ;
  code seul 0,746→0,599). Je n'ai pas arbitré ce résultat mixte moi-même — présenté tel quel, en
  attente de lecture par l'utilisateur.
- Notebook ré-exécuté (0 erreur, 58 cellules) après chaque changement de `fyc`/`benchmark.py` pour
  vérifier l'absence de régression.
- **Reste à faire** : arbitrage sur l'adoption de `label_v2` (§6) ; volet "détection de fix plus
  stricte" du Chantier 1 (pas encore traité, la granularité seule a été isolée pour le protocole) ;
  commit de cet état.

### 2026-09-09 — Session 3 — arbitrage, détection de fix stricte, clôture du Chantier 1
- Arbitrage tranché (par moi, sur consigne explicite de l'utilisateur : « fait ce qui est le plus
  pertinent entre les deux ») : **`label_v2` adopté** comme label par défaut. Raisonnement détaillé
  en §6 — 3 mesures sur 4 convergent avec marge confortable, le seul dissonant (M1) mesure une
  propriété différente de ce qui compte pour le modèle. Réserve M1 conservée, pas effacée.
- `fyc/labeling.py` : ajout de `make_fix_predicate` (exclusion + référence d'issue, composables),
  `tag_fixes` accepte maintenant un `fix_predicate` en plus de `fix_re` (rétrocompatible).
  Nettoyage au passage : paramètre `fix_re` mort retiré de `blame_line_fixes` (jamais utilisé,
  s'appuyait déjà sur `is_fix` pré-tagué).
- `results/chantier1_fix_detection.py` (nouveau) : 4 candidats mesurés sous label_v2 (déjà adopté),
  même méthode que le protocole précédent (comparer, pas supposer). Résultat détaillé en §6.
  **Décision : candidat B (exclusion typo/docs/changelog) adopté**, candidats avec référence
  d'issue exigée **rejetés** malgré une AUC brute plus haute — appliqué la même règle de garde-fou
  à 5 % de positifs que pour la granularité, par cohérence, pas par exception.
- `results/benchmark.py` : `FIX_PREDICATE` (candidat B) appliqué par défaut aux deux labels ;
  `--label` par défaut passé de `v1` à `v2` (le nouveau défaut du projet). `results.json` et
  `results_v2.json` régénérés avec la détection assainie.
- **Problème découvert en cours de route, documenté plutôt que caché** (§7 point 11) : les cellules
  49-53 du notebook (code inline du TP original, Partie 4.3) n'ont pas été mises à jour avec le
  correctif NaN ni la détection de fix assainie — elles divergent maintenant de `fyc`/`results.json`.
  Corrigé le label trompeur "BASE (Partie 3)" (cellule 52) qui prétendait une équivalence qui
  n'existe plus ; ajouté une note explicite de divergence dans la même cellule plutôt que de
  silencieusement laisser le lecteur croire que tout le notebook est à jour. Correction complète
  reportée au Chantier 7 (porter les correctifs dans les cellules inline).
- Notebook ré-exécuté après chaque changement, 0 erreur sur 58 cellules à chaque fois.
- **Chantier 1 CLOS** sur son périmètre défini (granularité + détection de fix). Deux réserves
  documentées et non résolues (M1, divergence notebook/fyc), explicitement pas effacées.
- **Reste à faire** : commit de cet état ; proposer un plan pour le Chantier 2 (industrialiser
  l'analyse de code) avant de l'implémenter, comme le veut la règle de collaboration du projet.
