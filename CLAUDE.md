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
- `.env` (créé le 2026-11-04, **non versionné**, ajouté à `.gitignore`) : contient un token
  GitHub personnel (`token=...`) fourni par l'utilisateur, nécessaire au Chantier 4 (API GitHub
  authentifiée, 5000 req/h contre 60 non authentifié). Lu par `fyc/github_signals.py`.

## 3. Architecture du code

- **`fyc/`** — modules réutilisables, extraits du notebook le 2026-09-09 (squelette minimal,
  seulement ce que le Chantier 1 va réellement utiliser) :
  - `git_extract.py` : `extraire_commits`, `get_head`, `get_parents`, `cat_file_batch`,
    `diff_raw`, `blame_porcelain` (ajoutés Chantier 1, pour le SZZ ligne).
  - `labeling.py` : SZZ fichier (v1, legacy) **et** SZZ ligne (v2, défaut — `label_commits_v2`,
    `blame_line_fixes`, `make_est_risque_v2`, `git blame -w -M -C`). `make_fix_predicate` /
    `tag_fixes(fix_predicate=...)` pour une détection de fix composable (exclusion de mots-clés,
    référence d'issue exigée).
  - `features.py` : `build_base_features`, `build_is_night`, `build_churn_features`,
    `build_code_features` (radon via `git cat-file`, NaN sur échec de parsing — §7 point 8),
    `churn_fix_correlation` (diagnostic de circularité, réutilisé pour M1 du Chantier 1).
  - `code_lint.py` (Chantier 2, 2026-09-09) : `lint_features` — Ruff + Bandit en un seul passage
    de matérialisation disque (batch par dossier temporaire), sur les mêmes versions « après »
    que `build_code_features`. `RUFF_COLS`, `BANDIT_COLS`.
  - `diff_features.py` (Chantier 3, 2026-09-09) : `build_diff_features` — dispersion du diff
    (`n_hunks`), changement de signature de fonction via diff d'AST avant/après (`sig_added`,
    `sig_removed`, `sig_modified`, `touches_signature`). `DIFF_COLS` = jeu **curaté** adopté ;
    `ALL_COLS` = tout ce qui a été mesuré, y compris `avg_hunk_size` et `test_line_ratio`,
    **rejetés** (§6) mais gardés dans le code pour la reproductibilité de la mesure négative.
  - `github_signals.py` (Chantier 4, 2026-11-04) : `fetch_commit_pr_signals` — API **GraphQL**
    GitHub (pas REST), commits batchés par alias dans une même requête (même principe que
    `git cat-file --batch` et le batching Ruff/Bandit, §4) : PR associée, reviewers distincts,
    dates de création/merge, `statusCheckRollup` (état CI agrégé, couvre Travis historique et
    GitHub Actions actuel en un seul champ). Cache disque resumable (`results/github_cache.json`,
    gitignore) — un sha absent du cache veut dire "pas encore mesuré" (budget épuisé avant d'y
    arriver), **différent** de `None` en cache (mesuré, confirmé sans PR) : même piège que
    `has_py`/Bandit (§7) anticipé cette fois plutôt que découvert après coup.
  - `process_features.py` (Chantier 4, 2026-09-09→2026-11-04) : `build_process_features` —
    `has_pr`, `n_reviewers`, `n_review_comments`, `merge_delay_hours`, `ci_failed`. Retourne aussi
    un masque de couverture pour sous-échantillonner les autres familles aux mêmes lignes si le
    cache GitHub est incomplet (voir `results/chantier4_github.py` et `results/benchmark.py`).
  - `evaluate.py` : `cv_auc`, `temporal_split_auc`, `baseline_majority_accuracy`.
  - **Chantier 4bis, vague 1 (2026-11-04)** : `pylint_features.py` (`PYLINT_COLS`),
    `vulture_features.py` (`VULTURE_COLS`, en process, un objet Vulture par commit),
    `mypy_features.py` (`MYPY_COLS`, un paquet par commit), `jit_kamei.py` (`KAMEI_COLS` : NS, ND,
    entropie, NDEV, REXP, SEXP), `coupling.py` (`COUPLING_COLS` : fan-in/fan-out par AST sur
    l'arbre complet de chaque commit). Scripts : `results/chantier4bis_common.py` (chargement +
    caches `*.pkl` non versionnés), `chantier4bis_wave1.py` → `chantier4bis_wave1.json`,
    `chantier4bis_combinations.py` → `chantier4bis_grid.json` (grille exhaustive 2048
    combinaisons, sélection imbriquée, réserve 20 %).

- **`results/benchmark.py`** — recalcule tous les jeux de features (BASE, churn, code radon,
  lint Ruff/Bandit, diff) et écrit **`results/results.json`** (label v1, legacy) ou
  **`results/results_v2.json`** (label v2, défaut depuis le 2026-09-09 — `--label v2`, qui est
  aussi le défaut de l'argument). AUC CV ± std, AUC split temporel, date, hash HEAD Flask, versions
  des libs, diagnostics de circularité/redondance. Source unique de vérité pour le tableau
  comparatif — plus aucun chiffre à recopier à la main dans le notebook pour cette table précise.
  `python results/benchmark.py` depuis la racine (venv actif) pour régénérer.
- **`results/chantier1_protocol.py`, `chantier1_fix_detection.py`, `chantier2_lint.py`,
  `chantier3_diff.py`, `chantier4_github.py`** — scripts de mesure dédiés à un chantier précis
  (protocole M1/M2/M2bis/M3, comparaison des détections de fix, redondance/circularité de
  Ruff+Bandit, dispersion du diff, signaux de process GitHub). Chacun écrit son propre
  `results/*.json` daté.
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
- **Ruff + Bandit retenus pour le Chantier 2, Lizard et Semgrep écartés pour l'instant.**
  *Pourquoi :* Lizard est multi-langages — sans valeur ajoutée sur un dépôt 100 % Python où radon
  couvre déjà la complexité cyclomatique ; Semgrep demande des règles personnalisées, plus lourd à
  justifier sans cas d'usage précis identifié. Les deux restent des pistes d'ouverture, pas un
  chantier fermé — à reconsidérer si le projet s'étend à un dépôt multi-langages (Chantier 6).
- **Ruff et Bandit batchés par dossier temporaire, pas un `subprocess` par fichier.** *Pourquoi :*
  même principe que `git cat-file --batch` — mesuré à 100,6 s pour ~3200 commits étiquetés (tout
  compris : matérialisation disque + un seul appel Ruff + un seul appel Bandit), contre des
  dizaines de minutes en appelant les deux outils fichier par fichier.
- **Bandit : normaliser par le nombre de fichiers réellement analysés, pas par le nombre de
  fichiers récupérés depuis Git.** *Pourquoi :* piège identique à `has_py`/`mi_avg=100.0` (§7
  point 8) — Bandit échoue silencieusement sur le même code Python 2 que radon, et son JSON
  contient alors une entrée sans les clés `SEVERITY.*`. Un `.get(cle, 0)` naïf aurait compté ça
  comme « zéro violation », pas comme un échec — bug réellement rencontré et corrigé en cours de
  Chantier 2 (voir §7).
- **API GitHub en GraphQL batché (alias de commits dans une même requête), pas un GET REST par
  commit.** *Pourquoi :* même principe que `git cat-file --batch` et le batching Ruff/Bandit —
  mesuré à 128 requêtes (coût ~1 point chacune, budget 5000/h) pour 4825 commits, contre un
  minimum de 4825 appels REST en série pour la seule étape « trouver la PR d'un commit » (plus
  1-2 appels REST supplémentaires par PR pour ses métadonnées). `statusCheckRollup` (champ
  GraphQL) donne en un seul champ l'état CI agrégé, couvrant aussi bien l'ancien système de
  statuts (Travis, utilisé sur l'historique ancien de Flask) que les GitHub Actions actuels —
  évite d'avoir à requêter deux API REST différentes (`status` legacy et `check-runs`) selon
  l'ancienneté du commit.
- **Cache GitHub distinguant explicitement "pas encore mesuré" de "mesuré, confirmé sans PR".**
  *Pourquoi :* anticipation du piège déjà rencontré deux fois (`has_py`, Bandit, §7) — un sha
  absent du cache (mesure pas encore faite, ex. arrêt sur budget épuisé) n'a pas la même
  signification qu'un sha en cache avec la valeur `None` (API interrogée avec succès, aucune PR
  associée trouvée). Conflater les deux aurait traité un "pas encore su" comme un "zéro" — même
  famille d'erreur que les deux précédentes, corrigée ici avant qu'elle ne se produise plutôt
  qu'après.

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
- **⚠️ Validée avec réserve forte — Ruff/Bandit ajoutent un signal réel mais largement redondant
  avec radon.** Mesuré le 2026-09-09 (`results/chantier2_lint.py`, label_v2 + détection de fix
  candidat B, run de référence) :

  | Jeu de features | AUC CV | Verdict |
  |---|---|---|
  | BASE | 0,787 | référence |
  | BASE + code (radon) | 0,791 | +0,004 |
  | BASE + Ruff | **0,798** | +0,011 — meilleur ajout seul que radon |
  | BASE + Bandit | 0,793 | +0,006 |
  | BASE + Ruff + Bandit | 0,798 | Bandit n'ajoute rien une fois Ruff présent |
  | BASE + radon + Ruff + Bandit | 0,796 | **combiner les trois fait moins bien que Ruff seul** |
  | Ruff + Bandit seuls (sans BASE) | 0,578 | signal faible en isolation |
  | Tout (BASE+churn+code+Ruff+Bandit) | 0,805 | **identique** à tout sans Ruff/Bandit (0,805) |

  **Diagnostic de redondance/circularité** : corrélation Ruff↔`loc_avg` (radon) = **0,704**,
  Bandit↔`loc_avg` = 0,431 — Ruff mesure en grande partie « ce fichier est volumineux », déjà
  capté par radon. Corrélation avec le churn (`file_commits_max`) = 0,198 (Ruff) / 0,057
  (Bandit) — **pas circulaire** comme l'était le churn (0,91) ou comme pouvait le laisser
  craindre radon au départ.
  **Lecture** : Ruff/Bandit ne sont pas un signal illusoire (corrélation au churn faible, gain
  individuel réel et supérieur à radon), mais leur valeur **marginale une fois combinés à tout le
  reste est quasi nulle** (0,805 → 0,805) — la redondance avec `loc_avg` explique l'essentiel de
  ce plafonnement. **Décision : les deux familles sont intégrées à `results/benchmark.py`** (jeu
  `BASE+lint`, `lint_only`, diagnostics de redondance/circularité dans `results/results_v2.json`)
  pour qu'on puisse suivre cette redondance dans le temps, mais **ne pas vendre Ruff/Bandit comme
  un gain net sur le modèle final** — c'est un problème de combinaison (Chantier 5), pas une
  feature inutile en soi.
- **❌ Réfutée — le ratio code/test en lignes bat le binaire `touches_tests`.** Mesuré le
  2026-09-09 (`results/chantier3_diff.py`), label_v2 + détection B. `test_line_ratio` seul donne
  **AUC = 0,499** (le hasard), contre **0,621** pour le `touches_tests` binaire déjà présent dans
  BASE — le raffinement proposé en ouverture du notebook (cellule 57) est **pire** que l'existant,
  pas meilleur. Cause probable : 80,7 % des valeurs de `test_line_ratio` sont exactement 0 ou 1,
  et la pondération par la taille totale du commit dilue le signal utile (« a-t-on touché un test
  du tout ? ») en écrasant les cas où un gros commit majoritairement non-test touche aussi,
  modestement, un fichier de test — un cas que le binaire capture proprement et que le ratio
  noie. **Décision : `touches_tests` conservé tel quel, `test_line_ratio` rejeté** — pas supprimé
  du code (`fyc/diff_features.py`, dans `ALL_COLS` mais pas dans `DIFF_COLS`), pour que ce résultat
  négatif reste reproductible plutôt que de disparaître silencieusement.
- **✅ Validée — dispersion du diff (hunks) et changement de signature de fonction (AST
  avant/après) apportent un signal réel, non circulaire.** Même mesure :

  | Feature | AUC seule | Corrélation churn (`file_commits_max`) | Verdict |
  |---|---|---|---|
  | `n_hunks` | 0,661 | 0,118 | adoptée — redondante à 0,74 avec `total_churn` (déjà dans BASE), mais pas circulaire |
  | `avg_hunk_size` | 0,550 | — | **rejetée** — signal trop faible pour être retenu |
  | `touches_signature` | 0,666 | 0,069 | adoptée — meilleur signal individuel du Chantier 3, quasi pas circulaire |
  | `sig_added`/`removed`/`modified` | 0,615–0,627 | — | adoptées (composantes de `touches_signature`) |

  **Résultat combiné** (jeu curaté `DIFF_COLS` = `n_hunks` + les 4 features de signature) :
  `BASE+diff` = 0,792 (contre 0,787 pour BASE seul, +0,005) ; `tout` (toutes familles) = 0,807
  contre 0,805 sans diff — petit gain net positif, contrairement à Ruff/Bandit dont la valeur
  marginale au niveau « tout » était nulle. **La couverture de `touches_signature` est de ~47 %**
  (NaN pour les commits sans `.py` touché ou dont l'AST avant/après a échoué à parser des deux
  côtés — même discipline NaN que radon/Bandit, pas de valeur par défaut déguisée).
- **✅ Validée — les signaux de processus GitHub (reviewers, délai de merge, état CI) apportent
  un signal réel, non circulaire, meilleur gain individuel que radon.** Mesuré le 2026-11-04
  (`results/chantier4_github.py`, label_v2 + détection B, couverture cache **100 %** — 3198/3198
  commits étiquetés) :

  | Jeu de features | AUC seule/ajoutée | Verdict |
  |---|---|---|
  | BASE | 0,787 | référence |
  | BASE + process | **0,799** | **+0,012** — meilleur ajout individuel des familles de code/process (radon +0,004, Ruff +0,011, diff +0,005) |
  | `process_only` (4 features) | 0,601 | signal réel, bien au-dessus du hasard |
  | `has_pr` seul | 0,585 | — |
  | `n_reviewers` seul | 0,593 | — |
  | `merge_delay_hours` seul | 0,605 | meilleur signal individuel de la famille |
  | `ci_failed` seul | 0,503 | quasi hasard — signal CI isolé trop faible (réserve ci-dessous) |
  | `tout` (toutes familles, sans process) | 0,805 | — |
  | `tout` (toutes familles, avec process, résultat final intégré) | **0,811** | petit gain net positif |

  **Circularité avec le churn fichier** (même diagnostic que pour Ruff/Bandit/diff) :
  `n_reviewers` ↔ `file_commits_max` = 0,0995, `merge_delay_hours` ↔ `file_commits_max` = −0,0043
  — **pas circulaire**, du même ordre que Ruff/diff (≤0,12), très loin du churn historique (0,91).
  **`part_has_pr` = 55,7 %** — l'autre moitié des commits n'a pas de PR associée retrouvée via
  l'API (rebase direct sur `main`, ou PR trop ancienne pour que GitHub l'associe) ; ces commits
  reçoivent `has_pr=0` et NaN sur les 4 autres colonnes (pas un zéro déguisé — discipline NaN
  identique à radon/Bandit/diff).
  **Réserve sur `ci_failed`** : signal quasi nul en isolation (0,503) alors que `merge_delay`/
  `n_reviewers` sont nettement au-dessus du hasard — possible dilution (`statusCheckRollup` agrège
  tous les checks d'une PR, un seul check instable peut suffire à faire basculer l'état sans
  relation avec le risque du commit lui-même) ou véritable absence de lien. **Non investigué plus
  avant** — gardé dans `PROCESS_COLS` pour la reproductibilité, mais ne pas vendre `ci_failed`
  comme un signal individuellement validé, seulement la famille dans son ensemble.
  **Décision : famille intégrée** à `results/benchmark.py` (jeux `BASE+process`/`process_only`,
  diagnostics de circularité et de couverture du cache dans `results/results_v2.json`). Sous
  `label_v1` (legacy, pour comparaison) : `BASE+process` = 0,791 (+0,023 sur BASE=0,768, gain
  individuel plus fort qu'en v2) mais `tout` = 0,877 **légèrement inférieur** à
  `all_sans_lint_diff_process` = 0,882 — sous le label circulaire, ajouter le process au reste
  n'apporte rien net (même lecture que Ruff/Bandit au Chantier 2 : le plafond est déjà saturé par
  le churn). Cohérent avec la préférence déjà actée pour `label_v2`.
- **⚠️ Vague 1 du Chantier 4bis — aucun des 5 outils ne démontre de gain sur BASE (tous < 1
  écart-type), signaux individuels réels mais modestes.** Mesuré le 2026-11-04
  (`results/chantier4bis_wave1.py` → `results/chantier4bis_wave1.json`, label_v2 + détection B,
  n = 3198, protocole §8 4bis). AUC(BASE) = 0,7867 ; écart-type entre plis ≈ 0,03 → seuil de
  conclusion du protocole ≈ +0,03 ; **aucun outil ne l'atteint**. Aucun seuil d'abandon
  déclenché (temps : Pylint 140 s, mypy 72 s, Vulture 34 s, couplage 35 s, Kamei 0,3 s, tous
  très sous les 15 min ; NaN ≤ 6 % parmi les commits touchant des `.py`).

  | Outil | BASE+outil | Gain | Meilleure(s) feature(s) seule(s) | Redondance / circularité |
  |---|---|---|---|---|
  | Pylint | 0,7958 | +0,009 | `pylint_error_avg` 0,638 ; `warning` 0,604 | `total` corrélé 0,86 à Ruff ; `convention` 0,87 à Bandit ; churn ≤ 0,16 |
  | mypy | 0,7937 | +0,007 | `mypy_total_avg` 0,591 | corrélé 0,64 à `cc_tot_avg` ; churn 0,01 (le plus indépendant) |
  | Couplage | 0,7897 | +0,003 | `fan_in_avg` 0,653 (**meilleur signal individuel du chantier**) ; `fan_in_max` 0,649 | `fan_in` peu redondant (corr. `loc_avg` −0,20) ; `fan_out_avg` corrélé 0,47 au churn (> plancher 0,3) |
  | Vulture | 0,7880 | +0,001 | `vulture_imports_avg` 0,635 | `total`/`funcs` corrélés 0,89/0,88 à Ruff et 0,74/0,77 à `loc_avg` = « nombre de définitions » (biais prévu) ; `imports` indépendant de `loc` (0,09) |
  | Kamei (entropie, NS, ND, NDEV, REXP, SEXP) | 0,7871 | +0,0004 | `nd` 0,631, `ns` 0,625 | **`ndev` corrélé 0,80 au churn = circulaire** ; `entropy`/`entropy_norm` ≈ hasard (0,478/0,510) ; `rexp`/`sexp` ≈ hasard et corrélés 0,90/0,84 à `author_experience` |

  **Lectures à retenir** : (1) l'**entropie du changement**, indicateur phare de la littérature
  JIT, ne prédit rien ici (AUC seule 0,478–0,510) — résultat négatif, conservé dans le code ;
  (2) le **fan-in** (combien de fichiers dépendent du module modifié) est le signal individuel le
  plus fort du chantier et n'est pas redondant avec la taille — mais n'apporte que +0,003 sur BASE
  une fois combiné ; (3) un AUC « seul » < 0,5 (`rexp` 0,445) est du bruit de CV autour du hasard,
  pas un signal inversé. **Les AUC seules sont des CV à 1 graine, bruit ≈ ±0,03 : ne comparer
  que des écarts nettement plus grands.** La combinaison des outils est traitée dans l'entrée
  suivante (grille exhaustive).
- **❌ Pas d'amélioration démontrée par la combinaison d'outils — résultat négatif net,
  avec un enseignement sur le churn.** Mesuré le 2026-11-04 (`results/chantier4bis_combinations.py`
  → `results/chantier4bis_grid.json`, 2048 AUC stockées ; figures regénérables par
  `results/chantier4bis_figures.py`), protocole §8 4bis + amendement n°1. Données : 2558 commits
  (80 % anciens, 10,8 % de positifs) ; réserve de 640 commits (5,2 % de positifs, ≈ 33 positifs).
  - **Grille exhaustive (2^11 = 2048 combinaisons, CV 5 plis, 100 arbres) :** AUC de 0,7915 (BASE)
    à 0,8228, médiane 0,8098 ; 99,95 % des combinaisons battent BASE mais seulement 44,8 % de plus
    d'un écart-type de pli (0,019) — **ajouter presque n'importe quoi aide un peu, rien n'aide
    beaucoup**. Meilleure AUC par taille : 0,8045 (1 famille : churn), 0,8153 (2), 0,8226 (5),
    puis **plateau** (0,8186–0,8228 de 6 à 11 familles) : empiler les 11 (0,8204) ne bat pas
    5 familles.
  - **Malédiction du gagnant, mesurée :** le top-20 de la grille, re-mesuré sur 5 graines neuves
    (300 arbres), tombe de 0,8206–0,8228 à **0,8108–0,8156** (−0,010 environ) ; BASE re-mesurée =
    0,7942. Les 20 meilleures combinaisons sont à ≤ 0,005 les unes des autres après re-mesure :
    **indiscernables, il n'y a pas « une » meilleure combinaison**. Gain réel plausible de la
    meilleure vs BASE : ≈ +0,02, sous 1 écart-type externe.
  - **Gain marginal moyen par famille** (sur toutes les combinaisons qui ne la contiennent pas) :
    churn **+0,0072** (positif dans 95,5 % des contextes), diff +0,0019, ruff +0,0016, pylint
    +0,0015, coupling +0,0011, mypy +0,0010, kamei +0,0009, process +0,0007, bandit +0,0006,
    radon +0,0004, **vulture −0,0008** (dégrade, positif dans 43 % seulement). Les familles de code
    ajoutent chacune ≤ 0,002 : cohérent avec la redondance mesurée aux Chantiers 2 et 4bis.
  - **⚠️ Le seul gain régulier vient du churn** — la famille dont la circularité avec le label a
    été mise en cause au Chantier 1 (r = 0,91 sous v1 ; réserve M1 non résolue sous v2 : r = 0,81).
    Toutes les meilleures combinaisons le contiennent. À présenter ainsi : le résultat « churn
    aide » est le plus robuste de la grille **et** le plus suspect ; ne pas en tirer
    « les outils d'analyse améliorent la prédiction ».
  - **Sélection pas-à-pas en CV imbriquée (estimation non biaisée de la procédure, 15 plis
    externes) : AUC = 0,8008 ± 0,0276.** Comparée à BASE re-mesurée (0,7942, plis différents donc
    comparaison non appariée) : **+0,007, sous le seuil du protocole (1 écart-type ≈ 0,02–0,03) →
    pas d'amélioration démontrée.** **Instabilité de la sélection** : fréquences de choix sur les
    15 plis — churn 40 %, process 27 %, ruff 20 %, mypy 13 %, radon/pylint 7 %, **bandit, diff,
    vulture, kamei, couplage 0 %** ; plusieurs plis choisissent « BASE seule ». La procédure ne
    converge pas vers une combinaison stable.
  - **Réserve 20 % (évaluée une seule fois) :** BASE 0,736 [0,659–0,812] ; tout empiler 0,742
    [0,659–0,816] ; meilleur de la grille 0,739 [0,647–0,817] ; sélection pas-à-pas
    (churn+pylint) 0,734 [0,643–0,814]. **Les quatre modèles sont indiscernables** (IC à 95 %
    de ±0,08 avec ≈ 33 positifs). **AUC réserve ≈ 0,74 contre ≈ 0,79–0,81 en CV sur le passé :**
    on ne peut pas dire si c'est de la dérive temporelle ou du bruit d'une queue à 5 % de
    positifs (biais de fin d'historique, §7 point 6) — **à ne pas annoncer comme une dégradation
    établie**, à garder comme question ouverte pour le Chantier 6A (la queue de test y est de
    30 %, avec garde-fou ≥ 40 positifs).
  - **Écarts au protocole, à dire :** (1) la sélection finale sur tout le 80 % utilise une CV
    interne à 5 plis (protocole : 3) ; (2) l'énumération utilise 100 arbres (amendement) ;
    (3) la comparaison nested vs BASE n'est pas appariée (BASE n'a pas été re-mesurée sur les
    mêmes 15 plis externes) ; (4) la vague 2 (Semgrep, Sonar, CodeQL, jscpd) et PySZZ ne sont pas
    faits. **Lecture pour le Chantier 5 :** empiler des familles mesurant la même chose ne donne
    rien ; le stacking par famille n'a de chance d'aider que si les familles sont réellement
    complémentaires — ce que la matrice de corrélation du Chantier 4bis ne suggère pas.
- **✅ T1 validé, ❌ T2 non validé, T3 mesuré — un modèle entraîné sur d'autres dépôts prédit un
  dépôt jamais vu ; l'avantage du « pré-entraîné + local » au démarrage à froid n'est pas
  démontré (extension 6A).** Mesuré le 2026-11-05 (`results/chantier6a_dataset.py` →
  `results/chantier6a_dataset.json` + jeu exporté `results/dataset_6a.csv.gz` ;
  `results/chantier6a_eval.py` → `results/chantier6a_eval.json`), protocole §8 6A, liste de
  7 dépôts figée avant toute mesure (commit 97d7284), 7/7 dépôts interprétables (aucun exclu par
  les garde-fous : part de risque 7,3–18,2 %, 44–147 positifs en queue de test). Leave-one-repo-out,
  queue de test = 30 % chronologiques récents, RF 300 arbres non réglé, 1 graine.

  | Dépôt | Étiquetés | Risque | Pos. test | AUC G [IC95] | AUC L (tout) | AUC G+L (tout) | Précision top 10 % (G) | Lift |
  |---|---|---|---|---|---|---|---|---|
  | flask | 3198 | 9,7 % | 44 | 0,757 [0,677–0,835] | 0,751 | 0,763 | 0,156 | 3,4× |
  | requests | 4233 | 9,3 % | 51 | 0,669 [0,591–0,749] | 0,663 | 0,659 | 0,126 | 3,2× |
  | click | 1808 | 12,2 % | 70 | 0,708 [0,649–0,767] | 0,673 | 0,711 | 0,315 | 2,4× |
  | black | 1976 | 9,5 % | 44 | 0,772 [0,698–0,839] | 0,764 | 0,794 | 0,254 | 3,4× |
  | tornado | 3446 | 7,3 % | 56 | 0,745 [0,678–0,809] | 0,724 | 0,771 | 0,155 | 2,9× |
  | pydantic | 4661 | 18,2 % | 147 | 0,779 [0,739–0,818] | 0,789 | 0,791 | 0,352 | 3,4× |
  | django-rest-framework | 5857 | 11,4 % | 80 | 0,745 [0,686–0,798] | 0,777 | 0,771 | 0,143 | 3,2× |

  (G = modèle global seul, entraîné sur les 6 autres dépôts ; L/G+L « tout » = tous les commits
  locaux de la partie d'entraînement ; lift = précision top 10 % / prévalence de la queue.)
  - **T1 (le transfert existe) : ✅ OUI.** Médiane AUC(G) = **0,745** (seuil 0,65). Les 7 dépôts
    > 0,66 ; le plus faible est requests (0,669, IC 0,59–0,75). Un modèle qui n'a **jamais vu**
    le dépôt classe ses commits récents à risque nettement au-dessus du hasard, avec les features
    en rang percentile, sans churn. **Les 4 variantes passent T1** (médianes 0,744–0,755).
  - **T2 (bénéfice au démarrage à froid) : ❌ ÉCHEC sur la configuration principale.** Médiane de
    AUC(G+L) − AUC(L) à n = 100 : **+0,009** (seuil +0,05). Variantes, rapportées sans les
    substituer (règle pré-enregistrée) : brut sans churn **+0,065** (passerait), percentile +
    churn +0,028, brut + churn +0,025 (échecs). **Le verdict T2 dépend donc de la variante** : à ne
    pas annoncer comme « le pré-entraînement aide au démarrage » ni comme « il n'aide pas » — il
    aide nettement dans un dépôt (pydantic : à n = 100, L 0,66 → G+L 0,77), pas dans d'autres
    (click, tornado, DRF ≈ 0 à n = 100).
  - **T3 (point de croisement) : mesure, bruitée.** `n*` (premier n où L ≥ G+L) : flask 200,
    requests 1600, black 800, tornado 100, DRF 200 ; **jamais** pour click et pydantic (G+L ≥ L
    à tous les n). Médiane où il existe : 200. **À ne pas sur-lire** : les courbes d'apprentissage
    ne sont pas monotones (ex. tornado L : 0,77 à n = 100, 0,69 à n = 800) — 1 graine, quelques
    dizaines de positifs en test (IC par dépôt de ± 0,05 à 0,08), un n* de 100 ou de 800 n'est pas
    distinguable. Ce qui est robuste : **à pleine taille, G+L ≥ L dans 5 dépôts sur 7** et jamais
    nettement pire (écart le plus défavorable −0,006 pour django-rest-framework, −0,004 pour requests ; le plus favorable +0,047 pour tornado) — ajouter les autres
    dépôts ne fait pas de mal.
  - **Normalisation en rang percentile : pas de gain.** Médiane AUC(G) 0,745 (percentile) contre
    0,755 (brut) ; l'écart vient surtout de requests (0,669 contre 0,705). **Le rang percentile ne
    rend pas les features plus transférables ici** — écart dans le bruit, mais pas de preuve du
    contraire de l'hypothèse de conception.
  - **Le churn n'explique pas le transfert.** Avec/sans churn (percentile) : médiane G 0,754
    contre 0,745 (+0,009, dans le bruit). La suspicion du Chantier 1 / de la grille 4bis (« le seul
    gain régulier vient du churn ») ne se confirme pas comme moteur du transfert inter-dépôts.
  - **Calibration : ❌ les probabilités brutes ne sont pas utilisables telles quelles.** Le Brier
    du modèle G est **moins bon que le simple prédicteur constant « prévalence réelle »** dans les
    7 dépôts (ex. flask 0,083 contre 0,044 ; tornado 0,127 contre 0,051 ; click 0,128 contre
    0,112) — attendu (`class_weight=balanced` gonfle les probabilités de la classe rare). **Le
    classement fonctionne (AUC, lift ≈ 3×), pas la valeur absolue** : un produit devrait
    recalibrer sur le taux de base local, ce qui ne demande que quelques dizaines de commits
    étiquetés, pas un réentraînement. À présenter comme la conclusion pratique du 6A.
  - **Lecture globale, honnête.** (1) Oui, un modèle global transfère, avec un lift ≈ 3× sur les
    10 % de commits les mieux classés, y compris sur un dépôt jamais vu — c'est l'argument le plus
    solide pour l'idée de modèle « pré-entraîné ». (2) Le bénéfice **au démarrage à froid** par
    rapport à un modèle local n'est pas établi avec ce protocole (T2 échoue sur la principale,
    passe sur une variante). (3) Les AUC en queue de test (0,67–0,78) sont du même ordre que les
    0,74 de la réserve 20 % du Chantier 4bis et **inférieures aux ≈ 0,79–0,81 des CV aléatoires** :
    cohérent avec une évaluation temporelle systématiquement plus dure que la CV mélangée, ce qui
    **renforce l'hypothèse de dérive** laissée ouverte en §6 (4bis) — **sans la démontrer** (les
    jeux de features diffèrent).
  - **Limites propres à ce résultat :** dépôts Python uniquement ; 7 dépôts (médianes peu
    précises, on donne les valeurs par dépôt) ; pas de dépôt data/ML (§8, critère ≤ 10 000
    commits) ; 1 graine et 1 queue de test par dépôt ; Flask recalculé à l'identique de la
    référence (3198 étiquetés, 310 positifs, 0 dépassement radon).

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
12. **Bandit échoue silencieusement sur le même code Python 2 que radon — même piège que
   `has_py`/`mi_avg=100.0`, trouvé et corrigé pendant le Chantier 2 (2026-09-09).** Sur un fichier
   non parsable, l'entrée `metrics` de Bandit existe mais ne contient que `loc`/`nosec`/
   `skipped_tests`, sans les clés `SEVERITY.*`. Un `.get(cle, 0)` naïf comptait ça comme « zéro
   violation » — un échec d'analyse redevenait indiscernable d'un fichier sans défaut de sécurité,
   et diluait la moyenne par fichier. **Corrigé** dans `fyc/code_lint.py::lint_features` : on ne
   normalise que par le nombre de fichiers où Bandit a réellement produit des métriques de
   sévérité (`bandit_ok_files`), NaN si aucun n'a réussi. **Leçon méthodologique répétée** : tout
   outil d'analyse statique externe sur de l'historique Git ancien doit être présumé capable
   d'échouer silencieusement sur du vieux code — vérifier explicitement, ne jamais faire confiance
   à un `except`/`.get(..., 0)` qui masque la différence entre « zéro trouvé » et « rien analysé ».
   Ruff, par contraste, ne fait pas cette erreur : il signale un échec de parsing comme une
   violation explicite (`code="invalid-syntax"`), pas un silence — donc pas corrigé de la même
   façon, mais à garder en tête que `ruff_total_avg` inclut ces échecs de syntaxe comme des
   « violations », ce qui n'est pas strictement la même chose qu'un vrai problème de qualité.
13. **mypy, Pylint, couplage, Python 3.14 — quatre pièges trouvés pendant la vague 1 du
   Chantier 4bis (2026-11-04).**
   - **mypy : 0 erreur ≠ code propre.** Par défaut mypy n'analyse pas le corps des fonctions non
     annotées (tout l'historique ancien de Flask). Premier essai : moyenne 0,59 erreur/fichier
     avec `core == total` — un zéro qui voulait dire « non vérifié », même famille que `has_py` et
     Bandit (§7 pts 8, 12). Corrigé par `--check-untyped-defs` (moyenne 3,9 sur le même
     échantillon). À re-vérifier pour tout outil qui a un mode « lenient » par défaut.
   - **mypy : une erreur bloquante avorte toute l'analyse.** Un `from . import x` sans paquet
     parent (fichiers aplatis dans un dossier) et un `SyntaxError` isolé font échouer le run entier,
     pas seulement le fichier fautif. Parades : un paquet à 3 niveaux par commit
     (`pk<i>/a/b/`) pour que `from ..x` résolve, et pré-filtrage `ast.parse`.
   - **Pylint : checks désactivés (documentés dans le module) :** `duplicate-code` (les versions
     de commits différents se « dupliqueraient » entre elles — mesure artificielle),
     `import-error`/`no-name-in-module`/`relative-beyond-top-level` (fichiers isolés, bruit
     pur), `invalid-name` (noms de fichiers aplatis). Les échecs de syntaxe (E0001…) sont exclus
     du numérateur ET du dénominateur ; NaN si aucun fichier analysé.
   - **Python 3.14 accepte `except A, e:`** (PEP 758, tuple d'exceptions sans parenthèses) :
     vérifié, `ast.parse` ne lève pas d'erreur sur du Python 2 de la forme `except X, e:` en
     3.14.4 (mypy `--python-version=3.12` le rejette). Observation **non reliée à une mesure** :
     cela pourrait contribuer à l'écart de couverture 3.12 / 3.14 du point 7, mais ce n'est
     **pas testé** ici (il faudrait comparer le taux de parsing avec/sans ce motif) — hypothèse,
     pas conclusion. Conséquence concrète : sous 3.14, du code Python 2 est parfois analysé
     comme du Python 3 valide *avec une sémantique différente* (`except A, e` attrape deux
     classes au lieu de nommer l'exception) — les métriques de ces fichiers sont à prendre avec
     cette réserve.
   - **Couplage : fan-in sous-estimé pour les blobs non parsables** (exclus de l'index) —
     mesuré à **3,8 %** des fichiers d'un arbre en moyenne, donc faible. Vérifié sur un cas : fan-in
     de `flask.globals` au HEAD = 17 avec l'outil et 17 avec `grep`.
14. **Fichiers générés gigantesques : radon bloquait indéfiniment sur black (6A, 2026-11-04).**
   Le dépôt `psf/black` versionne des fichiers de benchmark générés (`profiling/mix_huge.py`
   1,4 Mo, `dict_huge.py` 1,1 Mo, `list_huge.py` 0,9 Mo, `dict_big.py`, `mix_big.py`) : une seule
   expression littérale géante. Radon met **> 20 s par fichier** (contre 1,2 s pour le plus gros
   vrai fichier source, `tests/test_black.py`, 187 Ko) ; le calcul de black tournait depuis
   47 minutes (click, de taille comparable : 5 min) quand j'ai cherché la cause en chronométrant
   chaque étape et chaque gros fichier séparément (`py-spy` refusé par les permissions, pas
   contourné). **Correctif global, pas propre à black** : `fyc/features.py::_metriques_radon`
   plafonne chaque fichier à `RADON_TIMEOUT_S = 10` s (SIGALRM) ; un dépassement vaut échec
   d'analyse = **NaN** (même règle que le code illisible, point 8), jamais une valeur par défaut ;
   `RADON_STATS["timeouts"]` compte les dépassements et chaque dépôt les rapporte
   (`radon_timeouts_fichiers` dans `results/chantier6a_dataset.json`). `diff_features` attrape en
   plus `RecursionError`/`MemoryError` à l'AST. **Conséquence assumée** : le correctif change
   `fyc/` pour tous les dépôts ; les 3 dépôts déjà calculés (flask, requests, click) ont été
   **recalculés** pour que les 7 passent par le même code (aucun mélange de versions). Pour
   Flask, les chiffres du Chantier 1-4bis ne sont pas recalculés : ils n'ont pas touché de
   fichier > 10 s (le calcul d'origine s'était terminé normalement) — à confirmer par
   `radon_timeouts_fichiers` de Flask dans le JSON 6A. **Piège de méthode** : une extraction
   « qui tourne » n'est pas une extraction « qui progresse » — tout traitement par lot sur un
   nouveau dépôt devrait journaliser sa progression et plafonner le temps par unité.
15. **Mémoire : 7 dépôts en parallèle ont fait planter la machine de l'utilisateur ; cause =
   le JSON de Ruff (6A, 2026-11-05).** Plusieurs « redémarrages de WSL » constatés (`uptime`
   à 0 min) étaient en réalité des **plantages du PC par saturation mémoire** (l'utilisateur me l'a
   dit — je les avais d'abord attribués à WSL, hypothèse non vérifiée et fausse). **Mesure** par
   étape sur click (pic cumulé) : radon 289 Mo, **Ruff + Bandit 1811 Mo**, diff 1811 Mo — soit
   ~1,5 Go pour 1808 commits et **5,6 Go pour requests** (4233 commits). Cause : `ruff
   --select=ALL` écrit des millions de violations en JSON, parsées d'un bloc en objets Python.
   **Correctif** (`fyc/code_lint.py`) : Ruff et Bandit par **lots de 400 fichiers**, agrégation au
   fil de l'eau. **Équivalence vérifiée** : sur click, ancienne et nouvelle version donnent des
   tableaux strictement identiques (`assert_frame_equal`, Ruff et Bandit, NaN compris), pour
   495 Mo de pic au lieu de 1795 Mo, à temps égal — donc les dépôts calculés avant le correctif
   n'ont pas été recalculés. Pics mesurés ensuite : tornado 0,74 Go, DRF 0,92 Go.
   **Mesures de protection ajoutées** : exécution **strictement séquentielle** (un dépôt à la
   fois), garde-fou qui tue le calcul si la mémoire disponible < 900 Mo (déclenché une fois, sur
   tornado, avant le correctif — il a fait son travail), pic de RAM journalisé
   (`/usr/bin/time`), points de reprise par étape (`cache_6a_<dépôt>__<étape>.pkl`), évaluation
   limitée à 3 processus. **Leçon** : mesurer la mémoire par étape **avant** de paralléliser un
   traitement par lot ; ne pas conclure « WSL redémarre » sans chercher la cause.

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
2. **Chantier 2 — CLOS (2026-09-09).** Ruff + Bandit intégrés (`fyc/code_lint.py`,
   `results/chantier2_lint.py`), mesurés, documentés en §6 (gain réel mais largement redondant
   avec radon une fois combiné à tout le reste — pas un échec, un résultat nuancé). Lizard et
   Semgrep écartés pour ce passage (§4) — restent des pistes d'ouverture pour un dépôt
   multi-langages (Chantier 6). Bug Bandit trouvé et corrigé en cours de route (§7 point 12).
3. **Chantier 3 — CLOS (2026-09-09).** `fyc/diff_features.py` : dispersion du diff (`n_hunks`) et
   changement de signature de fonction (AST avant/après) adoptés — signal réel, non circulaire,
   petit gain net positif même combiné à tout le reste (0,807 vs 0,805). Ratio code/test en lignes
   testé et **rejeté** (pire que le binaire existant, résultat négatif documenté en §6) ;
   `avg_hunk_size` rejeté (signal trop faible). Intégré dans `results/benchmark.py`.
4. **Chantier 4 — CLOS (2026-11-04).** Signaux de processus via l'API GitHub (reviewers, délai de
   merge, état CI). Bloqué au démarrage le 2026-09-09 (pas de token, API non authentifiée limitée
   à 60 req/h) ; débloqué le 2026-11-04 par un token GitHub personnel fourni par l'utilisateur
   (option (a) des trois arbitrées). **Approche GraphQL batchée** (pas REST par commit, §4) :
   `fyc/github_signals.py`, cache disque resumable `results/github_cache.json`. Mesuré
   (`results/chantier4_github.py`, détail en §6) : signal réel et non circulaire, meilleur gain
   individuel que radon (+0,012 sur BASE), petit gain net positif combiné à tout (0,807→0,811).
   Couverture cache 100 % sur les 3198 commits étiquetés. Intégré dans `results/benchmark.py`
   (`fyc/process_features.py`, jeux `BASE+process`/`process_only`). Réserve documentée sur
   `ci_failed` (signal isolé trop faible, non investigué plus avant).
4bis. **Chantier 4bis — état de l'art : outils supplémentaires et recherche de la meilleure
   combinaison (AJOUTÉ le 2026-11-04, à traiter avant le Chantier 5, NON COMMENCÉ).**
   *Origine :* l'encadrant a conseillé d'utiliser les outils existants d'analyse de code et de les
   additionner. *Tension à garder en tête et à présenter honnêtement :* nos mesures (Chantier 2)
   montrent que additionner ne suffit pas (radon+Ruff+Bandit 0,796 < Ruff seul 0,798) — l'objectif
   de ce chantier est donc de **combiner** (sélectionner, éventuellement stacker), pas d'empiler.

   **Candidats à tester** (liste rédigée le 2026-11-04 ; estimations de coût/valeur à mesurer, pas
   des faits) :
   - *Analyse statique Python* : **Pylint** (duplication, conception — seul vrai complément
     probable à Ruff/radon, lent : chronométrer sur échantillon d'abord), **mypy/pyright** (types ;
     bruit attendu sur fichiers isolés de l'historique), **Vulture** (code mort, signal
     probablement faible), **CodeQL** (sécurité profonde, base de données à construire par version
     → échantillon seulement).
   - *Duplication / couplage* : **jscpd**, couplage fan-in/fan-out via AST ou **import-linter**.
   - *Littérature défaut JIT* : **les 14 indicateurs de Kamei et al. (2013)** — il manque surtout
     l'**entropie du changement** (dispersion des modifications entre fichiers ; sans outil externe,
     calcul sur le diff, cousin de `n_hunks`). **PySZZ** (variantes B-/AG-/MA-/R-SZZ) pour comparer
     `label_v2` à d'autres SZZ — validation de l'étiquetage. PyDriller écarté (n'apporte rien par
     rapport à `git_extract.py`). Modèles neuronaux (CodeBERT, DeepJIT, CC2Vec, JITLine) : à citer
     comme état de l'art, implémentation non prioritaire (311 positifs environ → risque de
     surapprentissage).
   - *Réévaluation de pistes différées* : **Semgrep avec jeux de règles prêts à l'emploi**
     (`p/python`, `p/security-audit` — l'objection « règles à écrire » du §4 est à revérifier),
     **Lizard** (utile surtout au Chantier 6), **SonarQube/SonarCloud** sur échantillon de 200–300
     commits (comparaison avec un outil industriel).
   - *Coûteux, a priori hors périmètre* : coverage.py + mutmut (exécution des tests de chaque
     version), DeepSource/Codacy (services en ligne, redondants avec Sonar).

   **Consigne pour l'instance de Claude qui traitera ce chantier** (décision de l'utilisateur, le
   2026-11-04 : il ne choisit pas les outils à sa place) : tester les outils **ensemble** et
   documenter **la meilleure combinaison trouvée**. Méthode obligatoire, à écrire AVANT
   d'implémenter, comme au Chantier 1 :
   1. Mesurer chaque outil **seul**, puis BASE+outil, avec le patron établi : module
      `fyc/<outil>.py`, traitement par lots (`git cat-file --batch`, dossier temporaire, un seul
      appel), **NaN et jamais 0 en cas d'échec d'analyse** (piège §7 points 8 et 12), jeu curaté
      (features rejetées gardées dans le code), script `results/chantier4bis_*.py` daté, intégration
      dans `results/benchmark.py`.
   2. Pour chaque outil : redondance (corrélation avec `loc_avg` et entre outils) et circularité
      (corrélation avec `file_commits_max`) avant toute conclusion.
   3. **Piège de la recherche de combinaison :** choisir la meilleure combinaison parmi de nombreuses
      sur la même validation croisée donne un chiffre optimiste (surapprentissage de la sélection,
      avec ~311 positifs). Fixer le protocole de sélection à l'avance et valider la combinaison
      retenue sur des données **non utilisées pour la sélection** (CV imbriquée, ou split temporel
      réservé) ; annoncer l'AUC de ce second chiffre, pas celui de la sélection. Gain à comparer à
      l'écart-type (~0,03) : en dessous, ne pas conclure à une amélioration.
   4. Les résultats négatifs (outil inutile, bruit, redondant) sont documentés en §6 au même titre
      que les positifs — l'évaluation porte sur la démarche, pas sur le score.
   5. Seuil d'abandon d'un outil (à fixer avant la mesure) : coût de calcul prohibitif ou
      couverture NaN trop élevée pour être interprétable.
   Ce chantier prépare le Chantier 5 (stacking par famille d'outils, matrice de désaccord).

   **PROTOCOLE PRÉ-ENREGISTRÉ — Chantier 4bis, v1 (rédigé le 2026-11-04, AVANT toute
   implémentation ; en attente de validation par l'utilisateur).**
   - **Périmètre en vagues** (le coût décide l'ordre, pas l'intérêt supposé) :
     - *Vague 1 — pleine échelle (~3200 commits étiquetés), calcul local :* **Pylint**
       (chronométré d'abord sur 100 commits), **Vulture**, **entropie du changement + indicateurs
       Kamei manquants** (calcul sur diff/`file_stats` : NS, ND, entropie, LA/LD/LT, NUC, EXP,
       REXP, SEXP — ceux déjà couverts par BASE/churn ne sont pas recalculés), **couplage
       fan-in/fan-out** par AST (imports). **mypy** : test de bruit sur 100 commits d'abord.
     - *Vague 2 — échantillon seulement (200–300 commits), résultats rapportés à part, jamais
       mélangés au tableau pleine échelle :* **Semgrep** (`p/python`, `p/security-audit`),
       **SonarQube/SonarCloud**, **CodeQL**, **jscpd**. Objectif : comparer à un outil
       industriel, pas alimenter le modèle final.
     - *Hors périmètre :* coverage.py/mutmut, DeepSource/Codacy, modèles neuronaux (cités comme
       état de l'art uniquement).
   - **Seuils d'abandon fixés maintenant** : (a) extrapolation > 15 min pour les ~3200 commits →
     passe en vague 2 (échantillon) ; (b) parmi les commits touchant des `.py`, > 50 % de NaN sur
     les features de l'outil → abandonné (non interprétable) ; (c) outil dont la meilleure
     feature a |AUC − 0,5| < 0,02 seule ET corrélation > 0,7 avec une feature déjà présente →
     rejeté comme redondant/bruit (code gardé, §6 documenté).
   - **Étape 1 par outil** (avant toute combinaison) : AUC de chaque feature seule, BASE+outil,
     corrélation avec `loc_avg` (redondance), avec `file_commits_max` (circularité, plancher de
     vigilance 0,3), et entre outils. NaN et jamais 0 sur échec d'analyse. Jeu curaté dans
     `DIFF_COLS`-style, rejetés conservés.
   - **Recherche de combinaison — protocole fixé à l'avance** :
     - *Unité de sélection = la famille* (BASE, churn, radon, Ruff, Bandit, diff, process + les
       nouvelles), pas la feature individuelle — limite l'espace de recherche (≈ 10 familles) et
       donc le surapprentissage de la sélection. BASE est toujours incluse.
     - *Algorithme :* sélection pas-à-pas ascendante par famille ; une famille n'est ajoutée que
       si son gain d'AUC en CV interne (3 plis) dépasse **0,01** (≈ 1 écart-type d'un pli
       interne, pas le 0,03 de la CV externe, trop conservateur pour un seuil de sélection) ;
       arrêt au premier tour sans ajout. Modèle fixe (RandomForest 300 arbres, hyperparamètres
       non réglés — régler serait une seconde recherche).
     - *Estimation honnête :* **CV imbriquée** (externe 5 plis × 3 graines différentes, sélection
       refaite dans chaque pli externe sur le train seul). Chiffre annoncé = AUC externe moyenne ±
       écart-type, pas l'AUC de la sélection. La combinaison « finale » = celle sélectionnée sur
       tout le jeu, annoncée avec la fréquence de sélection de chaque famille sur les 15 plis
       externes (stabilité de la sélection).
     - *Réserve temporelle :* le dernier 20 % chronologique est évalué **une seule fois**, à la
       fin, sur la combinaison finale, comparé à BASE et à « tout empiler » (0,811 sous v2).
       **Limite assumée dès maintenant :** les familles déjà existantes (churn, radon, Ruff,
       Bandit, diff, process) ont déjà été vues en CV complète sur ces données lors des chantiers
       1–4 ; la réserve n'est donc propre que pour les familles nouvelles — à dire tel quel.
     - *Règle de conclusion :* amélioration affirmée seulement si l'AUC externe de la combinaison
       dépasse BASE d'**au moins 1 écart-type externe** (~0,02) ; sinon on annonce « pas
       d'amélioration démontrée ». Comparer aussi à « tout empiler » : si la sélection ne bat pas
       l'empilement, le dire (c'est le résultat attendu d'après le Chantier 2).
   - **AMENDEMENT n°1 (2026-11-04, demandé par l'utilisateur, écrit AVANT d'exécuter aucune
     combinaison) — énumération EXHAUSTIVE des combinaisons, en complément de la sélection
     imbriquée (qui reste l'estimation de référence).** Raison : illustrer le projet par des
     statistiques sur de nombreuses combinaisons d'outils, pas seulement la gagnante.
     - *Espace :* les 11 familles hors BASE (churn, radon, Ruff, Bandit, diff, process, Pylint,
       Vulture, mypy, Kamei, couplage), BASE toujours incluse → 2^11 = **2048 combinaisons**.
     - *Données de l'énumération :* uniquement les **80 % chronologiques anciens** ; les 20 %
       récents restent intacts pour l'évaluation finale unique (sinon on consommerait la réserve
       temporelle sur 2048 candidats). CV 5 plis, 1 graine, **100 arbres** (coût) — le reste du
       projet utilise 300 : écart assumé, les ré-évaluations ci-dessous repassent à 300.
     - *Biais assumé :* le « meilleur sur 2048 » est optimiste (malédiction du gagnant). Parades
       fixées maintenant : (1) ré-évaluer le top-20 sur **5 graines de CV neuves** (300 arbres) ;
       (2) évaluer UNE fois sur les 20 % réservés : le meilleur de la grille, la sélection
       pas-à-pas, « tout empiler », BASE ; (3) l'AUC annoncée comme « meilleure combinaison » est
       celle de (2), jamais celle de la grille. La CV imbriquée de la sélection pas-à-pas
       (protocole initial) reste la mesure non biaisée de la *procédure*.
     - *Statistiques stockées pour l'illustration (toutes, pas seulement le top) :* les 2048 AUC
       (± écart-type par pli) ; meilleure AUC par nombre de familles ; gain marginal moyen de
       chaque famille sur toutes les combinaisons qui ne la contiennent pas ; top/bottom 20 ;
       part de combinaisons battant BASE de plus d'un écart-type. Fichier :
       `results/chantier4bis_grid.json`.
   - **Règle de stockage (demandée par l'utilisateur, 2026-11-04) :** tout résultat de ce
     chantier — positif, négatif ou nul — est écrit dans un JSON daté sous `results/` ET résumé
     en §6 ; toute décision de méthode (y compris les pièges rencontrés) en §7 ; chaque session
     dans le journal §9. Les caches `*.pkl` ne sont pas versionnés (recalculables par les
     scripts) ; les JSON de résultats le sont.
   - **Validation du label (sous-volet, après les features) :** PySZZ (au moins B-SZZ et MA-SZZ)
     vs `label_v2` — taux d'accord sur les commits étiquetés. Mesure de robustesse du label, pas
     un candidat à remplacer `label_v2` sans repasser par le protocole du Chantier 1.
   - **Livrables :** `fyc/<outil>.py` par outil, `results/chantier4bis_*.py` datés (un par vague +
     un pour la sélection imbriquée), intégration au `benchmark.py`, §6 mis à jour (positifs ET
     négatifs), `requirements.txt` mis à jour.
5. **Chantier 5 — croisement et combinaison** (stacking, matrice de désaccord entre modèles).
6. **Chantier 6 — validation externe** sur un deuxième dépôt.

   **EXTENSION 6A — Transférabilité inter-dépôts et courbe d'apprentissage (AJOUTÉE le
   2026-11-04 à la demande de l'utilisateur ; PROTOCOLE v1 VALIDÉ par l'utilisateur le
   2026-11-04 ; liste et jeu de familles figés (commit 97d7284) ; MESURÉE le 2026-11-05 sur les
   7 dépôts — résultats en §6, verdict : T1 OK, T2 ÉCHEC sur la configuration principale, T3
   mesuré).**
   *Origine :* idée de produit/agent « pré-entraîné » branchable sur un dépôt d'entreprise, utile
   surtout aux petits dépôts où l'entraînement local est impossible (≈30 positifs pour 500
   commits). *Question de recherche :* **un modèle entraîné sur d'autres dépôts prédit-il les
   commits à risque d'un dépôt jamais vu, et à partir de combien de commits locaux l'entraînement
   local fait-il mieux ?** Le produit/agent lui-même est **hors périmètre** (ouverture
   discutée en soutenance : l'« agent » LLM ne serait que l'interface autour du modèle
   statistique, qui reste ce que l'on évalue ici). *Bénéfice annexe pour le cours :* un jeu de
   données multi-dépôts exporté (JSON/pickle — `*.csv` est ignoré par git).
   - **Dépendance :** utilise le jeu de familles final du Chantier 4bis (figé avant de démarrer).
   - **Corpus :** 5 à 10 dépôts Python publics. **Règle de sélection fixée avant tout calcul**
     (liste commitée AVANT la première mesure, aucun dépôt retiré après avoir vu un résultat) :
     ≥ 2000 commits, ≥ 3 ans d'historique, activité récente, domaines/architectures variés
     (bibliothèque, framework web, outil CLI, data/ML). Flask peut en faire partie.
   - **JEU DE FAMILLES FIGÉ (2026-11-04, validé par l'utilisateur) :** BASE + diff (`DIFF_COLS`)
     + radon + Ruff. **Interprétation que j'ai faite et que je signale** : le churn n'est PAS
     dans le jeu principal (il est la famille suspecte du Chantier 1 et le seul gain régulier
     de la grille 4bis) ; il est mesuré en **ablation** `+churn (rang percentile)`, sans coût de
     données supplémentaire, pour voir si le transfert dépend de lui. Pylint, mypy, Vulture,
     Kamei, couplage, Bandit, process : exclus (gain nul ou négatif en grille, ou non
     universels). Chaque feature dépendante de l'échelle existe en deux versions (brute /
     rang percentile passé seul).
   - **LISTE DES DÉPÔTS — FIGÉE le 2026-11-04, VALIDÉE par l'utilisateur, commitée AVANT toute
     mesure du 6A (aucune mesure n'existe à cette date). Toute modification ultérieure doit être
     datée et justifiée ici, jamais faite en silence.** Nombres de commits lus via
     l'API GitHub le 2026-11-04 (pas de mémoire) : `pallets/flask` 5566 (référence, domaine :
     framework web), `psf/requests` 6496 (client HTTP), `pallets/click` 3379 (CLI),
     `psf/black` 2488 (outil de formatage), `tornadoweb/tornado` 5316 (réseau asynchrone),
     `pydantic/pydantic` 5918 (validation/typage), `encode/django-rest-framework` 9086 (API
     REST). **Critère pratique ajouté à ceux du protocole : ≤ 10 000 commits** — les
     extractions chargent en mémoire toutes les versions de fichiers (7 Go de RAM
     disponibles) ; conséquence assumée : pas de dépôt data/ML (scikit-learn 34 123 commits,
     pandas…), donc ce domaine n'est pas couvert, à dire en soutenance. **Réserve (fixée
     maintenant, utilisée dans cet ordre si un dépôt échoue au critère « ≥ 3 ans d'historique »,
     vérifié au clonage, ou « activité récente »)** : `python-poetry/poetry` 3900,
     `fastapi/fastapi` 7991 (nombreux commits de docs/traductions, à surveiller). Écartés :
     `encode/httpx` (1523 < 2000), `pallets/jinja` (dernier push 2025-06, inactif), tous les
     dépôts > 10 000 commits (pytest, aiohttp, sphinx, celery, django, mitmproxy, Pillow, pip,
     scikit-learn). **La sélection est discrétionnaire (un dépôt par domaine, jugé à l'œil) ;
     la liste est figée AVANT tout résultat pour que ce choix ne puisse pas dépendre des
     AUC.**
   - **Étiquetage :** identique à Flask (`label_v2`, fenêtre 90 j, détection de fix B), **sans
     réglage par dépôt**. Rapporter pour chaque dépôt `part_risque` et la part de commits
     reconnus « fix » : un dépôt dont les conventions de message cassent la détection (ex.
     `PROJ-123`, squash merges) est **signalé, pas corrigé en douce** ; garde-fou : dépôt exclu des
     conclusions si `part_risque` < 5 % ou < 40 positifs dans sa queue de test (§6, règle des 5 %).
   - **Jeu de features « transférable » :** BASE + diff + radon/Ruff (Python) ; **signaux de
     process GitHub exclus** (non universels), mesurés à part si disponibles. Les features
     dépendantes de l'échelle du dépôt (churn, `author_experience`, tailles) sont remplacées par
     leur **rang percentile dans le dépôt, calculé en fenêtre croissante sur le passé seul**
     (aucune fuite du futur). Comparaison systématique avec la version brute, pour mesurer ce que
     coûte/rapporte la normalisation.
   - **Protocole d'évaluation (leave-one-repo-out) :** pour chaque dépôt cible, queue de test =
     **30 % chronologiques récents**, jamais utilisés pour entraîner. Trois modèles, mêmes
     hyperparamètres que le reste du projet (RF 300 arbres, non réglé) :
     (G) **global seul** : entraîné sur les autres dépôts, restreints aux commits antérieurs à la
     date de début de la queue de test de la cible (pas de fuite temporelle entre dépôts) ;
     (L) **local seul** : entraîné sur les `n` premiers commits de la cible ;
     (G+L) **global + local** : union des deux, sans pondération (pas de réglage de poids).
     `n ∈ {50, 100, 200, 400, 800, 1600}` (plafonné à 70 % du dépôt).
   - **Métriques :** AUC avec IC bootstrap ; **Brier et courbe de calibration** (l'AUC est
     insensible au taux de base, qui varie d'un dépôt à l'autre — or c'est ce qui compte pour un
     produit) ; précision sur les 10 % de commits les mieux classés.
   - **Précisions d'implémentation pré-enregistrées (2026-11-04, écrites AVANT la première
     évaluation, après la construction du jeu de données) :**
     (1) **configuration principale = features en rang percentile, SANS churn** ; les trois
     autres (brut sans churn, percentile + churn, brut + churn) sont rapportées comme variantes,
     jamais substituées a posteriori ; T1/T2/T3 se jugent sur la principale ;
     (2) **un modèle local entraîné sur une seule classe (aucun positif parmi les `n` premiers
     commits) est un prédicteur constant : son AUC vaut 0,5** (et non « non défini » : ça
     avantagerait artificiellement le local en le retirant des comparaisons) ;
     (3) Brier calculé sur les probabilités brutes de la forêt (`class_weight=balanced` les
     décale par construction) ET comparé à deux références constantes (prévalence
     d'entraînement, prévalence réelle de la queue) — sans ces références le Brier n'est pas
     interprétable ; (4) un seul entraînement par cas (graine 42), IC bootstrap (1000 tirages)
     sur la queue de test ; (5) la queue de test d'un dépôt reste identique pour toutes les
     variantes (30 % chronologiques récents).
   - **Seuils de conclusion pré-enregistrés :**
     T1 (le transfert existe) : médiane sur les dépôts de AUC(G) ≥ 0,65.
     T2 (bénéfice au démarrage à froid) : à n = 100, médiane de AUC(G+L) − AUC(L) ≥ 0,05.
     T3 (point de croisement) : plus petit `n*` tel que AUC(L) ≥ AUC(G+L), rapporté par dépôt
     et en médiane — pas de seuil, c'est une mesure.
   - **Règle de résultat négatif :** si T1 échoue, on documente « pas de transfert avec ce jeu de
     features » en §6 et on teste **une seule** variante de repli fixée maintenant (features Git
     seules, sans outils Python). Pas d'autre repli ensuite.
   - **Limites assumées d'avance :** dépôts Python uniquement (les outils de code le sont) ; un
     modèle global dans un produit réel poserait en plus la question de la confidentialité du
     code d'entreprise, hors périmètre ; 5–10 dépôts donnent une médiane peu précise, on
     rapporte les valeurs par dépôt.
   - **Livrables :** `fyc/` généralisé (le chemin du dépôt est déjà un paramètre partout),
     `results/chantier6a_*.py` datés + JSON, jeu multi-dépôts exporté, §6 mis à jour (positifs
     ET négatifs), entrée dans le journal §9.
   **EXTENSION 6B — Recalibration locale d'un modèle global (AJOUTÉE le 2026-11-05 ; ordre
   choisi par moi sur délégation explicite de l'utilisateur : « l'ordre le plus pertinent » ;
   PROTOCOLE v1 rédigé AVANT toute exécution).**
   *Origine :* le 6A montre que le classement du modèle global transfère (lift ≈ 3×) mais que ses
   probabilités brutes sont **moins bonnes que le prédicteur constant « prévalence »** (Brier,
   7 dépôts sur 7). *Question :* **combien de commits étiquetés locaux faut-il pour recalibrer les
   probabilités d'un modèle global sans le réentraîner, au point de battre le prédicteur constant ?**
   - **Modèle de base :** G de la configuration principale du 6A (rang percentile, sans churn),
     mêmes cibles, même queue de test (30 % récents), mêmes données d'entraînement de G. Le
     classement (AUC) ne change pas : toute recalibration monotone le laisse intact — vérifié en
     sortie (garde-fou : tout écart d'AUC > 0,001 invalide la mesure).
   - **Méthodes (aucun hyperparamètre réglé) :** R0 = brut ; **R1 = correction de prior** (G est
     entraîné avec `class_weight=balanced`, donc son prior effectif vaut 0,5) :
     `odds' = odds(s) × π̂/(1−π̂)`, π̂ = (positifs+1)/(n+2) (Laplace) ; **R2 = Platt** (régression
     logistique à 1 variable sur logit(s)), repli sur R1 si < 5 positifs dans l'échantillon de
     calibration ; **R-oracle = R1 avec la vraie prévalence du test** (borne haute de ce que la
     correction de prior peut apporter, jamais une méthode utilisable).
   - **Échantillon de calibration — deux fenêtres, la première est la principale :**
     (a) **récente** = les `n` derniers commits étiquetés **antérieurs à (début du test − 90 j)**
     (le trou de 90 j évite que les étiquettes, calculées avec 90 j de futur, voient la période de
     test ; c'est la situation d'un vrai déploiement) ; (b) **ancienne** = les `n` premiers
     commits, diagnostic de dérive de prévalence (la queue de test a une prévalence plus basse que
     l'historique : flask 4,6 % contre 9,7 %). `n ∈ {25, 50, 100, 200, 400, 800}`.
   - **Références constantes :** « loyale » = π̂ estimé sur le même échantillon (la vraie
     concurrente d'un modèle recalibré) ; « oracle » = vraie prévalence du test.
   - **Métriques :** Brier ; **score de compétence BSS = 1 − Brier(modèle)/Brier(constante
     loyale)** (> 0 = le modèle recalibré apporte de l'information en plus du taux de base) ; ECE sur
     5 classes d'effectif égal ; AUC (garde-fou).
   - **Seuils de conclusion pré-enregistrés :**
     C1 (la recalibration corrige) : à n = 100, médiane sur les dépôts de Brier(R1) < Brier(R0).
     C2 (utilité) : plus petit `n` tel que la médiane de BSS(R1 ou R2, fenêtre récente) > 0 ET BSS > 0
     dans ≥ 5 dépôts sur 7 ; rapporté par dépôt. C3 : comparer fenêtre récente / ancienne — si
     « ancienne » est nettement pire, la dérive de prévalence est une limite pratique du produit.
   - **Règle de résultat négatif :** si C2 n'est atteint pour aucun `n`, on documente « la
     recalibration par prior/Platt ne suffit pas à battre la constante » en §6, sans autre
     méthode essayée (isotonique, etc.) — elle serait une seconde recherche.
   - **Limites d'avance :** 1 graine, ~45–150 positifs par queue de test ; Brier bruité à faible
     prévalence ; les données d'entraînement de G (autres dépôts) incluent des commits dans les
     90 j précédant le test, dont les étiquettes ont vu un peu de futur *dans leur propre dépôt* —
     fuite jugée négligeable pour le dépôt cible, à signaler aussi pour le 6A.
   - **Livrables :** `results/chantier6b_calibration.py` → `results/chantier6b_calibration.json`,
     §6, journal §9.
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

### 2026-09-09 — Session 3 (suite) — Chantier 2 : Ruff + Bandit
- Plan proposé (Ruff + Bandit, Lizard/Semgrep écartés avec justification) puis exécuté sur
  consigne explicite de l'utilisateur de continuer sans attendre d'autre validation.
- `fyc/code_lint.py` (nouveau) : `lint_features` — Ruff et Bandit en un seul passage de
  matérialisation disque, batch par dossier temporaire (mesuré : 100,6 s pour ~3200 commits
  étiquetés, tout compris).
- **Bug trouvé et corrigé en cours de route** (pas découvert après coup) : premier test donnait
  `bandit_total_avg = 0` sur tous les commits. Cause : Bandit écrit une barre de progression sur
  **stdout** (pas stderr), cassant le parsing JSON — un `except json.JSONDecodeError: return {}`
  trop permissif avalait l'erreur silencieusement. Corrigé (`-q`) et surtout **l'exception
  silencieuse retirée** : une sortie non-JSON lève maintenant une erreur explicite plutôt que de
  se faire passer pour "aucune violation".
- **Second bug, même famille, trouvé en vérifiant au lieu de supposer** : Bandit échoue
  silencieusement sur le même vieux code Python 2 que radon — l'entrée `metrics` pour un fichier
  en échec existe mais sans les clés `SEVERITY.*`. Un `.get(cle, 0)` comptait ça comme zéro.
  Corrigé par un compte explicite des fichiers réellement analysés (`bandit_ok_files`), NaN sinon.
  Documenté en §7 point 12 comme leçon méthodologique générale, pas juste un fix ponctuel.
- Mesuré (`results/chantier2_lint.py`, détail en §6) : Ruff ajoute le meilleur gain individuel des
  trois familles de code (+0,011 sur BASE, mieux que radon +0,004), non circulaire avec le churn
  (corrélation 0,20 contre 0,91 pour le churn historique), mais fortement redondant avec `loc_avg`
  de radon (corrélation 0,70) — combiner les trois familles de code ne bat pas Ruff seul, et
  n'ajoute rien une fois churn+radon déjà présents (0,805 avec ou sans Ruff/Bandit).
- Intégré dans `results/benchmark.py` (jeux `BASE+lint`/`lint_only`, diagnostics de redondance et
  de circularité dans le JSON) plutôt que laissé dans un script isolé — même discipline que pour
  le Chantier 1 (pas de chiffre qui ne soit pas reproductible et daté).
- `requirements.txt` : ajout de `bandit==1.9.4`.
- Notebook ré-exécuté, 0 erreur.
- **Chantier 2 CLOS** sur le périmètre Ruff+Bandit. Lizard/Semgrep explicitement différés, pas
  oubliés (§4, §8).
- **Reste à faire** : commit de cet état ; Chantier 3 (analyse du diff) ou autre, sur arbitrage de
  l'utilisateur.

### 2026-09-09 — Session 3 (suite) — Chantier 3, puis blocage identifié au Chantier 4
- Consigne reçue : « tu peux passer au chantier 3 puis 4 ». Avant de démarrer le 4, vérifié la
  faisabilité (pas supposé) : pas de `gh` CLI, pas de token GitHub — API limitée à 60 req/h.
  Chantier 4 bloqué au démarrage, documenté en §8, pas tenté à moitié ni silencieusement ignoré.
- `fyc/git_extract.py` : `extraire_commits` étendu avec `file_stats` (détail par fichier des
  lignes ajoutées/supprimées) — ajout additif, `add`/`del`/`files` inchangés, rien de cassé.
- `fyc/diff_features.py` (nouveau) : `build_diff_features` — dispersion (hunks), ratio code/test
  en lignes, changement de signature via diff d'AST avant/après. Testé sur 300 commits avant
  l'échelle complète (7,7 ms/ligne, acceptable), puis sur les ~3200 commits étiquetés (31,8 s).
  Même discipline NaN que radon/Bandit (un échec AST n'imite pas "pas de fonction").
- Mesuré feature par feature avant de bundler (`results/chantier3_diff.py`), pas seulement la
  famille entière — a révélé que le bundle initial masquait un résultat négatif net.
- **Résultat négatif trouvé et gardé, pas corrigé en douce** : `test_line_ratio` (le raffinement
  suggéré par le notebook lui-même, cellule 57) donne AUC = 0,499 seul, **pire** que le
  `touches_tests` binaire existant (0,621) qu'il devait améliorer. Diagnostiqué (80,7 % de valeurs
  à 0 ou 1, dilution du signal par la taille du commit) et documenté en §6 comme hypothèse
  réfutée, pas supprimée du code (`ALL_COLS` vs `DIFF_COLS` dans `fyc/diff_features.py`).
  `avg_hunk_size` rejeté aussi (signal trop faible, 0,550).
- `n_hunks` et les features de signature de fonction adoptées — signal réel, non circulaire
  (corrélation churn ≤ 0,12), petit gain net positif même au niveau "tout" (0,805 → 0,807),
  contrairement à Ruff/Bandit (Chantier 2) dont la valeur marginale y était nulle.
- Intégré dans `results/benchmark.py` avec le jeu curaté (pas tout ce qui a été mesuré).
- Notebook ré-exécuté, 0 erreur.
- **Chantier 3 CLOS.** **Chantier 4 bloqué**, en attente d'arbitrage utilisateur (token GitHub,
  échantillon restreint, ou report).
- **Projet mis en pause par l'utilisateur** : arbitrage tranché côté utilisateur en faveur de
  l'option (a) — fournira un token GitHub personnel la prochaine session pour débloquer le
  Chantier 4 à l'échelle complète (5000 req/h authentifié). Rien d'autre en attente côté code —
  état propre, tout committé (`71dca24`). **Reste à faire au retour** : récupérer le token,
  l'utiliser pour l'intégration API GitHub (reviewers, délai merge, échecs CI), puis suivre la
  même discipline que les chantiers précédents (mesurer, comparer à BASE, vérifier redondance/
  circularité avant de conclure).

### 2026-11-04 — Session 4 — Chantier 4 débloqué et clos
- Token GitHub personnel reçu de l'utilisateur, placé dans `.env` (variable `token`). **Ajouté
  immédiatement à `.gitignore`** avant toute autre action — le fichier n'était pas ignoré au
  moment où il a été créé, risque de fuite évité en vérifiant `git status`/`git check-ignore`
  avant de continuer. Token testé (`GET /rate_limit`) : 5000 req/h confirmés.
  Premier essai avec un nom de variable supposé (`GITHUB_TOKEN`) → 401 ; la vraie variable
  s'appelle `token` — vérifié en lisant le fichier plutôt qu'en re-essayant à l'aveugle.
- **Choix d'architecture fait avant de coder a grande echelle** (mesuré sur 5557 commits) : seuls
  326/5557 messages de commit contiennent déjà un numéro de PR (`(#NNNN)`) — pas assez pour s'en
  passer. Plutôt qu'un GET REST par commit pour trouver sa PR (minimum 3198+ appels en série),
  utilisé l'API **GraphQL** avec des commits aliasés par lots de 25 dans une même requête — même
  principe que `git cat-file --batch` déjà retenu au Chantier 1 et le batching Ruff/Bandit au
  Chantier 2. Coût mesuré : ~1 point GraphQL par lot de 25, soit ~190 points pour 4825 commits
  (bien sous le budget de 5000/h) — testé d'abord sur 50 commits avant l'échelle complète.
- `fyc/github_signals.py` (nouveau) : `fetch_commit_pr_signals`, cache disque resumable
  (`results/github_cache.json`, gitignore). Distinction explicite anticipée (pas découverte après
  coup cette fois) entre "sha absent du cache" (pas encore mesuré) et "sha en cache avec `None`"
  (mesuré, confirmé sans PR associée) — même classe de piège que `has_py`/Bandit (§7), traitée en
  amont dès la conception du module.
- `fyc/process_features.py` (nouveau) : `build_process_features` — `has_pr`, `n_reviewers`,
  `n_review_comments`, `merge_delay_hours`, `ci_failed`. Retourne un masque de couverture pour
  sous-échantillonner les autres familles de features aux mêmes lignes si le cache GitHub est
  incomplet (n'a pas été nécessaire : couverture 100 % obtenue en un seul run, 3198/3198).
- `results/chantier4_github.py` (nouveau) : mesure dédiée, même méthode que les chantiers
  précédents (AUC par feature isolée avant le jeu combiné, circularité vs churn, couverture du
  cache vérifiée et affichée avant toute conclusion). Résultat détaillé en §6 : signal réel, non
  circulaire, meilleur gain individuel (+0,012 sur BASE) que les trois autres familles de
  code/process mesurées jusqu'ici. Réserve notée sur `ci_failed` (signal isolé proche du hasard).
- Intégré dans `results/benchmark.py` (`PROCESS_COLS`, jeux `BASE+process`/`process_only`,
  diagnostics de circularité/couverture). `results_v2.json` régénéré : `all` passe de 0,807 à
  **0,811**. `results.json` (v1, legacy) aussi régénéré pour verifier la couverture du cache sous
  l'autre label (100 % également, même ensemble de commits éligibles) — sous v1, `tout` (0,877)
  est légèrement **inférieur** à `tout sans process` (0,882), cohérent avec la préférence déjà
  actée pour `label_v2` (le plafond est déjà saturé par la circularité du churn sous v1).
- **Chantier 4 CLOS.** Pas de notebook à ré-exécuter pour ce chantier (le tableau de la cellule 52
  ne référence pas les nouvelles clés `BASE+process`/`process_only`/`all_sans_lint_diff_process`,
  seulement `all` dont le nom est resté inchangé — vérifié avant de conclure qu'aucune mise à jour
  n'était nécessaire).
- **Reste à faire** : commit de cet état ; arbitrage de l'utilisateur sur la suite (Chantier 5 —
  croisement/combinaison — ou autre).

### 2026-11-04 — Session 5 — Chantier 4bis (vague 1 + combinaisons) ; extension 6A ajoutée
- Push GitHub bloqué en début de session : diagnostiqué sans modification de config. Clé SSH
  correcte et enregistrée (vérifié via l'API publique) ; l'authentification SSH est
  **intermittente côté GitHub/réseau** (même commande, succès une fois sur deux, indépendamment
  des options). Contournement : relancer `git push`. Cause exacte non identifiée.
- **Protocole du Chantier 4bis rédigé AVANT implémentation et validé par l'utilisateur** (§8).
  Amendement n°1 (à la demande de l'utilisateur, avant toute exécution) : énumération exhaustive
  des 2048 combinaisons pour illustrer, en complément de la CV imbriquée ; règle de stockage
  de tous les résultats ajoutée au protocole.
- Installé `pylint 4.1.3`, `vulture 2.16`, `mypy 2.4.1` (`requirements.txt`). Nouveaux modules :
  `fyc/pylint_features.py`, `vulture_features.py`, `mypy_features.py`, `jit_kamei.py`,
  `coupling.py` (§3). Scripts : `chantier4bis_common.py`, `_wave1.py`, `_combinations.py`,
  `_figures.py`.
- **Pièges trouvés en vérifiant plutôt qu'en supposant** (§7 point 13) : mypy renvoie 0 sur le
  code non annoté par défaut (un 0 qui voulait dire « non vérifié ») ; une erreur bloquante mypy
  avorte tout le run ; Python 3.14 accepte `except A, e:` (PEP 758) — observation non reliée à
  l'écart de couverture 3.12/3.14, hypothèse non testée. Couplage vérifié contre `grep`
  (17 = 17 pour `flask.globals`).
- **Résultats (détail §6) :** vague 1 — aucun outil ne dépasse 1 écart-type sur BASE ; l'entropie
  du changement ne prédit rien ; le fan-in est le meilleur signal individuel. Grille — plateau dès
  4-5 familles, malédiction du gagnant de −0,010 mesurée, sélection instable, **quatre modèles
  indiscernables sur la réserve**. Le seul gain régulier est le churn (suspect, §6).
- Aucun résultat n'a été retouché après coup ; les écarts au protocole sont listés en §6.
- **Extension 6A ajoutée au §8** (transférabilité inter-dépôts, courbe d'apprentissage) à la
  demande de l'utilisateur. **Protocole, jeu de familles et liste de 7 dépôts validés par
  l'utilisateur et figés dans le commit de cette session, avant toute mesure.** Implémentation
  non commencée.
- **Reste à faire :** intégrer les 5 familles dans `results/benchmark.py` (non fait) ; vague 2
  (échantillon) et PySZZ ; commit de cet état ; valider puis démarrer 6A (nécessite de figer le
  jeu de familles du 4bis).

### 2026-11-05 — Session 6 — Extension 6A : construction du jeu multi-dépôts et évaluation
- Reprise après plusieurs interruptions : les calculs en arrière-plan mouraient avec la machine.
  L'utilisateur a indiqué la cause réelle (RAM saturée par 7 processus parallèles) ; diagnostic
  par profilage mémoire par étape → Ruff (§7 point 15). Correctif en lots, équivalence stricte
  vérifiée sur click, exécution séquentielle avec garde-fou mémoire.
- Incident antérieur de la même session : radon bloquait sur black (fichiers générés de
  `profiling/`, > 20 s chacun, §7 point 14) → plafond de 10 s par fichier = NaN, global à tous les
  dépôts ; flask/requests/click recalculés avec le même code.
- **Résultats (détail §6) :** T1 ✅ (médiane AUC(G) 0,745), T2 ❌ sur la configuration principale
  (+0,009 ; +0,065 en brut sans churn), T3 médiane n* = 200 (bruité), normalisation percentile
  sans gain, churn ne conduit pas le transfert, **probabilités brutes mal calibrées (Brier pire
  que le prédicteur constant) mais classement bon (lift ≈ 3×)**.
- Flask retrouve exactement la référence (3198 étiquetés, 310 positifs, 0 dépassement radon).
- Aucun résultat retouché après coup ; configuration principale et règles de T1/T2/T3 fixées avant
  la première évaluation (§8 6A, précisions pré-enregistrées).
- Jeu multi-dépôts exporté pour le cours : `results/dataset_6a.csv.gz` (portable, 25 179 lignes,
  7 dépôts, 56 colonnes dont les versions `__pct`).
- **Reste à faire :** intégrer les familles du 4bis dans `results/benchmark.py` ; vague 2 du 4bis
  (échantillon) et PySZZ ; recalibration locale (la suite logique du 6A : estimer combien de
  commits étiquetés suffisent à recalibrer) ; Chantier 5 ; Chantier 7.
