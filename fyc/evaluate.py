"""Evaluation : AUC en validation croisee, split temporel, baseline. Porte les cellules
26, 35, 44 du notebook.
"""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


def _make_model(n_estimators=300, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state,
        class_weight="balanced", n_jobs=-1,
    )


def cv_auc(df, cols, target="risky", n_splits=5, cv_random_state=0, model_random_state=42,
           n_estimators=300):
    """AUC en validation croisee, moyenne et ecart-type."""
    X, y = df[cols], df[target]
    model = _make_model(n_estimators=n_estimators, random_state=model_random_state)
    scores = cross_val_score(
        model, X, y,
        cv=StratifiedKFold(n_splits, shuffle=True, random_state=cv_random_state),
        scoring="roc_auc", n_jobs=-1,
    )
    return float(scores.mean()), float(scores.std())


def temporal_split_auc(df, cols, target="risky", test_frac=0.2, model_random_state=42,
                        n_estimators=300):
    """AUC sur un split chronologique (80% ancien -> train, 20% recent -> test).

    Suppose `df` deja trie chronologiquement (c'est le cas de toutes les fonctions
    de fyc.features, qui parcourent `commits` dans l'ordre).
    """
    cut = int(len(df) * (1 - test_frac))
    Xtr, Xte = df[cols].iloc[:cut], df[cols].iloc[cut:]
    ytr, yte = df[target].iloc[:cut], df[target].iloc[cut:]

    model = _make_model(n_estimators=n_estimators, random_state=model_random_state)
    model.fit(Xtr, ytr)
    return float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))


def baseline_majority_accuracy(df, target="risky", test_frac=0.2, random_state=42):
    """Accuracy d'un DummyClassifier(most_frequent) sur un split aleatoire stratifie."""
    from sklearn.model_selection import train_test_split

    X, y = df.drop(columns=target), df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=random_state, stratify=y,
    )
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    return float(baseline.score(X_test, y_test))
