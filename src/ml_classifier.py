"""
Traditional machine learning classifiers for vibration fault diagnosis.

Reference implementations:
  - Full sklearn classifier framework:
    https://github.com/fboldt/bearing-fault-diagnosis
  - SVM/KNN/XGBoost on CWRU with cross-validation:
    https://github.com/LGDiMaggio/CWRU-bearing-fault-classification-ML

Classifiers:
  - SVM (RBF kernel): sklearn.svm.SVC with GridSearchCV
  - Random Forest: sklearn.ensemble.RandomForestClassifier
  - XGBoost: xgboost.XGBClassifier
  - KNN (k=5): sklearn.neighbors.KNeighborsClassifier
  - LDA (baseline): sklearn.discriminant_analysis.LinearDiscriminantAnalysis

Unified interface:
  train_classifier(X_train, y_train, method='svm', **kwargs) -> model
  evaluate_classifier(model, X_test, y_test) -> dict
"""

import os
import hashlib
import warnings
import numpy as np


_MODEL_CACHE_DIR = "outputs/models"


def _model_cache_path(method, X_train_hash, **kwargs):
    suffix = hashlib.md5(X_train_hash.encode()).hexdigest()[:12]
    return os.path.join(_MODEL_CACHE_DIR, f"{method}_{suffix}.joblib")


def _hash_array(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()


def load_or_train_classifier(X_train, y_train, method="svm",
                              cache_dir=None, force_retrain=False, **kwargs):
    """
    Load cached model or train a new one, with joblib persistence.

    Parameters
    ----------
    X_train : ndarray (n_samples, n_features)
    y_train : ndarray (n_samples,)
    method : str
        "svm" | "rf" | "xgb" | "knn" | "lda"
    cache_dir : str or None
        Directory for cached models. None uses outputs/models/.
    force_retrain : bool
        If True, ignore cache and retrain.
    **kwargs :
        Passed to the classifier constructor.

    Returns
    -------
    model : trained sklearn-compatible model.
    """
    import joblib

    if cache_dir is None:
        cache_dir = _MODEL_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)

    h = _hash_array(np.concatenate([X_train.ravel(), y_train.ravel()]))
    cache_path = os.path.join(cache_dir, f"{method}_{h[:12]}.joblib")

    if not force_retrain and os.path.exists(cache_path):
        try:
            return joblib.load(cache_path)
        except Exception:
            pass

    model = train_classifier(X_train, y_train, method=method, **kwargs)
    joblib.dump(model, cache_path)
    return model


def train_classifier(X_train, y_train, method="svm", **kwargs):
    """
    Train a traditional ML classifier.

    Parameters
    ----------
    X_train : ndarray (n_samples, n_features)
        Training features.
    y_train : ndarray (n_samples,)
        Training labels.
    method : str
        "svm" — SVM with RBF kernel + GridSearchCV over C, gamma
        "rf"  — Random Forest (n_estimators=200)
        "xgb" — XGBoost
        "knn" — K-Nearest Neighbors (k=5)
        "lda" — Linear Discriminant Analysis (baseline)
    **kwargs :
        Override default hyperparameters for each classifier.

    Returns
    -------
    model : trained sklearn-compatible model.
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder

    # Encode string labels to integers
    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    method = method.lower()

    if method == "svm":
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV

        n_features = X_train.shape[1]
        n_samples = X_train.shape[0]

        if n_samples > 5000:
            # Fall back to fixed parameters for speed
            C_val = kwargs.get("C", 10.0)
            gamma_val = kwargs.get("gamma", 1.0 / max(n_features, 1))
            model = SVC(kernel="rbf", C=C_val, gamma=gamma_val,
                        probability=True, random_state=42)
            model.fit(X_scaled, y_enc)
        elif n_samples < 20:
            # Too few samples for GridSearchCV — use default params
            model = SVC(kernel="rbf", probability=True, random_state=42)
            model.fit(X_scaled, y_enc)
        else:
            param_grid = {
                "C": kwargs.get("C_grid", [0.1, 1, 10, 100]),
                "gamma": kwargs.get("gamma_grid", [1e-3, 1e-2, 0.1, 1.0]),
            }
            svc = SVC(kernel="rbf", probability=True, random_state=42)
            model = GridSearchCV(svc, param_grid, cv=min(5, max(2, n_samples // 10)),
                                  scoring="f1_macro", n_jobs=-1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_scaled, y_enc)
            model = model.best_estimator_

    elif method == "rf":
        from sklearn.ensemble import RandomForestClassifier
        n_estimators = kwargs.get("n_estimators", 200)
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=42,
            n_jobs=-1, **{k: v for k, v in kwargs.items()
                          if k != "n_estimators"}
        )
        model.fit(X_scaled, y_enc)

    elif method == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError(
                "XGBoost requires xgboost. Install with:\n"
                "  pip install xgboost\n"
                "GitHub: https://github.com/dmlc/xgboost"
            )
        n_estimators = kwargs.get("n_estimators", 200)
        model = XGBClassifier(
            n_estimators=n_estimators, use_label_encoder=False,
            eval_metric="mlogloss", random_state=42,
            **{k: v for k, v in kwargs.items()
               if k not in ("n_estimators",)}
        )
        model.fit(X_scaled, y_enc)

    elif method == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        n_neighbors = kwargs.get("n_neighbors", 5)
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            **{k: v for k, v in kwargs.items() if k != "n_neighbors"}
        )
        model.fit(X_scaled, y_enc)

    elif method == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        model = LinearDiscriminantAnalysis(**kwargs)
        model.fit(X_scaled, y_enc)

    else:
        raise ValueError(
            f"Unknown classifier method: {method}. "
            f"Supported: svm, rf, xgb, knn, lda"
        )

    # Attach preprocessor extras to model
    model._scaler = scaler
    model._label_encoder = le
    model._method = method

    return model


def evaluate_classifier(model, X_test, y_test):
    """
    Evaluate a trained classifier on test data.

    Parameters
    ----------
    model : sklearn-compatible model
        Trained model (from train_classifier).
    X_test : ndarray (n_samples, n_features)
        Test features.
    y_test : ndarray (n_samples,)
        True labels.

    Returns
    -------
    metrics : dict
        {"accuracy": float, "f1_macro": float, "confusion_matrix": ndarray,
         "predictions": ndarray, "true_labels": ndarray}
    """
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    scaler = getattr(model, "_scaler", None)
    le = getattr(model, "_label_encoder", None)

    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    if le is not None:
        y_enc = le.transform(y_test)
    else:
        y_enc = y_test

    y_pred_enc = model.predict(X_test_scaled)

    if le is not None:
        y_pred = le.inverse_transform(y_pred_enc)
        y_true = y_test
    else:
        y_pred = y_pred_enc
        y_true = y_test

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # Use encoded labels for confusion matrix
    cm = confusion_matrix(y_enc, y_pred_enc)

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "confusion_matrix": cm,
        "predictions": y_pred,
        "true_labels": y_true,
        "label_names": list(le.classes_) if le is not None else None,
    }
