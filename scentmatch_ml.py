"""Outils communs au notebook, à l'entraînement et à l'application ScentMatch."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata as metadata
import json
import platform
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

SEED = 42
DATASET_REPO = "doevent/perfume"


def reproducibility_metadata(dataset_path: str | Path) -> dict:
    """Décrit les entrées, réglages et versions qui déterminent un entraînement."""
    path = Path(dataset_path)
    return {
        "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "configuration": {
            "seed": SEED,
            "test_size": 0.20,
            "split": "train_test_split avec stratification sur family",
            "cross_validation": "StratifiedKFold(n_splits=3, shuffle=True, random_state=42)",
            "selection_metric": "f1_macro",
            "grid_search_jobs": 1,
            "tfidf_placement": "dans le Pipeline ajuste sur chaque pli d'entraînement",
            "model_selection_uses_test": False,
            "calibration_fraction_train": 0.15,
            "calibration_random_state": SEED + 1,
            "threadpool_limit": 1,
            "tfidf": {
                "min_df": 2, "max_df": 0.98, "max_features": 18000,
                "ngram_range": [1, 2], "sublinear_tf": True,
                "strip_accents": "unicode", "lowercase": True,
            },
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "joblib": metadata.version("joblib"),
            "matplotlib": metadata.version("matplotlib"),
            "threadpoolctl": metadata.version("threadpoolctl"),
        },
    }
FAMILY_FR = {
    "FLORAL": "Floral", "WOODY": "Boisé", "AMBER": "Ambré", "AMBERY": "Ambré",
    "AMBERY (ORIENTAL)": "Ambré / oriental",
    "ORIENTAL": "Ambré / oriental", "CITRUS": "Hespéridé", "CHYPRE": "Chypré",
    "FOUGERE": "Fougère", "AROMATIC FOUGERE": "Fougère aromatique", "LEATHER": "Cuiré",
}

# Dictionnaire volontairement restreint : ce n'est PAS un traducteur généraliste.
# Il améliore seulement la saisie de quelques mots olfactifs fréquents en français.
FR_NOTES = {
    "bois de santal": "sandalwood", "fleur d oranger": "orange blossom",
    "fleur d'oranger": "orange blossom", "feve tonka": "tonka bean",
    "fève tonka": "tonka bean", "poivre rose": "pink pepper",
    "bois de cedre": "cedarwood", "bois de cèdre": "cedarwood",
    "bergamote": "bergamot", "vanille": "vanilla", "ambre": "amber", "cedre": "cedar",
    "cèdre": "cedar", "musc": "musk", "rose": "rose", "jasmin": "jasmine",
    "lavande": "lavender", "citron": "lemon", "orange": "orange", "patchouli": "patchouli",
    "vetiver": "vetiver", "vétiver": "vetiver", "epice": "spicy", "épice": "spicy",
    "cannelle": "cinnamon", "cuir": "leather", "boise": "woody", "boisé": "woody",
    "floral": "floral", "frais": "fresh", "sucre": "sweet", "sucré": "sweet",
    "fruité": "fruity", "fruite": "fruity", "encens": "incense", "santal": "sandalwood",
    "notes marines": "marine notes", "pamplemousse": "grapefruit", "menthe": "mint",
}


def normalize_text(value: object, *, translate_notes: bool = False) -> str:
    """Normalise une description sans inclure la réponse (family/subfamily)."""
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return ""
    s = str(value).strip().lower()
    if translate_notes:
        for fr, en in sorted(FR_NOTES.items(), key=lambda item: -len(item[0])):
            s = re.sub(r"(?<!\w)" + re.escape(fr) + r"(?!\w)", en, s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s'-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_ingredients(value: object) -> str:
    """Accepte une liste Python/JSON ou une chaîne issue du CSV."""
    if isinstance(value, (list, tuple)):
        return " ".join(str(x) for x in value if x is not None)
    if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "[]"}:
        return ""
    if s.startswith("["):
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, (list, tuple)):
                return " ".join(str(x) for x in obj if x is not None)
        except (ValueError, SyntaxError):
            try:
                obj = json.loads(s)
                if isinstance(obj, list):
                    return " ".join(str(x) for x in obj if x is not None)
            except (ValueError, TypeError):
                pass
    return s.replace("[", " ").replace("]", " ").replace("'", " ").replace('"', " ")


def obtain_dataset(local_path: str | Path = "data/perfumes.csv") -> Path:
    """Privilégie un CSV local ; sinon télécharge depuis Hugging Face."""
    local_path = Path(local_path)
    if local_path.exists():
        return local_path.resolve()
    try:
        from huggingface_hub import hf_hub_download
        cached = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename="perfumes.csv")
        return Path(cached)
    except Exception as exc:
        raise RuntimeError(
            "Le dataset n'a pas pu être téléchargé. Téléchargez perfumes.csv depuis "
            "https://huggingface.co/datasets/doevent/perfume/tree/main "
            "et placez-le dans le dossier data/ du projet. "
            f"Détail : {exc}"
        ) from exc


def load_raw_data(local_path: str | Path = "data/perfumes.csv") -> pd.DataFrame:
    path = obtain_dataset(local_path)
    # Le fichier officiel est séparé par des barres verticales (|), non des virgules.
    # Détection du séparateur pour accepter aussi une éventuelle version CSV classique.
    with open(path, "r", encoding="utf-8-sig") as fh:
        header = fh.readline()
    separator = "|" if header.count("|") > header.count(",") else ","
    try:
        df = pd.read_csv(path, sep=separator, low_memory=False, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=separator, encoding="latin1", low_memory=False)
    df.columns = df.columns.astype(str).str.strip().str.lower()
    required = {"family", "description", "ingredients"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colonnes requises absentes : {sorted(missing)} ; colonnes trouvées : {list(df.columns)}")
    return df


def prepare_dataset(raw: pd.DataFrame, *, min_class_count: int = 30) -> tuple[pd.DataFrame, dict]:
    """Nettoie sans jamais introduire family, subfamily ou fragrances dans X.

    Un texte identique portant plusieurs étiquettes est écarté ; cela évite
    d'avoir la même entrée à la fois en train et en test.
    """
    df = raw.copy()
    start = len(df)
    df["family"] = df["family"].fillna("").astype(str).str.strip().str.upper()
    df["description"] = df["description"].fillna("").astype(str)
    df["ingredients_text"] = df["ingredients"].apply(parse_ingredients)
    # L'entrée du modèle contient UNIQUEMENT la description et les ingrédients.
    df["text"] = (df["ingredients_text"] + " " + df["description"]).map(normalize_text)
    df = df[(df["family"] != "") & (df["text"].str.split().str.len() >= 3)].copy()
    # S'il existe plusieurs familles pour exactement le même texte, cas ambigu.
    n_families_per_text = df.groupby("text")["family"].transform("nunique")
    ambiguous_count = int((n_families_per_text > 1).sum())
    df = df[n_families_per_text == 1].copy()
    df = df.drop_duplicates(subset=["text"], keep="first").copy()
    counts = df["family"].value_counts()
    df = df[df["family"].isin(counts[counts >= min_class_count].index)].copy()
    df = df.reset_index(drop=True)
    if df.empty or df["family"].nunique() < 2:
        raise ValueError("Données insuffisantes pour la classification après nettoyage.")
    stats = {
        "observations_brutes": int(start),
        "observations_retenues": int(len(df)),
        "observations_ambiguës_ecartees": ambiguous_count,
        "classes": df["family"].value_counts().to_dict(),
        "variables_texte": ["ingredients", "description"],
        "variables_exclues_pour_eviter_la_fuite": ["family", "subfamily", "fragrances"],
    }
    return df, stats


def stratified_split(df: pd.DataFrame, test_size: float = 0.20):
    return train_test_split(
        df["text"], df["family"], test_size=test_size, stratify=df["family"], random_state=SEED
    )


def build_pipeline(name: str, params: dict | None = None) -> Pipeline:
    models = {
        "Naive Bayes": MultinomialNB(alpha=1.0),
        "Régression logistique": LogisticRegression(C=1.0, max_iter=1800, random_state=SEED),
        "SVM linéaire": LinearSVC(C=1.0, random_state=SEED),
    }
    if name not in models:
        raise ValueError(f"Modèle inconnu : {name}")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, max_df=0.98, max_features=18000,
                                  ngram_range=(1, 2), sublinear_tf=True,
                                  strip_accents="unicode", lowercase=True)),
        ("model", models[name]),
    ])
    if params:
        pipe.set_params(**params)
    return pipe


def get_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)


def compare_models(X_train, y_train, *, verbose=True) -> pd.DataFrame:
    rows = []
    with threadpool_limits(limits=1):
        for name in ["Naive Bayes", "Régression logistique", "SVM linéaire"]:
            res = cross_validate(
                build_pipeline(name), X_train, y_train, cv=get_cv(),
                scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"}, n_jobs=1,
                error_score="raise"
            )
            row = {
                "modèle": name,
                "f1_macro_cv": round(float(np.mean(res["test_f1_macro"])), 4),
                "écart_type_cv": round(float(np.std(res["test_f1_macro"])), 4),
                "accuracy_cv": round(float(np.mean(res["test_accuracy"])), 4),
            }
            rows.append(row)
            if verbose:
                print(f"{name:24s} | F1 macro CV : {row['f1_macro_cv']:.4f} | Accuracy : {row['accuracy_cv']:.4f}")
    return pd.DataFrame(rows).sort_values("f1_macro_cv", ascending=False).reset_index(drop=True)


def tune_model(name: str, X_train, y_train):
    grids = {
        "Naive Bayes": {"tfidf__ngram_range": [(1, 1), (1, 2)], "model__alpha": [0.3, 1.0, 2.0]},
        "Régression logistique": {"tfidf__ngram_range": [(1, 1), (1, 2)], "model__C": [0.5, 1.0, 2.0]},
        "SVM linéaire": {"tfidf__ngram_range": [(1, 1), (1, 2)], "model__C": [0.3, 1.0, 3.0]},
    }
    search = GridSearchCV(build_pipeline(name), grids[name], cv=get_cv(), scoring="f1_macro",
                          n_jobs=1, refit=True, error_score="raise", verbose=2)
    with threadpool_limits(limits=1):
        search.fit(X_train, y_train)
    return search


def probability_model(best_pipeline: Pipeline, name: str, X_train, y_train):
    """Le SVM nécessite une calibration pour produire un vecteur de probabilités."""
    if name == "SVM linéaire":
        # Je garde 15 % du Train uniquement pour calibrer les probabilités.
        # Cette étape n'utilise jamais le jeu de test final.
        X_fit, X_cal, y_fit, y_cal = train_test_split(
            X_train, y_train, test_size=0.15, random_state=SEED + 1, stratify=y_train
        )
        with threadpool_limits(limits=1):
            fitted_svm = clone(best_pipeline).fit(X_fit, y_fit)
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(fitted_svm), method="sigmoid"
        )
        with threadpool_limits(limits=1):
            calibrated.fit(X_cal, y_cal)
        return calibrated, "SVM calibré (sigmoïde) sur 15 % du Train réservés à la calibration"
    return best_pipeline, "Probabilités fournies par le classifieur"


def test_report(model, X_test, y_test) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    with threadpool_limits(limits=1):
        pred = model.predict(X_test)
    labels = [str(c) for c in model.classes_]
    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, pred, average="weighted")),
        "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        "classes": labels,
    }
    confusion = pd.DataFrame(confusion_matrix(y_test, pred, labels=labels), index=labels, columns=labels)
    errors = pd.DataFrame({"texte": list(X_test), "réel": list(y_test), "prédit": pred})
    errors = errors[errors["réel"] != errors["prédit"]].reset_index(drop=True)
    return metrics, confusion, errors


def family_display(family: str) -> str:
    key = str(family).strip().upper()
    return FAMILY_FR.get(key, key.title())
