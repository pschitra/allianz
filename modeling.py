from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = [
    "age_at_offer",
    "premium",
    "vehicle_age",
    "URB",
    "INCOME",
    "EDU_HIGH",
    "DINK",
    "OWN_HOUSE",
    "AVG_HOUSE",
    "STAGE_OF_LIFE",
    "SAVINGS",
    "SHOP_ONLINE",
    "CAR",
]

CATEGORICAL_FEATURES = ["cover_type", "brand", "PROVINCE", "buildmonth_car"]


@dataclass
class ClassificationResult:
    pipeline: Pipeline
    model_name: str
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    probabilities: np.ndarray
    metrics: dict[str, float]
    confusion: np.ndarray
    roc: pd.DataFrame
    feature_importance: pd.DataFrame


@dataclass
class SegmentationResult:
    assignments: pd.DataFrame
    profile: pd.DataFrame
    pca_explained_variance: float


def _available_features(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [col for col in NUMERIC_FEATURES if col in df.columns]
    categorical = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    return numeric, categorical


def _reduce_brand_cardinality(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    out = df.copy()
    if "brand" in out.columns:
        top = out["brand"].value_counts(dropna=True).head(top_n).index
        out["brand"] = out["brand"].where(out["brand"].isin(top), "Other")
    return out


def build_classifier(model_name: str, random_state: int = 42) -> object:
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=9,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=3,
            random_state=random_state,
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name]


def _build_pipeline(df: pd.DataFrame, model_name: str, random_state: int) -> tuple[Pipeline, list[str], list[str]]:
    numeric, categorical = _available_features(df)
    if not numeric and not categorical:
        raise ValueError("No usable model features were found.")

    transformers = []
    if numeric:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric))

    if categorical:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", build_classifier(model_name, random_state=random_state)),
        ]
    )
    return pipeline, numeric, categorical


def _feature_importance(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    try:
        names = preprocessor.get_feature_names_out()
    except Exception:
        return pd.DataFrame(columns=["feature", "importance"])

    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_).ravel())
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance = pd.DataFrame({"feature": names, "importance": values})
    importance["feature"] = (
        importance["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
        .str.replace("_", " ")
    )
    return importance.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def evaluate_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> tuple[dict[str, float], np.ndarray]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }
    return metrics, confusion_matrix(y_true, predictions, labels=[0, 1])


def train_classifier(
    df: pd.DataFrame,
    model_name: str = "Logistic Regression",
    threshold: float = 0.50,
    test_size: float = 0.25,
    random_state: int = 42,
) -> ClassificationResult:
    working = _reduce_brand_cardinality(df)
    numeric, categorical = _available_features(working)
    features = numeric + categorical
    model_df = working.loc[working["converted"].isin([0, 1]), features + ["converted"]].copy()
    if len(model_df) < 100 or model_df["converted"].nunique() < 2:
        raise ValueError("Not enough labelled observations to train the model.")

    X = model_df[features]
    y = model_df["converted"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    pipeline, _, _ = _build_pipeline(model_df, model_name, random_state)
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics, confusion = evaluate_threshold(y_test, probabilities, threshold)
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    roc = pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr, "threshold": thresholds})

    return ClassificationResult(
        pipeline=pipeline,
        model_name=model_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        probabilities=probabilities,
        metrics=metrics,
        confusion=confusion,
        roc=roc,
        feature_importance=_feature_importance(pipeline),
    )


def segment_policyholders(
    policies: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> SegmentationResult:
    feature_map = {
        "policyholder_age": "Age",
        "total_annual_premium": "Annual premium",
        "worth_car": "Car value",
        "bonus_malus_percent": "Bonus-malus",
        "policy_tenure_years": "Policy tenure",
        "vehicle_age": "Vehicle age",
    }
    features = [col for col in feature_map if col in policies.columns]
    working = policies[features].copy()
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    matrix = scaler.fit_transform(imputer.fit_transform(working))

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(matrix)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(matrix)

    assignments = policies.copy()
    assignments["segment"] = labels + 1
    assignments["pca_1"] = coords[:, 0]
    assignments["pca_2"] = coords[:, 1]

    profile = (
        assignments.groupby("segment")[features]
        .mean(numeric_only=True)
        .rename(columns=feature_map)
        .round(1)
    )
    profile.insert(0, "Customers", assignments.groupby("segment").size())
    profile = profile.reset_index()

    return SegmentationResult(
        assignments=assignments,
        profile=profile,
        pca_explained_variance=float(pca.explained_variance_ratio_.sum()),
    )
