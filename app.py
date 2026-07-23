from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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

REFERENCE_DATE = pd.Timestamp("2019-10-16")

FUNNEL_FILE = "W27307-XLS-ENG.xlsx"
POLICY_FILE = "W27308-XLS-ENG.xlsx"
REGIONAL_FILE = "W27309-XLS-ENG.xlsx"

STAGE_OF_LIFE_LABELS = {
    1: "Young singles",
    2: "Adult singles",
    3: "Older singles",
    4: "Families with young children",
    6: "Families with older children",
    7: "Young couples without children",
    8: "Adult couples without children",
    9: "Older couples without children",
}

STATUS_STAGE_MAP = {
    "Incompleterequest": "Form incomplete",
    "Calculatenewpremium": "Price calculated / stalled",
    "Policycreated": "Policy created",
    "Personaloffer": "Offer sent / no action",
    "Personalofferrejected": "Offer rejected / withdrawn",
    "Tailoredofferrequested": "Offer / approval journey",
    "Tailoredofferrejected": "Offer rejected / withdrawn",
    "Tailoredofferwithdrawn": "Offer rejected / withdrawn",
    "Adaptedproposalwithdrawn": "Offer rejected / withdrawn",
    "Waitresponsebackoffice": "Back-office / approval",
    "Waitforapproval": "Back-office / approval",
    "Requestaccepted": "Back-office / approval",
    "Requestrejected": "Offer rejected / withdrawn",
    "Requestwithdrawn": "Offer rejected / withdrawn",
}

STAGE_ORDER = [
    "Form incomplete",
    "Price calculated / stalled",
    "Offer sent / no action",
    "Offer / approval journey",
    "Back-office / approval",
    "Offer rejected / withdrawn",
    "Policy created",
]


@dataclass(frozen=True)
class DataBundle:
    funnel: pd.DataFrame
    policies: pd.DataFrame
    regional: pd.DataFrame


def _assert_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _read_case_excel(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path.name}. Keep the three Excel files inside the data/ folder."
        )
    return pd.read_excel(
        path,
        sheet_name=sheet_name,
        na_values=["NA", "N/A", "na", ""],
        keep_default_na=True,
        engine="openpyxl",
    )


def _yes_flag(series: pd.Series) -> pd.Series:
    return series.astype("string").str.upper().eq("Y").fillna(False).to_numpy(dtype=bool)


def _safe_age(later: pd.Series | pd.Timestamp, earlier: pd.Series) -> pd.Series:
    if isinstance(later, pd.Timestamp):
        delta_days = (later - earlier).dt.days
    else:
        delta_days = (later - earlier).dt.days
    age = np.floor(delta_days / 365.25)
    return pd.Series(age, index=earlier.index).where(lambda s: s.between(18, 100))


def prepare_funnel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {
        "affinity_name",
        "status_report",
        "offer_number",
        "zipcode_link",
        "birth_date",
        "date_offer",
        "premium",
        "buildyear_car",
        "wa",
        "wa_bep_ca",
        "wa_ca",
    }
    _assert_columns(df, required, "Funnel data")

    for col in ["birth_date", "date_offer", "date_request", "policy_start_date", "updated_on"]:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    for col in ["premium", "buildyear_car", "zipcode_link", "zip4"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["converted"] = df["status_report"].eq("Policycreated").astype(int)
    df["age_at_offer"] = _safe_age(df["date_offer"], df["birth_date"])
    df["vehicle_age"] = (df["date_offer"].dt.year - df["buildyear_car"]).where(
        lambda s: s.between(0, 60)
    )

    cover_conditions = [_yes_flag(df["wa"]), _yes_flag(df["wa_bep_ca"]), _yes_flag(df["wa_ca"])]
    cover_values = ["Liability only", "Liability + limited casco", "Liability + full casco"]
    df["cover_type"] = np.select(cover_conditions, cover_values, default="Not specified")

    df["age_group"] = pd.cut(
        df["age_at_offer"],
        bins=[17, 29, 39, 49, 59, 69, 100],
        labels=["18-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    )
    df["premium_band"] = pd.cut(
        df["premium"],
        bins=[-0.01, 400, 600, 800, 1000, np.inf],
        labels=["€0-400", "€401-600", "€601-800", "€801-1,000", "€1,001+"],
    )
    df["offer_month"] = df["date_offer"].dt.to_period("M").dt.to_timestamp()
    df["journey_status"] = df["status_report"].map(STATUS_STAGE_MAP).fillna("Other")
    return df


def prepare_policies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {
        "policy_number",
        "policy_start_date",
        "premium_wa",
        "premium_other_incl_discount",
        "zipcode_link",
        "birth_date",
        "bonus_malus_percent",
        "other_cover",
        "worth_car",
        "brand",
    }
    _assert_columns(df, required, "Policy data")

    for col in [
        "policy_continuation_date",
        "policy_start_date",
        "policy_lastchange_date",
        "birth_date",
        "builddate_car",
    ]:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    numeric_columns = [
        "premium_wa",
        "premium_other",
        "premium_other_incl_discount",
        "bonus_malus_percent",
        "worth_car",
        "weight_car",
        "mileage_car",
        "power_car",
        "zipcode_link",
    ]
    for col in numeric_columns:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["policyholder_age"] = _safe_age(REFERENCE_DATE, df["birth_date"])
    df["policy_tenure_years"] = ((REFERENCE_DATE - df["policy_start_date"]).dt.days / 365.25).clip(lower=0)
    df["vehicle_age"] = ((REFERENCE_DATE - df["builddate_car"]).dt.days / 365.25).where(
        lambda s: s.between(0, 60)
    )
    df["total_annual_premium"] = df[["premium_wa", "premium_other_incl_discount"]].sum(
        axis=1, min_count=1
    )
    df["age_group"] = pd.cut(
        df["policyholder_age"],
        bins=[17, 29, 39, 49, 59, 69, 100],
        labels=["18-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    )
    df["cover_label"] = df["other_cover"].map(
        {"BEP": "Limited casco", "CAS": "Full casco"}
    ).fillna("Liability only / no add-on")
    df["car_value_band"] = pd.cut(
        df["worth_car"],
        bins=[-0.01, 10000, 20000, 35000, 50000, np.inf],
        labels=["≤€10k", "€10-20k", "€20-35k", "€35-50k", "€50k+"],
    )
    return df


def prepare_regional(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = {
        "zipcode_link",
        "PROVINCE",
        "URB",
        "INCOME",
        "STAGE_OF_LIFE",
        "SAVINGS",
        "SHOP_ONLINE",
        "CAR",
    }
    _assert_columns(df, required, "Regional data")

    for col in df.columns:
        if col not in {"PROVINCE"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["life_stage_label"] = df["STAGE_OF_LIFE"].map(STAGE_OF_LIFE_LABELS).fillna("Unknown")
    df["urbanization_label"] = df["URB"].map(
        {
            1: "Very highly urban",
            2: "Highly urban",
            3: "Urban",
            4: "Moderately urban",
            5: "Less urban",
            6: "Rural",
            7: "Very rural",
        }
    ).fillna("Unknown")
    df["income_label"] = df["INCOME"].map(
        {1: "High", 2: "Upper-middle", 3: "Middle", 4: "Lower-middle", 5: "Minimal", 6: "Mixed"}
    ).fillna("Unknown")
    return df


def load_case_data(data_dir: str | Path) -> DataBundle:
    data_dir = Path(data_dir)
    funnel = _read_case_excel(data_dir / FUNNEL_FILE, "funnel_data")
    policies = _read_case_excel(data_dir / POLICY_FILE, "policies_data")
    regional = _read_case_excel(data_dir / REGIONAL_FILE, "regional_data")
    return DataBundle(
        funnel=prepare_funnel(funnel),
        policies=prepare_policies(policies),
        regional=prepare_regional(regional),
    )


def make_modeling_frame(funnel: pd.DataFrame, regional: pd.DataFrame, affinity: str = "T&B") -> pd.DataFrame:
    base = funnel.loc[funnel["affinity_name"].eq(affinity)].copy()
    regional_features = [
        "zipcode_link",
        "PROVINCE",
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
    available = [c for c in regional_features if c in regional.columns]
    merged = base.merge(regional[available], on="zipcode_link", how="left", validate="m:1")
    return merged


def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(dtype) for dtype in df.dtypes],
                "missing_count": df.isna().sum().to_numpy(),
                "missing_percent": (df.isna().mean().to_numpy() * 100).round(1),
                "unique_values": [df[col].nunique(dropna=True) for col in df.columns],
            }
        )
        .sort_values(["missing_percent", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )


def channel_summary(funnel: pd.DataFrame) -> pd.DataFrame:
    summary = (
        funnel.groupby("affinity_name", dropna=False)
        .agg(quotes=("offer_number", "count"), policies=("converted", "sum"), avg_premium=("premium", "mean"))
        .reset_index()
    )
    summary["conversion_rate"] = np.where(summary["quotes"] > 0, summary["policies"] / summary["quotes"], 0)
    return summary.sort_values("conversion_rate", ascending=False).reset_index(drop=True)


def tb_funnel_counts(funnel: pd.DataFrame) -> pd.DataFrame:
    tb = funnel.loc[funnel["affinity_name"].eq("T&B")]
    started = len(tb)
    completed_request = started - int(tb["status_report"].eq("Incompleterequest").sum())
    moved_beyond_price = completed_request - int(tb["status_report"].eq("Calculatenewpremium").sum())
    policies = int(tb["converted"].sum())
    return pd.DataFrame(
        {
            "stage": ["Quote requests started", "Request completed", "Moved beyond price", "Policy created"],
            "customers": [started, completed_request, moved_beyond_price, policies],
        }
    )


def monthly_performance(funnel: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        funnel.dropna(subset=["offer_month"])
        .groupby(["offer_month", "affinity_name"])
        .agg(quotes=("offer_number", "count"), policies=("converted", "sum"))
        .reset_index()
    )
    monthly["conversion_rate"] = np.where(monthly["quotes"] > 0, monthly["policies"] / monthly["quotes"], 0)
    return monthly


def grouped_conversion(df: pd.DataFrame, group_col: str, min_quotes: int = 20) -> pd.DataFrame:
    grouped = (
        df.groupby(group_col, observed=False, dropna=False)
        .agg(quotes=("offer_number", "count"), policies=("converted", "sum"), avg_premium=("premium", "mean"))
        .reset_index()
    )
    grouped["conversion_rate"] = np.where(grouped["quotes"] > 0, grouped["policies"] / grouped["quotes"], 0)
    return grouped.loc[grouped["quotes"].ge(min_quotes)].copy()


def status_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    status = df["journey_status"].value_counts(dropna=False).rename_axis("journey_status").reset_index(name="quotes")
    status["share"] = status["quotes"] / status["quotes"].sum()
    return status


def top_brand_conversion(df: pd.DataFrame, top_n: int = 12, min_quotes: int = 30) -> pd.DataFrame:
    brands = grouped_conversion(df.dropna(subset=["brand"]), "brand", min_quotes=min_quotes)
    return brands.sort_values(["quotes", "conversion_rate"], ascending=[False, False]).head(top_n)


def customer_profile_summary(funnel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for converted, label in [(1, "Converted"), (0, "Not converted")]:
        part = funnel.loc[funnel["converted"].eq(converted)]
        rows.append(
            {
                "outcome": label,
                "quotes": len(part),
                "average_age": part["age_at_offer"].mean(),
                "median_age": part["age_at_offer"].median(),
                "average_premium": part["premium"].mean(),
                "median_premium": part["premium"].median(),
            }
        )
    return pd.DataFrame(rows)


def policy_kpis(policies: pd.DataFrame) -> dict[str, float]:
    return {
        "active_policies": float(len(policies)),
        "median_age": float(policies["policyholder_age"].median()),
        "avg_total_premium": float(policies["total_annual_premium"].mean()),
        "median_car_value": float(policies["worth_car"].median()),
        "avg_tenure": float(policies["policy_tenure_years"].mean()),
    }


def regional_opportunity(regional: pd.DataFrame) -> pd.DataFrame:
    df = regional.copy()

    def normalize(series: pd.Series, reverse: bool = False) -> pd.Series:
        s = pd.to_numeric(series, errors="coerce")
        lo, hi = s.min(), s.max()
        if pd.isna(lo) or pd.isna(hi) or hi == lo:
            out = pd.Series(0.5, index=s.index)
        else:
            out = (s - lo) / (hi - lo)
        return 1 - out if reverse else out

    # Income is coded 1=high and 5=minimal; reverse it so a higher score means stronger spending power.
    df["opportunity_score"] = (
        0.40 * normalize(df["SHOP_ONLINE"])
        + 0.25 * normalize(df["CAR"])
        + 0.20 * normalize(df["INCOME"], reverse=True)
        + 0.15 * normalize(df["SAVINGS"])
    ) * 100
    df["opportunity_score"] = df["opportunity_score"].round(1)
    return df


def province_summary(regional: pd.DataFrame) -> pd.DataFrame:
    scored = regional_opportunity(regional)
    return (
        scored.groupby("PROVINCE", dropna=False)
        .agg(
            postal_areas=("zipcode_link", "count"),
            online_shopping=("SHOP_ONLINE", "mean"),
            car_ownership=("CAR", "mean"),
            income_code=("INCOME", "mean"),
            opportunity_score=("opportunity_score", "mean"),
        )
        .reset_index()
        .sort_values("opportunity_score", ascending=False)
    )


def incremental_policy_scenario(funnel: pd.DataFrame, target_conversion: float) -> dict[str, float]:
    tb = funnel.loc[funnel["affinity_name"].eq("T&B")]
    quotes = len(tb)
    current_policies = int(tb["converted"].sum())
    target_policies = int(round(quotes * target_conversion))
    incremental = max(target_policies - current_policies, 0)
    avg_converter_premium = tb.loc[tb["converted"].eq(1), "premium"].mean()
    estimated_premium = incremental * (avg_converter_premium if pd.notna(avg_converter_premium) else 0)
    return {
        "quotes": quotes,
        "current_policies": current_policies,
        "target_policies": target_policies,
        "incremental_policies": incremental,
        "avg_converter_premium": float(avg_converter_premium),
        "estimated_incremental_premium": float(estimated_premium),
    }


def apply_filters(
    df: pd.DataFrame,
    affinities: Iterable[str] | None = None,
    statuses: Iterable[str] | None = None,
    covers: Iterable[str] | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    age_range: tuple[float, float] | None = None,
    premium_range: tuple[float, float] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if affinities:
        out = out.loc[out["affinity_name"].isin(list(affinities))]
    if statuses:
        out = out.loc[out["status_report"].isin(list(statuses))]
    if covers:
        out = out.loc[out["cover_type"].isin(list(covers))]
    if date_range:
        start, end = date_range
        out = out.loc[out["date_offer"].between(pd.Timestamp(start), pd.Timestamp(end), inclusive="both")]
    if age_range:
        lo, hi = age_range
        out = out.loc[out["age_at_offer"].between(lo, hi, inclusive="both") | out["age_at_offer"].isna()]
    if premium_range:
        lo, hi = premium_range
        out = out.loc[out["premium"].between(lo, hi, inclusive="both") | out["premium"].isna()]
    return out


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


def resolve_data_dir(root: Path) -> Path:
    """Locate the three case workbooks in either data/ or the repository root."""
    required = {FUNNEL_FILE, POLICY_FILE, REGIONAL_FILE}
    for candidate in (root / "data", root):
        if all((candidate / filename).exists() for filename in required):
            return candidate
    # Preserve the clearest error path for the loader if files are missing.
    return root / "data"


ROOT = Path(__file__).resolve().parent
DATA_DIR = resolve_data_dir(ROOT)

st.set_page_config(
    page_title="Allianz CVM Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
:root {
    --navy: #003781;
    --navy-dark: #001f4d;
    --cyan: #00a6d6;
    --sky: #eaf7fc;
    --ink: #152238;
    --muted: #617189;
    --border: #dce5ef;
    --surface: #ffffff;
}
html, body, [class*="css"] {font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;}
[data-testid="stAppViewContainer"] {background: linear-gradient(180deg, #f7faff 0%, #ffffff 32%);}
[data-testid="stSidebar"] {background: linear-gradient(180deg, var(--navy-dark) 0%, var(--navy) 100%);}
[data-testid="stSidebar"] * {color: #ffffff;}
[data-testid="stSidebar"] [data-baseweb="radio"] label {padding: 0.35rem 0.2rem;}
.block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1500px;}
.hero {
    position: relative;
    overflow: hidden;
    padding: 1.8rem 2rem;
    border-radius: 24px;
    background: linear-gradient(120deg, #002f73 0%, #0050a4 56%, #00a6d6 140%);
    box-shadow: 0 18px 45px rgba(0, 55, 129, 0.18);
    color: white;
    margin-bottom: 1.25rem;
}
.hero:after {
    content: "";
    position: absolute;
    width: 280px;
    height: 280px;
    border-radius: 50%;
    right: -100px;
    top: -130px;
    border: 36px solid rgba(255,255,255,0.12);
}
.hero h1 {font-size: 2.15rem; line-height: 1.1; margin: 0 0 0.5rem 0; color: white;}
.hero p {font-size: 1rem; max-width: 820px; margin: 0; color: rgba(255,255,255,0.88);}
.eyebrow {text-transform: uppercase; font-size: 0.72rem; letter-spacing: 0.14em; font-weight: 800; opacity: 0.8; margin-bottom: 0.65rem;}
.kpi-card {
    background: rgba(255,255,255,0.96);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    min-height: 126px;
    box-shadow: 0 8px 24px rgba(21,34,56,0.07);
}
.kpi-label {font-size: 0.77rem; color: var(--muted); font-weight: 750; text-transform: uppercase; letter-spacing: 0.06em;}
.kpi-value {font-size: 1.85rem; color: var(--navy); font-weight: 800; margin-top: 0.28rem; line-height: 1.1;}
.kpi-note {font-size: 0.78rem; color: var(--muted); margin-top: 0.38rem;}
.insight-card {
    background: linear-gradient(135deg, #eef8ff 0%, #ffffff 100%);
    border: 1px solid #cfe7f7;
    border-left: 5px solid var(--cyan);
    border-radius: 16px;
    padding: 1rem 1.2rem;
    color: var(--ink);
    margin: 0.4rem 0 1rem 0;
}
.insight-card b {color: var(--navy);}
.section-title {font-size: 1.25rem; font-weight: 800; color: var(--ink); margin: 1rem 0 0.15rem 0;}
.section-subtitle {font-size: 0.87rem; color: var(--muted); margin-bottom: 0.8rem;}
.small-note {font-size: 0.78rem; color: var(--muted);}
hr {border-color: #e8eef5 !important;}
[data-testid="stMetric"] {background: white; border: 1px solid var(--border); border-radius: 16px; padding: 0.8rem 1rem;}
.stTabs [data-baseweb="tab-list"] {gap: 0.45rem;}
.stTabs [data-baseweb="tab"] {border-radius: 12px; padding: 0.55rem 0.95rem; background: #edf3f9;}
.stTabs [aria-selected="true"] {background: #dceeff; color: var(--navy); font-weight: 750;}
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_bundle():
    return load_case_data(DATA_DIR)


@st.cache_resource(show_spinner=False)
def cached_classifier(model_df: pd.DataFrame, model_name: str, test_size: float, random_state: int):
    return train_classifier(
        model_df,
        model_name=model_name,
        threshold=0.50,
        test_size=test_size,
        random_state=random_state,
    )


@st.cache_resource(show_spinner=False)
def cached_segmentation(policies: pd.DataFrame, n_clusters: int):
    return segment_policyholders(policies, n_clusters=n_clusters)


def fmt_int(value: float | int) -> str:
    return f"{int(round(value)):,}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    return f"{value * 100:.{decimals}f}%"


def fmt_eur(value: float, decimals: int = 0) -> str:
    if pd.isna(value):
        return "—"
    return f"€{value:,.{decimals}f}"


def kpi(label: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, eyebrow: str = "Customer Value Management") -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def chart_style(fig: go.Figure, height: int = 430, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color="#24344d"),
        title_font=dict(size=18, color="#152238"),
        hoverlabel=dict(bgcolor="white", font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1) if legend else None,
    )
    fig.update_xaxes(showgrid=False, linecolor="#dfe7f0", tickfont=dict(size=11))
    fig.update_yaxes(gridcolor="#e9eff5", zeroline=False, tickfont=dict(size=11))
    return fig


def sidebar_brand() -> None:
    st.sidebar.markdown(
        """
        <div style="padding: 0.4rem 0 1.1rem 0;">
          <div style="font-size:1.25rem;font-weight:850;letter-spacing:-0.02em;">Allianz CVM Lab</div>
          <div style="font-size:0.76rem;opacity:0.75;margin-top:0.25rem;">Acquisition intelligence dashboard</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def overview_page(funnel: pd.DataFrame, policies: pd.DataFrame, regional: pd.DataFrame) -> None:
    hero(
        "From quote traffic to customer value",
        "An interactive diagnosis of Allianz Benelux's T&B digital acquisition funnel, customer profile, geo-demographic opportunity and machine-learning strategy.",
    )

    channels = channel_summary(funnel)
    tb_row = channels.loc[channels["affinity_name"].eq("T&B")].iloc[0]
    benchmark = channels.loc[channels["affinity_name"].isin(["Insuro", "Seguros International Ltd."]), "conversion_rate"].mean()
    scenario_18 = incremental_policy_scenario(funnel, 0.18)

    cols = st.columns(4)
    with cols[0]:
        kpi("Total quote records", fmt_int(len(funnel)), "Across four affinity/channel groups")
    with cols[1]:
        kpi("T&B conversion", fmt_pct(tb_row["conversion_rate"]), f"{fmt_int(tb_row['policies'])} policies from {fmt_int(tb_row['quotes'])} quotes")
    with cols[2]:
        kpi("Gap to sister channels", f"{(benchmark - tb_row['conversion_rate']) * 100:.1f} pp", "Insuro and Seguros average about 25%")
    with cols[3]:
        kpi("Policies at 18%", f"+{fmt_int(scenario_18['incremental_policies'])}", "Incremental policies from current T&B traffic")

    st.markdown(
        """
        <div class="insight-card"><b>Executive read:</b> T&B generates the most quote traffic but converts the least. The highest-return move is to repair the form, pricing hand-off and follow-up before buying more traffic.</div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.15, 0.85])
    with left:
        fig = go.Figure()
        ordered = channels.sort_values("quotes", ascending=False)
        fig.add_bar(
            x=ordered["affinity_name"],
            y=ordered["quotes"],
            name="Quote volume",
            marker_color="#b9dff2",
            hovertemplate="%{x}<br>Quotes: %{y:,}<extra></extra>",
        )
        fig.add_scatter(
            x=ordered["affinity_name"],
            y=ordered["conversion_rate"] * 100,
            name="Conversion rate",
            mode="lines+markers",
            marker=dict(size=10, color="#003781"),
            line=dict(width=3, color="#003781"),
            yaxis="y2",
            hovertemplate="%{x}<br>Conversion: %{y:.1f}%<extra></extra>",
        )
        fig.update_layout(
            title="High traffic, weak close rate",
            yaxis=dict(title="Quotes"),
            yaxis2=dict(title="Conversion", overlaying="y", side="right", ticksuffix="%", range=[0, max(30, ordered["conversion_rate"].max() * 120)]),
            barmode="group",
        )
        st.plotly_chart(chart_style(fig), use_container_width=True, config={"displayModeBar": False})

    with right:
        funnel_counts = tb_funnel_counts(funnel)
        fig = go.Figure(
            go.Funnel(
                y=funnel_counts["stage"],
                x=funnel_counts["customers"],
                textinfo="value+percent initial",
                marker=dict(color=["#003781", "#0058a8", "#00a6d6", "#7fd0e8"]),
                connector=dict(line=dict(color="#a8b8cb", width=1)),
                hovertemplate="%{y}<br>%{x:,} customers<extra></extra>",
            )
        )
        fig.update_layout(title="T&B acquisition funnel")
        st.plotly_chart(chart_style(fig, legend=False), use_container_width=True, config={"displayModeBar": False})

    section("Conversion scenario simulator", "Test how a modest conversion lift changes policy volume and premium opportunity.")
    c1, c2 = st.columns([0.38, 0.62])
    with c1:
        target_pct = st.slider("Target T&B conversion", min_value=14.0, max_value=25.0, value=18.0, step=0.5) / 100
        scenario = incremental_policy_scenario(funnel, target_pct)
        st.metric("Incremental policies", fmt_int(scenario["incremental_policies"]))
        st.metric("Indicative premium opportunity", fmt_eur(scenario["estimated_incremental_premium"]))
        st.caption("Premium opportunity is a directional estimate using the average premium among current T&B converters, not profit or CLV.")
    with c2:
        sim_data = pd.DataFrame(
            {
                "Scenario": ["Current", f"Target {target_pct:.1%}"],
                "Policies": [scenario["current_policies"], scenario["target_policies"]],
            }
        )
        fig = px.bar(sim_data, x="Scenario", y="Policies", text="Policies", title="Policy volume from the same quote base")
        fig.update_traces(marker_color=["#a8c8e8", "#00a6d6"], texttemplate="%{text:,}", textposition="outside")
        st.plotly_chart(chart_style(fig, height=350, legend=False), use_container_width=True, config={"displayModeBar": False})

    section("Traffic and conversion over time", "Monthly patterns help separate persistent structural leakage from temporary fluctuations.")
    monthly = monthly_performance(funnel)
    selected_channels = st.multiselect(
        "Channels",
        options=channels["affinity_name"].tolist(),
        default=["T&B", "Insuro", "Seguros International Ltd."],
        key="overview_month_channels",
    )
    monthly_view = monthly.loc[monthly["affinity_name"].isin(selected_channels)]
    fig = px.line(
        monthly_view,
        x="offer_month",
        y="conversion_rate",
        color="affinity_name",
        markers=True,
        title="Monthly quote-to-policy conversion",
        labels={"offer_month": "Month", "conversion_rate": "Conversion rate", "affinity_name": "Channel"},
    )
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(chart_style(fig), use_container_width=True, config={"displayModeBar": False})


def funnel_page(funnel: pd.DataFrame) -> None:
    hero(
        "Funnel diagnostics",
        "Filter the acquisition journey, locate the value leakage and compare the customers who buy with those who walk away.",
        eyebrow="Interactive quote journey",
    )

    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            affinities = st.multiselect("Affinity/channel", sorted(funnel["affinity_name"].dropna().unique()), default=["T&B"])
            covers = st.multiselect("Cover selected", sorted(funnel["cover_type"].dropna().unique()), default=[])
        with f2:
            date_min = funnel["date_offer"].min().date()
            date_max = funnel["date_offer"].max().date()
            date_range = st.date_input("Offer date", value=(date_min, date_max), min_value=date_min, max_value=date_max)
            statuses = st.multiselect("Final quote status", sorted(funnel["status_report"].dropna().unique()), default=[])
        with f3:
            ages = funnel["age_at_offer"].dropna()
            age_range = st.slider("Age", int(ages.min()), int(ages.max()), (int(ages.min()), int(ages.max())))
            premiums = funnel["premium"].dropna()
            premium_range = st.slider(
                "Premium (€)",
                float(np.floor(premiums.quantile(0.01))),
                float(np.ceil(premiums.quantile(0.99))),
                (float(np.floor(premiums.quantile(0.01))), float(np.ceil(premiums.quantile(0.99)))),
                step=25.0,
            )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        parsed_date_range = (pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1))
    else:
        parsed_date_range = None

    filtered = apply_filters(
        funnel,
        affinities=affinities,
        statuses=statuses,
        covers=covers,
        date_range=parsed_date_range,
        age_range=age_range,
        premium_range=premium_range,
    )
    if filtered.empty:
        st.warning("No quotes match the selected filters.")
        return

    quote_count = len(filtered)
    policy_count = int(filtered["converted"].sum())
    conv = policy_count / quote_count if quote_count else 0
    avg_age = filtered["age_at_offer"].mean()
    avg_premium = filtered.loc[filtered["premium"].gt(0), "premium"].mean()

    cols = st.columns(4)
    with cols[0]: kpi("Filtered quotes", fmt_int(quote_count), "Current interactive selection")
    with cols[1]: kpi("Policies created", fmt_int(policy_count), "Final status = Policycreated")
    with cols[2]: kpi("Conversion", fmt_pct(conv), "Quote-to-policy rate")
    with cols[3]: kpi("Average quoted premium", fmt_eur(avg_premium), f"Average customer age: {avg_age:.1f}")

    t1, t2 = st.tabs(["Leakage map", "Customer signals"])
    with t1:
        left, right = st.columns([0.9, 1.1])
        with left:
            status = status_breakdown(filtered)
            status["journey_status"] = pd.Categorical(status["journey_status"], categories=STAGE_ORDER + ["Other"], ordered=True)
            status = status.sort_values("journey_status")
            fig = px.bar(
                status,
                x="quotes",
                y="journey_status",
                orientation="h",
                text="quotes",
                title="Where quotes finish",
                labels={"quotes": "Quotes", "journey_status": "Journey outcome"},
            )
            fig.update_traces(marker_color="#0058a8", texttemplate="%{text:,}", textposition="outside")
            st.plotly_chart(chart_style(fig, height=470, legend=False), use_container_width=True, config={"displayModeBar": False})
        with right:
            status_raw = filtered["status_report"].value_counts().head(12).rename_axis("status").reset_index(name="quotes")
            fig = px.treemap(status_raw, path=["status"], values="quotes", title="Detailed status mix")
            fig.update_traces(root_color="lightgrey", textinfo="label+value+percent root")
            st.plotly_chart(chart_style(fig, height=470, legend=False), use_container_width=True, config={"displayModeBar": False})

        monthly = (
            filtered.dropna(subset=["offer_month"])
            .groupby("offer_month")
            .agg(quotes=("offer_number", "count"), policies=("converted", "sum"))
            .reset_index()
        )
        monthly["conversion_rate"] = monthly["policies"] / monthly["quotes"]
        fig = go.Figure()
        fig.add_bar(x=monthly["offer_month"], y=monthly["quotes"], name="Quotes", marker_color="#c8e6f3")
        fig.add_scatter(
            x=monthly["offer_month"], y=monthly["conversion_rate"] * 100, name="Conversion", mode="lines+markers",
            line=dict(color="#003781", width=3), marker=dict(size=8), yaxis="y2"
        )
        fig.update_layout(
            title="Monthly traffic and conversion",
            yaxis=dict(title="Quotes"),
            yaxis2=dict(title="Conversion", overlaying="y", side="right", ticksuffix="%"),
        )
        st.plotly_chart(chart_style(fig), use_container_width=True, config={"displayModeBar": False})

    with t2:
        a, b = st.columns(2)
        with a:
            age_conv = grouped_conversion(filtered.dropna(subset=["age_group"]), "age_group", min_quotes=5)
            fig = px.bar(age_conv, x="age_group", y="conversion_rate", text="quotes", title="Conversion by age group")
            fig.update_yaxes(tickformat=".0%")
            fig.update_traces(marker_color="#00a6d6", texttemplate="n=%{text:,}", textposition="outside")
            st.plotly_chart(chart_style(fig), use_container_width=True, config={"displayModeBar": False})
        with b:
            premium_conv = grouped_conversion(filtered.dropna(subset=["premium_band"]), "premium_band", min_quotes=5)
            fig = px.bar(premium_conv, x="premium_band", y="conversion_rate", text="quotes", title="Conversion by premium band")
            fig.update_yaxes(tickformat=".0%")
            fig.update_traces(marker_color="#0058a8", texttemplate="n=%{text:,}", textposition="outside")
            st.plotly_chart(chart_style(fig), use_container_width=True, config={"displayModeBar": False})

        a, b = st.columns(2)
        with a:
            cover = grouped_conversion(filtered, "cover_type", min_quotes=5).sort_values("quotes", ascending=False)
            fig = px.scatter(
                cover,
                x="avg_premium",
                y="conversion_rate",
                size="quotes",
                color="cover_type",
                title="Cover mix: price, volume and conversion",
                labels={"avg_premium": "Average premium (€)", "conversion_rate": "Conversion rate", "quotes": "Quotes"},
                hover_data={"quotes": ":,", "policies": ":,"},
            )
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(chart_style(fig), use_container_width=True, config={"displayModeBar": False})
        with b:
            brands = top_brand_conversion(filtered, top_n=12, min_quotes=max(10, int(len(filtered) * 0.005)))
            fig = px.bar(
                brands.sort_values("conversion_rate"),
                x="conversion_rate",
                y="brand",
                orientation="h",
                text="quotes",
                title="Conversion among high-volume vehicle brands",
            )
            fig.update_xaxes(tickformat=".0%")
            fig.update_traces(marker_color="#003781", texttemplate="n=%{text:,}", textposition="outside")
            st.plotly_chart(chart_style(fig, height=470, legend=False), use_container_width=True, config={"displayModeBar": False})

        profile = customer_profile_summary(filtered)
        st.dataframe(
            profile.style.format(
                {"average_age": "{:.1f}", "median_age": "{:.1f}", "average_premium": "€{:,.0f}", "median_premium": "€{:,.0f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )


def customer_value_page(funnel: pd.DataFrame, policies: pd.DataFrame) -> None:
    hero(
        "Customer value and policy book",
        "Move beyond conversion volume: understand who the active customers are, what they insure and where value can be expanded through coverage and retention.",
        eyebrow="Policyholder intelligence",
    )
    metrics = policy_kpis(policies)
    digital_buyers = funnel.loc[(funnel["affinity_name"].eq("T&B")) & funnel["converted"].eq(1)]
    buyer_age = digital_buyers["age_at_offer"].mean()

    cols = st.columns(5)
    with cols[0]: kpi("Active policies", fmt_int(metrics["active_policies"]), "Outstanding T&B contracts")
    with cols[1]: kpi("Median policyholder age", f"{metrics['median_age']:.0f}", f"Digital converter average: {buyer_age:.0f}")
    with cols[2]: kpi("Average annual premium", fmt_eur(metrics["avg_total_premium"]), "Liability plus discounted add-on")
    with cols[3]: kpi("Median vehicle value", fmt_eur(metrics["median_car_value"]), "Current insured book")
    with cols[4]: kpi("Average policy tenure", f"{metrics['avg_tenure']:.1f} yrs", "As of 16 October 2019")

    st.markdown(
        f"""
        <div class="insight-card"><b>Portfolio tension:</b> the active policy book has a median age of {metrics['median_age']:.0f}, while the average T&B digital buyer is about {buyer_age:.0f}. The digital channel can refresh an ageing book, but acquisition should be paired with coverage expansion and retention.</div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Policy filters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            genders = st.multiselect("Gender", sorted(policies["gender"].dropna().astype(str).unique()), default=[])
            covers = st.multiselect("Additional cover", sorted(policies["cover_label"].dropna().unique()), default=[])
        with c2:
            fuel = st.multiselect("Fuel", sorted(policies["fuel_car"].dropna().astype(str).unique()), default=[])
            transmission = st.multiselect("Transmission", sorted(policies["transmission"].dropna().astype(str).unique()), default=[])
        with c3:
            ages = policies["policyholder_age"].dropna()
            age_range = st.slider("Policyholder age", int(ages.min()), int(ages.max()), (int(ages.min()), int(ages.max())), key="policy_age")
            car_values = policies["worth_car"].dropna()
            car_value_range = st.slider(
                "Vehicle value (€)",
                float(np.floor(car_values.quantile(0.01))),
                float(np.ceil(car_values.quantile(0.99))),
                (float(np.floor(car_values.quantile(0.01))), float(np.ceil(car_values.quantile(0.99)))),
                step=1000.0,
            )

    p = policies.copy()
    if genders: p = p.loc[p["gender"].astype(str).isin(genders)]
    if covers: p = p.loc[p["cover_label"].isin(covers)]
    if fuel: p = p.loc[p["fuel_car"].astype(str).isin(fuel)]
    if transmission: p = p.loc[p["transmission"].astype(str).isin(transmission)]
    p = p.loc[p["policyholder_age"].between(*age_range) | p["policyholder_age"].isna()]
    p = p.loc[p["worth_car"].between(*car_value_range) | p["worth_car"].isna()]
    if p.empty:
        st.warning("No policies match the selected filters.")
        return

    t1, t2, t3 = st.tabs(["Portfolio profile", "Value relationships", "Segmentation"])
    with t1:
        a, b = st.columns(2)
        with a:
            age_dist = p["age_group"].value_counts(sort=False).rename_axis("age_group").reset_index(name="customers")
            fig = px.bar(age_dist, x="age_group", y="customers", text="customers", title="Age structure of the active policy book")
            fig.update_traces(marker_color="#003781", texttemplate="%{text:,}", textposition="outside")
            st.plotly_chart(chart_style(fig), use_container_width=True, config={"displayModeBar": False})
        with b:
            cover_mix = p["cover_label"].value_counts().rename_axis("cover").reset_index(name="policies")
            fig = px.pie(cover_mix, names="cover", values="policies", hole=0.58, title="Additional coverage mix")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(chart_style(fig), use_container_width=True, config={"displayModeBar": False})

        a, b = st.columns(2)
        with a:
            top_brands = p["brand"].value_counts().head(12).rename_axis("brand").reset_index(name="policies")
            fig = px.bar(top_brands.sort_values("policies"), x="policies", y="brand", orientation="h", title="Top insured vehicle brands")
            fig.update_traces(marker_color="#00a6d6")
            st.plotly_chart(chart_style(fig, height=470, legend=False), use_container_width=True, config={"displayModeBar": False})
        with b:
            premium_cover = (
                p.groupby("cover_label")
                .agg(policies=("policy_number", "count"), avg_premium=("total_annual_premium", "mean"), median_car_value=("worth_car", "median"))
                .reset_index()
            )
            fig = px.bar(premium_cover, x="cover_label", y="avg_premium", text="policies", title="Average premium by cover type")
            fig.update_traces(marker_color="#0058a8", texttemplate="n=%{text:,}", textposition="outside")
            fig.update_yaxes(tickprefix="€")
            st.plotly_chart(chart_style(fig, height=470, legend=False), use_container_width=True, config={"displayModeBar": False})

    with t2:
        eligible = p.dropna(subset=["worth_car", "total_annual_premium"])
        sample = eligible.sample(min(2500, len(eligible)), random_state=42).copy()
        if sample["power_car"].notna().any():
            sample["power_car_size"] = sample["power_car"].fillna(sample["power_car"].median()).clip(lower=1)
            size_column = "power_car_size"
        else:
            size_column = None
        fig = px.scatter(
            sample,
            x="worth_car",
            y="total_annual_premium",
            color="cover_label",
            size=size_column,
            opacity=0.58,
            title="Vehicle value versus annual premium",
            labels={"worth_car": "Vehicle value (€)", "total_annual_premium": "Annual premium (€)", "cover_label": "Cover"},
            hover_data=["brand", "policyholder_age", "bonus_malus_percent"],
        )
        fig.update_xaxes(tickprefix="€")
        fig.update_yaxes(tickprefix="€")
        st.plotly_chart(chart_style(fig, height=520), use_container_width=True, config={"displayModeBar": False})

        bonus = p.dropna(subset=["bonus_malus_percent"]).copy()
        bonus["bonus_band"] = pd.cut(bonus["bonus_malus_percent"], bins=[-np.inf, 25, 50, 75, np.inf], labels=["≤25", "26-50", "51-75", "76+"])
        bonus_summary = bonus.groupby("bonus_band", observed=False).agg(policies=("policy_number", "count"), avg_premium=("total_annual_premium", "mean")).reset_index()
        fig = px.line(bonus_summary, x="bonus_band", y="avg_premium", markers=True, text="policies", title="Premium pattern by bonus-malus band")
        fig.update_yaxes(tickprefix="€")
        fig.update_traces(line=dict(color="#003781", width=3), marker=dict(size=10), texttemplate="n=%{text:,}", textposition="top center")
        st.plotly_chart(chart_style(fig, legend=False), use_container_width=True, config={"displayModeBar": False})

    with t3:
        n_clusters = st.slider("Number of customer segments", 2, 6, 4)
        with st.spinner("Building customer segments…"):
            segments = cached_segmentation(p, n_clusters)
        st.caption(f"The two-dimensional view retains {segments.pca_explained_variance:.0%} of the standardized feature variance.")
        sample_seg = segments.assignments.sample(min(4000, len(segments.assignments)), random_state=42)
        fig = px.scatter(
            sample_seg,
            x="pca_1",
            y="pca_2",
            color=sample_seg["segment"].astype(str),
            hover_data=["policyholder_age", "total_annual_premium", "worth_car", "brand"],
            title="K-means customer segments (PCA view)",
            labels={"color": "Segment", "pca_1": "Customer profile dimension 1", "pca_2": "Customer profile dimension 2"},
            opacity=0.65,
        )
        st.plotly_chart(chart_style(fig, height=530), use_container_width=True, config={"displayModeBar": False})
        st.dataframe(
            segments.profile.style.format({c: "{:,.1f}" for c in segments.profile.columns if c not in {"segment", "Customers"}}),
            use_container_width=True,
            hide_index=True,
        )


def regional_page(regional: pd.DataFrame) -> None:
    hero(
        "Geo-demographic opportunity",
        "Use postal-area signals to identify digital-ready markets and design distinct acquisition messages by income, life stage and online-shopping propensity.",
        eyebrow="Regional targeting",
    )
    scored = regional_opportunity(regional)
    provinces = sorted(scored["PROVINCE"].dropna().unique())
    selected = st.multiselect("Province", provinces, default=provinces)
    view = scored.loc[scored["PROVINCE"].isin(selected)] if selected else scored.iloc[0:0]
    if view.empty:
        st.warning("Select at least one province.")
        return

    online_top = view["SHOP_ONLINE"].ge(6).mean()
    middle_income = view["INCOME"].isin([2, 3]).mean()
    car_high = view["CAR"].ge(4).mean()
    avg_score = view["opportunity_score"].mean()
    cols = st.columns(4)
    with cols[0]: kpi("Postal areas", fmt_int(len(view)), "Current geographic selection")
    with cols[1]: kpi("Highest online-shopping bracket", fmt_pct(online_top), "SHOP_ONLINE = 6")
    with cols[2]: kpi("Upper-middle / middle income", fmt_pct(middle_income), "Income codes 2-3")
    with cols[3]: kpi("Opportunity score", f"{avg_score:.1f}/100", "Heuristic digital acquisition index")

    st.caption("The opportunity score is an exploratory index: 40% online shopping, 25% car ownership, 20% income strength and 15% savings. It is not a trained propensity model.")

    a, b = st.columns([1.05, 0.95])
    with a:
        prov = province_summary(view)
        fig = px.bar(
            prov.sort_values("opportunity_score"),
            x="opportunity_score",
            y="PROVINCE",
            orientation="h",
            text="postal_areas",
            title="Regional acquisition opportunity",
            labels={"opportunity_score": "Opportunity score", "PROVINCE": "Province", "postal_areas": "Postal areas"},
        )
        fig.update_traces(marker_color="#003781", texttemplate="n=%{text:,}", textposition="outside")
        st.plotly_chart(chart_style(fig, height=530, legend=False), use_container_width=True, config={"displayModeBar": False})
    with b:
        online = view["SHOP_ONLINE"].value_counts().sort_index().rename_axis("online_bracket").reset_index(name="postal_areas")
        fig = px.area(online, x="online_bracket", y="postal_areas", markers=True, title="Online-shopping propensity distribution")
        fig.update_traces(line=dict(color="#00a6d6", width=3), fillcolor="rgba(0,166,214,0.18)")
        fig.update_xaxes(dtick=1, title="SHOP_ONLINE bracket (higher = more digital)")
        st.plotly_chart(chart_style(fig, height=530, legend=False), use_container_width=True, config={"displayModeBar": False})

    a, b = st.columns(2)
    with a:
        life = view["life_stage_label"].value_counts().head(9).rename_axis("life_stage").reset_index(name="postal_areas")
        fig = px.bar(life.sort_values("postal_areas"), x="postal_areas", y="life_stage", orientation="h", title="Largest life-stage groups")
        fig.update_traces(marker_color="#0058a8")
        st.plotly_chart(chart_style(fig, height=500, legend=False), use_container_width=True, config={"displayModeBar": False})
    with b:
        heat = (
            view.groupby(["income_label", "life_stage_label"])
            .size()
            .reset_index(name="postal_areas")
            .pivot(index="life_stage_label", columns="income_label", values="postal_areas")
            .fillna(0)
        )
        fig = px.imshow(heat, aspect="auto", text_auto=True, title="Life stage by income profile", labels=dict(x="Income", y="Life stage", color="Postal areas"))
        st.plotly_chart(chart_style(fig, height=500, legend=False), use_container_width=True, config={"displayModeBar": False})

    section("Highest-opportunity postal areas", "Use these areas as candidates for look-alike targeting or controlled campaign tests.")
    display_cols = ["zipcode_link", "zip4", "PROVINCE", "income_label", "life_stage_label", "SHOP_ONLINE", "CAR", "SAVINGS", "opportunity_score"]
    st.dataframe(view.nlargest(50, "opportunity_score")[display_cols], use_container_width=True, hide_index=True)


def ml_page(funnel: pd.DataFrame, regional: pd.DataFrame) -> None:
    hero(
        "Machine-learning lab",
        "Compare classification models, tune the business decision threshold and inspect which quote and regional signals contribute most to predicted conversion.",
        eyebrow="Supervised acquisition scoring",
    )
    model_df = make_modeling_frame(funnel, regional, affinity="T&B")
    class_rate = model_df["converted"].mean()
    st.markdown(
        f"""
        <div class="insight-card"><b>Modelling population:</b> {len(model_df):,} T&B quote records with a {class_rate:.1%} conversion rate. Features exclude the final journey status to avoid target leakage.</div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.selectbox("Model", ["Logistic Regression", "Random Forest", "Gradient Boosting"])
    with c2:
        test_size = st.slider("Test-set share", 0.20, 0.35, 0.25, 0.05)
    with c3:
        random_state = st.number_input("Random seed", min_value=1, max_value=999, value=42, step=1)

    with st.spinner(f"Training {model_name}…"):
        result = cached_classifier(model_df, model_name, test_size, int(random_state))

    threshold = st.slider("Decision threshold", 0.10, 0.90, 0.50, 0.05, help="Lower thresholds catch more potential buyers; higher thresholds protect follow-up capacity.")
    metrics, confusion = evaluate_threshold(result.y_test, result.probabilities, threshold)

    cols = st.columns(5)
    names = [("ROC AUC", metrics["roc_auc"]), ("Accuracy", metrics["accuracy"]), ("Precision", metrics["precision"]), ("Recall", metrics["recall"]), ("F1 score", metrics["f1"])]
    for col, (label, value) in zip(cols, names):
        with col: kpi(label, f"{value:.3f}", f"Threshold = {threshold:.2f}" if label != "ROC AUC" else "Ranking quality across thresholds")

    a, b = st.columns(2)
    with a:
        roc_fig = go.Figure()
        roc_fig.add_scatter(
            x=result.roc["false_positive_rate"], y=result.roc["true_positive_rate"], mode="lines", name=f"{model_name} (AUC {metrics['roc_auc']:.3f})", line=dict(color="#003781", width=3)
        )
        roc_fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(color="#a8b4c2", dash="dash"))
        roc_fig.update_layout(title="ROC curve")
        roc_fig.update_xaxes(title="False-positive rate", tickformat=".0%")
        roc_fig.update_yaxes(title="True-positive rate", tickformat=".0%")
        st.plotly_chart(chart_style(roc_fig), use_container_width=True, config={"displayModeBar": False})
    with b:
        cm_df = pd.DataFrame(confusion, index=["Actual: no", "Actual: yes"], columns=["Predicted: no", "Predicted: yes"])
        cm_fig = px.imshow(cm_df, text_auto=True, aspect="auto", title="Confusion matrix", color_continuous_scale="Blues")
        st.plotly_chart(chart_style(cm_fig, legend=False), use_container_width=True, config={"displayModeBar": False})

    a, b = st.columns([1.05, 0.95])
    with a:
        if result.feature_importance.empty:
            st.info("This model does not expose native feature importance.")
        else:
            imp = result.feature_importance.sort_values("importance")
            fig = px.bar(imp, x="importance", y="feature", orientation="h", title="Top model signals")
            fig.update_traces(marker_color="#00a6d6")
            st.plotly_chart(chart_style(fig, height=560, legend=False), use_container_width=True, config={"displayModeBar": False})
    with b:
        capacity = st.slider("Follow-up capacity (% of leads)", 5, 50, 20, 5)
        scored = pd.DataFrame({"actual": result.y_test.to_numpy(), "probability": result.probabilities}).sort_values("probability", ascending=False)
        top_n = max(1, int(len(scored) * capacity / 100))
        selected = scored.head(top_n)
        captured = selected["actual"].sum()
        total_buyers = scored["actual"].sum()
        capture_rate = captured / total_buyers if total_buyers else 0
        lift = selected["actual"].mean() / scored["actual"].mean() if scored["actual"].mean() else 0
        st.markdown("#### Campaign capacity simulation")
        st.metric("Prospects contacted", f"{top_n:,}", f"Top {capacity}% by score")
        st.metric("Actual buyers captured", f"{int(captured):,}", f"{capture_rate:.1%} of buyers in the test sample")
        st.metric("Conversion lift", f"{lift:.2f}×", "Versus contacting a random lead")
        st.caption("This is an out-of-sample simulation on the held-out test set, not a production forecast.")

    with st.expander("Model design and responsible-use notes"):
        st.markdown(
            """
            - The target is whether the quote ended as **Policycreated**.
            - Inputs use quote, vehicle and postcode-level regional variables; final status fields are excluded.
            - Gender is not included in the model. The case notes that gender does not affect pricing.
            - Precision and recall should be chosen according to follow-up cost and the cost of missing a likely buyer.
            - Before deployment, Allianz should test stability, calibration, fairness, drift and consent/privacy controls.
            """
        )


def data_page(funnel: pd.DataFrame, policies: pd.DataFrame, regional: pd.DataFrame) -> None:
    hero(
        "Data explorer and quality audit",
        "Inspect the three linked case data sets, examine missingness and export filtered views for deeper analysis.",
        eyebrow="CRISP-DM data understanding",
    )
    datasets = {"Funnel data": funnel, "Policy data": policies, "Regional data": regional}
    selected_name = st.selectbox("Dataset", list(datasets))
    df = datasets[selected_name]
    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi("Rows", fmt_int(len(df)), "Observations")
    with c2: kpi("Columns", fmt_int(df.shape[1]), "Including engineered fields")
    with c3: kpi("Missing cells", fmt_pct(df.isna().mean().mean()), "Average cell missingness")
    with c4: kpi("Duplicate rows", fmt_int(df.duplicated().sum()), "Full-row duplicates")

    t1, t2 = st.tabs(["Explore rows", "Quality profile"])
    with t1:
        columns = st.multiselect("Columns to display", df.columns.tolist(), default=df.columns.tolist()[: min(12, len(df.columns))])
        search = st.text_input("Search displayed columns", placeholder="Type a brand, province, status or policy number…")
        view = df[columns].copy() if columns else df.iloc[:, 0:0]
        if search and not view.empty:
            mask = view.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            view = view.loc[mask]
        st.caption(f"Showing {min(len(view), 1000):,} of {len(view):,} matching rows.")
        st.dataframe(view.head(1000), use_container_width=True, hide_index=True, height=520)
        csv = view.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", csv, file_name=f"{selected_name.lower().replace(' ', '_')}_filtered.csv", mime="text/csv")
    with t2:
        quality = data_quality_summary(df)
        fig = px.bar(
            quality.head(20).sort_values("missing_percent"),
            x="missing_percent",
            y="column",
            orientation="h",
            title="Highest-missingness fields",
            labels={"missing_percent": "Missing values (%)", "column": "Column"},
        )
        fig.update_traces(marker_color="#0058a8")
        st.plotly_chart(chart_style(fig, height=560, legend=False), use_container_width=True, config={"displayModeBar": False})
        st.dataframe(quality, use_container_width=True, hide_index=True)


def about_page() -> None:
    hero(
        "About this dashboard",
        "A classroom analytics application built around the Allianz Benelux customer-acquisition case and its three supporting data sets.",
        eyebrow="Project guide",
    )
    st.markdown(
        """
        ### Business question
        T&B is Allianz's self-service digital auto-insurance channel. It produces the largest quote volume but converts materially below Insuro and Seguros. The dashboard asks where prospects leave, who is most likely to buy, which regional markets look digitally ready and how analytics can improve customer acquisition without defaulting to blanket discounting.

        ### Analytical structure
        The application follows **CRISP-DM**: business understanding, data understanding, preparation, modelling, evaluation and an outline for deployment. It combines descriptive funnel analytics, customer value analysis, K-means segmentation and supervised classification.

        ### Important limitations
        The case data are disguised and sampled. Regional variables are postcode-level proxies, not individual attributes. The opportunity score is heuristic, while model results are demonstrations rather than a production underwriting or pricing system. Decisions should be validated through controlled experiments and governance checks.
        """
    )
    st.info("Keep the repository private or follow your institution's licensing instructions before uploading the case data to GitHub.")


def main() -> None:
    sidebar_brand()
    try:
        bundle = load_bundle()
    except Exception as exc:
        st.error("The dashboard could not load the case data.")
        st.exception(exc)
        st.stop()

    page = st.sidebar.radio(
        "Navigate",
        [
            "Executive overview",
            "Funnel diagnostics",
            "Customer value",
            "Regional opportunity",
            "Machine-learning lab",
            "Data explorer",
            "About",
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("Data: W27307, W27308, W27309 · Reference date: 16 Oct 2019")

    if page == "Executive overview":
        overview_page(bundle.funnel, bundle.policies, bundle.regional)
    elif page == "Funnel diagnostics":
        funnel_page(bundle.funnel)
    elif page == "Customer value":
        customer_value_page(bundle.funnel, bundle.policies)
    elif page == "Regional opportunity":
        regional_page(bundle.regional)
    elif page == "Machine-learning lab":
        ml_page(bundle.funnel, bundle.regional)
    elif page == "Data explorer":
        data_page(bundle.funnel, bundle.policies, bundle.regional)
    else:
        about_page()


if __name__ == "__main__":
    main()
