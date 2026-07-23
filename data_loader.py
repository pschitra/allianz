from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

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
