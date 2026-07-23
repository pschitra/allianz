from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


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
