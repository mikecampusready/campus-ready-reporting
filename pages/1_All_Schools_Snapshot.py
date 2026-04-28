import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import load_data, YEAR_COLORS, CURRENT_YEAR, DISPLAY_YEARS, check_password


def compute_snapshot(orders, summary, day_n, metric, normalize, fold_predate=False):
    if fold_predate:
        orders = orders.copy()
        orders["days_since_drop"] = orders["days_since_drop"].clip(lower=0)
    window = orders[
        (orders["days_since_drop"] >= 0) & (orders["days_since_drop"] <= day_n)
    ]

    if metric == "Orders":
        snap = window.groupby(["school_name", "year"]).size().reset_index(name="value")
    else:
        snap = window.groupby(["school_name", "year"])["revenue"].sum().reset_index(name="value")

    if normalize:
        circ = summary[["school_name", "year", "total_circ"]].dropna(subset=["total_circ"])
        # summary has school_name via the same translation table — join on school_name + year
        # school_name in summary may be null for unmapped rows; only mapped schools matter here
        snap = snap.merge(circ, on=["school_name", "year"], how="left")
        snap["value"] = (snap["value"] / snap["total_circ"] * 1000).round(1)

    return snap


def main():
    st.set_page_config(page_title="All Schools Snapshot", layout="wide")
    check_password()
    st.title("All Schools — Snapshot at Day N")
    st.caption(
        "Cumulative domestic orders from first order date (Day 0) through Day N, "
        "for all 2025 schools. Sorted by 2025 performance."
    )

    orders, summary = load_data()
    orders = orders[orders["year"].isin(DISPLAY_YEARS)]

    # Restrict to schools that have 2025 data
    schools_2025 = set(orders[orders["year"] == CURRENT_YEAR]["school_name"].unique())
    orders_filtered = orders[orders["school_name"].isin(schools_2025)].copy()

    with st.sidebar:
        st.header("Filters")
        day_n = st.slider("Days since first order", min_value=0, max_value=120, value=30)
        metric = st.radio("Metric", ["Orders", "Revenue"])
        normalize = st.checkbox("Per 1,000 CIRC", value=False)
        fold_predate = st.checkbox("Include pre-date orders at Day 0", value=True)

    snap = compute_snapshot(orders_filtered, summary, day_n, metric, normalize, fold_predate)

    # Sort schools by 2025 value descending
    order_2025 = (
        snap[snap["year"] == CURRENT_YEAR]
        .sort_values("value", ascending=False)["school_name"]
        .tolist()
    )
    # Append any 2025 schools with zero orders at this day (not in snap at all)
    zero_schools = sorted(s for s in schools_2025 if s not in order_2025)
    school_order = order_2025 + zero_schools

    years = sorted(snap["year"].unique())
    fig = go.Figure()

    for year in years:
        yr = snap[snap["year"] == year].set_index("school_name")
        values = [yr.loc[s, "value"] if s in yr.index else 0 for s in school_order]
        color = YEAR_COLORS.get(int(year), "#AAAAAA")

        fig.add_trace(
            go.Bar(
                name=str(year),
                x=school_order,
                y=values,
                marker_color=color,
                opacity=1.0 if int(year) == CURRENT_YEAR else 0.7,
            )
        )

    y_label = metric + (" per 1,000 CIRC" if normalize else "")

    fig.update_layout(
        barmode="group",
        xaxis_title="School",
        yaxis_title=y_label,
        hovermode="x unified",
        height=540,
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=140),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Pivot table below the chart
    st.subheader(f"Cumulative {metric.lower()} at Day {day_n}")
    pivot = snap.pivot(index="school_name", columns="year", values="value")
    pivot = pivot.reindex(school_order)
    pivot.index.name = "School"
    pivot.columns = [str(int(c)) for c in pivot.columns]
    st.dataframe(pivot.fillna(0).style.format("{:.1f}" if normalize else "{:.0f}"), use_container_width=True)


if __name__ == "__main__":
    main()
