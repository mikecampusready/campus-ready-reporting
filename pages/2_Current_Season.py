import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date
from utils import load_data, YEAR_COLORS, CURRENT_YEAR, DISPLAY_YEARS, check_password


def get_school_day_map(orders, as_of):
    """Return {school_name: days_since_first_order} for 2025 schools with a first order date before as_of."""
    as_of_ts = pd.Timestamp(as_of)
    first_dates = (
        orders[orders["year"] == CURRENT_YEAR]
        .groupby("school_name")["order_first_date"]
        .first()
        .reset_index()
    )
    first_dates["days_since_first_today"] = (as_of_ts - first_dates["order_first_date"]).dt.days
    started = first_dates[first_dates["days_since_first_today"] >= 0].dropna(subset=["order_first_date"])
    return started.set_index("school_name")["days_since_first_today"].to_dict()


def compute_snapshot(orders, school_day_map, metric, fold_predate=False):
    """For each school at its current day N, compute cumulative orders/revenue for all years."""
    school_names = list(school_day_map.keys())
    day_map_df = pd.DataFrame(
        {"school_name": school_names, "day_n": [school_day_map[s] for s in school_names]}
    )

    filtered = orders[orders["school_name"].isin(school_names)].copy()
    filtered = filtered.merge(day_map_df, on="school_name", how="left")
    if fold_predate:
        filtered["days_since_drop"] = filtered["days_since_drop"].clip(lower=0)
    filtered = filtered[
        (filtered["days_since_drop"] >= 0) & (filtered["days_since_drop"] <= filtered["day_n"])
    ]

    if metric == "Orders":
        snap = filtered.groupby(["school_name", "year", "day_n"]).size().reset_index(name="value")
    else:
        snap = filtered.groupby(["school_name", "year", "day_n"])["revenue"].sum().reset_index(name="value")

    return snap


def main():
    st.set_page_config(page_title="Current Season", layout="wide")
    check_password()
    st.title("Current Season — Each School at Today's Day")
    st.caption(
        "Each school shown at its own days since first order as of the selected date. "
        "2025 in red, prior OCM years in blue. Sorted by days in season (longest first)."
    )

    orders, summary = load_data()
    orders = orders[orders["year"].isin(DISPLAY_YEARS)]

    with st.sidebar:
        st.header("Filters")
        as_of = st.date_input(
            "As of date",
            value=date(2025, 7, 1),
            min_value=date(2025, 1, 1),
            max_value=date(2025, 12, 31),
        )
        metric = st.radio("Metric", ["Orders", "Revenue"])
        normalize = st.checkbox("Per 1,000 CIRC", value=False)
        fold_predate = st.checkbox("Include pre-date orders at Day 0", value=True)

    school_day_map = get_school_day_map(orders, as_of)

    if not school_day_map:
        st.info("No 2025 schools have reached their first order date by the selected date.")
        return

    snap = compute_snapshot(orders, school_day_map, metric, fold_predate)

    if normalize:
        circ = summary[["school_name", "year", "total_circ"]].dropna(subset=["total_circ"])
        snap = snap.merge(circ, on=["school_name", "year"], how="left")
        snap["value"] = (snap["value"] / snap["total_circ"] * 1000).round(1)

    # Sort schools by days since first order descending (longest in season first)
    school_order = sorted(school_day_map, key=lambda s: school_day_map[s], reverse=True)

    # X-axis labels: "School Name (Day N)"
    def label(school):
        day = int(school_day_map[school])
        return f"{school}<br>(Day {day})"

    x_labels = [label(s) for s in school_order]

    years = sorted(snap["year"].unique())
    fig = go.Figure()

    for year in years:
        yr = snap[snap["year"] == year].set_index("school_name")
        values = [yr.loc[s, "value"] if s in yr.index else 0 for s in school_order]
        color = YEAR_COLORS.get(int(year), "#AAAAAA")

        fig.add_trace(
            go.Bar(
                name=str(year),
                x=x_labels,
                y=values,
                marker_color=color,
                opacity=1.0 if int(year) == CURRENT_YEAR else 0.7,
                hovertemplate="%{x}<br>" + str(year) + ": %{y:,.0f}<extra></extra>",
            )
        )

    y_label = metric + (" per 1,000 CIRC" if normalize else "")

    fig.update_layout(
        barmode="group",
        xaxis_title="School (Days Since First Order)",
        yaxis_title=y_label,
        hovermode="x unified",
        height=560,
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=160),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detail table
    st.subheader("Detail")
    snap["day_n"] = snap["day_n"].astype(int)
    pivot = snap.pivot_table(index=["school_name", "day_n"], columns="year", values="value", aggfunc="sum")
    pivot.columns = [str(int(c)) for c in pivot.columns]
    pivot = pivot.reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"school_name": "School", "day_n": "Days Since First Order"})
    pivot["Days Since First Order"] = pivot["Days Since First Order"].astype(int)
    pivot["_order"] = pivot["School"].map({s: i for i, s in enumerate(school_order)})
    pivot = pivot.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    fmt = "{:.1f}" if normalize else "{:.0f}"
    year_cols = [c for c in pivot.columns if c.isdigit()]
    st.dataframe(
        pivot.style.format({c: fmt for c in year_cols}),
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
