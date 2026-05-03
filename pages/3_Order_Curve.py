import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from utils import load_data, check_password

SEP_SCHOOLS = {
    "University of California, Los Angeles",
    "University of California Santa Barbara",
    "Cal Poly Pomona",
}

BUCKET_COLORS = {
    "May-Aug": "#2196F3",
    "Jun-Aug": "#43A047",
    "Jul-Aug": "#FB8C00",
}


def assign_bucket(drop_month, end_month):
    if end_month != 8:
        return None
    if drop_month in (4, 5):
        return "May-Aug"
    if drop_month == 6:
        return "Jun-Aug"
    if drop_month == 7:
        return "Jul-Aug"
    return None


def build_validation_table(orders, summary, may_aug_median_curve):
    """2018 partial orders vs actual full-season totals from schools_summary."""
    pct14 = may_aug_median_curve[14]
    pct30 = may_aug_median_curve[30]

    hist18 = orders[(orders["year"] == 2018) & orders["days_since_drop"].notna()].copy()
    hist18["days_since_drop"] = hist18["days_since_drop"].clip(lower=0)
    if hist18.empty:
        return pd.DataFrame(), pct14, pct30

    drop_dates = hist18.groupby("school_name")["order_first_date"].first().reset_index()
    drop_dates["drop_month"] = drop_dates["order_first_date"].dt.month
    drop_dates = drop_dates[
        drop_dates["drop_month"].isin([4, 5, 6])
        & ~drop_dates["school_name"].isin(SEP_SCHOOLS)
    ]

    hist18 = hist18[hist18["school_name"].isin(drop_dates["school_name"])]
    max_day_avail = hist18.groupby("school_name")["days_since_drop"].max().reset_index(name="max_day")
    cnt14 = hist18[hist18["days_since_drop"] <= 14].groupby("school_name").size().reset_index(name="orders_day14")
    cnt30 = hist18[hist18["days_since_drop"] <= 30].groupby("school_name").size().reset_index(name="orders_day30")

    actual = (
        summary[(summary["year"] == 2018) & summary["school_name"].isin(drop_dates["school_name"]) & summary["total_orders"].notna()]
        .groupby("school_name")["total_orders"].sum()
        .reset_index(name="actual_total")
    )

    df = actual.merge(drop_dates[["school_name", "order_first_date"]], on="school_name", how="inner")
    df = df.merge(max_day_avail, on="school_name", how="left")
    df = df.merge(cnt14, on="school_name", how="left")
    df = df.merge(cnt30, on="school_name", how="left")
    df["orders_day14"] = df["orders_day14"].fillna(0).astype(int)
    df["orders_day30"] = df["orders_day30"].fillna(0).astype(int)

    df["pred14"] = np.where(df["max_day"] >= 14, (df["orders_day14"] / pct14 * 100).round(0), np.nan)
    df["pred30"] = np.where(df["max_day"] >= 30, (df["orders_day30"] / pct30 * 100).round(0), np.nan)
    df["err14_pct"] = ((df["pred14"] - df["actual_total"]) / df["actual_total"] * 100).round(1)
    df["err30_pct"] = ((df["pred30"] - df["actual_total"]) / df["actual_total"] * 100).round(1)

    df = df[df["actual_total"] > 40].sort_values("actual_total", ascending=False).reset_index(drop=True)
    return df, pct14, pct30


def build_curves(orders, years, max_day):
    hist = orders[orders["year"].isin(years) & orders["days_since_drop"].notna()].copy()
    hist["days_since_drop"] = hist["days_since_drop"].clip(lower=0)
    hist["month"] = hist["order_date"].dt.month

    totals = hist.groupby(["school_name", "year"]).size().reset_index(name="total_orders")

    by_month = hist.groupby(["school_name", "year", "month"]).size().reset_index(name="cnt")
    by_month = by_month.merge(totals, on=["school_name", "year"])
    by_month["pct"] = by_month["cnt"] / by_month["total_orders"] * 100
    end_months = (
        by_month[by_month["pct"] >= 5]
        .groupby(["school_name", "year"])["month"]
        .max()
        .reset_index(name="end_month")
    )

    drop_months = (
        hist.groupby(["school_name", "year"])["order_first_date"]
        .first()
        .dt.month
        .reset_index(name="drop_month")
    )

    meta = totals.merge(drop_months, on=["school_name", "year"], how="left")
    meta = meta.merge(end_months, on=["school_name", "year"], how="left")
    meta = meta[(meta["total_orders"] > 40) & (~meta["school_name"].isin(SEP_SCHOOLS))]
    meta["bucket"] = meta.apply(
        lambda r: assign_bucket(r["drop_month"], r["end_month"]), axis=1
    )
    meta = meta.dropna(subset=["bucket"])

    day_grid = np.arange(0, max_day + 1)
    curves = {b: [] for b in BUCKET_COLORS}
    school_lists = {b: [] for b in BUCKET_COLORS}

    for _, row in meta.iterrows():
        sub = hist[
            (hist["school_name"] == row["school_name"])
            & (hist["year"] == row["year"])
            & (hist["days_since_drop"] <= max_day)
        ]
        if sub.empty:
            continue

        daily = sub.groupby("days_since_drop").size().reindex(day_grid, fill_value=0)
        pct_curve = daily.cumsum() / row["total_orders"] * 100

        bucket = row["bucket"]
        curves[bucket].append(pct_curve.values)
        school_lists[bucket].append(f"{row['school_name']} ({int(row['year'])})")

    return curves, school_lists, day_grid


def main():
    st.set_page_config(page_title="Order Curve", layout="wide")
    check_password()
    st.title("Order Curve by Season Type")
    st.caption(
        "Normalized cumulative order curve by season type (drop month + end month). "
        "Shows what % of total season orders are captured by each day since first order."
    )

    orders, summary = load_data()

    with st.sidebar:
        st.header("Filters")
        years = st.multiselect("Years", [2016, 2017], default=[2016, 2017])
        max_day = st.slider("Days to show", 60, 150, 120)
        show_bands = st.checkbox("Show 25th-75th percentile band", value=True)
        show_schools = st.checkbox("Show school list per bucket", value=False)

    if not years:
        st.info("Select at least one year.")
        return

    curves, school_lists, day_grid = build_curves(orders, years, max_day)

    fig = go.Figure()

    for bucket, color in BUCKET_COLORS.items():
        arr_list = curves[bucket]
        if not arr_list:
            continue
        arr = np.array(arr_list)
        median_curve = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        n = len(arr_list)

        if show_bands:
            fig.add_trace(go.Scatter(
                x=np.concatenate([day_grid, day_grid[::-1]]),
                y=np.concatenate([p75, p25[::-1]]),
                fill="toself",
                fillcolor=color,
                opacity=0.15,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

        fig.add_trace(go.Scatter(
            x=day_grid,
            y=median_curve,
            mode="lines",
            name=f"{bucket} (n={n})",
            line=dict(color=color, width=2.5),
            hovertemplate=f"{bucket}<br>Day %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))

    fig.update_layout(
        xaxis_title="Days Since First Order",
        yaxis_title="% of Total Season Orders",
        yaxis=dict(range=[0, 105], ticksuffix="%"),
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=40, b=40),
    )
    fig.add_hline(y=100, line_dash="dot", line_color="gray", opacity=0.4)

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("% captured at key days")
    key_days = [d for d in [7, 14, 21, 30, 45, 60, 90, 120] if d <= max_day]
    rows = []
    for bucket in BUCKET_COLORS:
        if not curves[bucket]:
            continue
        arr = np.array(curves[bucket])
        median_curve = np.median(arr, axis=0)
        row = {"Bucket": bucket, "n": len(curves[bucket])}
        for d in key_days:
            row[f"Day {d}"] = f"{median_curve[d]:.1f}%"
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Season Prediction Tool")
    st.caption(
        "Enter where your school is in the season to estimate total orders. "
        "Predicted total = current orders ÷ curve % at that day. "
        "Range uses the p25–p75 band of training schools."
    )

    pt_col1, pt_col2 = st.columns(2)
    with pt_col1:
        pred_day = st.number_input(
            "Days since first order", min_value=1, max_value=max_day, value=min(30, max_day)
        )
    with pt_col2:
        pred_orders = st.number_input("Orders so far", min_value=1, value=50, step=10)

    pred_results = []
    for bucket in BUCKET_COLORS:
        arr_list = curves[bucket]
        if not arr_list:
            continue
        arr = np.array(arr_list)
        med = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        if pred_day >= len(med) or med[pred_day] <= 0:
            continue
        pct_med = med[pred_day]
        pred_median = round(pred_orders / pct_med * 100)
        pred_low = round(pred_orders / p75[pred_day] * 100) if p75[pred_day] > 0 else None
        pred_high = round(pred_orders / p25[pred_day] * 100) if p25[pred_day] > 0 else None
        pred_results.append({
            "bucket": bucket,
            "n": len(arr_list),
            "pct_med": pct_med,
            "pred_median": pred_median,
            "pred_low": pred_low,
            "pred_high": pred_high,
        })

    if pred_results:
        metric_cols = st.columns(len(pred_results))
        for i, r in enumerate(pred_results):
            with metric_cols[i]:
                st.metric(
                    label=f"{r['bucket']} (n={r['n']})",
                    value=f"{r['pred_median']:,}",
                )
                range_str = (
                    f"Range: {r['pred_low']:,} – {r['pred_high']:,}"
                    if r["pred_low"] else ""
                )
                st.caption(f"Curve at day {int(pred_day)}: {r['pct_med']:.1f}%  |  {range_str}")
    else:
        st.info("No curve data available for the selected years.")

    if show_schools:
        st.subheader("Schools in each bucket")
        for bucket in BUCKET_COLORS:
            items = school_lists[bucket]
            if not items:
                continue
            with st.expander(f"{bucket} — {len(items)} school/years"):
                st.write("\n".join(f"- {s}" for s in sorted(items)))

    # 2018 validation
    st.divider()
    st.subheader("2018 Prediction Validation")
    st.caption(
        "May-Aug curve applied to partial 2018 order data (through June 13). "
        "Predicted totals at Day 14 and Day 30 compared to actual 2018 full-season totals."
    )

    may_aug_list = curves.get("May-Aug", [])
    if not may_aug_list:
        st.info("May-Aug curve not available with the selected years.")
    else:
        may_aug_median = np.median(np.array(may_aug_list), axis=0)
        val_df, pct14, pct30 = build_validation_table(orders, summary, may_aug_median)

        if val_df.empty:
            st.info("No 2018 validation data available.")
        else:
            has14 = val_df[val_df["pred14"].notna()]
            has30 = val_df[val_df["pred30"].notna()]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Day 14 — schools tested", len(has14))
            col2.metric("Day 14 — median abs error", f"{has14['err14_pct'].abs().median():.1f}%")
            col3.metric("Day 30 — schools tested", len(has30))
            col4.metric("Day 30 — median abs error", f"{has30['err30_pct'].abs().median():.1f}%")

            display = val_df.copy()
            display["Drop Date"] = display["order_first_date"].dt.strftime("%m/%d")
            display["Actual Total"] = display["actual_total"].astype(int)
            display["Orders @ Day 14"] = display["orders_day14"]
            display["Predicted @ Day 14"] = display["pred14"].apply(lambda x: int(x) if pd.notna(x) else "—")
            display["Error % @ Day 14"] = display["err14_pct"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")
            display["Orders @ Day 30"] = display["orders_day30"]
            display["Predicted @ Day 30"] = display["pred30"].apply(lambda x: int(x) if pd.notna(x) else "—")
            display["Error % @ Day 30"] = display["err30_pct"].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "—")

            show_cols = ["school_name", "Drop Date", "Actual Total",
                         "Orders @ Day 14", "Predicted @ Day 14", "Error % @ Day 14",
                         "Orders @ Day 30", "Predicted @ Day 30", "Error % @ Day 30"]
            display = display[show_cols].rename(columns={"school_name": "School"})

            st.dataframe(display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
