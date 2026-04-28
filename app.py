import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from utils import load_data, YEAR_COLORS, CURRENT_YEAR, DISPLAY_YEARS, check_password


def build_cumulative(school_orders, metric, fold_predate=False):
    years = sorted(school_orders["year"].unique())
    traces = []

    for year in years:
        yr = school_orders[school_orders["year"] == year].copy()
        if fold_predate:
            yr["days_since_drop"] = yr["days_since_drop"].clip(lower=0)
        yr = yr.sort_values("days_since_drop")

        if metric == "Orders":
            daily = yr.groupby("days_since_drop").size().reset_index(name="value")
        else:
            daily = yr.groupby("days_since_drop")["revenue"].sum().reset_index(name="value")

        daily["cumulative"] = daily["value"].cumsum()

        color = YEAR_COLORS.get(year, "#AAAAAA")
        width = 3 if year == CURRENT_YEAR else 1.5
        opacity = 1.0 if year == CURRENT_YEAR else 0.7

        traces.append(
            go.Scatter(
                x=daily["days_since_drop"],
                y=daily["cumulative"],
                mode="lines",
                name=str(year),
                line=dict(color=color, width=width),
                opacity=opacity,
            )
        )

    return traces


def build_summary_table(school_name, summary, display_years):
    rows = summary[summary["school_name"] == school_name].copy()
    # Ensure all display years appear, filling missing ones with NaN
    all_years = pd.DataFrame({"year": display_years})
    rows = all_years.merge(rows, on="year", how="left").sort_values("year")
    display = rows[["year", "order_first_date", "total_orders", "total_revenue", "total_circ"]].copy()
    display.columns = ["Year", "First Order Date", "Total Orders", "Total Revenue ($)", "CIRC"]
    display["Total Revenue ($)"] = display["Total Revenue ($)"].apply(
        lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
    )
    display["Total Orders"] = display["Total Orders"].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) else "0"
    )
    display["CIRC"] = display["CIRC"].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) else "—"
    )
    display["First Order Date"] = pd.to_datetime(display["First Order Date"]).apply(
        lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else "—"
    )
    return display.reset_index(drop=True)


def main():
    st.set_page_config(page_title="Campus Ready Sales", layout="wide")
    check_password()
    st.title("Campus Ready — Days Since First Order")
    st.caption("Cumulative orders by days since first order date. Includes any school active in 2016, 2017, or 2025. 2025 shown in red.")

    orders, summary = load_data()
    orders = orders[orders["year"].isin(DISPLAY_YEARS)]

    # All schools with summary data in any display year (covers 2016/2017 clients not in 2025)
    schools = sorted(
        summary[summary["year"].isin(DISPLAY_YEARS)]["school_name"].dropna().unique()
    )

    with st.sidebar:
        st.header("Filters")
        selected_school = st.selectbox("School", schools)
        metric = st.radio("Metric", ["Orders", "Revenue"])
        day_min, day_max = st.slider(
            "Days since first order window",
            min_value=-30,
            max_value=150,
            value=(-10, 120),
        )
        fold_predate = st.checkbox("Include pre-date orders at Day 0", value=True)

    school_orders = orders[orders["school_name"] == selected_school].copy()
    if fold_predate:
        school_orders = school_orders[school_orders["days_since_drop"] <= day_max]
    else:
        school_orders = school_orders[
            (school_orders["days_since_drop"] >= day_min)
            & (school_orders["days_since_drop"] <= day_max)
        ]

    curve_orders = school_orders[school_orders["days_since_drop"].notna()]

    if curve_orders.empty:
        st.info("No order curve available for this school (missing first-order date).")
    else:
        traces = build_cumulative(curve_orders, metric, fold_predate)
        y_label = "Cumulative Orders" if metric == "Orders" else "Cumulative Revenue ($)"

        fig = go.Figure(traces)
        fig.update_layout(
            xaxis_title="Days Since First Order",
            yaxis_title=y_label,
            hovermode="x unified",
            height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=40, b=40),
        )
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="First Order")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Annual Summary")
    table = build_summary_table(selected_school, summary, DISPLAY_YEARS)
    st.dataframe(table, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
