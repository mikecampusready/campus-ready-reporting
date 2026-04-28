import os
import pandas as pd
import streamlit as st


DATA_DIR = os.path.join(os.path.dirname(__file__), "data") + "/"

YEAR_COLORS = {
    2025: "#E63946",
    2017: "#6BA3BE",
    2016: "#A8DADC",
}

DISPLAY_YEARS = [2016, 2017, 2025]

CURRENT_YEAR = 2025


def check_password():
    if st.session_state.get("authenticated"):
        return
    st.text_input("Password", type="password", key="_pwd_input")
    if st.session_state.get("_pwd_input") == st.secrets["password"]:
        st.session_state.authenticated = True
        st.rerun()
    elif st.session_state.get("_pwd_input"):
        st.error("Incorrect password")
    st.stop()


@st.cache_data
def load_data():
    orders = pd.read_csv(DATA_DIR + "orders_clean.csv", low_memory=False)
    summary = pd.read_csv(DATA_DIR + "schools_summary.csv")

    orders["order_date"] = pd.to_datetime(orders["order_date"])
    summary["order_first_date"] = pd.to_datetime(summary["order_first_date"])

    # All order types for mapped schools (domestic + summer + international)
    orders = orders[orders["school_name"].notna()].copy()

    # Anchor = earliest order_first_date across all variants of the same school/year
    min_dates = (
        summary.dropna(subset=["school_name", "order_first_date"])
        .groupby(["school_name", "year"])["order_first_date"]
        .min()
        .reset_index()
    )
    min_dates["year"] = min_dates["year"].astype(int)
    orders["year"] = orders["year"].astype(int)

    orders = orders.merge(min_dates, on=["school_name", "year"], how="left")
    orders["days_since_drop"] = (orders["order_date"] - orders["order_first_date"]).dt.days

    return orders, summary
