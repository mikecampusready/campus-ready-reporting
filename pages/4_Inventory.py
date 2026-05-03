import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
from utils import check_password

DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
SEASON_START = pd.Timestamp("2026-04-30")


def file_freshness(path):
    mtime = os.path.getmtime(path)
    return pd.Timestamp(mtime, unit="s").strftime("%Y-%m-%d %H:%M")


def load_sources():
    woo       = pd.read_csv(os.path.join(DATA_DIR, "woo_orders_raw.csv"),     parse_dates=["date_created"])
    refunds   = pd.read_csv(os.path.join(DATA_DIR, "woo_refunds_raw.csv"))
    zoho      = pd.read_csv(os.path.join(DATA_DIR, "inventory_summary.csv"))
    composite = pd.read_csv(os.path.join(DATA_DIR, "composite_items.csv"))
    ship      = pd.read_csv(os.path.join(DATA_DIR, "shp_to_school_info.csv"))
    return woo, refunds, zoho, composite, ship


def compute_report(woo, refunds, zoho):
    # WOO: current season, processing + completed only (cancelled excluded entirely)
    woo_s = woo[
        (woo["date_created"] >= SEASON_START) &
        woo["order_status"].isin(["processing", "completed"])
    ].copy()
    woo_s["sku"] = woo_s["sku"].str.strip()

    proc     = woo_s[woo_s["order_status"] == "processing"]
    woo_done = woo_s[woo_s["order_status"] == "completed"]

    def sku_qty(df, qty_col, name):
        return df.groupby("sku")[qty_col].sum().rename(name)

    woo_all     = sku_qty(woo_s,                                                    "quantity", "woo_all")
    woo_home    = sku_qty(proc[proc["meta_chosen_shipping"] == "ship_to_home"],     "quantity", "woo_home")
    woo_school  = sku_qty(proc[proc["meta_chosen_shipping"] == "ship_to_school"],   "quantity", "woo_school")
    woo_shipped = sku_qty(woo_done,                                                 "quantity", "woo_shipped")

    # Returns: split by shipping type so we can reduce ship_asap / held_school correctly
    refunds_s = refunds.copy()
    refunds_s["sku"] = refunds_s["sku"].str.strip()
    ret_home   = sku_qty(refunds_s[refunds_s["meta_chosen_shipping"] == "ship_to_home"],   "quantity_returned", "ret_home")
    ret_school = sku_qty(refunds_s[refunds_s["meta_chosen_shipping"] == "ship_to_school"], "quantity_returned", "ret_school")
    ret_all    = sku_qty(refunds_s,                                                         "quantity_returned", "ret_all")

    # ZOHO: active SKUs are the master list
    zoho_a = zoho[zoho["Status"].str.lower() == "active"].copy()
    zoho_a["sku"] = zoho_a["sku"].str.strip()
    zoho_a = zoho_a.rename(columns={
        "item_name":                   "sku_name",
        "quantity_available":          "zoho_avail",
        "quantity_available_for_sale": "zoho_avail_sale",
        "quantity_ordered":            "zoho_ordered",
    })

    rpt = zoho_a[["sku", "sku_name", "is_combo_product", "zoho_avail", "zoho_avail_sale", "zoho_ordered"]].copy()
    rpt = (rpt
           .join(woo_all,     on="sku", how="left")
           .join(woo_home,    on="sku", how="left")
           .join(woo_school,  on="sku", how="left")
           .join(woo_shipped, on="sku", how="left")
           .join(ret_home,    on="sku", how="left")
           .join(ret_school,  on="sku", how="left")
           .join(ret_all,     on="sku", how="left"))

    for col in ["woo_all", "woo_home", "woo_school", "woo_shipped", "ret_home", "ret_school", "ret_all"]:
        rpt[col] = rpt[col].fillna(0)

    # ZOHO-direct committed orders (Zoho knows about these; WOO doesn't)
    zoho_direct = (rpt["zoho_avail"] - rpt["zoho_avail_sale"]).clip(lower=0)

    # Total Sold: gross WOO orders (processing + completed) + ZOHO-direct
    # Returns are shown separately — not netted out of this number
    rpt["total_sold"]     = rpt["woo_all"] + zoho_direct

    # Total Shipping ASAP: WOO processing ship_to_home, minus any home-order returns
    rpt["ship_asap"]      = (rpt["woo_home"] - rpt["ret_home"]).clip(lower=0)

    # Total Held for School: WOO processing ship_to_school minus school-order returns + ZOHO-direct
    rpt["held_school"]    = (rpt["woo_school"] - rpt["ret_school"]).clip(lower=0) + zoho_direct

    # Physical Inventory: ZOHO qty_available minus shipped, plus returned (restocked)
    rpt["physical_inv"]   = rpt["zoho_avail"] - rpt["woo_shipped"] + rpt["ret_all"]

    # Available for Sale: ZOHO qty_available_for_sale minus net sold (gross - returned)
    rpt["avail_for_sale"] = rpt["zoho_avail_sale"] - (rpt["woo_all"] - rpt["ret_all"]) - zoho_direct

    # In Transit: ZOHO-quantity_ordered
    rpt["in_transit"]     = rpt["zoho_ordered"]

    # Returned column for display (total across all shipping types)
    rpt["returned"]       = rpt["ret_all"]

    return rpt


def compute_assemblies(rpt, composite, include_in_transit=False):
    base = rpt.set_index("sku")["avail_for_sale"]
    if include_in_transit:
        base = base + rpt.set_index("sku")["in_transit"]
    avail = base.to_dict()

    rows = []
    for parent_sku, grp in composite.groupby("SKU"):
        max_can_make = float("inf")
        limiting_sku = ""
        for _, r in grp.iterrows():
            comp_sku = str(r["Mapped Item SKU"]).strip()
            qty_per  = float(r["Mapped Quantity"])
            if qty_per <= 0:
                continue
            can_make = float(avail.get(comp_sku, 0)) / qty_per
            if can_make < max_can_make:
                max_can_make = can_make
                limiting_sku = comp_sku

        assemblies = int(max_can_make) if max_can_make != float("inf") else 0
        rows.append({
            "sku":             parent_sku,
            "assembly_name":   grp["Composite Item Name"].iloc[0],
            "can_produce":     assemblies,
            "limiting_sku":    limiting_sku,
        })

    return pd.DataFrame(rows)


def main():
    st.set_page_config(page_title="Inventory Report", layout="wide")
    check_password()
    data_files = {
        "woo":       "woo_orders_raw.csv",
        "refunds":   "woo_refunds_raw.csv",
        "zoho":      "inventory_summary.csv",
        "ship":      "shp_to_school_info.csv",
        "composite": "composite_items.csv",
    }
    missing = [k for k, f in data_files.items() if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        st.error(f"Missing data files: {', '.join(missing)}. Add them to DEV/data/ and refresh.")
        return

    woo_updated  = file_freshness(os.path.join(DATA_DIR, data_files["woo"]))
    zoho_updated = file_freshness(os.path.join(DATA_DIR, data_files["zoho"]))

    st.title("Inventory Report")
    st.markdown(
        f"WOO orders as of **{woo_updated}**&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"ZOHO inventory as of **{zoho_updated}**"
    )

    woo, refunds, zoho, composite, ship = load_sources()

    rpt = compute_report(woo, refunds, zoho)

    # --- Sidebar filters ---
    with st.sidebar:
        st.header("Filters")
        search            = st.text_input("Search SKU or name")
        sku_type          = st.radio("Show", ["Both", "Assemblies only", "Individual SKUs only"], index=0)
        hide_zero         = st.checkbox("Hide SKUs with no activity", value=False)
        negative_net      = st.checkbox(
            "Show only negative even with in-transit",
            value=False,
            help="Filters to SKUs where Available for Sale + In Transit is still negative — these need attention even after incoming stock arrives.",
        )
        st.divider()
        st.subheader("Assemblies")
        include_in_transit = st.checkbox(
            "Include ZOHO in-transit inventory",
            value=False,
            help="Adds ZOHO-quantity_ordered (supplier orders not yet received) to available inventory when calculating assemblies producible.",
        )

    asm = compute_assemblies(rpt, composite, include_in_transit=include_in_transit)

    display = rpt[[
        "sku", "sku_name", "is_combo_product", "total_sold", "returned",
        "ship_asap", "held_school", "physical_inv", "avail_for_sale", "in_transit",
    ]].copy()

    num_cols = ["total_sold", "returned", "ship_asap", "held_school",
                "physical_inv", "avail_for_sale", "in_transit"]
    display[num_cols] = display[num_cols].round(0).astype(int)

    display = display.rename(columns={
        "sku":           "SKU",
        "sku_name":      "SKU Name",
        "total_sold":    "Total Sold",
        "returned":      "Returned",
        "ship_asap":     "Total Shipping ASAP",
        "held_school":   "Total Held for School",
        "physical_inv":  "Physical Inventory",
        "avail_for_sale":"Available for Sale",
        "in_transit":    "In Transit",
    })

    # is_combo_product comes through as string "true"/"false" from ZOHO CSV
    is_combo = display["is_combo_product"].astype(str).str.lower() == "true"

    if sku_type == "Assemblies only":
        display = display[is_combo]
    elif sku_type == "Individual SKUs only":
        display = display[~is_combo]

    display = display.drop(columns=["is_combo_product"])

    if search:
        mask = (
            display["SKU"].str.contains(search, case=False, na=False) |
            display["SKU Name"].str.contains(search, case=False, na=False)
        )
        display = display[mask]

    if hide_zero:
        display = display[display["Total Sold"] > 0]

    if negative_net:
        display = display[(display["Available for Sale"] + display["In Transit"]) < 0]

    # --- Main inventory table ---
    st.subheader("Inventory by SKU")
    st.dataframe(display, use_container_width=True, hide_index=True)

    # --- Assemblies detail section ---
    st.divider()
    st.subheader("Assemblies Producible — Detail")
    st.caption(
        "How many of each composite pack can be built from Available for Sale "
        "of component SKUs. Limiting Component SKU is the bottleneck."
    )

    asm_display = asm.rename(columns={
        "sku":           "SKU",
        "assembly_name": "Assembly Name",
        "can_produce":   "Can Produce",
        "limiting_sku":  "Limiting Component SKU",
    }).sort_values("Can Produce", ascending=False)

    if search:
        mask = (
            asm_display["SKU"].str.contains(search, case=False, na=False) |
            asm_display["Assembly Name"].str.contains(search, case=False, na=False)
        )
        asm_display = asm_display[mask]

    st.dataframe(asm_display, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
