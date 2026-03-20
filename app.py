import io
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils_processing import (
    APP_CONFIG,
    ensure_data_dirs,
    load_master_tables,
    save_master_tables,
    read_uploaded_excels,
    standardize_gap,
    standardize_pickups,
    consolidate_physical_pickups,
    append_dedup,
    build_ingestion_log_entries,
    append_ingestion_log,
    build_route_day_summary,
    match_pickups_to_gap,
)

st.set_page_config(page_title="FedEx Pickup Route Optimization", layout="wide")

st.title("FedEx Pickup Route Optimization")
st.caption("Upload new GAP and pickup files, update master data, and run analysis only when needed.")

page = st.sidebar.radio(
    "Navigation",
    [
        "Update Master Data",
        "Analyze Existing Master",
        "Settings / Exceptions",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write("Station:", APP_CONFIG["station_address"])
st.sidebar.write("W1 start:", APP_CONFIG["wave_starts"]["W1"])
st.sidebar.write("W2 start:", APP_CONFIG["wave_starts"]["W2"])
st.sidebar.write("Start station minutes:", APP_CONFIG["start_of_day_station_min"])
st.sidebar.write("End station minutes:", APP_CONFIG["end_of_day_station_min"])

storage_root = st.sidebar.text_input(
    "Data folder",
    value="streamlit_data",
    help="Local app storage folder. For cloud deployment, this should later be replaced with persistent storage."
)
paths = ensure_data_dirs(storage_root)

gap_master, pickup_master, pickup_stops_master, ingestion_log = load_master_tables(paths)

if page == "Update Master Data":
    st.header("Update Master Data")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Current master status")
        st.write(f"GAP master rows: **{len(gap_master):,}**")
        st.write(f"Pickup master rows: **{len(pickup_master):,}**")
        st.write(f"Pickup stops master rows: **{len(pickup_stops_master):,}**")
        st.write(f"Ingestion log rows: **{len(ingestion_log):,}**")

    with c2:
        if not gap_master.empty and "scan_date" in gap_master.columns:
            gap_dates = pd.to_datetime(gap_master["scan_date"], errors="coerce").dropna()
            if not gap_dates.empty:
                st.write("GAP dates:", f"{gap_dates.min().date()} to {gap_dates.max().date()}")
            else:
                st.write("GAP dates: none detected")

        if not pickup_master.empty and "pickup_date" in pickup_master.columns:
            pickup_dates = pd.to_datetime(pickup_master["pickup_date"], errors="coerce").dropna()
            if not pickup_dates.empty:
                st.write("Pickup dates:", f"{pickup_dates.min().date()} to {pickup_dates.max().date()}")
            else:
                st.write("Pickup dates: none detected")

    st.markdown("---")

    with st.form("update_form", clear_on_submit=False):
        st.subheader("Upload new Excel files")
        uploaded_files = st.file_uploader(
            "Upload GAP and pickup Excel files together",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            key="mixed_uploads",
        )

        run_update = st.form_submit_button("Update Master Data")

    if uploaded_files:
        detected_gap_files = []
        detected_pickup_files = []
        unknown_files = []

        for f in uploaded_files:
            try:
                preview_df = pd.read_excel(f, nrows=5)
                cols = set(preview_df.columns.astype(str))

                if {"Scan Date", "Route", "Stop Order", "Stop Type", "Activity"}.issubset(cols):
                    detected_gap_files.append(f.name)
                elif {"Pickup Date", "Work Area #", "Ready Pickup Time", "Close Pickup Time", "Pickup Time"}.issubset(cols):
                    detected_pickup_files.append(f.name)
                else:
                    unknown_files.append(f.name)
            except Exception:
                unknown_files.append(f.name)

        st.markdown("### Detected file types")
        st.write(f"GAP files detected: **{len(detected_gap_files)}**")
        if detected_gap_files:
            st.write(detected_gap_files)

        st.write(f"Pickup files detected: **{len(detected_pickup_files)}**")
        if detected_pickup_files:
            st.write(detected_pickup_files)

        if unknown_files:
            st.write(f"Unknown files: **{len(unknown_files)}**")
            st.write(unknown_files)

    if run_update:
        if not uploaded_files:
            st.warning("Please upload at least one Excel file.")
        else:
            gap_file_objs = []
            pickup_file_objs = []
            unknown_files = []

            for f in uploaded_files:
                try:
                    preview_df = pd.read_excel(f, nrows=5)
                    cols = set(preview_df.columns.astype(str))

                    if {"Scan Date", "Route", "Stop Order", "Stop Type", "Activity"}.issubset(cols):
                        gap_file_objs.append(f)
                    elif {"Pickup Date", "Work Area #", "Ready Pickup Time", "Close Pickup Time", "Pickup Time"}.issubset(cols):
                        pickup_file_objs.append(f)
                    else:
                        unknown_files.append(f.name)
                except Exception:
                    unknown_files.append(f.name)

            with st.spinner("Reading uploaded files..."):
                gap_new_raw = read_uploaded_excels(gap_file_objs)
                pickup_new_raw = read_uploaded_excels(pickup_file_objs)

            st.write("New GAP raw rows:", len(gap_new_raw))
            st.write("New pickup raw rows:", len(pickup_new_raw))

            if unknown_files:
                st.warning(f"Skipped unknown files: {unknown_files}")

            with st.spinner("Standardizing files..."):
                gap_new = standardize_gap(gap_new_raw) if not gap_new_raw.empty else pd.DataFrame()
                pickup_new = standardize_pickups(pickup_new_raw) if not pickup_new_raw.empty else pd.DataFrame()
                pickup_stops_new = (
                    consolidate_physical_pickups(pickup_new)
                    if not pickup_new.empty else pd.DataFrame()
                )

            st.write("Standardized GAP rows:", len(gap_new))
            st.write("Standardized pickup rows:", len(pickup_new))
            st.write("Consolidated pickup stops:", len(pickup_stops_new))

            with st.spinner("Appending and deduplicating master tables..."):
                gap_master_updated = append_dedup(
                    gap_master,
                    gap_new,
                    key_cols=["scan_date", "route", "stop_order", "address", "activity_dt"]
                )
                pickup_master_updated = append_dedup(
                    pickup_master,
                    pickup_new,
                    key_cols=["pickup_date", "route", "account_name", "address", "pickup_dt", "confirmation_no"]
                )
                pickup_stops_master_updated = append_dedup(
                    pickup_stops_master,
                    pickup_stops_new,
                    key_cols=["pickup_key"]
                )

                log_new = build_ingestion_log_entries(gap_file_objs, pickup_file_objs)
                ingestion_log_updated = append_ingestion_log(ingestion_log, log_new)

                save_master_tables(
                    paths,
                    gap_master_updated,
                    pickup_master_updated,
                    pickup_stops_master_updated,
                    ingestion_log_updated,
                )

            st.success("Master data updated and saved.")

            st.write("Updated GAP master rows:", len(gap_master_updated))
            st.write("Updated pickup master rows:", len(pickup_master_updated))
            st.write("Updated pickup stops master rows:", len(pickup_stops_master_updated))

            st.markdown("### Sample of updated pickup stops")
            if not pickup_stops_master_updated.empty:
                st.dataframe(pickup_stops_master_updated.head(25), use_container_width=True)
            else:
                st.info("No pickup stops are currently in master data.")

    st.markdown("---")
    st.subheader("Download current master files")

    if not gap_master.empty:
        st.download_button(
            "Download gap_master.csv",
            data=gap_master.to_csv(index=False).encode("utf-8"),
            file_name="gap_master.csv",
            mime="text/csv",
        )

    if not pickup_master.empty:
        st.download_button(
            "Download pickup_master.csv",
            data=pickup_master.to_csv(index=False).encode("utf-8"),
            file_name="pickup_master.csv",
            mime="text/csv",
        )

    if not pickup_stops_master.empty:
        st.download_button(
            "Download pickup_stops_master.csv",
            data=pickup_stops_master.to_csv(index=False).encode("utf-8"),
            file_name="pickup_stops_master.csv",
            mime="text/csv",
        )

    if not ingestion_log.empty:
        st.download_button(
            "Download ingestion_log.csv",
            data=ingestion_log.to_csv(index=False).encode("utf-8"),
            file_name="ingestion_log.csv",
            mime="text/csv",
        )

elif page == "Analyze Existing Master":
    st.header("Analyze Existing Master")

    if gap_master.empty or pickup_stops_master.empty:
        st.info("Load or create master data first on the 'Update Master Data' page.")
    else:
        c1, c2, c3 = st.columns(3)

        with c1:
            available_routes = sorted([x for x in gap_master["route"].dropna().unique().tolist()])
            selected_routes = st.multiselect(
                "Routes",
                options=available_routes,
                default=available_routes[:5] if len(available_routes) > 5 else available_routes,
            )

        with c2:
            min_gap_date = pd.to_datetime(gap_master["scan_date"]).min().date()
            max_gap_date = pd.to_datetime(gap_master["scan_date"]).max().date()
            date_range = st.date_input(
                "Date range",
                value=(min_gap_date, max_gap_date),
                min_value=min_gap_date,
                max_value=max_gap_date,
            )

        with c3:
            tolerance = st.number_input(
                "Pickup/GAP match tolerance (min)",
                min_value=1,
                max_value=30,
                value=APP_CONFIG["pickup_match_tolerance_min"],
                step=1,
            )

        run_analysis = st.button("Run Analysis")

        if run_analysis:
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range

            gap_f = gap_master.copy()
            pickup_stops_f = pickup_stops_master.copy()

            gap_f = gap_f[
                (pd.to_datetime(gap_f["scan_date"]).dt.date >= start_date) &
                (pd.to_datetime(gap_f["scan_date"]).dt.date <= end_date)
            ]
            pickup_stops_f = pickup_stops_f[
                (pd.to_datetime(pickup_stops_f["pickup_date"]).dt.date >= start_date) &
                (pd.to_datetime(pickup_stops_f["pickup_date"]).dt.date <= end_date)
            ]

            if selected_routes:
                gap_f = gap_f[gap_f["route"].isin(selected_routes)]
                pickup_stops_f = pickup_stops_f[pickup_stops_f["route"].isin(selected_routes)]

            with st.spinner("Building summaries..."):
                route_day_summary = build_route_day_summary(gap_f, pickup_stops_f)
                best_matches, match_report = match_pickups_to_gap(
                    gap_f,
                    pickup_stops_f,
                    tolerance_min=tolerance
                )

            st.subheader("Route-day summary")
            st.dataframe(route_day_summary, use_container_width=True)

            st.subheader("Pickup ↔ GAP match report")
            st.dataframe(match_report, use_container_width=True)

            st.subheader("Best matches")
            st.dataframe(best_matches.head(100), use_container_width=True)

            st.download_button(
                "Download route_day_summary.csv",
                data=route_day_summary.to_csv(index=False).encode("utf-8"),
                file_name="route_day_summary.csv",
                mime="text/csv",
            )

            st.download_button(
                "Download best_matches.csv",
                data=best_matches.to_csv(index=False).encode("utf-8"),
                file_name="best_matches.csv",
                mime="text/csv",
            )

elif page == "Settings / Exceptions":
    st.header("Settings / Exceptions")

    settings_path = Path(paths["base"]) / "route_exceptions.json"
    if settings_path.exists():
        current_settings = json.loads(settings_path.read_text())
    else:
        current_settings = {
            "bulk_route_overrides": [],
            "fixed_customer_route_assignments": [],
            "special_route_start_exceptions": [],
            "notes": ""
        }

    edited_json = st.text_area(
        "Edit route exceptions JSON",
        value=json.dumps(current_settings, indent=2),
        height=400,
    )

    if st.button("Save Exceptions"):
        try:
            parsed = json.loads(edited_json)
            settings_path.write_text(json.dumps(parsed, indent=2))
            st.success("Exceptions saved.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
