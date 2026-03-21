
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils_processing import (
    APP_CONFIG,
    ensure_data_dirs,
    load_master_tables,
    save_master_tables,
    load_anchor_references,
    append_or_replace_anchor_reference,
    normalize_work_area_key,
    format_work_area_display,
    read_uploaded_excels,
    read_stop_detail_file,
    standardize_gap,
    standardize_pickups,
    standardize_stop_detail,
    consolidate_physical_pickups,
    append_dedup,
    build_ingestion_log_entries,
    append_ingestion_log,
    build_route_day_summary,
    match_pickups_to_gap,
    cross_reference_stop_detail,
)

st.set_page_config(page_title="FedEx Pickup Route Optimization", layout="wide")

st.title("FedEx Pickup Route Optimization")
st.caption("Upload GAP, pickup, and stop-detail files, update master data, and run analysis only when needed.")

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

gap_master, pickup_master, pickup_stops_master, stop_detail_master, ingestion_log = load_master_tables(paths)
anchor_refs = load_anchor_references(paths)

def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

if page == "Update Master Data":
    st.header("Update Master Data")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Current master status")
        st.write(f"GAP master rows: **{len(gap_master):,}**")
        st.write(f"Pickup master rows: **{len(pickup_master):,}**")
        st.write(f"Pickup stops master rows: **{len(pickup_stops_master):,}**")
        st.write(f"Stop detail master rows: **{len(stop_detail_master):,}**")
        st.write(f"Ingestion log rows: **{len(ingestion_log):,}**")

    with c2:
        if not gap_master.empty and "scan_date" in gap_master.columns:
            gap_dates = pd.to_datetime(gap_master["scan_date"], errors="coerce").dropna()
            if not gap_dates.empty:
                st.write("GAP dates:", f"{gap_dates.min().date()} to {gap_dates.max().date()}")

        if not pickup_master.empty and "pickup_date" in pickup_master.columns:
            pickup_dates = pd.to_datetime(pickup_master["pickup_date"], errors="coerce").dropna()
            if not pickup_dates.empty:
                st.write("Pickup dates:", f"{pickup_dates.min().date()} to {pickup_dates.max().date()}")

        if not stop_detail_master.empty and "scan_date" in stop_detail_master.columns:
            stop_dates = pd.to_datetime(stop_detail_master["scan_date"], errors="coerce").dropna()
            if not stop_dates.empty:
                st.write("Stop detail dates:", f"{stop_dates.min().date()} to {stop_dates.max().date()}")

    st.markdown("---")

    with st.form("update_form", clear_on_submit=False):
        st.subheader("Upload new Excel files")
        uploaded_files = st.file_uploader(
            "Upload GAP, pickup, and stop-detail Excel files together",
            type=["xlsx", "xls"],
            accept_multiple_files=True,
            key="mixed_uploads",
        )
        run_update = st.form_submit_button("Update Master Data")

    def detect_file_type(file_obj):
        try:
            # read_html first because stop-detail .xls can be HTML export
            try:
                tables = pd.read_html(file_obj)
                if tables:
                    t = max(tables, key=lambda x: x.shape[0] * x.shape[1]).copy()
                    if t.shape[0] >= 2:
                        t.columns = t.iloc[0]
                        t = t.iloc[1:].copy()
                    cols = set(map(str, t.columns))
                    if {"Date", "Route", "Stop Order", "Stop Type"}.issubset(cols) and (
                        {"Activity Time", "Ready Time", "Close Time"}.intersection(cols)
                    ):
                        file_obj.seek(0)
                        return "stop_detail"
            except Exception:
                pass

            file_obj.seek(0)
            preview_df = pd.read_excel(file_obj, nrows=5)
            cols = set(preview_df.columns.astype(str))

            if {"Scan Date", "Route", "Stop Order", "Stop Type", "Activity"}.issubset(cols):
                file_obj.seek(0)
                return "gap"
            if {"Pickup Date", "Work Area #", "Ready Pickup Time", "Close Pickup Time", "Pickup Time"}.issubset(cols):
                file_obj.seek(0)
                return "pickup"
            if {"Date", "Route", "Stop Order", "Stop Type"}.issubset(cols) and (
                {"Activity Time", "Ready Time", "Close Time"}.intersection(cols)
            ):
                file_obj.seek(0)
                return "stop_detail"
        except Exception:
            pass
        try:
            file_obj.seek(0)
        except Exception:
            pass
        return "unknown"

    if uploaded_files:
        detected_gap_files = []
        detected_pickup_files = []
        detected_stop_detail_files = []
        unknown_files = []

        for f in uploaded_files:
            ftype = detect_file_type(f)
            if ftype == "gap":
                detected_gap_files.append(f.name)
            elif ftype == "pickup":
                detected_pickup_files.append(f.name)
            elif ftype == "stop_detail":
                detected_stop_detail_files.append(f.name)
            else:
                unknown_files.append(f.name)

        st.markdown("### Detected file types")
        st.write(f"GAP files detected: **{len(detected_gap_files)}**")
        if detected_gap_files:
            st.write(detected_gap_files)

        st.write(f"Pickup files detected: **{len(detected_pickup_files)}**")
        if detected_pickup_files:
            st.write(detected_pickup_files)

        st.write(f"Stop-detail files detected: **{len(detected_stop_detail_files)}**")
        if detected_stop_detail_files:
            st.write(detected_stop_detail_files)

        if unknown_files:
            st.write(f"Unknown files: **{len(unknown_files)}**")
            st.write(unknown_files)

    if run_update:
        if not uploaded_files:
            st.warning("Please upload at least one Excel file.")
        else:
            gap_file_objs = []
            pickup_file_objs = []
            stop_detail_file_objs = []
            unknown_files = []

            for f in uploaded_files:
                ftype = detect_file_type(f)
                if ftype == "gap":
                    gap_file_objs.append(f)
                elif ftype == "pickup":
                    pickup_file_objs.append(f)
                elif ftype == "stop_detail":
                    stop_detail_file_objs.append(f)
                else:
                    unknown_files.append(f.name)

            with st.spinner("Reading uploaded files..."):
                gap_new_raw = read_uploaded_excels(gap_file_objs)
                pickup_new_raw = read_uploaded_excels(pickup_file_objs)

                stop_detail_raw_frames = []
                for f in stop_detail_file_objs:
                    f.seek(0)
                    raw = read_stop_detail_file(f)
                    if not raw.empty:
                        stop_detail_raw_frames.append(raw)
                stop_detail_new_raw = pd.concat(stop_detail_raw_frames, ignore_index=True) if stop_detail_raw_frames else pd.DataFrame()

            st.write("New GAP raw rows:", len(gap_new_raw))
            st.write("New pickup raw rows:", len(pickup_new_raw))
            st.write("New stop-detail raw rows:", len(stop_detail_new_raw))

            if unknown_files:
                st.warning(f"Skipped unknown files: {unknown_files}")

            with st.spinner("Standardizing files..."):
                gap_new = standardize_gap(gap_new_raw) if not gap_new_raw.empty else pd.DataFrame()
                pickup_new = standardize_pickups(pickup_new_raw) if not pickup_new_raw.empty else pd.DataFrame()
                pickup_stops_new = consolidate_physical_pickups(pickup_new) if not pickup_new.empty else pd.DataFrame()
                stop_detail_new = standardize_stop_detail(stop_detail_new_raw) if not stop_detail_new_raw.empty else pd.DataFrame()

            st.write("Standardized GAP rows:", len(gap_new))
            st.write("Standardized pickup rows:", len(pickup_new))
            st.write("Consolidated pickup stops:", len(pickup_stops_new))
            st.write("Standardized stop-detail rows:", len(stop_detail_new))

            with st.spinner("Appending and deduplicating master tables..."):
                gap_master_updated = append_dedup(
                    gap_master, gap_new,
                    key_cols=["scan_date", "route", "stop_order", "address", "activity_dt"]
                )
                pickup_master_updated = append_dedup(
                    pickup_master, pickup_new,
                    key_cols=["pickup_date", "route", "account_name", "address", "pickup_dt", "confirmation_no"]
                )
                pickup_stops_master_updated = append_dedup(
                    pickup_stops_master, pickup_stops_new,
                    key_cols=["pickup_key"]
                )
                stop_detail_master_updated = append_dedup(
                    stop_detail_master, stop_detail_new,
                    key_cols=["scan_date", "route_key", "stop_order", "address_norm", "activity_dt", "stop_type"]
                )

                log_new = build_ingestion_log_entries(gap_file_objs, pickup_file_objs, stop_detail_file_objs)
                ingestion_log_updated = append_ingestion_log(ingestion_log, log_new)

                save_master_tables(
                    paths,
                    gap_master_updated,
                    pickup_master_updated,
                    pickup_stops_master_updated,
                    stop_detail_master_updated,
                    ingestion_log_updated,
                )

            st.success("Master data updated and saved.")
            st.write("Updated GAP master rows:", len(gap_master_updated))
            st.write("Updated pickup master rows:", len(pickup_master_updated))
            st.write("Updated pickup stops master rows:", len(pickup_stops_master_updated))
            st.write("Updated stop-detail master rows:", len(stop_detail_master_updated))

            if not stop_detail_master_updated.empty:
                st.markdown("### Sample of updated stop-detail master")
                st.dataframe(stop_detail_master_updated.head(25), use_container_width=True)


    st.markdown("---")
    st.subheader("Anchor Area Manager")
    st.caption("Upload anchor reference images/files by work area. New uploads for the same work area replace the active version and archive older ones.")

    with st.form("anchor_upload_form", clear_on_submit=True):
        ac1, ac2, ac3 = st.columns([1, 1, 1.4])
        with ac1:
            anchor_work_area_input = st.text_input(
                "Work Area / Route Number",
                help="Enter 745, 0745, or a string like W1-745. The app stores it internally as 745."
            )
        with ac2:
            anchor_wave = st.selectbox("Wave (optional)", ["", "W1", "W2"], index=0)
        with ac3:
            anchor_effective_date = st.date_input("Effective Date (optional)", value=None)

        anchor_notes = st.text_input("Notes (optional)")
        anchor_upload = st.file_uploader(
            "Upload anchor image/file",
            type=["png", "jpg", "jpeg", "webp", "pdf", "kml", "kmz", "geojson", "json"],
            accept_multiple_files=False,
            key="anchor_upload_single",
        )
        save_anchor = st.form_submit_button("Save / Replace Anchor")

    if save_anchor:
        if anchor_upload is None:
            st.warning("Please upload an anchor file before saving.")
        elif not normalize_work_area_key(anchor_work_area_input):
            st.warning("Please enter a valid Work Area / Route Number before saving.")
        else:
            updated_anchor_refs = append_or_replace_anchor_reference(
                paths=paths,
                existing_refs=anchor_refs,
                uploaded_file=anchor_upload,
                work_area_input=anchor_work_area_input,
                wave=anchor_wave,
                effective_date=anchor_effective_date,
                notes=anchor_notes,
            )
            anchor_refs = updated_anchor_refs
            st.success(
                f"Saved anchor for work area {format_work_area_display(normalize_work_area_key(anchor_work_area_input))}."
            )

    st.markdown("### Current Anchor References")
    if anchor_refs.empty:
        st.info("No anchor references uploaded yet.")
    else:
        anchor_display = anchor_refs.copy()
        if "work_area_key" in anchor_display.columns:
            anchor_display["display_work_area"] = anchor_display["work_area_key"].astype(str).map(format_work_area_display)
        display_cols = [
            "display_work_area",
            "work_area_key",
            "wave",
            "version",
            "is_active",
            "effective_date",
            "uploaded_at",
            "original_file_name",
            "saved_file_name",
            "notes",
        ]
        display_cols = [c for c in display_cols if c in anchor_display.columns]
        st.dataframe(
            anchor_display.sort_values(["work_area_key", "version"], ascending=[True, False])[display_cols],
            use_container_width=True,
        )

    st.subheader("Download current master files")

    if not gap_master.empty:
        st.download_button(
            "Download gap_master.csv",
            data=_csv_bytes(gap_master),
            file_name="gap_master.csv",
            mime="text/csv",
        )

    if not pickup_master.empty:
        st.download_button(
            "Download pickup_master.csv",
            data=_csv_bytes(pickup_master),
            file_name="pickup_master.csv",
            mime="text/csv",
        )

    if not pickup_stops_master.empty:
        st.download_button(
            "Download pickup_stops_master.csv",
            data=_csv_bytes(pickup_stops_master),
            file_name="pickup_stops_master.csv",
            mime="text/csv",
        )

    if not stop_detail_master.empty:
        st.download_button(
            "Download stop_detail_master.csv",
            data=_csv_bytes(stop_detail_master),
            file_name="stop_detail_master.csv",
            mime="text/csv",
        )

    if not ingestion_log.empty:
        st.download_button(
            "Download ingestion_log.csv",
            data=_csv_bytes(ingestion_log),
            file_name="ingestion_log.csv",
            mime="text/csv",
        )

elif page == "Analyze Existing Master":
    st.header("Analyze Existing Master")

    if gap_master.empty and pickup_stops_master.empty and stop_detail_master.empty:
        st.info("No master data available yet. Use Update Master Data first.")
    else:
        route_options = sorted([int(r) for r in pd.Series(pd.concat([
            gap_master["route"] if "route" in gap_master.columns else pd.Series(dtype="Int64"),
            pickup_stops_master["route"] if "route" in pickup_stops_master.columns else pd.Series(dtype="Int64")
        ], ignore_index=True)).dropna().unique()])

        c1, c2, c3 = st.columns(3)
        with c1:
            selected_routes = st.multiselect("Filter routes", options=route_options, default=[])

        with c2:
            all_dates = []
            for df, col in [
                (gap_master, "scan_date"),
                (pickup_stops_master, "pickup_date"),
                (stop_detail_master, "scan_date"),
            ]:
                if not df.empty and col in df.columns:
                    vals = pd.to_datetime(df[col], errors="coerce").dropna()
                    if not vals.empty:
                        all_dates.extend(vals.dt.date.tolist())

            if all_dates:
                min_date = min(all_dates)
                max_date = max(all_dates)
                date_range = st.date_input(
                    "Date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                )
            else:
                date_range = None

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
            stop_detail_f = stop_detail_master.copy()

            if start_date and end_date:
                if not gap_f.empty:
                    gap_f = gap_f[
                        (pd.to_datetime(gap_f["scan_date"]).dt.date >= start_date) &
                        (pd.to_datetime(gap_f["scan_date"]).dt.date <= end_date)
                    ]
                if not pickup_stops_f.empty:
                    pickup_stops_f = pickup_stops_f[
                        (pd.to_datetime(pickup_stops_f["pickup_date"]).dt.date >= start_date) &
                        (pd.to_datetime(pickup_stops_f["pickup_date"]).dt.date <= end_date)
                    ]
                if not stop_detail_f.empty:
                    stop_detail_f = stop_detail_f[
                        (pd.to_datetime(stop_detail_f["scan_date"]).dt.date >= start_date) &
                        (pd.to_datetime(stop_detail_f["scan_date"]).dt.date <= end_date)
                    ]

            if selected_routes:
                if not gap_f.empty and "route" in gap_f.columns:
                    gap_f = gap_f[gap_f["route"].isin(selected_routes)]
                if not pickup_stops_f.empty and "route" in pickup_stops_f.columns:
                    pickup_stops_f = pickup_stops_f[pickup_stops_f["route"].isin(selected_routes)]
                if not stop_detail_f.empty and "route_key" in stop_detail_f.columns:
                    stop_detail_f = stop_detail_f[stop_detail_f["route_key"].astype(str).isin([str(r) for r in selected_routes])]

            with st.spinner("Building summaries..."):
                route_day_summary = build_route_day_summary(gap_f, pickup_stops_f)
                best_matches, match_report = match_pickups_to_gap(
                    gap_f,
                    pickup_stops_f,
                    tolerance_min=tolerance
                )
                stop_detail_xref = cross_reference_stop_detail(stop_detail_f, gap_f, pickup_master)

            st.subheader("Route-day summary")
            st.dataframe(route_day_summary, use_container_width=True)

            st.subheader("Pickup ↔ GAP match report")
            st.dataframe(match_report, use_container_width=True)

            st.subheader("Best matches")
            st.dataframe(best_matches.head(100), use_container_width=True)

            if not stop_detail_f.empty:
                st.subheader("Stop-detail family summary")
                family_summary = (
                    stop_detail_f.groupby(["scan_date", "route_key", "stop_family"], dropna=False)
                    .agg(
                        stops=("stop_order", "count"),
                        total_pkgs=("package_count", "sum"),
                        avg_gap_minutes=("gap_minutes", "mean"),
                        pickup_stops=("is_pickup", "sum"),
                        delivery_stops=("is_delivery", "sum"),
                    )
                    .reset_index()
                    .sort_values(["scan_date", "route_key", "stop_family"])
                )
                st.dataframe(family_summary, use_container_width=True)

                st.subheader("Stop-detail cross-reference")
                st.dataframe(stop_detail_xref.head(200), use_container_width=True)

                cxa, cxb = st.columns(2)
                with cxa:
                    xref_summary = pd.DataFrame({
                        "metric": [
                            "stop_detail_rows",
                            "matched_to_gap",
                            "matched_to_pickup_sheet",
                            "matched_to_gap_pct",
                            "matched_to_pickup_pct",
                        ],
                        "value": [
                            len(stop_detail_xref),
                            int(stop_detail_xref["matched_to_gap"].fillna(False).sum()) if not stop_detail_xref.empty else 0,
                            int(stop_detail_xref["matched_to_pickup_sheet"].fillna(False).sum()) if not stop_detail_xref.empty else 0,
                            round(float(stop_detail_xref["matched_to_gap"].fillna(False).mean()) * 100, 1) if not stop_detail_xref.empty else 0.0,
                            round(float(stop_detail_xref["matched_to_pickup_sheet"].fillna(False).mean()) * 100, 1) if not stop_detail_xref.empty else 0.0,
                        ]
                    })
                    st.dataframe(xref_summary, use_container_width=True)

            st.download_button(
                "Download route_day_summary.csv",
                data=_csv_bytes(route_day_summary),
                file_name="route_day_summary.csv",
                mime="text/csv",
            )

            st.download_button(
                "Download best_matches.csv",
                data=_csv_bytes(best_matches),
                file_name="best_matches.csv",
                mime="text/csv",
            )

            if not stop_detail_f.empty:
                st.download_button(
                    "Download stop_detail_cross_reference.csv",
                    data=_csv_bytes(stop_detail_xref),
                    file_name="stop_detail_cross_reference.csv",
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
