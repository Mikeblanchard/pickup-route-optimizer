import json
from pathlib import Path

import pandas as pd
import streamlit as st

from utils_processing import (
    APP_CONFIG,
    append_dedup,
    append_ingestion_log,
    append_or_replace_anchor_reference,
    build_courier_day_changes,
    build_ingestion_log_entries,
    build_route_day_summary,
    consolidate_physical_pickups,
    cross_reference_stop_detail,
    ensure_data_dirs,
    enrich_gap_route_metrics_from_stops,
    format_work_area_display,
    load_anchor_references,
    load_master_tables,
    match_pickups_to_gap,
    normalize_work_area_key,
    read_gap_html_file,
    read_stop_detail_file,
    read_uploaded_excels,
    save_master_tables,
    save_uploaded_source_files,
    standardize_gap,
    standardize_gap_html_stops,
    standardize_gap_route_metrics,
    standardize_pickups,
    standardize_stop_detail,
)

st.set_page_config(page_title="Pickup Route Optimizer", layout="wide")

st.title("Pickup Route Optimizer")
st.caption("Master data, anchor references, and courier performance tracking from GAP / pickup / stop-detail files.")

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
st.sidebar.write("Saturday:", APP_CONFIG["weekend_starts"]["Saturday"])
st.sidebar.write("Sunday:", APP_CONFIG["weekend_starts"]["Sunday"])
st.sidebar.write("Start station minutes:", APP_CONFIG["start_of_day_station_min"])
st.sidebar.write("End station minutes:", APP_CONFIG["end_of_day_station_min"])

storage_root = st.sidebar.text_input(
    "Data folder",
    value="streamlit_data",
    help="Local app storage folder. For cloud deployment this should later be replaced with persistent storage.",
)
paths = ensure_data_dirs(storage_root)

gap_master, pickup_master, pickup_stops_master, stop_detail_master, gap_route_metrics_master, ingestion_log = load_master_tables(paths)
anchor_refs = load_anchor_references(paths)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _date_range_filter(df: pd.DataFrame, date_col: str, start_date, end_date) -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns or start_date is None or end_date is None:
        return df
    d = df.copy()
    vals = pd.to_datetime(d[date_col], errors="coerce")
    mask = (vals.dt.date >= start_date) & (vals.dt.date <= end_date)
    return d.loc[mask].copy()


def _route_filter(df: pd.DataFrame, route_col: str, selected_routes) -> pd.DataFrame:
    if df is None or df.empty or route_col not in df.columns or not selected_routes:
        return df
    return df[df[route_col].astype(str).isin([str(x) for x in selected_routes])].copy()


def _detect_file_type(file_obj) -> str:
    name = getattr(file_obj, "name", "").lower()

    try:
        file_obj.seek(0)
        head = file_obj.getvalue()[:20000]
        head_text = head.decode("utf-8", errors="ignore").lower()
    except Exception:
        head_text = ""

    if name.endswith((".html", ".htm")):
        if "gap reports for" in head_text and "total paid hrs" in head_text:
            return "gap_html"

    try:
        file_obj.seek(0)
        preview_tables = pd.read_html(file_obj)
        if preview_tables:
            biggest = max(preview_tables, key=lambda x: x.shape[0] * x.shape[1]).copy()
            if biggest.shape[0] >= 2:
                tmp = biggest.copy()
                tmp.columns = tmp.iloc[0]
                tmp = tmp.iloc[1:].copy()
                cols = set(map(str, tmp.columns))
                if {"Date", "Route", "Stop Order", "Stop Type"}.issubset(cols) and {"Activity Time", "Ready Time", "Close Time"}.intersection(cols):
                    file_obj.seek(0)
                    return "stop_detail"
    except Exception:
        pass

    for reader in ("excel", "csv"):
        try:
            file_obj.seek(0)
            if reader == "excel":
                preview_df = pd.read_excel(file_obj, nrows=5)
            else:
                preview_df = pd.read_csv(file_obj, nrows=5)
            cols = set(preview_df.columns.astype(str))
            if {"Scan Date", "Route", "Stop Order", "Stop Type", "Activity"}.issubset(cols):
                file_obj.seek(0)
                return "gap_excel"
            if {"Pickup Date", "Work Area #", "Ready Pickup Time", "Close Pickup Time", "Pickup Time"}.issubset(cols):
                file_obj.seek(0)
                return "pickup"
            if {"Date", "Route", "Stop Order", "Stop Type"}.issubset(cols) and {"Activity Time", "Ready Time", "Close Time"}.intersection(cols):
                file_obj.seek(0)
                return "stop_detail"
        except Exception:
            pass

    try:
        file_obj.seek(0)
    except Exception:
        pass
    return "unknown"


if page == "Update Master Data":
    st.header("Update Master Data")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Current master status")
        st.write(f"GAP stop rows: **{len(gap_master):,}**")
        st.write(f"Pickup master rows: **{len(pickup_master):,}**")
        st.write(f"Pickup stops master rows: **{len(pickup_stops_master):,}**")
        st.write(f"Stop detail master rows: **{len(stop_detail_master):,}**")
        st.write(f"Courier day metrics rows: **{len(gap_route_metrics_master):,}**")
        st.write(f"Ingestion log rows: **{len(ingestion_log):,}**")

    with c2:
        for label, df, col in [
            ("GAP dates", gap_master, "scan_date"),
            ("Pickup dates", pickup_master, "pickup_date"),
            ("Stop-detail dates", stop_detail_master, "scan_date"),
            ("Courier metrics dates", gap_route_metrics_master, "scan_date"),
        ]:
            if not df.empty and col in df.columns:
                vals = pd.to_datetime(df[col], errors="coerce").dropna()
                if not vals.empty:
                    st.write(label + ":", f"{vals.min().date()} to {vals.max().date()}")

    st.markdown("---")

    with st.form("update_form", clear_on_submit=False):
        st.subheader("Upload new source files")
        uploaded_files = st.file_uploader(
            "Upload GAP Excel, GAP saved HTML, pickup, and stop-detail files together",
            type=["xlsx", "xls", "csv", "html", "htm"],
            accept_multiple_files=True,
            key="mixed_uploads",
        )
        run_update = st.form_submit_button("Update Master Data")

    if uploaded_files:
        detected = {"gap_excel": [], "gap_html": [], "pickup": [], "stop_detail": [], "unknown": []}
        for f in uploaded_files:
            detected[_detect_file_type(f)].append(f.name)

        st.markdown("### Detected file types")
        st.write(f"GAP Excel files: **{len(detected['gap_excel'])}**")
        if detected["gap_excel"]:
            st.write(detected["gap_excel"])
        st.write(f"GAP saved HTML files: **{len(detected['gap_html'])}**")
        if detected["gap_html"]:
            st.write(detected["gap_html"])
        st.write(f"Pickup files: **{len(detected['pickup'])}**")
        if detected["pickup"]:
            st.write(detected["pickup"])
        st.write(f"Stop-detail files: **{len(detected['stop_detail'])}**")
        if detected["stop_detail"]:
            st.write(detected["stop_detail"])
        if detected["unknown"]:
            st.warning(f"Unknown files skipped if you run update: {detected['unknown']}")

    if run_update:
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            gap_excel_files, gap_html_files, pickup_files, stop_detail_files, unknown_files = [], [], [], [], []
            for f in uploaded_files:
                ftype = _detect_file_type(f)
                if ftype == "gap_excel":
                    gap_excel_files.append(f)
                elif ftype == "gap_html":
                    gap_html_files.append(f)
                elif ftype == "pickup":
                    pickup_files.append(f)
                elif ftype == "stop_detail":
                    stop_detail_files.append(f)
                else:
                    unknown_files.append(f.name)

            saved_sources = save_uploaded_source_files(paths, uploaded_files)

            with st.spinner("Reading uploaded files..."):
                gap_excel_raw = read_uploaded_excels(gap_excel_files)
                pickup_raw = read_uploaded_excels(pickup_files)

                html_stop_frames = []
                html_metric_frames = []
                for f in gap_html_files:
                    f.seek(0)
                    stop_raw, metric_raw = read_gap_html_file(f)
                    if not stop_raw.empty:
                        html_stop_frames.append(stop_raw)
                    if not metric_raw.empty:
                        html_metric_frames.append(metric_raw)
                gap_html_stops_raw = pd.concat(html_stop_frames, ignore_index=True) if html_stop_frames else pd.DataFrame()
                gap_html_metrics_raw = pd.concat(html_metric_frames, ignore_index=True) if html_metric_frames else pd.DataFrame()

                stop_detail_frames = []
                for f in stop_detail_files:
                    f.seek(0)
                    raw = read_stop_detail_file(f)
                    if not raw.empty:
                        stop_detail_frames.append(raw)
                stop_detail_raw = pd.concat(stop_detail_frames, ignore_index=True) if stop_detail_frames else pd.DataFrame()

            with st.spinner("Standardizing files..."):
                gap_excel_std = standardize_gap(gap_excel_raw) if not gap_excel_raw.empty else pd.DataFrame()
                gap_html_stops_std = standardize_gap_html_stops(gap_html_stops_raw) if not gap_html_stops_raw.empty else pd.DataFrame()
                gap_new = pd.concat([gap_excel_std, gap_html_stops_std], ignore_index=True) if not gap_excel_std.empty or not gap_html_stops_std.empty else pd.DataFrame()

                pickup_new = standardize_pickups(pickup_raw) if not pickup_raw.empty else pd.DataFrame()
                pickup_stops_new = consolidate_physical_pickups(pickup_new) if not pickup_new.empty else pd.DataFrame()
                stop_detail_new = standardize_stop_detail(stop_detail_raw) if not stop_detail_raw.empty else pd.DataFrame()

                gap_route_metrics_new = standardize_gap_route_metrics(gap_html_metrics_raw) if not gap_html_metrics_raw.empty else pd.DataFrame()
                if not gap_route_metrics_new.empty and not gap_html_stops_std.empty:
                    gap_route_metrics_new = enrich_gap_route_metrics_from_stops(gap_route_metrics_new, gap_html_stops_std)

            st.write("New GAP Excel raw rows:", len(gap_excel_raw))
            st.write("New GAP HTML stop rows:", len(gap_html_stops_raw))
            st.write("New GAP HTML route metrics rows:", len(gap_html_metrics_raw))
            st.write("New pickup raw rows:", len(pickup_raw))
            st.write("New stop-detail raw rows:", len(stop_detail_raw))
            if saved_sources is not None and not saved_sources.empty:
                st.write("Saved source files:", len(saved_sources))
                st.dataframe(saved_sources[[c for c in ["saved_name", "source_name", "source_type", "saved_at"] if c in saved_sources.columns]], use_container_width=True)
            if unknown_files:
                st.warning(f"Skipped unknown files: {unknown_files}")

            st.write("Standardized GAP stop rows:", len(gap_new))
            st.write("Standardized pickup rows:", len(pickup_new))
            st.write("Consolidated pickup stop rows:", len(pickup_stops_new))
            st.write("Standardized stop-detail rows:", len(stop_detail_new))
            st.write("Standardized courier day metric rows:", len(gap_route_metrics_new))

            with st.spinner("Appending and deduplicating master tables..."):
                gap_master_updated = append_dedup(
                    gap_master,
                    gap_new,
                    key_cols=["scan_date", "route", "fedex_id", "stop_order", "address", "activity_dt"],
                )
                pickup_master_updated = append_dedup(
                    pickup_master,
                    pickup_new,
                    key_cols=["pickup_date", "route", "account_name", "address", "pickup_dt", "confirmation_no"],
                )
                pickup_stops_master_updated = append_dedup(
                    pickup_stops_master,
                    pickup_stops_new,
                    key_cols=["pickup_key"],
                )
                stop_detail_master_updated = append_dedup(
                    stop_detail_master,
                    stop_detail_new,
                    key_cols=["scan_date", "route_key", "stop_order", "address_norm", "activity_dt", "stop_type"],
                )
                gap_route_metrics_master_updated = append_dedup(
                    gap_route_metrics_master,
                    gap_route_metrics_new,
                    key_cols=["scan_date", "route", "fedex_id"],
                )
                log_new = build_ingestion_log_entries(gap_excel_files, pickup_files, stop_detail_files, gap_html_files)
                ingestion_log_updated = append_ingestion_log(ingestion_log, log_new)

                save_master_tables(
                    paths,
                    gap_master_updated,
                    pickup_master_updated,
                    pickup_stops_master_updated,
                    stop_detail_master_updated,
                    gap_route_metrics_master_updated,
                    ingestion_log_updated,
                )

            gap_master = gap_master_updated
            pickup_master = pickup_master_updated
            pickup_stops_master = pickup_stops_master_updated
            stop_detail_master = stop_detail_master_updated
            gap_route_metrics_master = gap_route_metrics_master_updated
            ingestion_log = ingestion_log_updated

            st.success("Master data updated and saved.")

            if not gap_route_metrics_master.empty:
                st.markdown("### Sample courier day metrics")
                st.dataframe(gap_route_metrics_master.sort_values(["scan_date", "route"]).tail(25), use_container_width=True)

    st.markdown("---")
    st.subheader("Anchor Area Manager")
    st.caption("Anchor uploads are reference only for now, not geometry yet.")

    with st.form("anchor_upload_form", clear_on_submit=True):
        ac1, ac2, ac3 = st.columns([1, 1, 1.4])
        with ac1:
            anchor_work_area_input = st.text_input("Work Area / Route Number", help="Enter 745, 0745, or W1-745. Stored internally as 745.")
        with ac2:
            anchor_wave = st.selectbox("Wave (optional)", ["", "W1", "W2"], index=0)
        with ac3:
            anchor_effective_date = st.date_input("Effective Date (optional)", value=None)

        anchor_notes = st.text_input("Notes (optional)")
        anchor_upload = st.file_uploader(
            "Upload anchor image/file",
            type=["png", "jpg", "jpeg", "webp", "pdf", "kml", "kmz", "geojson", "json", "snag"],
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
            anchor_refs = append_or_replace_anchor_reference(
                paths=paths,
                existing_refs=anchor_refs,
                uploaded_file=anchor_upload,
                work_area_input=anchor_work_area_input,
                wave=anchor_wave,
                effective_date=anchor_effective_date,
                notes=anchor_notes,
            )
            st.success(f"Saved anchor for work area {format_work_area_display(normalize_work_area_key(anchor_work_area_input))}.")

    st.markdown("### Current Anchor References")
    if anchor_refs.empty:
        st.info("No anchor references uploaded yet.")
    else:
        anchor_display = anchor_refs.copy()
        anchor_display["display_work_area"] = anchor_display["work_area_key"].astype(str).map(format_work_area_display)
        display_cols = [
            "display_work_area", "work_area_key", "wave", "version", "is_active",
            "effective_date", "uploaded_at", "original_file_name", "saved_file_name", "notes",
        ]
        st.dataframe(anchor_display.sort_values(["work_area_key", "version"], ascending=[True, False])[display_cols], use_container_width=True)

    st.subheader("Download current master files")
    download_items = [
        ("gap_master.csv", gap_master),
        ("pickup_master.csv", pickup_master),
        ("pickup_stops_master.csv", pickup_stops_master),
        ("stop_detail_master.csv", stop_detail_master),
        ("gap_route_metrics_master.csv", gap_route_metrics_master),
        ("ingestion_log.csv", ingestion_log),
    ]
    for file_name, df in download_items:
        if df is not None and not df.empty:
            st.download_button(f"Download {file_name}", data=_csv_bytes(df), file_name=file_name, mime="text/csv")

elif page == "Analyze Existing Master":
    st.header("Analyze Existing Master")

    if gap_master.empty and pickup_stops_master.empty and stop_detail_master.empty and gap_route_metrics_master.empty:
        st.info("No master data available yet. Use Update Master Data first.")
    else:
        route_values = []
        for df, col in [(gap_master, "route"), (pickup_stops_master, "route"), (gap_route_metrics_master, "route")]:
            if not df.empty and col in df.columns:
                route_values.extend(pd.Series(df[col]).dropna().astype(str).tolist())
        route_options = sorted({int(float(r)) for r in route_values if str(r).strip() not in {"", "nan"}})

        courier_options = []
        if not gap_route_metrics_master.empty and "courier_name" in gap_route_metrics_master.columns:
            courier_options = sorted([c for c in gap_route_metrics_master["courier_name"].dropna().astype(str).unique().tolist() if c.strip()])

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            selected_routes = st.multiselect("Filter routes", options=route_options, default=[])
        with c2:
            all_dates = []
            for df, col in [
                (gap_master, "scan_date"),
                (pickup_stops_master, "pickup_date"),
                (stop_detail_master, "scan_date"),
                (gap_route_metrics_master, "scan_date"),
            ]:
                if not df.empty and col in df.columns:
                    vals = pd.to_datetime(df[col], errors="coerce").dropna()
                    if not vals.empty:
                        all_dates.extend(vals.dt.date.tolist())
            if all_dates:
                min_date, max_date = min(all_dates), max(all_dates)
                date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            else:
                date_range = None
        with c3:
            selected_couriers = st.multiselect("Filter couriers", options=courier_options, default=[])
        with c4:
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

            gap_f = _date_range_filter(gap_master, "scan_date", start_date, end_date)
            pickup_stops_f = _date_range_filter(pickup_stops_master, "pickup_date", start_date, end_date)
            stop_detail_f = _date_range_filter(stop_detail_master, "scan_date", start_date, end_date)
            metrics_f = _date_range_filter(gap_route_metrics_master, "scan_date", start_date, end_date)

            gap_f = _route_filter(gap_f, "route", selected_routes)
            pickup_stops_f = _route_filter(pickup_stops_f, "route", selected_routes)
            stop_detail_f = _route_filter(stop_detail_f, "route_key", selected_routes)
            metrics_f = _route_filter(metrics_f, "route", selected_routes)

            if selected_couriers and not metrics_f.empty and "courier_name" in metrics_f.columns:
                metrics_f = metrics_f[metrics_f["courier_name"].astype(str).isin(selected_couriers)].copy()
                if not gap_f.empty and "fedex_id" in gap_f.columns and "fedex_id" in metrics_f.columns:
                    gap_f = gap_f[gap_f["fedex_id"].astype(str).isin(metrics_f["fedex_id"].astype(str).unique())].copy()

            with st.spinner("Building summaries..."):
                route_day_summary = build_route_day_summary(gap_f, pickup_stops_f)
                best_matches, match_report = match_pickups_to_gap(gap_f, pickup_stops_f, tolerance_min=tolerance)
                stop_detail_xref = cross_reference_stop_detail(stop_detail_f, gap_f, pickup_master) if not stop_detail_f.empty else pd.DataFrame()
                courier_changes = build_courier_day_changes(metrics_f)

            st.subheader("Courier day performance")
            if metrics_f.empty:
                st.info("No GAP HTML courier metrics found in the current filter range yet.")
            else:
                show_cols = [
                    "scan_date", "route", "courier_name", "fedex_id", "total_stops_actual",
                    "actual_sth_oa", "actual_sth_or", "actual_hr_oa_min", "actual_hr_or_min",
                    "gap_sum_minutes_all", "gap_sum_minutes_customer", "customer_stop_count",
                    "pickup_stop_count", "delivery_stop_count", "miles_on_road", "total_paid_hours_min",
                    "source_file",
                ]
                show_cols = [c for c in show_cols if c in metrics_f.columns]
                st.dataframe(metrics_f.sort_values(["scan_date", "route", "courier_name"])[show_cols], use_container_width=True)

                k1, k2, k3, k4 = st.columns(4)
                with k1:
                    st.metric("Courier-day rows", len(metrics_f))
                with k2:
                    st.metric("Avg ST/H OA", round(float(metrics_f["actual_sth_oa"].dropna().mean()), 2) if metrics_f["actual_sth_oa"].notna().any() else "—")
                with k3:
                    st.metric("Avg ST/H OR", round(float(metrics_f["actual_sth_or"].dropna().mean()), 2) if metrics_f["actual_sth_or"].notna().any() else "—")
                with k4:
                    st.metric("Avg GAP sum/day", round(float(metrics_f["gap_sum_minutes_all"].dropna().mean()), 1) if "gap_sum_minutes_all" in metrics_f.columns and metrics_f["gap_sum_minutes_all"].notna().any() else "—")

            st.subheader("Courier changes over uploaded days")
            if courier_changes.empty:
                st.info("Not enough courier-day rows to calculate changes yet.")
            else:
                change_cols = [
                    "scan_date", "route", "courier_name", "fedex_id", "previous_date", "previous_route",
                    "actual_sth_oa", "delta_actual_sth_oa",
                    "actual_sth_or", "delta_actual_sth_or",
                    "gap_sum_minutes_all", "delta_gap_sum_minutes_all",
                    "gap_sum_minutes_customer", "delta_gap_sum_minutes_customer",
                    "total_stops_actual", "delta_total_stops_actual",
                    "customer_stop_count", "delta_customer_stop_count",
                ]
                change_cols = [c for c in change_cols if c in courier_changes.columns]
                st.dataframe(courier_changes[change_cols], use_container_width=True)
                st.download_button(
                    "Download courier_changes_over_days.csv",
                    data=_csv_bytes(courier_changes),
                    file_name="courier_changes_over_days.csv",
                    mime="text/csv",
                )

            st.subheader("Route-day summary")
            if route_day_summary.empty:
                st.info("No route-day summary available for the current filters.")
            else:
                st.dataframe(route_day_summary, use_container_width=True)
                st.download_button("Download route_day_summary.csv", data=_csv_bytes(route_day_summary), file_name="route_day_summary.csv", mime="text/csv")

            st.subheader("Pickup ↔ GAP match report")
            if match_report.empty:
                st.info("No pickup/GAP matches available for the current filters.")
            else:
                st.dataframe(match_report, use_container_width=True)

            if not best_matches.empty:
                st.subheader("Best matches")
                st.dataframe(best_matches.head(200), use_container_width=True)
                st.download_button("Download best_matches.csv", data=_csv_bytes(best_matches), file_name="best_matches.csv", mime="text/csv")

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
            "notes": "",
        }

    edited_json = st.text_area("Edit route exceptions JSON", value=json.dumps(current_settings, indent=2), height=400)

    if st.button("Save Exceptions"):
        try:
            parsed = json.loads(edited_json)
            settings_path.write_text(json.dumps(parsed, indent=2))
            st.success("Exceptions saved.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
