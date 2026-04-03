import json
from pathlib import Path

import pandas as pd
import streamlit as st

import utils_processing as up

st.set_page_config(page_title="Pickup Route Optimizer", layout="wide")

st.title("Pickup Route Optimizer")
st.caption("Recent route and pickup analysis for your workgroup, plus route-cut planning.")

page = st.sidebar.radio(
    "Navigation",
    [
        "Update Master Data",
        "Analyze Existing Master",
        "Cut Run Optimizer",
        "Settings / Exceptions",
    ],
)

st.sidebar.markdown("---")
st.sidebar.write("Station:", up.APP_CONFIG["station_address"])
st.sidebar.write("W1 start:", up.APP_CONFIG["wave_starts"]["W1"])
st.sidebar.write("W2 start:", up.APP_CONFIG["wave_starts"]["W2"])
st.sidebar.write("Saturday:", up.APP_CONFIG["weekend_starts"]["Saturday"])
st.sidebar.write("Sunday:", up.APP_CONFIG["weekend_starts"]["Sunday"])
st.sidebar.write("Start station minutes:", up.APP_CONFIG["start_of_day_station_min"])
st.sidebar.write("End station minutes:", up.APP_CONFIG["end_of_day_station_min"])

storage_root = st.sidebar.text_input(
    "Data folder",
    value="streamlit_data",
    help="Local app storage folder. For cloud deployment this should later be replaced with persistent storage.",
)
paths = up.ensure_data_dirs(storage_root)

gap_master, pickup_master, pickup_stops_master, stop_detail_master, gap_route_metrics_master, ingestion_log = up.load_master_tables(paths)
if not gap_master.empty:
    gap_route_metrics_master = up.rebuild_gap_metrics_from_master(gap_master, gap_route_metrics_master)
anchor_refs = up.load_anchor_references(paths)


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _date_range_filter(df: pd.DataFrame, date_col: str, start_date, end_date) -> pd.DataFrame:
    if df is None or df.empty or date_col not in df.columns or start_date is None or end_date is None:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    d = df.copy()
    vals = pd.to_datetime(d[date_col], errors="coerce")
    mask = (vals.dt.date >= start_date) & (vals.dt.date <= end_date)
    return d.loc[mask].copy()


def _route_filter(df: pd.DataFrame, route_col: str, selected_routes) -> pd.DataFrame:
    if df is None or df.empty or route_col not in df.columns or not selected_routes:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    route_series = pd.to_numeric(df[route_col], errors="coerce").astype("Int64")
    allowed = {int(r) for r in selected_routes}
    return df.loc[route_series.isin(list(allowed))].copy()


def _round_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    num_cols = out.select_dtypes(include=["number"]).columns
    out[num_cols] = out[num_cols].round(1)
    return out


def _detect_file_type(file_obj) -> str:
    name = getattr(file_obj, "name", "").lower()

    try:
        file_obj.seek(0)
        head = file_obj.getvalue()[:20000]
        head_text = head.decode("utf-8", errors="ignore").lower()
    except Exception:
        head_text = ""

    if name.endswith((".html", ".htm")):
        if "gap reports for" in head_text and ("total paid hrs" in head_text or "route:" in head_text):
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

    try:
        file_obj.seek(0)
        if name.endswith(".csv"):
            preview_df = pd.read_csv(file_obj, nrows=5)
        else:
            preview_df = pd.read_excel(file_obj, nrows=5)
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
        for key, label in [
            ("gap_excel", "GAP Excel files"),
            ("gap_html", "GAP saved HTML files"),
            ("pickup", "Pickup files"),
            ("stop_detail", "Stop-detail files"),
        ]:
            st.write(f"{label}: **{len(detected[key])}**")
            if detected[key]:
                st.write(detected[key])
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

            with st.spinner("Reading uploaded files..."):
                gap_excel_raw = up.read_uploaded_excels(gap_excel_files)
                pickup_raw = up.read_uploaded_excels(pickup_files)

                html_stop_frames = []
                html_metric_frames = []
                for f in gap_html_files:
                    f.seek(0)
                    stop_raw, metric_raw = up.read_gap_html_file(f)
                    if not stop_raw.empty:
                        html_stop_frames.append(stop_raw)
                    if not metric_raw.empty:
                        html_metric_frames.append(metric_raw)
                gap_html_stops_raw = pd.concat(html_stop_frames, ignore_index=True) if html_stop_frames else pd.DataFrame()
                gap_html_metrics_raw = pd.concat(html_metric_frames, ignore_index=True) if html_metric_frames else pd.DataFrame()

                stop_detail_frames = []
                for f in stop_detail_files:
                    f.seek(0)
                    raw = up.read_stop_detail_file(f)
                    if not raw.empty:
                        stop_detail_frames.append(raw)
                stop_detail_raw = pd.concat(stop_detail_frames, ignore_index=True) if stop_detail_frames else pd.DataFrame()

            with st.spinner("Standardizing files..."):
                gap_excel_std = up.standardize_gap(gap_excel_raw) if not gap_excel_raw.empty else pd.DataFrame()
                gap_html_stops_std = up.standardize_gap_html_stops(gap_html_stops_raw) if not gap_html_stops_raw.empty else pd.DataFrame()
                gap_new = pd.concat([gap_excel_std, gap_html_stops_std], ignore_index=True) if not gap_excel_std.empty or not gap_html_stops_std.empty else pd.DataFrame()

                pickup_new = up.standardize_pickups(pickup_raw) if not pickup_raw.empty else pd.DataFrame()
                pickup_stops_new = up.consolidate_physical_pickups(pickup_new) if not pickup_new.empty else pd.DataFrame()
                stop_detail_new = up.standardize_stop_detail(stop_detail_raw) if not stop_detail_raw.empty else pd.DataFrame()

                gap_route_metrics_std = up.standardize_gap_route_metrics(gap_html_metrics_raw) if not gap_html_metrics_raw.empty else pd.DataFrame()
                gap_route_metrics_new = up.enrich_gap_route_metrics_from_stops(gap_route_metrics_std, gap_html_stops_std)

            st.write("New GAP Excel raw rows:", len(gap_excel_raw))
            st.write("New GAP HTML stop rows:", len(gap_html_stops_raw))
            st.write("New GAP HTML route metrics rows:", len(gap_html_metrics_raw))
            st.write("New pickup raw rows:", len(pickup_raw))
            st.write("New stop-detail raw rows:", len(stop_detail_raw))
            if unknown_files:
                st.warning(f"Skipped unknown files: {unknown_files}")

            st.write("Standardized GAP stop rows:", len(gap_new))
            st.write("Standardized pickup rows:", len(pickup_new))
            st.write("Consolidated pickup stop rows:", len(pickup_stops_new))
            st.write("Standardized stop-detail rows:", len(stop_detail_new))
            st.write("Standardized courier day metric rows:", len(gap_route_metrics_new))

            with st.spinner("Appending and deduplicating master tables..."):
                gap_master_updated = up.append_dedup(
                    gap_master,
                    gap_new,
                    key_cols=["scan_date", "route", "fedex_id", "stop_order", "address", "activity_dt"],
                )
                pickup_master_updated = up.append_dedup(
                    pickup_master,
                    pickup_new,
                    key_cols=["pickup_date", "route", "account_name", "address", "pickup_dt", "confirmation_no"],
                )
                pickup_stops_master_updated = up.append_dedup(
                    pickup_stops_master,
                    pickup_stops_new,
                    key_cols=["pickup_key"],
                )
                stop_detail_master_updated = up.append_dedup(
                    stop_detail_master,
                    stop_detail_new,
                    key_cols=["scan_date", "route_key", "stop_order", "address_norm", "activity_dt", "stop_type"],
                )
                gap_route_metrics_master_updated = up.append_dedup(
                    gap_route_metrics_master,
                    gap_route_metrics_new,
                    key_cols=["scan_date", "route", "fedex_id"],
                )
                gap_route_metrics_master_updated = up.rebuild_gap_metrics_from_master(
                    gap_master_updated,
                    gap_route_metrics_master_updated,
                )
                log_new = up.build_ingestion_log_entries(gap_excel_files, pickup_files, stop_detail_files, gap_html_files)
                ingestion_log_updated = up.append_ingestion_log(ingestion_log, log_new)

                up.save_master_tables(
                    paths,
                    gap_master_updated,
                    pickup_master_updated,
                    pickup_stops_master_updated,
                    stop_detail_master_updated,
                    gap_route_metrics_master_updated,
                    ingestion_log_updated,
                )

            st.success("Master tables updated successfully.")
            st.info("Use Analyze Existing Master to review courier-day performance, route benchmarks, and large gaps.")

elif page == "Analyze Existing Master":
    st.header("Analyze Existing Master")

    if gap_master.empty and pickup_stops_master.empty and stop_detail_master.empty and gap_route_metrics_master.empty:
        st.info("No master data available yet. Use Update Master Data first.")
    else:
        gap_master_wg = up.filter_to_our_workgroup(gap_master, "route") if not gap_master.empty and "route" in gap_master.columns else gap_master
        pickup_stops_wg = up.filter_to_our_workgroup(pickup_stops_master, "route") if not pickup_stops_master.empty and "route" in pickup_stops_master.columns else pickup_stops_master
        stop_detail_wg = up.filter_to_our_workgroup(stop_detail_master, "route") if not stop_detail_master.empty and "route" in stop_detail_master.columns else stop_detail_master
        metrics_wg = up.filter_to_our_workgroup(gap_route_metrics_master, "route") if not gap_route_metrics_master.empty and "route" in gap_route_metrics_master.columns else gap_route_metrics_master

        route_values = []
        for df, col in [(gap_master_wg, "route"), (pickup_stops_wg, "route"), (stop_detail_wg, "route"), (metrics_wg, "route")]:
            if df is not None and not df.empty and col in df.columns:
                route_values.extend(pd.to_numeric(df[col], errors="coerce").dropna().astype(int).tolist())
        route_options = sorted({r for r in route_values if up.is_relevant_workgroup_route(r)})

        all_dates = []
        for df, col in [(gap_master_wg, "scan_date"), (pickup_stops_wg, "pickup_date"), (stop_detail_wg, "scan_date"), (metrics_wg, "scan_date")]:
            if df is not None and not df.empty and col in df.columns:
                vals = pd.to_datetime(df[col], errors="coerce").dropna()
                if not vals.empty:
                    all_dates.extend(vals.dt.date.tolist())

        c1, c2, c3 = st.columns(3)
        with c1:
            selected_routes = st.multiselect("Filter routes", options=route_options, default=[])
        with c2:
            if all_dates:
                min_date, max_date = min(all_dates), max(all_dates)
                date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            else:
                date_range = None
        with c3:
            courier_options = []
            courier_source = metrics_wg.copy() if metrics_wg is not None else pd.DataFrame()
            if selected_routes and not courier_source.empty and "route" in courier_source.columns:
                courier_source = courier_source[pd.to_numeric(courier_source["route"], errors="coerce").isin(selected_routes)]
            if not courier_source.empty and "courier_name" in courier_source.columns:
                courier_options = sorted([c for c in courier_source["courier_name"].dropna().astype(str).unique().tolist() if c.strip()])
            selected_couriers = st.multiselect("Filter couriers", options=courier_options, default=[])

        run_analysis = st.button("Run Analysis")

        if run_analysis:
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range

            gap_f = _route_filter(_date_range_filter(gap_master_wg, "scan_date", start_date, end_date), "route", selected_routes)
            pickup_stops_f = _route_filter(_date_range_filter(pickup_stops_wg, "pickup_date", start_date, end_date), "route", selected_routes)
            stop_detail_f = _route_filter(_date_range_filter(stop_detail_wg, "scan_date", start_date, end_date), "route", selected_routes)
            metrics_f = _route_filter(_date_range_filter(metrics_wg, "scan_date", start_date, end_date), "route", selected_routes)

            if selected_couriers and not metrics_f.empty and "courier_name" in metrics_f.columns:
                metrics_f = metrics_f[metrics_f["courier_name"].astype(str).isin(selected_couriers)].copy()
                if not gap_f.empty and "fedex_id" in gap_f.columns and "fedex_id" in metrics_f.columns:
                    fedex_ids = metrics_f["fedex_id"].astype(str).unique()
                    gap_f = gap_f[gap_f["fedex_id"].astype(str).isin(fedex_ids)].copy()
                if not stop_detail_f.empty and "fedex_id" in stop_detail_f.columns and "fedex_id" in metrics_f.columns:
                    fedex_ids = metrics_f["fedex_id"].astype(str).unique()
                    stop_detail_f = stop_detail_f[stop_detail_f["fedex_id"].astype(str).isin(fedex_ids)].copy()

            route_day_summary = up.build_route_day_summary(gap_f, pickup_stops_f)
            route_bench = up.build_route_performance_benchmarks(metrics_f, metrics_wg) if not metrics_f.empty else pd.DataFrame()
            large_gap_exceptions = up.build_large_gap_exceptions(gap_f, gap_master_wg) if not gap_f.empty else pd.DataFrame()

            metrics_show = _round_df(metrics_f)
            route_bench_show = _round_df(route_bench)
            large_gap_show = _round_df(large_gap_exceptions)
            route_day_show = _round_df(route_day_summary)

            st.subheader("Courier day performance")
            if metrics_show.empty:
                st.info("No GAP HTML courier metrics found in the current filter range yet.")
            else:
                show_cols = [c for c in [
                    "scan_date", "route", "courier_name", "fedex_id", "total_stops_actual",
                    "actual_sth_oa", "actual_sth_or", "actual_hr_oa_min", "actual_hr_or_min",
                    "gap_sum_minutes_all", "gap_sum_minutes_customer", "customer_stop_count",
                    "pickup_stop_count", "delivery_stop_count", "miles_on_road"
                ] if c in metrics_show.columns]
                st.dataframe(
                    metrics_show.sort_values([c for c in ["scan_date", "route", "courier_name"] if c in metrics_show.columns])[show_cols],
                    use_container_width=True,
                )
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Courier-day rows", len(metrics_show))
                k2.metric("Avg ST/H OA", round(float(metrics_show["actual_sth_oa"].dropna().mean()), 1) if "actual_sth_oa" in metrics_show.columns and metrics_show["actual_sth_oa"].notna().any() else "—")
                k3.metric("Avg ST/H OR", round(float(metrics_show["actual_sth_or"].dropna().mean()), 1) if "actual_sth_or" in metrics_show.columns and metrics_show["actual_sth_or"].notna().any() else "—")
                k4.metric("Large gap exceptions", len(large_gap_show))

            st.subheader("Route performance benchmarks")
            if route_bench_show.empty:
                st.info("Not enough route-day rows to calculate route benchmarks yet.")
            else:
                bench_cols = [c for c in [
                    "scan_date", "weekday", "route", "courier_name", "fedex_id", "actual_sth_oa", "actual_sth_or",
                    "gap_sum_minutes_all", "gap_sum_minutes_customer", "total_stops_actual",
                    "overall_avg_sth_oa", "overall_avg_sth_or", "route_all_history_avg_sth_oa", "route_all_history_avg_sth_or",
                    "route_prev_7_obs_avg_sth_oa", "route_prev_7_obs_avg_sth_or", "route_same_weekday_avg_sth_oa", "route_same_weekday_avg_sth_or",
                    "delta_vs_route_all_history_oa", "delta_vs_route_all_history_or", "delta_vs_prev_7_obs_oa", "delta_vs_prev_7_obs_or",
                    "delta_vs_same_weekday_oa", "delta_vs_same_weekday_or"
                ] if c in route_bench_show.columns]
                st.dataframe(route_bench_show[bench_cols], use_container_width=True)
                st.download_button("Download route_performance_benchmarks.csv", data=_csv_bytes(route_bench_show), file_name="route_performance_benchmarks.csv", mime="text/csv")

            st.subheader("Large gaps needing explanation")
            if large_gap_show.empty:
                st.info("No large customer-stop gaps for the current filters.")
            else:
                gap_cols = [c for c in [
                    "scan_date", "route", "courier_name", "before_address", "after_address", "before_time", "after_time",
                    "leg_elapsed_minutes", "break_minutes_between", "adjusted_gap_minutes", "stop_type",
                    "route_gap_median", "route_gap_p90", "route_gap_threshold", "severity_ratio", "severity_band"
                ] if c in large_gap_show.columns]
                st.dataframe(large_gap_show[gap_cols], use_container_width=True)
                st.download_button("Download large_gap_exceptions.csv", data=_csv_bytes(large_gap_show), file_name="large_gap_exceptions.csv", mime="text/csv")

            st.subheader("Route-day summary")
            if route_day_show.empty:
                st.info("No route-day summary rows for the current filters.")
            else:
                st.dataframe(route_day_show, use_container_width=True)
                st.download_button("Download route_day_summary.csv", data=_csv_bytes(route_day_show), file_name="route_day_summary.csv", mime="text/csv")

elif page == "Cut Run Optimizer":
    st.header("Cut Run Optimizer")
    st.caption("Additive planning page. Existing GAP analysis remains unchanged.")

    if gap_route_metrics_master.empty and pickup_stops_master.empty:
        st.info("No master data available yet. Load GAP HTML and pickup history first.")
    else:
        metrics_wg = up.filter_to_our_workgroup(gap_route_metrics_master, "route") if not gap_route_metrics_master.empty and "route" in gap_route_metrics_master.columns else gap_route_metrics_master
        pickup_stops_wg = up.filter_to_our_workgroup(pickup_stops_master, "route") if not pickup_stops_master.empty and "route" in pickup_stops_master.columns else pickup_stops_master
        route_profiles_all, _, _ = up.build_route_baseline_profiles(metrics_wg, pickup_stops_wg, day_name=None)
        route_options = sorted({int(r) for r in route_profiles_all.get("route", pd.Series(dtype="Int64")).dropna().astype(int).tolist() if up.is_relevant_workgroup_route(r)})
        anchor_catalog = up.get_anchor_catalog(anchor_refs=anchor_refs, pickup_stops_df=pickup_stops_wg)
        anchor_options = anchor_catalog["anchor_key"].dropna().astype(str).tolist() if not anchor_catalog.empty else []
        anchor_display_map = dict(zip(anchor_catalog.get("anchor_key", pd.Series(dtype=str)).astype(str), anchor_catalog.get("anchor_display", pd.Series(dtype=str)).astype(str)))

        c1, c2, c3 = st.columns(3)
        with c1:
            day_name = st.selectbox("Day of week", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
            mode_label = st.radio("Mode", options=["Test selected route", "Suggest best route to cut"], horizontal=False)
            mode = "test_selected_route" if mode_label == "Test selected route" else "suggest_best_route"
        with c2:
            scope_mode = st.radio("Candidate scope", options=["Entire workgroup", "Chosen subset"], horizontal=False)
            scope_routes = route_options
            if scope_mode == "Chosen subset":
                scope_routes = st.multiselect("Subset routes", options=route_options, default=route_options)
            test_route = st.selectbox("Route to test", options=route_options, index=0 if route_options else None) if route_options else None
        with c3:
            strategy_preset = st.selectbox("Strategy preset", options=["Balanced", "Pickup Safe", "Shortest Added Drive", "Highest Density"], index=0)
            package_soft_warning = st.number_input("Package soft warning threshold", min_value=100.0, max_value=1000.0, value=400.0, step=25.0)

        with st.expander("Advanced strategy weights"):
            base_weights = up.get_strategy_weights(strategy_preset)
            w1, w2, w3 = st.columns(3)
            with w1:
                pickup_safety = st.slider("Pickup safety", 0.0, 1.0, float(base_weights["pickup_safety"]), 0.01)
                added_time = st.slider("Total added time", 0.0, 1.0, float(base_weights["added_time"]), 0.01)
            with w2:
                proximity = st.slider("Proximity / travel efficiency", 0.0, 1.0, float(base_weights["proximity"]), 0.01)
                stop_density = st.slider("Stop density improvement", 0.0, 1.0, float(base_weights["stop_density"]), 0.01)
            with w3:
                package_burden = st.slider("Package burden", 0.0, 1.0, float(base_weights["package_burden"]), 0.01)
                fragmentation = st.slider("Area fragmentation", 0.0, 1.0, float(base_weights["fragmentation"]), 0.01)
            strategy_overrides = {
                "pickup_safety": pickup_safety,
                "added_time": added_time,
                "proximity": proximity,
                "stop_density": stop_density,
                "package_burden": package_burden,
                "fragmentation": fragmentation,
            }

        with st.expander("Anchor controls"):
            include_anchors = st.multiselect(
                "Include only these anchors",
                options=anchor_options,
                default=[],
                format_func=lambda x: f"{anchor_display_map.get(str(x), str(x))}",
            )
            prioritize_anchors = st.multiselect(
                "Prioritize these anchors",
                options=anchor_options,
                default=[],
                format_func=lambda x: f"{anchor_display_map.get(str(x), str(x))}",
            )
            exclude_anchors = st.multiselect(
                "Exclude / protect these anchors",
                options=anchor_options,
                default=[],
                format_func=lambda x: f"{anchor_display_map.get(str(x), str(x))}",
            )

        st.info("Current cut-run logic uses pickup anchor history plus GAP route-day performance. True geographic maps and traffic-aware reassignment will improve further once anchor geographies or geocoding are connected.")

        if st.button("Run Cut Optimizer"):
            result = up.build_cut_run_optimizer(
                gap_route_metrics_df=metrics_wg,
                pickup_stops_df=pickup_stops_wg,
                anchor_refs=anchor_refs,
                day_name=day_name,
                mode=mode,
                cut_route=test_route if mode == "test_selected_route" else None,
                candidate_routes=scope_routes,
                include_anchors=include_anchors,
                prioritize_anchors=prioritize_anchors,
                exclude_anchors=exclude_anchors,
                strategy_preset=strategy_preset,
                strategy_overrides=strategy_overrides,
                package_soft_warning=package_soft_warning,
            )

            suggestions = _round_df(result.get("suggestions", pd.DataFrame()))
            selected_plan = result.get("selected_plan", {})
            scorecard = _round_df(selected_plan.get("scorecard", pd.DataFrame()))
            cut_summary = _round_df(selected_plan.get("cut_route_summary", pd.DataFrame()))
            assignment_plan = _round_df(selected_plan.get("assignment_plan", pd.DataFrame()))
            receiver_summary = _round_df(selected_plan.get("receiver_summary", pd.DataFrame()))
            weights_show = _round_df(result.get("weights", pd.DataFrame()))

            st.subheader("Strategy weights used")
            st.dataframe(weights_show, use_container_width=True)

            st.subheader("Suggested cut routes")
            if suggestions.empty:
                st.info("No route suggestions available for the selected scope and day.")
            else:
                st.dataframe(suggestions, use_container_width=True)

            st.subheader("Selected plan summary")
            if scorecard.empty:
                st.warning(selected_plan.get("message", "No plan built."))
            else:
                s = scorecard.iloc[0]
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Cut route", int(s["cut_route"]) if pd.notna(s.get("cut_route")) else "—")
                k2.metric("Feasible", "Yes" if bool(s.get("feasible")) else "No")
                k3.metric("Overall score", round(float(s.get("overall_score", 0.0)), 1) if pd.notna(s.get("overall_score")) else "—")
                k4.metric("Total added minutes", round(float(s.get("total_added_minutes", 0.0)), 1) if pd.notna(s.get("total_added_minutes")) else "—")
                k5, k6, k7, k8 = st.columns(4)
                k5.metric("Anchors reassigned", int(s.get("anchors_reassigned", 0)))
                k6.metric("Receiving routes used", int(s.get("receiving_routes_used", 0)))
                k7.metric("Pickup risk flags", int(s.get("pickup_risk_flags", 0)))
                k8.metric("Package soft warnings", int(s.get("package_soft_warnings", 0)))
                st.caption(selected_plan.get("message", ""))

            st.subheader("Cut route baseline")
            if cut_summary.empty:
                st.info("No cut-route baseline available.")
            else:
                st.dataframe(cut_summary, use_container_width=True)

            st.subheader("Recommended anchor redistribution")
            if assignment_plan.empty:
                st.info("No anchor redistribution plan available.")
            else:
                assign_cols = [c for c in [
                    "anchor_display", "anchor_wave", "receiver_route", "receiver_wave", "avg_pickup_stops", "avg_packages",
                    "median_close_min", "overlap_share", "added_minutes", "pickup_risk_raw", "disqualify", "package_warning", "weighted_score"
                ] if c in assignment_plan.columns]
                st.dataframe(assignment_plan[assign_cols], use_container_width=True)
                st.download_button("Download cut_route_assignment_plan.csv", data=_csv_bytes(assignment_plan), file_name="cut_route_assignment_plan.csv", mime="text/csv")

            st.subheader("Receiving route before / after summary")
            if receiver_summary.empty:
                st.info("No receiving route summary available.")
            else:
                st.dataframe(receiver_summary, use_container_width=True)
                chart_df = receiver_summary.set_index("receiver_route")[[c for c in ["base_avg_stops", "new_avg_stops"] if c in receiver_summary.columns]]
                if not chart_df.empty:
                    st.bar_chart(chart_df)
                st.download_button("Download cut_route_receivers.csv", data=_csv_bytes(receiver_summary), file_name="cut_route_receivers.csv", mime="text/csv")

            st.subheader("Anchor flow view")
            if assignment_plan.empty:
                st.info("No anchor flow view available.")
            else:
                flow = assignment_plan.groupby(["receiver_route"], dropna=False).agg(
                    assigned_anchors=("anchor_display", "count"),
                    added_stops=("added_stops", "sum"),
                    added_packages=("added_packages", "sum"),
                    added_minutes=("added_minutes", "sum"),
                ).reset_index()
                st.dataframe(_round_df(flow), use_container_width=True)
                st.caption("This view is map-ready but not yet a true geographic map. Once anchor geographies or geocodes are available, this same plan can render spatial before/after maps.")

elif page == "Settings / Exceptions":
    st.header("Settings / Exceptions")

    st.subheader("Route exceptions JSON")
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

    edited_json = st.text_area("Edit route exceptions JSON", value=json.dumps(current_settings, indent=2), height=300)

    if st.button("Save Exceptions"):
        try:
            parsed = json.loads(edited_json)
            settings_path.write_text(json.dumps(parsed, indent=2))
            st.success("Exceptions saved.")
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    st.markdown("---")
    st.subheader("Anchor reference manager")
    st.caption("Upload anchor reference images / files so work-area ownership stays tied to the project.")

    if anchor_refs.empty:
        st.info("No anchor references uploaded yet.")
    else:
        show_anchor = anchor_refs.copy()
        for c in ["effective_date", "uploaded_at"]:
            if c in show_anchor.columns:
                show_anchor[c] = pd.to_datetime(show_anchor[c], errors="coerce")
        st.dataframe(show_anchor, use_container_width=True)

    with st.form("anchor_upload_form"):
        uploaded_anchor = st.file_uploader("Upload anchor reference", type=None, accept_multiple_files=False, key="anchor_upload")
        c1, c2, c3 = st.columns(3)
        with c1:
            work_area_input = st.text_input("Work area / anchor key")
        with c2:
            wave_input = st.selectbox("Wave", options=["", "W1", "W2"])
        with c3:
            effective_date = st.date_input("Effective date", value=None)
        notes = st.text_area("Notes", value="", height=80)
        save_anchor = st.form_submit_button("Save anchor reference")

    if save_anchor:
        if uploaded_anchor is None or not work_area_input.strip():
            st.warning("Please upload a file and enter a work area / anchor key.")
        else:
            try:
                anchor_refs = up.append_or_replace_anchor_reference(
                    paths=paths,
                    existing_refs=anchor_refs,
                    uploaded_file=uploaded_anchor,
                    work_area_input=work_area_input,
                    wave=wave_input,
                    effective_date=effective_date,
                    notes=notes,
                )
                st.success("Anchor reference saved.")
            except Exception as e:
                st.error(f"Could not save anchor reference: {e}")
