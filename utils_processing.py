import hashlib
from pathlib import Path
import re
from datetime import datetime, time

import numpy as np
import pandas as pd

APP_CONFIG = {
    "station_address": "45 Di Poce Way Vaughan, ON L4H 4J4",
    "wave_starts": {
        "W1": "08:15",
        "W2": "10:30",
    },
    "weekend_starts": {
        "Saturday": "08:15",
        "Sunday": "09:30",
    },
    "start_of_day_station_min": 30,
    "end_of_day_station_min": 15,
    "pickup_match_tolerance_min": 5,
    "consolidation_time_tolerance_min": 5,
    "reason_code_map": {
        910: "Pickup Cancelled"
    }
}

def ensure_data_dirs(base_dir="streamlit_data"):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    paths = {
        "base": base,
        "master": base / "master",
        "cache": base / "cache",
        "outputs": base / "outputs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths

def load_master_tables(paths):
    gap_path = paths["master"] / "gap_master.parquet"
    pickup_path = paths["master"] / "pickup_master.parquet"
    pickup_stops_path = paths["master"] / "pickup_stops_master.parquet"
    log_path = paths["master"] / "ingestion_log.csv"

    gap_master = pd.read_parquet(gap_path) if gap_path.exists() else pd.DataFrame()
    pickup_master = pd.read_parquet(pickup_path) if pickup_path.exists() else pd.DataFrame()
    pickup_stops_master = pd.read_parquet(pickup_stops_path) if pickup_stops_path.exists() else pd.DataFrame()
    ingestion_log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()

    for df, cols in [
        (gap_master, ["activity_dt"]),
        (pickup_master, ["ready_pickup_dt", "close_pickup_dt", "pickup_dt", "wave_start_dt"]),
        (pickup_stops_master, ["pickup_dt", "ready_pickup_dt", "close_pickup_dt", "pickup_dt_floor", "ready_dt_floor", "close_dt_floor"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    return gap_master, pickup_master, pickup_stops_master, ingestion_log



def _prepare_for_parquet(df):
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()

    for col in out.columns:
        s = out[col]

        if pd.api.types.is_datetime64_any_dtype(s):
            continue

        non_null = s.dropna()
        sample = non_null.iloc[:20] if len(non_null) else non_null

        if len(sample) == 0:
            continue

        # Python time/date objects and mixed object columns often break parquet writes.
        if sample.map(lambda x: isinstance(x, time)).any():
            out[col] = s.astype(str).replace({"NaT": np.nan, "None": np.nan, "nan": np.nan})
            continue

        if sample.map(lambda x: hasattr(x, "isoformat") and not isinstance(x, (str, bytes, pd.Timestamp))).any():
            try:
                out[col] = pd.to_datetime(s, errors="ignore")
            except Exception:
                out[col] = s.astype(str).replace({"NaT": np.nan, "None": np.nan, "nan": np.nan})
            continue

        if s.dtype == "object":
            py_types = {type(x).__name__ for x in sample}
            if len(py_types) > 1:
                out[col] = s.astype(str).replace({"NaT": np.nan, "None": np.nan, "nan": np.nan})

    return out

def save_master_tables(paths, gap_master, pickup_master, pickup_stops_master, ingestion_log):
    gap_master = _prepare_for_parquet(gap_master)
    pickup_master = _prepare_for_parquet(pickup_master)
    pickup_stops_master = _prepare_for_parquet(pickup_stops_master)

    gap_master.to_parquet(paths["master"] / "gap_master.parquet", index=False)
    pickup_master.to_parquet(paths["master"] / "pickup_master.parquet", index=False)
    pickup_stops_master.to_parquet(paths["master"] / "pickup_stops_master.parquet", index=False)
    ingestion_log.to_csv(paths["master"] / "ingestion_log.csv", index=False)

def clean_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def normalize_postal_code(x):
    if pd.isna(x):
        return np.nan
    s = str(x).upper().strip()
    s = re.sub(r"[^A-Z0-9]", "", s)
    if len(s) == 6:
        return s[:3] + " " + s[3:]
    return s

def to_date_col(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def to_datetime_col(s):
    return pd.to_datetime(s, errors="coerce")

def to_time_only(s):
    dt = pd.to_datetime(s, errors="coerce")
    return dt.dt.time

def parse_time_string(t):
    if pd.isna(t):
        return None
    if isinstance(t, time):
        return t
    t = str(t).strip()
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"):
        try:
            return datetime.strptime(t, fmt).time()
        except:
            pass
    return None

def combine_date_and_time(date_val, time_val):
    if pd.isna(date_val) or time_val is None:
        return pd.NaT
    try:
        return pd.Timestamp(datetime.combine(pd.Timestamp(date_val).date(), time_val))
    except:
        return pd.NaT

def weekday_name_from_date(d):
    if pd.isna(d):
        return None
    return pd.Timestamp(d).day_name()

def infer_wave_start(work_area, pickup_date):
    day_name = weekday_name_from_date(pickup_date)
    if day_name in APP_CONFIG["weekend_starts"]:
        return parse_time_string(APP_CONFIG["weekend_starts"][day_name]), day_name

    wa = "" if pd.isna(work_area) else str(work_area).upper().strip()
    m = re.match(r"^(W\d+)", wa)
    if m:
        wave = m.group(1)
        if wave in APP_CONFIG["wave_starts"]:
            return parse_time_string(APP_CONFIG["wave_starts"][wave]), wave

    return None, "UNKNOWN"

def norm_addr(s):
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"[^A-Z0-9 ]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s

def floor_dt_to_tolerance(dt_series, tolerance_min=5):
    if dt_series.isna().all():
        return dt_series
    return dt_series.dt.floor(f"{tolerance_min}min")

def uploaded_file_hash(uploaded_file):
    content = uploaded_file.getvalue()
    return hashlib.md5(content).hexdigest()

def build_ingestion_log_entries(gap_uploads, pickup_uploads):
    rows = []

    for f in gap_uploads or []:
        rows.append({
            "file_name": f.name,
            "file_type": "gap",
            "file_size": f.size,
            "file_hash": uploaded_file_hash(f),
            "ingested_at": pd.Timestamp.now(),
        })

    for f in pickup_uploads or []:
        rows.append({
            "file_name": f.name,
            "file_type": "pickup",
            "file_size": f.size,
            "file_hash": uploaded_file_hash(f),
            "ingested_at": pd.Timestamp.now(),
        })

    return pd.DataFrame(rows)

def append_ingestion_log(existing_log, new_log):
    if existing_log.empty:
        return new_log.copy()
    if new_log.empty:
        return existing_log.copy()

    out = pd.concat([existing_log, new_log], ignore_index=True)
    out = out.drop_duplicates(subset=["file_name", "file_size", "file_hash"], keep="first")
    return out

def read_uploaded_excels(uploaded_files):
    if not uploaded_files:
        return pd.DataFrame()

    frames = []
    for f in uploaded_files:
        try:
            df = pd.read_excel(f)
            df["source_file"] = f.name
            frames.append(df)
        except Exception:
            pass

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def append_dedup(master_df, new_df, key_cols):
    if new_df is None or new_df.empty:
        return master_df.copy() if master_df is not None else pd.DataFrame()

    if master_df is None or master_df.empty:
        out = new_df.copy()
    else:
        out = pd.concat([master_df, new_df], ignore_index=True)

    existing_keys = [c for c in key_cols if c in out.columns]
    if existing_keys:
        out = out.drop_duplicates(subset=existing_keys, keep="first")

    return out.reset_index(drop=True)

def standardize_gap(df):
    if df.empty:
        return df.copy()

    g = df.copy()
    g = g.rename(columns={
        "Scan Date": "scan_date",
        "Loc": "loc",
        "Route": "route",
        "Stop Order": "stop_order",
        "Stop Type": "stop_type",
        "Ready Time": "ready_time",
        "Close Time": "close_time",
        "Activity": "activity_time",
        "GAP": "gap_minutes",
        "FedEx ID": "fedex_id",
        "Address": "address",
        "ZIP": "postal_code",
        "RecByDADS": "rec_by_dads",
        "STAT": "stat",
        "FXE PKGS": "fxe_pkgs",
        "FXG PKGS": "fxg_pkgs",
        "FO": "fo",
        "PO": "po",
        "SO": "so",
        "ES": "es",
        "XS": "xs",
        "FXG Service": "fxg_service",
    })

    g["scan_date"] = to_date_col(g["scan_date"])
    g["route"] = pd.to_numeric(g["route"], errors="coerce").astype("Int64")
    g["stop_order"] = pd.to_numeric(g["stop_order"], errors="coerce")
    g["gap_minutes"] = pd.to_numeric(g["gap_minutes"], errors="coerce")

    if "address" in g.columns:
        g["address"] = g["address"].apply(clean_text)
    if "postal_code" in g.columns:
        g["postal_code"] = g["postal_code"].apply(normalize_postal_code)

    if "stop_type" in g.columns:
        g["stop_type"] = g["stop_type"].astype(str).str.upper().str.strip()

    g["activity_dt"] = to_datetime_col(g["activity_time"]) if "activity_time" in g.columns else pd.NaT
    if "activity_dt" in g.columns and g["activity_dt"].notna().sum() == 0 and "activity_time" in g.columns:
        g["activity_dt"] = pd.to_datetime(
            g["scan_date"].astype(str) + " " + g["activity_time"].astype(str),
            errors="coerce"
        )

    g["activity_time_only"] = g["activity_dt"].dt.time
    g["is_pickup_like"] = g["stop_type"].str.contains("PU", na=False) if "stop_type" in g.columns else False
    g["is_delivery_like"] = g["stop_type"].str.contains("DL", na=False) if "stop_type" in g.columns else False

    return g

def standardize_pickups(df):
    if df.empty:
        return df.copy()

    p = df.copy()
    p = p.rename(columns={
        "Station": "station",
        "Scheduled Ground Account #": "scheduled_ground_acct",
        "Scheduled Express Account #": "scheduled_express_acct",
        "Scanned Ground Account #": "scanned_ground_acct",
        "Scanned Express Account #": "scanned_express_acct",
        "Pickup Type": "pickup_type",
        "Account Name": "account_name",
        "Address": "address",
        "Address 2": "address_2",
        "Address 3": "address_3",
        "City": "city",
        "State": "state",
        "Postal Code": "postal_code",
        "LOCID": "locid",
        "Customer Contact": "customer_contact",
        "Phone Number": "phone_number",
        "Ready Pickup Time": "ready_pickup_time",
        "Close Pickup Time": "close_pickup_time",
        "Pickup Location": "pickup_location",
        "Creation Source": "creation_source",
        "Confirmation #": "confirmation_no",
        "Key Sequence #": "key_sequence_no",
        "PU Listing / ID #": "pu_listing_id",
        "Service Area #": "service_area_no",
        "Work Area": "work_area",
        "Work Area #": "work_area_no",
        "Residential": "residential",
        "Alternate Address": "alternate_address",
        "Pickup Date": "pickup_date",
        "Pickup Time": "pickup_time",
        "Reason Code": "reason_code",
        "Reconciliation Source": "reconciliation_source",
        "Expected Packages": "expected_packages",
        "Total Stop Packages Picked Up": "packages",
        "STAR Comments": "star_comments",
    })

    for col in ["account_name", "address", "address_2", "address_3", "city", "state", "work_area", "pickup_type"]:
        if col in p.columns:
            p[col] = p[col].apply(clean_text)

    if "postal_code" in p.columns:
        p["postal_code"] = p["postal_code"].apply(normalize_postal_code)

    p["pickup_date"] = to_date_col(p["pickup_date"]) if "pickup_date" in p.columns else pd.NaT

    p["ready_pickup_time_only"] = to_time_only(p["ready_pickup_time"]) if "ready_pickup_time" in p.columns else None
    p["close_pickup_time_only"] = to_time_only(p["close_pickup_time"]) if "close_pickup_time" in p.columns else None
    p["pickup_time_only"] = to_time_only(p["pickup_time"]) if "pickup_time" in p.columns else None

    p["ready_pickup_dt"] = p.apply(lambda r: combine_date_and_time(r["pickup_date"], r["ready_pickup_time_only"]), axis=1)
    p["close_pickup_dt"] = p.apply(lambda r: combine_date_and_time(r["pickup_date"], r["close_pickup_time_only"]), axis=1)
    p["pickup_dt"] = p.apply(lambda r: combine_date_and_time(r["pickup_date"], r["pickup_time_only"]), axis=1)

    for col in ["work_area_no", "reason_code", "expected_packages", "packages"]:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")

    inferred = p.apply(lambda r: infer_wave_start(r.get("work_area"), r.get("pickup_date")), axis=1)
    p["wave_start_time"] = [x[0] for x in inferred]
    p["wave_label"] = [x[1] for x in inferred]
    p["wave_start_dt"] = p.apply(lambda r: combine_date_and_time(r["pickup_date"], r["wave_start_time"]), axis=1)

    p["route"] = p["work_area_no"].astype("Int64") if "work_area_no" in p.columns else pd.Series(dtype="Int64")
    p["reason_text"] = p["reason_code"].map(APP_CONFIG["reason_code_map"]).fillna("") if "reason_code" in p.columns else ""

    return p

def consolidate_physical_pickups(p):
    if p.empty:
        return p.copy()

    df = p.copy()

    df["pickup_dt_floor"] = floor_dt_to_tolerance(df["pickup_dt"], APP_CONFIG["consolidation_time_tolerance_min"])
    df["ready_dt_floor"] = floor_dt_to_tolerance(df["ready_pickup_dt"], APP_CONFIG["consolidation_time_tolerance_min"])
    df["close_dt_floor"] = floor_dt_to_tolerance(df["close_pickup_dt"], APP_CONFIG["consolidation_time_tolerance_min"])

    group_cols = [
        "pickup_date",
        "route",
        "work_area",
        "account_name",
        "address",
        "city",
        "state",
        "postal_code",
        "pickup_dt_floor",
        "ready_dt_floor",
        "close_dt_floor",
    ]

    agg = {
        "station": "first",
        "pickup_type": lambda s: " | ".join(sorted(set([str(x) for x in s.dropna()]))),
        "confirmation_no": lambda s: " | ".join(sorted(set([str(x) for x in s.dropna()]))),
        "service_area_no": "first",
        "pickup_location": "first",
        "residential": "first",
        "reason_code": "first",
        "reason_text": "first",
        "packages": "sum",
        "expected_packages": "sum",
        "source_file": lambda s: " | ".join(sorted(set(s.astype(str)))),
        "pickup_dt": "min",
        "ready_pickup_dt": "min",
        "close_pickup_dt": "max",
        "scanned_ground_acct": "count",
    }

    consolidated = (
        df.groupby(group_cols, dropna=False, as_index=False)
          .agg(agg)
          .rename(columns={"scanned_ground_acct": "raw_rows_merged"})
    )

    consolidated["pickup_key"] = (
        consolidated["pickup_date"].astype(str) + " | " +
        consolidated["route"].astype(str) + " | " +
        consolidated["address"].fillna("") + " | " +
        consolidated["pickup_dt_floor"].astype(str)
    )

    consolidated["is_cancelled"] = consolidated["reason_code"].eq(910)
    consolidated["is_completed"] = consolidated["pickup_dt"].notna() & ~consolidated["is_cancelled"]

    return consolidated

def build_route_day_summary(gap_df, pickup_stops):
    if gap_df.empty:
        return pd.DataFrame()

    route_day_summary = (
        gap_df.groupby(["scan_date", "route"], dropna=False)
        .agg(
            first_activity_dt=("activity_dt", "min"),
            last_activity_dt=("activity_dt", "max"),
            stop_count=("stop_order", "count"),
            unique_addresses=("address", "nunique"),
            pickup_like_stops=("is_pickup_like", "sum"),
            delivery_like_stops=("is_delivery_like", "sum"),
            avg_gap_minutes=("gap_minutes", "mean"),
        )
        .reset_index()
    )

    route_day_summary["route_span_minutes"] = (
        route_day_summary["last_activity_dt"] - route_day_summary["first_activity_dt"]
    ).dt.total_seconds() / 60.0

    if pickup_stops is not None and not pickup_stops.empty:
        pickup_day_summary = (
            pickup_stops.groupby(["pickup_date", "route"], dropna=False)
            .agg(
                consolidated_pickup_stops=("pickup_key", "count"),
                completed_pickups=("is_completed", "sum"),
                cancelled_pickups=("is_cancelled", "sum"),
                total_packages=("packages", "sum"),
            )
            .reset_index()
            .rename(columns={"pickup_date": "scan_date"})
        )

        route_day_summary = route_day_summary.merge(
            pickup_day_summary,
            on=["scan_date", "route"],
            how="left",
        )

        for col in ["consolidated_pickup_stops", "completed_pickups", "cancelled_pickups", "total_packages"]:
            if col in route_day_summary.columns:
                route_day_summary[col] = route_day_summary[col].fillna(0)

    return route_day_summary

def match_pickups_to_gap(gap_df, pickup_stops, tolerance_min=5):
    if gap_df.empty or pickup_stops.empty:
        return pd.DataFrame(), pd.DataFrame()

    gap_match_base = gap_df[
        ["scan_date", "route", "address", "postal_code", "activity_dt", "stop_type", "stop_order"]
    ].copy()

    pickup_match_base = pickup_stops[pickup_stops["is_completed"]].copy()

    match_candidates = pickup_match_base.merge(
        gap_match_base,
        left_on=["pickup_date", "route", "postal_code"],
        right_on=["scan_date", "route", "postal_code"],
        how="left",
        suffixes=("_pu", "_gap"),
    )

    match_candidates["addr_norm_pu"] = match_candidates["address_pu"].map(norm_addr)
    match_candidates["addr_norm_gap"] = match_candidates["address_gap"].map(norm_addr)
    match_candidates["address_exact_match"] = match_candidates["addr_norm_pu"] == match_candidates["addr_norm_gap"]

    match_candidates["pickup_vs_gap_min"] = (
        match_candidates["activity_dt"] - match_candidates["pickup_dt"]
    ).dt.total_seconds() / 60.0

    match_candidates = match_candidates[
        match_candidates["address_exact_match"] &
        match_candidates["pickup_vs_gap_min"].abs().le(tolerance_min)
    ].copy()

    match_candidates["abs_time_diff"] = match_candidates["pickup_vs_gap_min"].abs()

    best_matches = (
        match_candidates.sort_values(["pickup_key", "abs_time_diff", "stop_order"])
        .drop_duplicates(subset=["pickup_key"], keep="first")
        .copy()
    )

    matched_keys = set(best_matches["pickup_key"])
    pickup_match_report = pickup_stops.copy()
    pickup_match_report["matched_to_gap"] = pickup_match_report["pickup_key"].isin(matched_keys)

    report = (
        pickup_match_report.groupby(["pickup_date", "route"], dropna=False)
        .agg(
            consolidated_pickups=("pickup_key", "count"),
            completed_pickups=("is_completed", "sum"),
            matched_pickups=("matched_to_gap", "sum"),
            unmatched_pickups=("matched_to_gap", lambda s: (~s).sum()),
        )
        .reset_index()
    )

    return best_matches, report