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
        "anchors": base / "anchors",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths

def load_master_tables(paths):
    gap_pkl = paths["master"] / "gap_master.pkl"
    pickup_pkl = paths["master"] / "pickup_master.pkl"
    pickup_stops_pkl = paths["master"] / "pickup_stops_master.pkl"
    stop_detail_pkl = paths["master"] / "stop_detail_master.pkl"
    log_path = paths["master"] / "ingestion_log.csv"

    gap_master = pd.read_pickle(gap_pkl) if gap_pkl.exists() else pd.DataFrame()
    pickup_master = pd.read_pickle(pickup_pkl) if pickup_pkl.exists() else pd.DataFrame()
    pickup_stops_master = pd.read_pickle(pickup_stops_pkl) if pickup_stops_pkl.exists() else pd.DataFrame()
    stop_detail_master = pd.read_pickle(stop_detail_pkl) if stop_detail_pkl.exists() else pd.DataFrame()
    ingestion_log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()

    for df, cols in [
        (gap_master, ["activity_dt"]),
        (pickup_master, ["ready_pickup_dt", "close_pickup_dt", "pickup_dt", "wave_start_dt"]),
        (pickup_stops_master, ["pickup_dt", "ready_pickup_dt", "close_pickup_dt", "pickup_dt_floor", "ready_dt_floor", "close_dt_floor"]),
        (stop_detail_master, ["ready_dt", "close_dt", "activity_dt"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    return gap_master, pickup_master, pickup_stops_master, stop_detail_master, ingestion_log

def save_master_tables(paths, gap_master, pickup_master, pickup_stops_master, stop_detail_master, ingestion_log):
    gap_master.to_pickle(paths["master"] / "gap_master.pkl")
    pickup_master.to_pickle(paths["master"] / "pickup_master.pkl")
    pickup_stops_master.to_pickle(paths["master"] / "pickup_stops_master.pkl")
    stop_detail_master.to_pickle(paths["master"] / "stop_detail_master.pkl")
    ingestion_log.to_csv(paths["master"] / "ingestion_log.csv", index=False)


def normalize_work_area_key(x):
    if pd.isna(x):
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    m = re.search(r"(\d{3,4})", s)
    if not m:
        return None
    try:
        return str(int(m.group(1)))
    except Exception:
        return m.group(1).lstrip("0") or m.group(1)

def format_work_area_display(x):
    key = normalize_work_area_key(x)
    if key is None:
        return ""
    return key.zfill(4)

def load_anchor_references(paths):
    anchor_meta_path = paths["anchors"] / "anchor_references.csv"
    if not anchor_meta_path.exists():
        return pd.DataFrame(columns=[
            "work_area_key", "wave", "version", "is_active", "effective_date",
            "uploaded_at", "original_file_name", "saved_file_name", "saved_path", "notes"
        ])
    refs = pd.read_csv(anchor_meta_path)
    for c in ["effective_date", "uploaded_at"]:
        if c in refs.columns:
            refs[c] = pd.to_datetime(refs[c], errors="coerce")
    if "is_active" in refs.columns:
        refs["is_active"] = refs["is_active"].fillna(False).astype(bool)
    return refs

def save_anchor_references(paths, anchor_refs):
    anchor_meta_path = paths["anchors"] / "anchor_references.csv"
    out = anchor_refs.copy()
    out.to_csv(anchor_meta_path, index=False)

def append_or_replace_anchor_reference(paths, existing_refs, uploaded_file, work_area_input, wave="", effective_date=None, notes=""):
    work_area_key = normalize_work_area_key(work_area_input)
    if work_area_key is None:
        raise ValueError("Invalid work area input.")

    refs = existing_refs.copy() if existing_refs is not None else pd.DataFrame()
    if refs.empty:
        refs = pd.DataFrame(columns=[
            "work_area_key", "wave", "version", "is_active", "effective_date",
            "uploaded_at", "original_file_name", "saved_file_name", "saved_path", "notes"
        ])

    if "work_area_key" in refs.columns:
        same_mask = refs["work_area_key"].astype(str) == str(work_area_key)
        refs.loc[same_mask, "is_active"] = False
        existing_versions = pd.to_numeric(refs.loc[same_mask, "version"], errors="coerce").dropna()
        next_version = int(existing_versions.max()) + 1 if not existing_versions.empty else 1
    else:
        next_version = 1

    original_name = getattr(uploaded_file, "name", "anchor_upload")
    suffix = Path(original_name).suffix.lower() or ".bin"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_name).stem)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    saved_file_name = f"anchor_{work_area_key}_v{next_version}_{timestamp}_{safe_name}{suffix}"
    saved_path = paths["anchors"] / saved_file_name

    content = uploaded_file.getvalue()
    saved_path.write_bytes(content)

    new_row = pd.DataFrame([{
        "work_area_key": str(work_area_key),
        "wave": clean_text(wave) if wave else "",
        "version": next_version,
        "is_active": True,
        "effective_date": pd.Timestamp(effective_date) if effective_date else pd.NaT,
        "uploaded_at": pd.Timestamp.now(),
        "original_file_name": original_name,
        "saved_file_name": saved_file_name,
        "saved_path": str(saved_path),
        "notes": clean_text(notes) if notes else "",
    }])

    refs = pd.concat([refs, new_row], ignore_index=True)
    save_anchor_references(paths, refs)
    return refs

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

def build_ingestion_log_entries(gap_uploads, pickup_uploads, stop_detail_uploads=None):
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

    for f in stop_detail_uploads or []:
        rows.append({
            "file_name": f.name,
            "file_type": "stop_detail",
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

def normalize_route_key(x):
    s = clean_text(x)
    if pd.isna(s):
        return np.nan
    s = str(s)
    m = re.search(r"(\d{3,4})", s)
    if not m:
        return s.strip()
    try:
        return str(int(m.group(1)))
    except Exception:
        return m.group(1)

def normalize_address_for_match(x):
    if pd.isna(x):
        return ""
    s = str(x).upper().strip()
    replacements = {
        r"\bSTREET\b": "ST",
        r"\bROAD\b": "RD",
        r"\bAVENUE\b": "AVE",
        r"\bBOULEVARD\b": "BLVD",
        r"\bDRIVE\b": "DR",
        r"\bCOURT\b": "CRT",
        r"\bPLACE\b": "PL",
        r"\bLANE\b": "LN",
        r"\bSUITE\b": "STE",
        r"\bAPARTMENT\b": "APT",
    }
    for pattern, repl in replacements.items():
        s = re.sub(pattern, repl, s)
    s = re.sub(r"[^A-Z0-9 ]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _combine_address_parts(parts):
    vals = []
    for x in parts:
        cleaned = clean_text(x)
        if pd.notna(cleaned) and str(cleaned).strip():
            vals.append(str(cleaned).strip())
    return ", ".join(vals)

def read_stop_detail_file(file_obj):
    """
    Reads the stop-detail spreadsheet. Supports:
    - HTML-table .xls exports via read_html
    - regular Excel files via read_excel fallback
    """
    name = getattr(file_obj, "name", "")
    frames = []

    try:
        html_tables = pd.read_html(file_obj)
        if html_tables:
            raw = max(html_tables, key=lambda t: t.shape[0] * t.shape[1]).copy()
            if raw.shape[0] >= 2:
                raw.columns = raw.iloc[0]
                raw = raw.iloc[1:].copy()
            frames.append(raw)
    except Exception:
        pass

    if not frames:
        try:
            file_obj.seek(0)
            raw = pd.read_excel(file_obj)
            frames.append(raw)
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    df = frames[0].copy()
    df["source_file"] = name
    try:
        file_obj.seek(0)
    except Exception:
        pass
    return df

def standardize_stop_detail(df):
    if df.empty:
        return df.copy()

    s = df.copy()

    cols = list(s.columns)
    address_idx = [i for i, c in enumerate(cols) if str(c).strip().upper() == "ADDRESS"]
    for n, idx in enumerate(address_idx, start=1):
        cols[idx] = f"ADDRESS_{n}"
    s.columns = cols

    rename_map = {
        "Date": "scan_date",
        "Route": "route",
        "Stop Order": "stop_order",
        "Stop Type": "stop_type",
        "Ready Time": "ready_time",
        "Close Time": "close_time",
        "Activity Time": "activity_time",
        "Activity": "activity_time",
        "GAP": "gap_minutes",
        "FedEx ID": "fedex_id",
        "STAT": "stat",
        "ZIP": "postal_code",
        "RecBy DADS": "rec_by_dads",
        "RecByDADS": "rec_by_dads",
        "Pkgs": "package_count",
    }
    s = s.rename(columns=rename_map)

    address_cols = [c for c in s.columns if str(c).startswith("ADDRESS_")]
    if "address" not in s.columns:
        s["address"] = s[address_cols].apply(lambda row: _combine_address_parts(row.tolist()), axis=1) if address_cols else np.nan

    for col in ["route", "stop_type", "fedex_id", "stat", "address", "postal_code"]:
        if col in s.columns:
            s[col] = s[col].apply(clean_text)

    if "route" in s.columns:
        s["route_key"] = s["route"].apply(normalize_route_key)

    if "postal_code" in s.columns:
        s["postal_code"] = s["postal_code"].apply(normalize_postal_code)
        s["postal_code_norm"] = s["postal_code"]

    if "address" in s.columns:
        s["address_norm"] = s["address"].apply(normalize_address_for_match)

    if "scan_date" in s.columns:
        s["scan_date"] = to_date_col(s["scan_date"])

    for c in ["stop_order", "package_count", "gap_minutes"]:
        if c in s.columns:
            s[c] = pd.to_numeric(s[c], errors="coerce")

    for c in ["ready_time", "close_time", "activity_time"]:
        if c in s.columns:
            s[c] = s[c].apply(clean_text)

    if {"scan_date", "ready_time"}.issubset(s.columns):
        s["ready_dt"] = pd.to_datetime(s["scan_date"].astype(str) + " " + s["ready_time"].astype(str), errors="coerce")
    if {"scan_date", "close_time"}.issubset(s.columns):
        s["close_dt"] = pd.to_datetime(s["scan_date"].astype(str) + " " + s["close_time"].astype(str), errors="coerce")

    if "activity_time" in s.columns:
        s["activity_dt"] = pd.to_datetime(s["activity_time"], errors="coerce")
        if s["activity_dt"].notna().sum() == 0 and "scan_date" in s.columns:
            s["activity_dt"] = pd.to_datetime(s["scan_date"].astype(str) + " " + s["activity_time"].astype(str), errors="coerce")

    stop_type_upper = s["stop_type"].astype(str).str.upper().fillna("") if "stop_type" in s.columns else pd.Series("", index=s.index)
    s["is_pickup"] = stop_type_upper.str.startswith("PU")
    s["is_delivery"] = stop_type_upper.str.startswith("DL")
    s["is_oncall_pickup"] = stop_type_upper.eq("PU ONC")
    s["is_commercial_delivery"] = stop_type_upper.eq("DL COM")
    s["is_ground_delivery"] = stop_type_upper.eq("DL GRD")

    s["stop_family"] = np.where(
        s["is_oncall_pickup"], "Pickup On-Call",
        np.where(
            s["is_pickup"], "Pickup Other",
            np.where(
                s["is_commercial_delivery"], "Delivery Commercial",
                np.where(
                    s["is_ground_delivery"], "Delivery Ground",
                    np.where(s["is_delivery"], "Delivery Other", "Unknown")
                )
            )
        )
    )

    keep_cols = [
        "scan_date", "route", "route_key", "stop_order", "stop_type", "stop_family",
        "ready_time", "close_time", "activity_time", "ready_dt", "close_dt", "activity_dt",
        "gap_minutes", "fedex_id", "stat", "address", "postal_code",
        "address_norm", "postal_code_norm", "package_count",
        "is_pickup", "is_delivery", "is_oncall_pickup",
        "is_commercial_delivery", "is_ground_delivery", "source_file"
    ]
    keep_cols = [c for c in keep_cols if c in s.columns]
    return s[keep_cols].copy()

def prep_gap_for_matching(gap_df):
    if gap_df is None or gap_df.empty:
        return pd.DataFrame()
    g = gap_df.copy()
    if "scan_date" in g.columns:
        g["scan_date"] = to_date_col(g["scan_date"])
    route_col = "route" if "route" in g.columns else None
    addr_col = "address" if "address" in g.columns else None
    postal_col = "postal_code" if "postal_code" in g.columns else None

    g["route_key"] = g[route_col].apply(normalize_route_key) if route_col else np.nan
    g["address_norm"] = g[addr_col].apply(normalize_address_for_match) if addr_col else ""
    g["postal_code_norm"] = g[postal_col].apply(normalize_postal_code) if postal_col else np.nan
    return g

def prep_pickups_for_matching(pickup_df):
    if pickup_df is None or pickup_df.empty:
        return pd.DataFrame()
    p = pickup_df.copy()
    if "pickup_date" in p.columns:
        p["scan_date"] = to_date_col(p["pickup_date"])
    route_col = "route" if "route" in p.columns else "work_area_no" if "work_area_no" in p.columns else None
    addr_col = "address" if "address" in p.columns else None
    postal_col = "postal_code" if "postal_code" in p.columns else None

    p["route_key"] = p[route_col].apply(normalize_route_key) if route_col else np.nan
    p["address_norm"] = p[addr_col].apply(normalize_address_for_match) if addr_col else ""
    p["postal_code_norm"] = p[postal_col].apply(normalize_postal_code) if postal_col else np.nan
    return p

def cross_reference_stop_detail(stop_detail_df, gap_df, pickup_df, time_tolerance_min=10):
    if stop_detail_df is None or stop_detail_df.empty:
        return pd.DataFrame()

    sd = stop_detail_df.copy()
    gap = prep_gap_for_matching(gap_df)
    pu = prep_pickups_for_matching(pickup_df)

    if "route_key" not in sd.columns and "route" in sd.columns:
        sd["route_key"] = sd["route"].apply(normalize_route_key)
    if "address_norm" not in sd.columns and "address" in sd.columns:
        sd["address_norm"] = sd["address"].apply(normalize_address_for_match)
    if "postal_code_norm" not in sd.columns and "postal_code" in sd.columns:
        sd["postal_code_norm"] = sd["postal_code"].apply(normalize_postal_code)

    gap_match = gap[[c for c in ["scan_date", "route_key", "address_norm", "postal_code_norm", "activity_dt", "stop_type"] if c in gap.columns]].drop_duplicates() if not gap.empty else pd.DataFrame()
    pu_match = pu[[c for c in ["scan_date", "route_key", "address_norm", "postal_code_norm", "pickup_dt", "pickup_type"] if c in pu.columns]].drop_duplicates() if not pu.empty else pd.DataFrame()

    if not gap_match.empty:
        sd = sd.merge(
            gap_match,
            how="left",
            on=[c for c in ["scan_date", "route_key", "address_norm", "postal_code_norm"] if c in sd.columns and c in gap_match.columns],
            suffixes=("", "_gap")
        )
        sd["matched_to_gap"] = sd.get("activity_dt_gap", pd.Series(pd.NaT, index=sd.index)).notna()
    else:
        sd["matched_to_gap"] = False

    if not pu_match.empty:
        sd = sd.merge(
            pu_match,
            how="left",
            on=[c for c in ["scan_date", "route_key", "address_norm", "postal_code_norm"] if c in sd.columns and c in pu_match.columns],
            suffixes=("", "_pickup")
        )
        # after merge, left pickup_dt exists only if stop_detail has one; right becomes pickup_dt_pickup
        sd["matched_to_pickup_sheet"] = sd.get("pickup_dt_pickup", pd.Series(pd.NaT, index=sd.index)).notna()
    else:
        sd["matched_to_pickup_sheet"] = False

    return sd
