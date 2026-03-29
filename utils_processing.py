import hashlib
import io
import re
from datetime import datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


APP_CONFIG = {
    "station_address": "45 Di Poce Way Vaughan, ON L4H 4J4",
    "wave_starts": {"W1": "08:15", "W2": "10:30"},
    "weekend_starts": {"Saturday": "08:15", "Sunday": "09:30"},
    "start_of_day_station_min": 30,
    "end_of_day_station_min": 15,
    "pickup_match_tolerance_min": 10,
    "consolidation_time_tolerance_min": 5,
    "reason_code_map": {
        910: "Cancelled",
        0: "Completed",
    },
}


# -----------------------------
# Files / storage
# -----------------------------
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


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def load_master_tables(paths):
    gap_master = pd.read_pickle(paths["master"] / "gap_master.pkl") if (paths["master"] / "gap_master.pkl").exists() else pd.DataFrame()
    pickup_master = pd.read_pickle(paths["master"] / "pickup_master.pkl") if (paths["master"] / "pickup_master.pkl").exists() else pd.DataFrame()
    pickup_stops_master = pd.read_pickle(paths["master"] / "pickup_stops_master.pkl") if (paths["master"] / "pickup_stops_master.pkl").exists() else pd.DataFrame()
    stop_detail_master = pd.read_pickle(paths["master"] / "stop_detail_master.pkl") if (paths["master"] / "stop_detail_master.pkl").exists() else pd.DataFrame()
    gap_route_metrics_master = pd.read_pickle(paths["master"] / "gap_route_metrics_master.pkl") if (paths["master"] / "gap_route_metrics_master.pkl").exists() else pd.DataFrame()
    ingestion_log = _safe_read_csv(paths["master"] / "ingestion_log.csv")

    for df, cols in [
        (gap_master, ["activity_dt", "ready_dt", "close_dt"]),
        (pickup_master, ["ready_pickup_dt", "close_pickup_dt", "pickup_dt", "wave_start_dt"]),
        (pickup_stops_master, ["pickup_dt", "ready_pickup_dt", "close_pickup_dt", "pickup_dt_floor", "ready_dt_floor", "close_dt_floor"]),
        (stop_detail_master, ["ready_dt", "close_dt", "activity_dt"]),
        (gap_route_metrics_master, ["scan_date", "leave_building_dt", "on_area_start_dt", "on_area_end_dt", "return_to_bldg_dt"]),
    ]:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
    return gap_master, pickup_master, pickup_stops_master, stop_detail_master, gap_route_metrics_master, ingestion_log


def save_master_tables(paths, gap_master, pickup_master, pickup_stops_master, stop_detail_master, gap_route_metrics_master, ingestion_log):
    gap_master.to_pickle(paths["master"] / "gap_master.pkl")
    pickup_master.to_pickle(paths["master"] / "pickup_master.pkl")
    pickup_stops_master.to_pickle(paths["master"] / "pickup_stops_master.pkl")
    stop_detail_master.to_pickle(paths["master"] / "stop_detail_master.pkl")
    gap_route_metrics_master.to_pickle(paths["master"] / "gap_route_metrics_master.pkl")
    ingestion_log.to_csv(paths["master"] / "ingestion_log.csv", index=False)


# -----------------------------
# Anchor references
# -----------------------------
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
    return str(key).zfill(4)


def load_anchor_references(paths):
    anchor_meta_path = paths["anchors"] / "anchor_references.csv"
    if not anchor_meta_path.exists() or anchor_meta_path.stat().st_size == 0:
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
    anchor_refs.to_csv(anchor_meta_path, index=False)


def append_or_replace_anchor_reference(paths, existing_refs, uploaded_file, work_area_input, wave="", effective_date=None, notes=""):
    work_area_key = normalize_work_area_key(work_area_input)
    if work_area_key is None:
        raise ValueError("Invalid work area input.")

    refs = existing_refs.copy() if existing_refs is not None and not existing_refs.empty else pd.DataFrame(columns=[
        "work_area_key", "wave", "version", "is_active", "effective_date",
        "uploaded_at", "original_file_name", "saved_file_name", "saved_path", "notes"
    ])

    same_mask = refs["work_area_key"].astype(str).eq(str(work_area_key)) if not refs.empty else pd.Series(False, index=refs.index)
    if not refs.empty:
        refs.loc[same_mask, "is_active"] = False
    existing_versions = pd.to_numeric(refs.loc[same_mask, "version"], errors="coerce").dropna() if not refs.empty else pd.Series(dtype=float)
    next_version = int(existing_versions.max()) + 1 if not existing_versions.empty else 1

    original_name = getattr(uploaded_file, "name", "anchor_upload")
    suffix = Path(original_name).suffix.lower() or ".bin"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(original_name).stem)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    saved_file_name = f"anchor_{work_area_key}_v{next_version}_{timestamp}_{safe_name}{suffix}"
    saved_path = paths["anchors"] / saved_file_name
    saved_path.write_bytes(uploaded_file.getvalue())

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


# -----------------------------
# Generic cleaning / parsing
# -----------------------------
def clean_text(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace("\xa0", " ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def normalize_postal_code(x):
    if pd.isna(x):
        return np.nan
    s = str(x).upper().strip()
    s = re.sub(r"[^A-Z0-9]", "", s)
    if len(s) == 6:
        return f"{s[:3]} {s[3:]}"
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
    txt = clean_text(t)
    if not txt:
        return None
    for fmt in ("%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"):
        try:
            return datetime.strptime(txt, fmt).time()
        except Exception:
            pass
    return None


def parse_duration_to_minutes(value):
    if pd.isna(value):
        return np.nan
    txt = clean_text(value)
    if txt in (None, "", "-", "--"):
        return np.nan
    txt = txt.replace(" ", "")
    m = re.fullmatch(r"(-?)(\d+):(\d{2})(?::(\d{2}))?", txt)
    if m:
        sign = -1 if m.group(1) == "-" else 1
        first = int(m.group(2))
        second = int(m.group(3))
        third = int(m.group(4) or 0)
        if m.group(4) is not None:
            total = first * 60 + second + third / 60.0
        else:
            total = first * 60 + second
        return sign * total
    try:
        return float(txt)
    except Exception:
        return np.nan


def combine_date_and_time(date_val, time_val):
    if pd.isna(date_val) or time_val is None:
        return pd.NaT
    try:
        return pd.Timestamp(datetime.combine(pd.Timestamp(date_val).date(), time_val))
    except Exception:
        return pd.NaT


def minutes_to_hhmm(minutes_val):
    if pd.isna(minutes_val):
        return np.nan
    try:
        total = int(round(float(minutes_val)))
    except Exception:
        return np.nan
    sign = "-" if total < 0 else ""
    total = abs(total)
    hours = total // 60
    minutes = total % 60
    return f"{sign}{hours:02d}:{minutes:02d}"


def _safe_rate_per_hour(count_val, minutes_val):
    try:
        count = float(count_val)
        mins = float(minutes_val)
    except Exception:
        return np.nan
    if pd.isna(count) or pd.isna(mins) or mins <= 0:
        return np.nan
    return count / (mins / 60.0)


def _sum_break_minutes(event_df):
    if event_df is None or event_df.empty or "activity_dt" not in event_df.columns:
        return 0.0
    e = event_df.copy().sort_values("activity_dt")
    labels = e.get("event_type", pd.Series("", index=e.index)).astype(str).str.upper()
    starts = e.loc[labels.str.contains("BEGIN BREAK", na=False), "activity_dt"].tolist()
    ends = e.loc[labels.str.contains("END BREAK", na=False), "activity_dt"].tolist()
    if not starts or not ends:
        return 0.0
    total = 0.0
    end_idx = 0
    for start in starts:
        while end_idx < len(ends) and ends[end_idx] <= start:
            end_idx += 1
        if end_idx < len(ends):
            total += max((ends[end_idx] - start).total_seconds() / 60.0, 0.0)
            end_idx += 1
    return total


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
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def build_ingestion_log_entries(gap_uploads, pickup_uploads, stop_detail_uploads=None, gap_html_uploads=None):
    rows = []
    for files, label in [
        (gap_uploads, "gap_excel"),
        (gap_html_uploads, "gap_html"),
        (pickup_uploads, "pickup"),
        (stop_detail_uploads, "stop_detail"),
    ]:
        for f in files or []:
            rows.append({
                "file_name": f.name,
                "file_type": label,
                "file_size": getattr(f, "size", len(f.getvalue())),
                "file_hash": uploaded_file_hash(f),
                "ingested_at": pd.Timestamp.now(),
            })
    return pd.DataFrame(rows)


def append_ingestion_log(existing_log, new_log):
    if existing_log is None or existing_log.empty:
        return new_log.copy() if new_log is not None else pd.DataFrame()
    if new_log is None or new_log.empty:
        return existing_log.copy()
    out = pd.concat([existing_log, new_log], ignore_index=True)
    out = out.drop_duplicates(subset=["file_name", "file_size", "file_hash"], keep="first")
    return out.reset_index(drop=True)


def read_uploaded_excels(uploaded_files):
    if not uploaded_files:
        return pd.DataFrame()
    frames = []
    for f in uploaded_files:
        try:
            f.seek(0)
            df = pd.read_excel(f)
            df["source_file"] = getattr(f, "name", "uploaded_excel")
            frames.append(df)
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def append_dedup(master_df, new_df, key_cols):
    if new_df is None or new_df.empty:
        return master_df.copy() if master_df is not None else pd.DataFrame()
    out = new_df.copy() if master_df is None or master_df.empty else pd.concat([master_df, new_df], ignore_index=True)
    existing_keys = [c for c in key_cols if c in out.columns]
    if existing_keys:
        out = out.drop_duplicates(subset=existing_keys, keep="first")
    return out.reset_index(drop=True)


# -----------------------------
# Existing GAP Excel / pickup / stop-detail logic
# -----------------------------
def standardize_gap(df):
    if df.empty:
        return df.copy()

    g = df.copy().rename(columns={
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

    if "scan_date" in g.columns:
        g["scan_date"] = to_date_col(g["scan_date"])
    if "route" in g.columns:
        g["route"] = pd.to_numeric(g["route"], errors="coerce").astype("Int64")
        g["route_key"] = g["route"].apply(normalize_route_key)
    if "stop_order" in g.columns:
        g["stop_order"] = pd.to_numeric(g["stop_order"], errors="coerce")
    if "gap_minutes" in g.columns:
        g["gap_minutes"] = pd.to_numeric(g["gap_minutes"], errors="coerce")

    for c in ["address", "stop_type", "fedex_id", "stat", "loc", "source_file"]:
        if c in g.columns:
            g[c] = g[c].apply(clean_text)
    if "postal_code" in g.columns:
        g["postal_code"] = g["postal_code"].apply(normalize_postal_code)
        g["postal_code_norm"] = g["postal_code"]
    if "address" in g.columns:
        g["address_norm"] = g["address"].apply(normalize_address_for_match)

    if "activity_time" in g.columns:
        g["activity_time"] = g["activity_time"].apply(clean_text)
        g["activity_dt"] = pd.to_datetime(g["activity_time"], errors="coerce")
        if g["activity_dt"].notna().sum() == 0 and "scan_date" in g.columns:
            g["activity_dt"] = pd.to_datetime(g["scan_date"].astype(str) + " " + g["activity_time"].astype(str), errors="coerce")
    if {"scan_date", "ready_time"}.issubset(g.columns):
        g["ready_dt"] = pd.to_datetime(g["scan_date"].astype(str) + " " + g["ready_time"].astype(str), errors="coerce")
    if {"scan_date", "close_time"}.issubset(g.columns):
        g["close_dt"] = pd.to_datetime(g["scan_date"].astype(str) + " " + g["close_time"].astype(str), errors="coerce")

    stop_type_upper = g.get("stop_type", pd.Series("", index=g.index)).astype(str).str.upper().fillna("")
    g["is_pickup_like"] = stop_type_upper.str.contains("PU", na=False)
    g["is_delivery_like"] = stop_type_upper.str.contains("DL", na=False)
    g["is_event_row"] = g.get("address", pd.Series("", index=g.index)).astype(str).str.startswith("*")
    return g


def standardize_pickups(df):
    if df.empty:
        return df.copy()

    p = df.copy().rename(columns={
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
    if "pickup_date" in p.columns:
        p["pickup_date"] = to_date_col(p["pickup_date"])

    p["ready_pickup_time_only"] = to_time_only(p["ready_pickup_time"]) if "ready_pickup_time" in p.columns else None
    p["close_pickup_time_only"] = to_time_only(p["close_pickup_time"]) if "close_pickup_time" in p.columns else None
    p["pickup_time_only"] = to_time_only(p["pickup_time"]) if "pickup_time" in p.columns else None

    p["ready_pickup_dt"] = p.apply(lambda r: combine_date_and_time(r.get("pickup_date"), r.get("ready_pickup_time_only")), axis=1)
    p["close_pickup_dt"] = p.apply(lambda r: combine_date_and_time(r.get("pickup_date"), r.get("close_pickup_time_only")), axis=1)
    p["pickup_dt"] = p.apply(lambda r: combine_date_and_time(r.get("pickup_date"), r.get("pickup_time_only")), axis=1)

    for col in ["work_area_no", "reason_code", "expected_packages", "packages"]:
        if col in p.columns:
            p[col] = pd.to_numeric(p[col], errors="coerce")

    inferred = p.apply(lambda r: infer_wave_start(r.get("work_area"), r.get("pickup_date")), axis=1)
    p["wave_start_time"] = [x[0] for x in inferred]
    p["wave_label"] = [x[1] for x in inferred]
    p["wave_start_dt"] = p.apply(lambda r: combine_date_and_time(r.get("pickup_date"), r.get("wave_start_time")), axis=1)
    p["route"] = p["work_area_no"].astype("Int64") if "work_area_no" in p.columns else pd.Series(dtype="Int64")
    p["reason_text"] = p.get("reason_code", pd.Series(dtype=float)).map(APP_CONFIG["reason_code_map"]).fillna("")
    return p


def consolidate_physical_pickups(p):
    if p.empty:
        return p.copy()
    df = p.copy()
    df["pickup_dt_floor"] = floor_dt_to_tolerance(df["pickup_dt"], APP_CONFIG["consolidation_time_tolerance_min"])
    df["ready_dt_floor"] = floor_dt_to_tolerance(df["ready_pickup_dt"], APP_CONFIG["consolidation_time_tolerance_min"])
    df["close_dt_floor"] = floor_dt_to_tolerance(df["close_pickup_dt"], APP_CONFIG["consolidation_time_tolerance_min"])

    group_cols = [
        "pickup_date", "route", "work_area", "account_name", "address", "city", "state", "postal_code",
        "pickup_dt_floor", "ready_dt_floor", "close_dt_floor",
    ]
    agg = {
        "station": "first",
        "pickup_type": lambda s: " | ".join(sorted(set(str(x) for x in s.dropna()))),
        "confirmation_no": lambda s: " | ".join(sorted(set(str(x) for x in s.dropna()))),
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
    consolidated = df.groupby(group_cols, dropna=False, as_index=False).agg(agg).rename(columns={"scanned_ground_acct": "raw_rows_merged"})
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
    if gap_df is None or gap_df.empty:
        return pd.DataFrame()

    use_gap = gap_df.copy()
    if "is_event_row" in use_gap.columns:
        use_gap = use_gap[~use_gap["is_event_row"].fillna(False)].copy()

    if use_gap.empty:
        return pd.DataFrame()

    route_day_summary = (
        use_gap.groupby(["scan_date", "route"], dropna=False)
        .agg(
            first_activity_dt=("activity_dt", "min"),
            last_activity_dt=("activity_dt", "max"),
            stop_count=("stop_order", "count"),
            unique_addresses=("address", "nunique"),
            pickup_like_stops=("is_pickup_like", "sum"),
            delivery_like_stops=("is_delivery_like", "sum"),
            avg_gap_minutes=("gap_minutes", "mean"),
            total_gap_minutes=("gap_minutes", "sum"),
        )
        .reset_index()
    )
    route_day_summary["route_span_minutes"] = (route_day_summary["last_activity_dt"] - route_day_summary["first_activity_dt"]).dt.total_seconds() / 60.0

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
        route_day_summary = route_day_summary.merge(pickup_day_summary, on=["scan_date", "route"], how="left")
        for col in ["consolidated_pickup_stops", "completed_pickups", "cancelled_pickups", "total_packages"]:
            if col in route_day_summary.columns:
                route_day_summary[col] = route_day_summary[col].fillna(0)

    return route_day_summary.sort_values(["scan_date", "route"]).reset_index(drop=True)


def match_pickups_to_gap(gap_df, pickup_stops, tolerance_min=5):
    if gap_df is None or gap_df.empty or pickup_stops is None or pickup_stops.empty:
        return pd.DataFrame(), pd.DataFrame()

    gap_match_base = gap_df[[c for c in ["scan_date", "route", "address", "postal_code", "activity_dt", "stop_type", "stop_order"] if c in gap_df.columns]].copy()
    pickup_match_base = pickup_stops[pickup_stops["is_completed"]].copy()
    if gap_match_base.empty or pickup_match_base.empty:
        return pd.DataFrame(), pd.DataFrame()

    match_candidates = pickup_match_base.merge(
        gap_match_base,
        left_on=["pickup_date", "route", "postal_code"],
        right_on=["scan_date", "route", "postal_code"],
        how="left",
        suffixes=("_pu", "_gap"),
    )
    match_candidates["addr_norm_pu"] = match_candidates["address_pu"].map(norm_addr)
    match_candidates["addr_norm_gap"] = match_candidates["address_gap"].map(norm_addr)
    match_candidates["addr_match"] = match_candidates["addr_norm_pu"].eq(match_candidates["addr_norm_gap"])
    match_candidates["time_delta_min"] = (
        (match_candidates["pickup_dt"] - match_candidates["activity_dt"]).dt.total_seconds().abs() / 60.0
    )
    match_candidates["within_tolerance"] = match_candidates["time_delta_min"] <= tolerance_min
    best_matches = match_candidates[match_candidates["addr_match"] & match_candidates["within_tolerance"]].copy()
    if best_matches.empty:
        report = (
            pickup_match_base.groupby(["pickup_date", "route"], dropna=False)
            .agg(completed_pickups=("is_completed", "sum"))
            .reset_index()
            .assign(matched_pickups=0, unmatched_pickups=lambda d: d["completed_pickups"])
        )
        return best_matches, report

    best_matches = best_matches.sort_values(["pickup_key", "time_delta_min", "stop_order"]).drop_duplicates(subset=["pickup_key"], keep="first")
    best_matches["matched_to_gap"] = True
    report = (
        pickup_match_base.merge(best_matches[["pickup_key", "matched_to_gap"]], on="pickup_key", how="left")
        .assign(matched_to_gap=lambda d: d["matched_to_gap"].fillna(False))
        .groupby(["pickup_date", "route"], dropna=False)
        .agg(
            completed_pickups=("is_completed", "sum"),
            matched_pickups=("matched_to_gap", "sum"),
            unmatched_pickups=("matched_to_gap", lambda s: (~s).sum()),
        )
        .reset_index()
    )
    return best_matches.reset_index(drop=True), report.reset_index(drop=True)


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
    name = getattr(file_obj, "name", "")
    frames = []
    try:
        file_obj.seek(0)
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

    s = s.rename(columns={
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
    })

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

    stop_type_upper = s.get("stop_type", pd.Series("", index=s.index)).astype(str).str.upper().fillna("")
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
                np.where(s["is_ground_delivery"], "Delivery Ground", np.where(s["is_delivery"], "Delivery Other", "Unknown"))
            )
        )
    )
    keep_cols = [
        "scan_date", "route", "route_key", "stop_order", "stop_type", "stop_family",
        "ready_time", "close_time", "activity_time", "ready_dt", "close_dt", "activity_dt",
        "gap_minutes", "fedex_id", "stat", "address", "postal_code",
        "address_norm", "postal_code_norm", "package_count",
        "is_pickup", "is_delivery", "is_oncall_pickup", "is_commercial_delivery", "is_ground_delivery", "source_file"
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
            suffixes=("", "_gap"),
        )
        sd["matched_to_gap"] = sd.get("activity_dt_gap", pd.Series(pd.NaT, index=sd.index)).notna()
    else:
        sd["matched_to_gap"] = False

    if not pu_match.empty:
        sd = sd.merge(
            pu_match,
            how="left",
            on=[c for c in ["scan_date", "route_key", "address_norm", "postal_code_norm"] if c in sd.columns and c in pu_match.columns],
            suffixes=("", "_pickup"),
        )
        sd["matched_to_pickup_sheet"] = sd.get("pickup_dt_pickup", pd.Series(pd.NaT, index=sd.index)).notna()
    else:
        sd["matched_to_pickup_sheet"] = False

    return sd


# -----------------------------
# GAP HTML parsing
# -----------------------------
STOP_HEADERS = [
    "Date", "Loc", "Route", "Stop Order", "Stop Type", "Ready Time", "Close Time",
    "Activity Time", "GAP", "FS280 Tol", "FedEx ID", "Address", "ZIP", "RecBy DADS",
    "STAT", "Pkgs", "Box", "Doc", "FO", "PO", "SO", "ES", "XS", "GD", "Other",
]


def _extract_route_header_fields(text: str) -> dict:
    txt = clean_text(text) or ""
    out = {}
    m = re.search(r"LOC:\s*([A-Z0-9]+)\s*/\s*(\d+)\s+DATE:\s*([0-9/]+)\s+ROUTE:\s*(\d+)", txt, flags=re.I)
    if m:
        out["loc"] = m.group(1)
        out["loc_nbr"] = m.group(2)
        out["scan_date"] = pd.to_datetime(m.group(3), errors="coerce")
        out["route"] = pd.to_numeric(m.group(4), errors="coerce")
    m2 = re.search(r"FEDEX ID:\s*(\d+)\s*-\s*([^\|]+?)\s+TOTAL PAID HRS:\s*([0-9:]+)", txt, flags=re.I)
    if m2:
        out["fedex_id"] = m2.group(1)
        out["courier_short_name"] = clean_text(m2.group(2))
        out["total_paid_hours_text"] = m2.group(3)
    return out


def _extract_identity_block(text: str) -> dict:
    txt = clean_text(text) or ""
    out = {}
    m = re.search(r"Name:\s*([^|]+?)\s*\|\s*FedEx ID:\s*(\d+)", txt, flags=re.I)
    if m:
        out["courier_name"] = clean_text(m.group(1))
        out["fedex_id"] = m.group(2)
    return out


def read_gap_html_file(file_obj):
    """Parse FWR GAP saved HTML into stop rows + route-level metrics."""
    try:
        file_obj.seek(0)
        html = file_obj.getvalue().decode("utf-8", errors="ignore")
    except Exception:
        file_obj.seek(0)
        html = file_obj.read().decode("utf-8", errors="ignore")

    soup = BeautifulSoup(html, "html.parser")
    print_table = soup.find("table", attrs={"width": "1060"})
    if print_table is None:
        return pd.DataFrame(), pd.DataFrame()

    trs = print_table.find_all("tr")
    route_start_idxs = []
    for i, tr in enumerate(trs):
        row_text = " ".join(td.get_text(" ", strip=True) for td in tr.find_all("td"))
        if "LOC:" in row_text and "ROUTE:" in row_text and "TOTAL PAID HRS:" in row_text:
            route_start_idxs.append(i)
    if not route_start_idxs:
        return pd.DataFrame(), pd.DataFrame()
    route_start_idxs.append(len(trs))

    stop_rows = []
    metric_rows = []
    source_file = getattr(file_obj, "name", "gap_html")

    for s_idx, e_idx in zip(route_start_idxs[:-1], route_start_idxs[1:]):
        section_rows = trs[s_idx:e_idx]
        header_text = " ".join(td.get_text(" ", strip=True) for td in section_rows[0].find_all("td"))
        route_info = _extract_route_header_fields(header_text)

        # Detailed identity/name block if present
        for tr in section_rows[:8]:
            block_text = " ".join(td.get_text(" ", strip=True) for td in tr.find_all("td"))
            route_info.update({k: v for k, v in _extract_identity_block(block_text).items() if v})

        stop_header_idx = None
        summary_map = {}
        last_map = {}
        section_text = " ".join(clean_text(td.get_text(" ", strip=True)) or "" for tr in section_rows for td in tr.find_all("td"))
        no_plan_information_found = "NO PLAN INFORMATION FOUND FOR THIS ROUTE" in section_text.upper()

        for local_idx, tr in enumerate(section_rows):
            cells = [clean_text(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
            if cells[:5] == STOP_HEADERS[:5]:
                stop_header_idx = local_idx
                break
            if len(cells) == 5 and cells[0] not in {"Element", "Plan"} and not str(cells[0]).startswith("LOC:"):
                summary_map[cells[0]] = cells[1]
            elif len(cells) == 2 and cells[0] not in {"FEDEX ID: "+str(route_info.get("fedex_id", "")), "TOTAL PAID HRS: "+str(route_info.get("total_paid_hours_text", ""))}:
                if cells[0] and cells[1]:
                    last_map[cells[0]] = cells[1]

        if stop_header_idx is None:
            continue

        # Stop rows
        for tr in section_rows[stop_header_idx + 1:]:
            cells = [clean_text(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
            if len(cells) != len(STOP_HEADERS):
                continue
            if cells[0] == "Date":
                continue
            row = dict(zip(STOP_HEADERS, cells))
            row["source_file"] = source_file
            stop_rows.append(row)

        metric_row = {
            "scan_date": route_info.get("scan_date"),
            "route": route_info.get("route"),
            "route_key": normalize_route_key(route_info.get("route")),
            "loc": route_info.get("loc"),
            "loc_nbr": route_info.get("loc_nbr"),
            "fedex_id": route_info.get("fedex_id"),
            "courier_name": route_info.get("courier_name") or route_info.get("courier_short_name"),
            "total_paid_hours_text": route_info.get("total_paid_hours_text"),
            "leave_building_time": summary_map.get("Leave Building"),
            "hr_to_area_text": summary_map.get("HR To Area"),
            "on_area_start_time": summary_map.get("On Area Start"),
            "st_lt_1030_actual": pd.to_numeric(summary_map.get("ST < 1030") or summary_map.get("ST <\xa01030"), errors="coerce"),
            "st_gt_1030_actual": pd.to_numeric(summary_map.get("ST > 1030"), errors="coerce"),
            "total_stops_actual": pd.to_numeric(summary_map.get("Total Stops"), errors="coerce"),
            "on_area_end_time": summary_map.get("On Area End"),
            "hr_from_area_text": summary_map.get("HR From Area"),
            "return_to_bldg_time": summary_map.get("Return To Bldg"),
            "hr_break_or_text": summary_map.get("HR Break OR"),
            "actual_hr_oa_text": summary_map.get("HR OA"),
            "actual_hr_or_text": summary_map.get("HR OR"),
            "actual_sth_oa": pd.to_numeric(summary_map.get("ST/H OA"), errors="coerce"),
            "actual_sth_or": pd.to_numeric(summary_map.get("ST/H OR"), errors="coerce"),
            "miles_on_road": pd.to_numeric(last_map.get("Miles On Road"), errors="coerce"),
            "pk_undelivered": pd.to_numeric(last_map.get("PK Undelivered"), errors="coerce"),
            "summary_available": bool(summary_map),
            "summary_missing_reason": "no_plan_information_found" if no_plan_information_found and not summary_map else "",
            "source_file": source_file,
        }
        metric_rows.append(metric_row)

    stop_df = pd.DataFrame(stop_rows)
    metrics_df = pd.DataFrame(metric_rows)
    return stop_df, metrics_df


def standardize_gap_html_stops(df):
    if df is None or df.empty:
        return pd.DataFrame()
    g = df.copy().rename(columns={
        "Date": "scan_date",
        "Loc": "loc_nbr",
        "Route": "route",
        "Stop Order": "stop_order",
        "Stop Type": "stop_type",
        "Ready Time": "ready_time",
        "Close Time": "close_time",
        "Activity Time": "activity_time",
        "GAP": "gap_raw",
        "FS280 Tol": "fs280_tol_raw",
        "FedEx ID": "fedex_id",
        "Address": "address",
        "ZIP": "postal_code",
        "RecBy DADS": "rec_by_dads",
        "STAT": "stat",
        "Pkgs": "package_count",
        "Box": "box",
        "Doc": "doc",
        "FO": "fo",
        "PO": "po",
        "SO": "so",
        "ES": "es",
        "XS": "xs",
        "GD": "gd",
        "Other": "other",
    })

    g["scan_date"] = to_date_col(g["scan_date"])
    g["route"] = pd.to_numeric(g["route"], errors="coerce").astype("Int64")
    g["route_key"] = g["route"].apply(normalize_route_key)
    g["stop_order"] = pd.to_numeric(g["stop_order"], errors="coerce")
    g["gap_minutes"] = g["gap_raw"].apply(parse_duration_to_minutes)
    g["fs280_tol_minutes"] = g["fs280_tol_raw"].apply(parse_duration_to_minutes)
    g["package_count"] = pd.to_numeric(g["package_count"], errors="coerce")

    for c in ["loc_nbr", "stop_type", "fedex_id", "address", "stat", "rec_by_dads", "source_file"]:
        if c in g.columns:
            g[c] = g[c].apply(clean_text)
    if "postal_code" in g.columns:
        g["postal_code"] = g["postal_code"].apply(normalize_postal_code)
        g["postal_code_norm"] = g["postal_code"]
    g["address_norm"] = g["address"].apply(normalize_address_for_match)

    for col in ["ready_time", "close_time", "activity_time"]:
        if col in g.columns:
            g[col] = g[col].apply(clean_text)
    g["ready_dt"] = pd.to_datetime(g["scan_date"].astype(str) + " " + g["ready_time"].astype(str), errors="coerce")
    g["close_dt"] = pd.to_datetime(g["scan_date"].astype(str) + " " + g["close_time"].astype(str), errors="coerce")
    g["activity_dt"] = pd.to_datetime(g["scan_date"].astype(str) + " " + g["activity_time"].astype(str), errors="coerce")

    stop_type_upper = g["stop_type"].astype(str).str.upper().fillna("")
    g["is_pickup_like"] = stop_type_upper.str.contains("PU", na=False)
    g["is_delivery_like"] = stop_type_upper.str.contains("DL", na=False)
    g["is_event_row"] = g["address"].astype(str).str.startswith("*")
    g["event_type"] = np.where(g["is_event_row"], g["address"].astype(str).str.replace(r"^\*\d+-", "", regex=True).str.upper(), "")
    return g


def standardize_gap_route_metrics(df):
    if df is None or df.empty:
        return pd.DataFrame()
    m = df.copy()
    if "scan_date" in m.columns:
        m["scan_date"] = pd.to_datetime(m["scan_date"], errors="coerce").dt.normalize()
    if "route" in m.columns:
        m["route"] = pd.to_numeric(m["route"], errors="coerce").astype("Int64")
        m["route_key"] = m["route"].apply(normalize_route_key)
    for c in ["fedex_id", "courier_name", "loc", "loc_nbr", "source_file", "leave_building_time", "on_area_start_time", "on_area_end_time", "return_to_bldg_time", "total_paid_hours_text", "actual_hr_oa_text", "actual_hr_or_text", "hr_to_area_text", "hr_from_area_text", "hr_break_or_text", "summary_missing_reason"]:
        if c in m.columns:
            m[c] = m[c].apply(clean_text)

    for out_col, src_col in [
        ("total_paid_hours_min", "total_paid_hours_text"),
        ("actual_hr_oa_min", "actual_hr_oa_text"),
        ("actual_hr_or_min", "actual_hr_or_text"),
        ("hr_to_area_min", "hr_to_area_text"),
        ("hr_from_area_min", "hr_from_area_text"),
        ("hr_break_or_min", "hr_break_or_text"),
    ]:
        if src_col in m.columns:
            m[out_col] = m[src_col].apply(parse_duration_to_minutes)

    for out_col, src_col in [
        ("leave_building_dt", "leave_building_time"),
        ("on_area_start_dt", "on_area_start_time"),
        ("on_area_end_dt", "on_area_end_time"),
        ("return_to_bldg_dt", "return_to_bldg_time"),
    ]:
        if src_col in m.columns and "scan_date" in m.columns:
            m[out_col] = pd.to_datetime(m["scan_date"].dt.strftime("%Y-%m-%d") + " " + m[src_col].astype(str), errors="coerce")

    for c in ["actual_sth_oa", "actual_sth_or", "total_stops_actual", "st_lt_1030_actual", "st_gt_1030_actual", "miles_on_road", "pk_undelivered"]:
        if c in m.columns:
            m[c] = pd.to_numeric(m[c], errors="coerce")
    if "summary_available" in m.columns:
        m["summary_available"] = m["summary_available"].fillna(False).astype(bool)
    return m


def enrich_gap_route_metrics_from_stops(gap_route_metrics_df, gap_stops_df):
    if gap_route_metrics_df is None or gap_route_metrics_df.empty:
        return pd.DataFrame()
    out = gap_route_metrics_df.copy()
    if gap_stops_df is None or gap_stops_df.empty:
        return out

    g = gap_stops_df.copy()
    if "scan_date" in g.columns:
        g["scan_date"] = pd.to_datetime(g["scan_date"], errors="coerce").dt.normalize()
    if "route" in g.columns:
        g["route"] = pd.to_numeric(g["route"], errors="coerce").astype("Int64")
    if "fedex_id" in g.columns:
        g["fedex_id"] = g["fedex_id"].astype(str)
    if "activity_dt" in g.columns:
        g["activity_dt"] = pd.to_datetime(g["activity_dt"], errors="coerce")
    if "stop_order" in g.columns:
        g["stop_order"] = pd.to_numeric(g["stop_order"], errors="coerce")
    if "gap_minutes" in g.columns:
        g["gap_minutes"] = pd.to_numeric(g["gap_minutes"], errors="coerce")
    if "is_event_row" not in g.columns:
        g["is_event_row"] = g.get("address", pd.Series("", index=g.index)).astype(str).str.startswith("*")
    if "event_type" not in g.columns:
        g["event_type"] = np.where(
            g["is_event_row"],
            g.get("address", pd.Series("", index=g.index)).astype(str).str.replace(r"^\*\d+-", "", regex=True).str.upper(),
            "",
        )
    if "is_delivery_like" not in g.columns:
        stop_type_upper = g.get("stop_type", pd.Series("", index=g.index)).astype(str).str.upper()
        g["is_delivery_like"] = stop_type_upper.str.contains("DL", na=False)
    if "is_pickup_like" not in g.columns:
        stop_type_upper = g.get("stop_type", pd.Series("", index=g.index)).astype(str).str.upper()
        g["is_pickup_like"] = stop_type_upper.str.contains("PU", na=False)

    key_cols = ["scan_date", "route", "fedex_id"]

    def _summarize_group(df):
        df = df.sort_values([c for c in ["activity_dt", "stop_order"] if c in df.columns]).copy()
        events = df[df["is_event_row"].fillna(False)].copy()
        cust = df[~df["is_event_row"].fillna(False)].copy()

        leave_dt = events.loc[events["event_type"].astype(str).str.contains("LEAVE BUILDING", na=False), "activity_dt"].min()
        return_dt = events.loc[events["event_type"].astype(str).str.contains("RETURN TO BUILDING", na=False), "activity_dt"].max()
        first_customer_dt = cust["activity_dt"].min() if not cust.empty else pd.NaT
        last_customer_dt = cust["activity_dt"].max() if not cust.empty else pd.NaT
        break_min = _sum_break_minutes(events)

        hr_or_min_calc = np.nan
        if pd.notna(leave_dt) and pd.notna(return_dt):
            hr_or_min_calc = max((return_dt - leave_dt).total_seconds() / 60.0 - break_min, 0.0)

        hr_oa_min_calc = np.nan
        if pd.notna(first_customer_dt) and pd.notna(last_customer_dt):
            hr_oa_min_calc = max((last_customer_dt - first_customer_dt).total_seconds() / 60.0 - break_min, 0.0)

        total_stops_calc = float(cust["stop_order"].notna().sum()) if "stop_order" in cust.columns else float(len(cust))
        st_lt_1030_calc = float((cust["activity_dt"].dt.time <= time(10, 30)).sum()) if not cust.empty else np.nan
        st_gt_1030_calc = float((cust["activity_dt"].dt.time > time(10, 30)).sum()) if not cust.empty else np.nan

        return pd.Series({
            "gap_sum_minutes_all": df["gap_minutes"].sum(min_count=1),
            "gap_avg_minutes_all": df["gap_minutes"].mean(),
            "gap_sum_minutes_customer": cust["gap_minutes"].sum(min_count=1),
            "customer_stop_count": len(cust),
            "event_row_count": len(events),
            "first_activity_dt": df["activity_dt"].min(),
            "last_activity_dt": df["activity_dt"].max(),
            "delivery_stop_count": cust["is_delivery_like"].sum() if "is_delivery_like" in cust.columns else np.nan,
            "pickup_stop_count": cust["is_pickup_like"].sum() if "is_pickup_like" in cust.columns else np.nan,
            "leave_building_dt_calc": leave_dt,
            "return_to_bldg_dt_calc": return_dt,
            "on_area_start_dt_calc": first_customer_dt,
            "on_area_end_dt_calc": last_customer_dt,
            "hr_break_or_min_calc": break_min,
            "actual_hr_or_min_calc": hr_or_min_calc,
            "actual_hr_oa_min_calc": hr_oa_min_calc,
            "total_stops_actual_calc": total_stops_calc,
            "actual_sth_or_calc": _safe_rate_per_hour(total_stops_calc, hr_or_min_calc),
            "actual_sth_oa_calc": _safe_rate_per_hour(total_stops_calc, hr_oa_min_calc),
            "st_lt_1030_actual_calc": st_lt_1030_calc,
            "st_gt_1030_actual_calc": st_gt_1030_calc,
        })

    agg = g.groupby(key_cols, dropna=False).apply(_summarize_group).reset_index()
    out = out.merge(agg, on=key_cols, how="left")

    if "summary_available" not in out.columns:
        out["summary_available"] = True
    metrics_to_fill = {
        "total_stops_actual": "total_stops_actual_calc",
        "actual_hr_oa_min": "actual_hr_oa_min_calc",
        "actual_hr_or_min": "actual_hr_or_min_calc",
        "actual_sth_oa": "actual_sth_oa_calc",
        "actual_sth_or": "actual_sth_or_calc",
        "st_lt_1030_actual": "st_lt_1030_actual_calc",
        "st_gt_1030_actual": "st_gt_1030_actual_calc",
        "leave_building_dt": "leave_building_dt_calc",
        "on_area_start_dt": "on_area_start_dt_calc",
        "on_area_end_dt": "on_area_end_dt_calc",
        "return_to_bldg_dt": "return_to_bldg_dt_calc",
        "hr_break_or_min": "hr_break_or_min_calc",
    }
    for target, calc in metrics_to_fill.items():
        if calc not in out.columns:
            continue
        if target not in out.columns:
            out[target] = np.nan
        fallback_mask = (~out["summary_available"].fillna(False)) | out[target].isna()
        out.loc[fallback_mask, target] = out.loc[fallback_mask, target].where(out.loc[fallback_mask, target].notna(), out.loc[fallback_mask, calc])

    time_text_pairs = {
        "leave_building_time": "leave_building_dt",
        "on_area_start_time": "on_area_start_dt",
        "on_area_end_time": "on_area_end_dt",
        "return_to_bldg_time": "return_to_bldg_dt",
    }
    for text_col, dt_col in time_text_pairs.items():
        if dt_col in out.columns:
            if text_col not in out.columns:
                out[text_col] = np.nan
            fallback_mask = (~out["summary_available"].fillna(False)) | out[text_col].isna()
            out.loc[fallback_mask, text_col] = out.loc[fallback_mask, dt_col].apply(lambda x: pd.Timestamp(x).strftime("%H:%M") if pd.notna(x) else np.nan)

    duration_text_pairs = {
        "actual_hr_oa_text": "actual_hr_oa_min",
        "actual_hr_or_text": "actual_hr_or_min",
        "hr_break_or_text": "hr_break_or_min",
    }
    for text_col, min_col in duration_text_pairs.items():
        if min_col in out.columns:
            if text_col not in out.columns:
                out[text_col] = np.nan
            fallback_mask = (~out["summary_available"].fillna(False)) | out[text_col].isna()
            out.loc[fallback_mask, text_col] = out.loc[fallback_mask, min_col].apply(minutes_to_hhmm)

    missing_summary_mask = ~out["summary_available"].fillna(False)
    if "derived_from_stop_rows" not in out.columns:
        out["derived_from_stop_rows"] = False
    out.loc[missing_summary_mask, "derived_from_stop_rows"] = True
    if "summary_missing_reason" not in out.columns:
        out["summary_missing_reason"] = ""
    out.loc[missing_summary_mask & out["summary_missing_reason"].isna(), "summary_missing_reason"] = "summary_block_missing"

    return out

def build_courier_day_changes(gap_route_metrics_df):
    if gap_route_metrics_df is None or gap_route_metrics_df.empty:
        return pd.DataFrame()
    df = gap_route_metrics_df.copy()
    df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce")
    df["fedex_id"] = df["fedex_id"].astype(str)
    sort_cols = [c for c in ["fedex_id", "scan_date", "route"] if c in df.columns]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    diff_cols = [
        "actual_sth_oa", "actual_sth_or", "actual_hr_oa_min", "actual_hr_or_min",
        "gap_sum_minutes_all", "gap_sum_minutes_customer", "total_stops_actual", "customer_stop_count", "miles_on_road",
    ]
    group = df.groupby("fedex_id", dropna=False)
    df["previous_date"] = group["scan_date"].shift(1)
    if "route" in df.columns:
        df["previous_route"] = group["route"].shift(1)
    for c in diff_cols:
        if c in df.columns:
            df[f"prev_{c}"] = group[c].shift(1)
            df[f"delta_{c}"] = df[c] - df[f"prev_{c}"]
    return df



def build_route_performance_benchmarks(gap_route_metrics_df, metrics_history=None):
    """Build one row per route-day with route-level benchmarks and deltas.
    Uses full metrics_history when provided; otherwise falls back to the filtered frame.
    """
    if gap_route_metrics_df is None or len(gap_route_metrics_df) == 0:
        return pd.DataFrame()

    df = gap_route_metrics_df.copy()
    history = metrics_history.copy() if metrics_history is not None and len(metrics_history) else df.copy()

    for x in (df, history):
        if 'scan_date' in x.columns:
            x['scan_date'] = pd.to_datetime(x['scan_date'], errors='coerce')
        if 'route' in x.columns:
            x['route'] = x['route'].astype(str).str.strip()

    for col in ['actual_sth_oa','actual_sth_or','gap_sum_minutes_all','gap_sum_minutes_customer','total_stops_actual']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if col in history.columns:
            history[col] = pd.to_numeric(history[col], errors='coerce')

    if 'weekday' not in df.columns:
        df['weekday'] = df['scan_date'].dt.day_name()
    if 'weekday' not in history.columns:
        history['weekday'] = history['scan_date'].dt.day_name()

    overall_avg_sth_oa = history['actual_sth_oa'].dropna().mean() if 'actual_sth_oa' in history.columns else np.nan
    overall_avg_sth_or = history['actual_sth_or'].dropna().mean() if 'actual_sth_or' in history.columns else np.nan

    if 'route' in history.columns:
        route_hist = history.groupby('route', dropna=False).agg(
            route_all_history_avg_sth_oa=('actual_sth_oa','mean'),
            route_all_history_avg_sth_or=('actual_sth_or','mean'),
        ).reset_index()
        df = df.merge(route_hist, on='route', how='left')
    else:
        df['route_all_history_avg_sth_oa'] = np.nan
        df['route_all_history_avg_sth_or'] = np.nan

    if {'route','weekday'}.issubset(history.columns):
        same_wd = history.groupby(['route','weekday'], dropna=False).agg(
            route_same_weekday_avg_sth_oa=('actual_sth_oa','mean'),
            route_same_weekday_avg_sth_or=('actual_sth_or','mean'),
        ).reset_index()
        df = df.merge(same_wd, on=['route','weekday'], how='left')
    else:
        df['route_same_weekday_avg_sth_oa'] = np.nan
        df['route_same_weekday_avg_sth_or'] = np.nan

    # previous 7 route observations, excluding current row by date match
    df = df.sort_values(['route','scan_date']).reset_index(drop=True)
    history = history.sort_values(['route','scan_date']).reset_index(drop=True)
    prev7_oa = []
    prev7_or = []
    for _, row in df.iterrows():
        route = row.get('route')
        date = row.get('scan_date')
        hist = history[history['route'].astype(str).str.strip() == str(route).strip()].copy()
        if pd.notna(date):
            hist = hist[hist['scan_date'] < date]
        hist = hist.tail(7)
        prev7_oa.append(hist['actual_sth_oa'].dropna().mean() if 'actual_sth_oa' in hist.columns and len(hist) else np.nan)
        prev7_or.append(hist['actual_sth_or'].dropna().mean() if 'actual_sth_or' in hist.columns and len(hist) else np.nan)
    df['route_prev_7_obs_avg_sth_oa'] = prev7_oa
    df['route_prev_7_obs_avg_sth_or'] = prev7_or

    df['overall_avg_sth_oa'] = overall_avg_sth_oa
    df['overall_avg_sth_or'] = overall_avg_sth_or

    for actual, bench, out in [
        ('actual_sth_oa','route_all_history_avg_sth_oa','delta_vs_route_all_history_oa'),
        ('actual_sth_or','route_all_history_avg_sth_or','delta_vs_route_all_history_or'),
        ('actual_sth_oa','route_prev_7_obs_avg_sth_oa','delta_vs_prev_7_obs_oa'),
        ('actual_sth_or','route_prev_7_obs_avg_sth_or','delta_vs_prev_7_obs_or'),
        ('actual_sth_oa','route_same_weekday_avg_sth_oa','delta_vs_same_weekday_oa'),
        ('actual_sth_or','route_same_weekday_avg_sth_or','delta_vs_same_weekday_or'),
    ]:
        if actual in df.columns and bench in df.columns:
            df[out] = pd.to_numeric(df[actual], errors='coerce') - pd.to_numeric(df[bench], errors='coerce')
            pct = out.replace('delta_','pct_')
            denom = pd.to_numeric(df[bench], errors='coerce').replace(0, np.nan)
            df[pct] = (df[out] / denom) * 100.0

    return df


def build_large_gap_exceptions(gap_df, gap_history=None, floor_minutes=10.0, exclude_first_stop_gap=True):
    """Customer-stop-only large gaps with previous/current stop context.
    Uses gap_history for route thresholds when provided.
    """
    if gap_df is None or len(gap_df) == 0:
        return pd.DataFrame()

    df = gap_df.copy()
    history_source = gap_history.copy() if gap_history is not None and len(gap_history) else df.copy()

    for x in (df, history_source):
        if 'scan_date' in x.columns:
            x['scan_date'] = pd.to_datetime(x['scan_date'], errors='coerce')
        if 'activity_dt' in x.columns:
            x['activity_dt'] = pd.to_datetime(x['activity_dt'], errors='coerce')
        if 'gap_minutes' in x.columns:
            x['gap_minutes'] = pd.to_numeric(x['gap_minutes'], errors='coerce')
        if 'stop_order' in x.columns:
            x['stop_order'] = pd.to_numeric(x['stop_order'], errors='coerce')
        if 'route' in x.columns:
            x['route'] = x['route'].astype(str).str.strip()

    # customer stops only, exclude event rows
    def _customer_only(x):
        if 'is_event_row' in x.columns:
            x = x[~x['is_event_row'].fillna(False)].copy()
        if 'stop_order' in x.columns:
            x = x[x['stop_order'].notna()].copy()
        return x

    df = _customer_only(df)
    history_source = _customer_only(history_source)
    if df.empty:
        return pd.DataFrame()

    # optional: exclude first customer stop from exception logic
    df = df.sort_values([c for c in ['scan_date','route','fedex_id','activity_dt','stop_order'] if c in df.columns]).copy()
    grp_cols = [c for c in ['scan_date','route','fedex_id'] if c in df.columns]
    if exclude_first_stop_gap and grp_cols:
        df['_rownum_in_day'] = df.groupby(grp_cols, dropna=False).cumcount() + 1
        df = df[df['_rownum_in_day'] > 1].copy()

    # previous stop context from filtered customer-stop series
    base = _customer_only(gap_df.copy())
    for col in ['scan_date','activity_dt']:
        if col in base.columns:
            base[col] = pd.to_datetime(base[col], errors='coerce')
    if 'stop_order' in base.columns:
        base['stop_order'] = pd.to_numeric(base['stop_order'], errors='coerce')
    if 'route' in base.columns:
        base['route'] = base['route'].astype(str).str.strip()
    base = base.sort_values([c for c in ['scan_date','route','fedex_id','activity_dt','stop_order'] if c in base.columns]).copy()
    base_grp = [c for c in ['scan_date','route','fedex_id'] if c in base.columns]
    if base_grp:
        base['prev_stop_order'] = base.groupby(base_grp, dropna=False)['stop_order'].shift(1)
        base['prev_stop_type'] = base.groupby(base_grp, dropna=False)['stop_type'].shift(1)
        base['prev_address'] = base.groupby(base_grp, dropna=False)['address'].shift(1)
        base['prev_activity_dt'] = base.groupby(base_grp, dropna=False)['activity_dt'].shift(1)

    merge_keys = [c for c in ['scan_date','route','fedex_id','stop_order'] if c in df.columns and c in base.columns]
    if merge_keys:
        df = df.merge(base[merge_keys + [c for c in ['prev_stop_order','prev_stop_type','prev_address','prev_activity_dt'] if c in base.columns]], on=merge_keys, how='left')

    # route thresholds from history
    if history_source.empty or 'gap_minutes' not in history_source.columns:
        return pd.DataFrame()
    route_stats = history_source.groupby('route', dropna=False)['gap_minutes'].agg(
        route_gap_median='median',
        route_gap_p90=lambda s: s.quantile(0.90)
    ).reset_index()
    route_stats['route_gap_threshold'] = route_stats.apply(
        lambda r: max(float(r['route_gap_p90']) if pd.notna(r['route_gap_p90']) else 0.0,
                      float(r['route_gap_median']) * 2.0 if pd.notna(r['route_gap_median']) else 0.0,
                      float(floor_minutes)), axis=1)
    df = df.merge(route_stats, on='route', how='left')
    df = df[pd.to_numeric(df['gap_minutes'], errors='coerce') >= pd.to_numeric(df['route_gap_threshold'], errors='coerce')].copy()
    if df.empty:
        return df

    df['severity_ratio'] = pd.to_numeric(df['gap_minutes'], errors='coerce') / pd.to_numeric(df['route_gap_threshold'], errors='coerce').replace(0, np.nan)
    df['severity_band'] = np.select(
        [df['severity_ratio'] >= 2.0, df['severity_ratio'] >= 1.5, df['severity_ratio'] >= 1.0],
        ['High','Medium','Watch'],
        default='Watch'
    )
    if 'prev_address' in df.columns and 'address' in df.columns:
        df['before_address'] = df['prev_address']
        df['after_address'] = df['address']
        df['gap_from_to'] = df['before_address'].fillna('') + ' -> ' + df['after_address'].fillna('')
    if 'prev_activity_dt' in df.columns:
        df['before_time'] = pd.to_datetime(df['prev_activity_dt'], errors='coerce').dt.strftime('%H:%M')
    if 'activity_dt' in df.columns:
        df['after_time'] = pd.to_datetime(df['activity_dt'], errors='coerce').dt.strftime('%H:%M')
    if 'before_time' in df.columns and 'after_time' in df.columns:
        df['gap_time_window'] = df['before_time'].fillna('') + ' -> ' + df['after_time'].fillna('')
    if 'scan_date' in df.columns:
        df['scan_date'] = pd.to_datetime(df['scan_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    return df.sort_values(['severity_ratio','gap_minutes'], ascending=[False,False]).reset_index(drop=True)
