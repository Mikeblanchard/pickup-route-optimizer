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
    "consolidation_time_tolerance_min": 5,
    "reason_code_map": {
        910: "Cancelled",
        0: "Completed",
    },
}


OUR_WORKGROUP_EXPLICIT = {482, 483, 488, 489, 924}
OUR_WORKGROUP_RANGE = range(702, 800)
OUR_WORKGROUP_EXCLUDE = {721}

def is_relevant_workgroup_route(x):
    try:
        if pd.isna(x):
            return False
        r = int(float(x))
    except Exception:
        return False
    if r in OUR_WORKGROUP_EXCLUDE:
        return False
    return r in OUR_WORKGROUP_EXPLICIT or r in OUR_WORKGROUP_RANGE

def filter_to_our_workgroup(df, route_col='route'):
    if df is None or df.empty or route_col not in df.columns:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out = df.copy()
    mask = out[route_col].apply(is_relevant_workgroup_route)
    return out.loc[mask].copy()

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


def rebuild_gap_metrics_from_master(gap_master: pd.DataFrame, gap_route_metrics_master: pd.DataFrame | None = None) -> pd.DataFrame:
    """Recompute / backfill courier-day GAP metrics from saved stop rows.

    Derived values are preferred for:
    - total_stops_actual (Stop Order count)
    - actual_hr_oa_min
    - actual_hr_or_min
    - actual_sth_oa
    - actual_sth_or
    """
    metrics_std = standardize_gap_route_metrics(gap_route_metrics_master) if gap_route_metrics_master is not None and not gap_route_metrics_master.empty else pd.DataFrame()
    return enrich_gap_route_metrics_from_stops(metrics_std, gap_master)


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
            name = getattr(f, "name", "").lower()
            f.seek(0)
            if name.endswith('.csv'):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
            df["source_file"] = getattr(f, "name", "uploaded_file")
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


def _gap_source_priority(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.lower()
    return np.select(
        [s.eq("gap_html") | s.str.endswith(".html") | s.str.endswith(".htm"),
         s.eq("gap_excel") | s.str.endswith(".xlsx") | s.str.endswith(".xls") | s.str.endswith(".csv")],
        [2, 1],
        default=0,
    )


def _gap_row_richness(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["address", "postal_code", "stop_type", "activity_dt", "gap_minutes", "fedex_id", "stat", "fxe_pkgs", "fxg_pkgs", "event_type"] if c in df.columns]
    if not cols:
        return pd.Series(0, index=df.index, dtype=float)
    return df[cols].notna().sum(axis=1)


def build_gap_stop_dedup_key(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=str)
    d = df.copy()
    scan_date = pd.to_datetime(d.get("scan_date"), errors="coerce").dt.strftime("%Y-%m-%d").fillna("")
    route = pd.to_numeric(d.get("route"), errors="coerce").astype("Int64").astype(str).replace("<NA>", "")
    fedex_id = d.get("fedex_id", pd.Series("", index=d.index)).fillna("").astype(str).str.strip()
    stop_order = pd.to_numeric(d.get("stop_order"), errors="coerce").astype("Int64").astype(str).replace("<NA>", "")
    address_norm = d.get("address_norm", d.get("address", pd.Series("", index=d.index))).fillna("").astype(str)
    address_norm = address_norm.apply(normalize_address_for_match)
    stop_type = d.get("stop_type", pd.Series("", index=d.index)).fillna("").astype(str).str.upper().str.strip()
    activity_dt = pd.to_datetime(d.get("activity_dt"), errors="coerce")
    activity_min = activity_dt.dt.floor("min").dt.strftime("%Y-%m-%d %H:%M").fillna("")
    activity_txt = d.get("activity_time", pd.Series("", index=d.index)).fillna("").astype(str).str.strip()
    is_event = d.get("is_event_row", d.get("address", pd.Series("", index=d.index)).fillna("").astype(str).str.startswith("*"))
    event_name = d.get("event_type", d.get("address", pd.Series("", index=d.index))).fillna("").astype(str).str.upper().str.strip()

    customer_key = scan_date + "|" + route + "|" + fedex_id + "|C|" + stop_order + "|" + address_norm + "|" + activity_min + "|" + stop_type
    event_key = scan_date + "|" + route + "|" + fedex_id + "|E|" + event_name + "|" + activity_min + "|" + activity_txt
    return pd.Series(np.where(is_event.fillna(False), event_key, customer_key), index=d.index)


def merge_gap_stop_sources(master_df: pd.DataFrame | None, new_df: pd.DataFrame | None) -> pd.DataFrame:
    frames = []
    if master_df is not None and not master_df.empty:
        frames.append(master_df.copy())
    if new_df is not None and not new_df.empty:
        frames.append(new_df.copy())
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    if "address_norm" not in out.columns and "address" in out.columns:
        out["address_norm"] = out["address"].apply(normalize_address_for_match)
    if "is_event_row" not in out.columns and "address" in out.columns:
        out["is_event_row"] = out["address"].fillna("").astype(str).str.startswith("*")
    if "source_dataset" not in out.columns:
        out["source_dataset"] = out.get("source_file", "")

    out["gap_stop_dedup_key"] = build_gap_stop_dedup_key(out)
    out["_source_priority"] = _gap_source_priority(out.get("source_dataset", out.get("source_file", pd.Series("", index=out.index))))
    out["_row_richness"] = _gap_row_richness(out)
    out["_orig_idx"] = range(len(out))
    out = out.sort_values(["gap_stop_dedup_key", "_source_priority", "_row_richness", "_orig_idx"], ascending=[True, False, False, True]).reset_index(drop=True)

    keep_keys = []
    merged_rows = []
    for _, grp in out.groupby("gap_stop_dedup_key", dropna=False, sort=False):
        base = grp.iloc[0].copy()
        for col in grp.columns:
            vals = grp[col]
            nonnull = vals.dropna()
            if nonnull.empty:
                continue
            if base.get(col) is None or pd.isna(base.get(col)) or (isinstance(base.get(col), str) and str(base.get(col)).strip() == ""):
                base[col] = nonnull.iloc[0]
        merged_rows.append(base)
        keep_keys.append(base["gap_stop_dedup_key"])

    out = pd.DataFrame(merged_rows)
    drop_cols = [c for c in ["_source_priority", "_row_richness", "_orig_idx"] if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
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
    g["source_dataset"] = "gap_excel"
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
    p = filter_to_our_workgroup(p, "route")
    return p


def consolidate_physical_pickups(p):
    if p.empty:
        return p.copy()
    df = p.copy()
    if "source_file" not in df.columns:
        df["source_file"] = ""
    if "scanned_ground_acct" not in df.columns:
        df["scanned_ground_acct"] = 1
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
    g["source_dataset"] = "gap_html"
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
    for c in ["fedex_id", "courier_name", "loc", "loc_nbr", "source_file", "leave_building_time", "on_area_start_time", "on_area_end_time", "return_to_bldg_time", "total_paid_hours_text", "actual_hr_oa_text", "actual_hr_or_text", "hr_to_area_text", "hr_from_area_text", "hr_break_or_text"]:
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
    if "is_event_row" not in g.columns:
        g["is_event_row"] = g.get("address", pd.Series("", index=g.index)).astype(str).str.startswith("*")

    agg = (
        g.groupby(["scan_date", "route", "fedex_id"], dropna=False)
        .agg(
            gap_sum_minutes_all=("gap_minutes", "sum"),
            gap_avg_minutes_all=("gap_minutes", "mean"),
            gap_sum_minutes_customer=("gap_minutes", lambda s: s[g.loc[s.index, "is_event_row"] == False].sum()),
            customer_stop_count=("is_event_row", lambda s: (~s).sum()),
            event_row_count=("is_event_row", "sum"),
            first_activity_dt=("activity_dt", "min"),
            last_activity_dt=("activity_dt", "max"),
            delivery_stop_count=("is_delivery_like", "sum"),
            pickup_stop_count=("is_pickup_like", "sum"),
        )
        .reset_index()
    )
    out = out.merge(agg, on=["scan_date", "route", "fedex_id"], how="left")
    return out


def build_courier_day_changes(gap_route_metrics_df):
    if gap_route_metrics_df is None or gap_route_metrics_df.empty:
        return pd.DataFrame()
    df = gap_route_metrics_df.copy()
    df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce")
    df["fedex_id"] = df.get("fedex_id", pd.Series("", index=df.index)).astype(str)
    if "route" in df.columns:
        df["route"] = pd.to_numeric(df["route"], errors="coerce").astype("Int64")
    df = df.sort_values([c for c in ["fedex_id", "scan_date", "route"] if c in df.columns]).reset_index(drop=True)
    diff_cols = [
        "actual_sth_oa", "actual_sth_or", "actual_hr_oa_min", "actual_hr_or_min",
        "gap_sum_minutes_all", "gap_sum_minutes_customer", "total_stops_actual",
        "customer_stop_count", "miles_on_road",
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
    if gap_route_metrics_df is None or gap_route_metrics_df.empty:
        return pd.DataFrame()
    cur = gap_route_metrics_df.copy()
    hist = metrics_history.copy() if metrics_history is not None and not metrics_history.empty else cur.copy()
    for df in (cur, hist):
        if "scan_date" in df.columns:
            df["scan_date"] = pd.to_datetime(df["scan_date"], errors="coerce")
        if "route" in df.columns:
            df["route"] = pd.to_numeric(df["route"], errors="coerce").astype("Int64")
        if "courier_name" in df.columns:
            df["courier_name"] = df["courier_name"].astype(str)
    cur["weekday"] = cur["scan_date"].dt.day_name()
    hist["weekday"] = hist["scan_date"].dt.day_name()
    overall_oa = hist["actual_sth_oa"].dropna().mean() if "actual_sth_oa" in hist.columns else np.nan
    overall_or = hist["actual_sth_or"].dropna().mean() if "actual_sth_or" in hist.columns else np.nan
    cur["overall_avg_sth_oa"] = overall_oa
    cur["overall_avg_sth_or"] = overall_or
    route_hist = hist.groupby("route", dropna=False).agg(
        route_all_history_avg_sth_oa=("actual_sth_oa", "mean"),
        route_all_history_avg_sth_or=("actual_sth_or", "mean"),
    ).reset_index()
    cur = cur.merge(route_hist, on="route", how="left")
    prev7_oa, prev7_or, samewd_oa, samewd_or = [], [], [], []
    for _, row in cur.iterrows():
        h = hist[hist["route"] == row["route"]].copy()
        h = h[h["scan_date"] < row["scan_date"]].sort_values("scan_date")
        prev7 = h.tail(7)
        prev7_oa.append(prev7["actual_sth_oa"].dropna().mean() if "actual_sth_oa" in prev7.columns and not prev7.empty else np.nan)
        prev7_or.append(prev7["actual_sth_or"].dropna().mean() if "actual_sth_or" in prev7.columns and not prev7.empty else np.nan)
        same = h[h["weekday"] == row["weekday"]]
        samewd_oa.append(same["actual_sth_oa"].dropna().mean() if "actual_sth_oa" in same.columns and not same.empty else np.nan)
        samewd_or.append(same["actual_sth_or"].dropna().mean() if "actual_sth_or" in same.columns and not same.empty else np.nan)
    cur["route_prev_7_obs_avg_sth_oa"] = prev7_oa
    cur["route_prev_7_obs_avg_sth_or"] = prev7_or
    cur["route_same_weekday_avg_sth_oa"] = samewd_oa
    cur["route_same_weekday_avg_sth_or"] = samewd_or
    for lhs, rhs, out in [
        ("actual_sth_oa", "overall_avg_sth_oa", "delta_vs_overall_oa"),
        ("actual_sth_or", "overall_avg_sth_or", "delta_vs_overall_or"),
        ("actual_sth_oa", "route_all_history_avg_sth_oa", "delta_vs_route_all_history_oa"),
        ("actual_sth_or", "route_all_history_avg_sth_or", "delta_vs_route_all_history_or"),
        ("actual_sth_oa", "route_prev_7_obs_avg_sth_oa", "delta_vs_prev_7_obs_oa"),
        ("actual_sth_or", "route_prev_7_obs_avg_sth_or", "delta_vs_prev_7_obs_or"),
        ("actual_sth_oa", "route_same_weekday_avg_sth_oa", "delta_vs_same_weekday_oa"),
        ("actual_sth_or", "route_same_weekday_avg_sth_or", "delta_vs_same_weekday_or"),
    ]:
        if lhs in cur.columns and rhs in cur.columns:
            cur[out] = cur[lhs] - cur[rhs]
    return cur


def build_large_gap_exceptions(gap_df, gap_history=None, floor_minutes=10.0, exclude_first_stop_gap=True):
    if gap_df is None or gap_df.empty:
        return pd.DataFrame()

    def _prep(df):
        d = df.copy()
        if "scan_date" in d.columns:
            d["scan_date"] = pd.to_datetime(d["scan_date"], errors="coerce").dt.normalize()
        if "route" in d.columns:
            d["route"] = pd.to_numeric(d["route"], errors="coerce").astype("Int64")
        d["fedex_id"] = d.get("fedex_id", pd.Series("", index=d.index)).astype(str)
        d["courier_name"] = d.get("courier_name", pd.Series("", index=d.index)).astype(str)
        d["activity_dt"] = pd.to_datetime(d.get("activity_dt"), errors="coerce")
        d["stop_order"] = pd.to_numeric(d.get("stop_order"), errors="coerce")
        d["gap_minutes"] = pd.to_numeric(d.get("gap_minutes"), errors="coerce")
        d["address"] = d.get("address", pd.Series("", index=d.index)).astype(str)
        d["stop_type"] = d.get("stop_type", pd.Series("", index=d.index)).astype(str)
        if "is_event_row" not in d.columns:
            d["is_event_row"] = d["address"].str.startswith("*")
        d["event_type"] = np.where(d["is_event_row"], d["address"].str.replace(r"^\*\d+-", "", regex=True).str.upper(), "")
        d["is_customer_stop"] = ~d["is_event_row"].fillna(False)
        return d

    def _customer_legs(d):
        d = _prep(d)
        grp_cols = [c for c in ["scan_date", "route", "fedex_id"] if c in d.columns]
        rows = []
        if not grp_cols:
            return pd.DataFrame()
        for _, grp in d.sort_values([c for c in grp_cols + ["activity_dt", "stop_order"] if c in d.columns]).groupby(grp_cols, dropna=False):
            grp = grp.reset_index(drop=True)
            cust_idx = grp.index[grp["is_customer_stop"] & grp["activity_dt"].notna()].tolist()
            if exclude_first_stop_gap and cust_idx:
                cust_idx = cust_idx[1:]
            for cur_idx in cust_idx:
                prev_candidates = grp.index[(grp.index < cur_idx) & grp["is_customer_stop"] & grp["activity_dt"].notna()].tolist()
                if not prev_candidates:
                    continue
                prev_idx = prev_candidates[-1]
                prev = grp.loc[prev_idx]
                cur_row = grp.loc[cur_idx]
                between = grp.loc[(grp.index > prev_idx) & (grp.index < cur_idx)].copy()
                begin_times = between.loc[between["event_type"].str.contains("BEGIN BREAK", na=False), "activity_dt"].tolist()
                end_times = between.loc[between["event_type"].str.contains("END BREAK", na=False), "activity_dt"].tolist()
                break_mins = 0.0
                for bt in begin_times:
                    et = next((x for x in end_times if pd.notna(x) and pd.notna(bt) and x >= bt), None)
                    if et is not None:
                        break_mins += max((et - bt).total_seconds()/60.0, 0.0)
                leg_elapsed = (cur_row["activity_dt"] - prev["activity_dt"]).total_seconds()/60.0 if pd.notna(cur_row["activity_dt"]) and pd.notna(prev["activity_dt"]) else np.nan
                adjusted = leg_elapsed - break_mins if pd.notna(leg_elapsed) else np.nan
                rows.append({
                    "scan_date": cur_row.get("scan_date"),
                    "route": cur_row.get("route"),
                    "fedex_id": cur_row.get("fedex_id"),
                    "courier_name": cur_row.get("courier_name"),
                    "stop_order": cur_row.get("stop_order"),
                    "stop_type": cur_row.get("stop_type"),
                    "before_address": prev.get("address"),
                    "after_address": cur_row.get("address"),
                    "before_time": prev.get("activity_dt").strftime("%H:%M") if pd.notna(prev.get("activity_dt")) else "",
                    "after_time": cur_row.get("activity_dt").strftime("%H:%M") if pd.notna(cur_row.get("activity_dt")) else "",
                    "leg_elapsed_minutes": leg_elapsed,
                    "break_minutes_between": break_mins,
                    "adjusted_gap_minutes": adjusted,
                    "gap_from_to": f"{prev.get('address','')} -> {cur_row.get('address','')}",
                    "gap_time_window": f"{prev.get('activity_dt').strftime('%H:%M') if pd.notna(prev.get('activity_dt')) else ''} -> {cur_row.get('activity_dt').strftime('%H:%M') if pd.notna(cur_row.get('activity_dt')) else ''}",
                    "source_file": cur_row.get("source_file"),
                })
        return pd.DataFrame(rows)

    cur_legs = _customer_legs(gap_df)
    hist_legs = _customer_legs(gap_history if gap_history is not None and not gap_history.empty else gap_df)
    if cur_legs.empty or hist_legs.empty:
        return pd.DataFrame()
    route_stats = hist_legs.groupby("route", dropna=False)["adjusted_gap_minutes"].agg(route_gap_median="median", route_gap_mean="mean", route_gap_count="count").reset_index()
    p90 = hist_legs.groupby("route", dropna=False)["adjusted_gap_minutes"].quantile(0.90).reset_index(name="route_gap_p90")
    route_stats = route_stats.merge(p90, on="route", how="left")
    route_stats["route_gap_threshold"] = route_stats.apply(lambda r: max(float(r["route_gap_p90"]) if pd.notna(r["route_gap_p90"]) else 0.0, float(r["route_gap_median"]) * 2.0 if pd.notna(r["route_gap_median"]) else 0.0, float(floor_minutes)), axis=1)
    out = cur_legs.merge(route_stats, on="route", how="left")
    out["route_gap_threshold"] = out["route_gap_threshold"].fillna(float(floor_minutes))
    out = out[out["adjusted_gap_minutes"] >= out["route_gap_threshold"]].copy()
    if out.empty:
        return pd.DataFrame()
    out["severity_ratio"] = np.where(out["route_gap_threshold"] > 0, out["adjusted_gap_minutes"] / out["route_gap_threshold"], np.nan)
    out["severity_band"] = np.select([out["severity_ratio"] >= 2.0, out["severity_ratio"] >= 1.5], ["High", "Medium"], default="Watch")
    keep_cols = ["scan_date","route","courier_name","fedex_id","before_address","after_address","before_time","after_time","leg_elapsed_minutes","break_minutes_between","adjusted_gap_minutes","stop_type","route_gap_median","route_gap_p90","route_gap_threshold","severity_ratio","severity_band","gap_from_to","gap_time_window","source_file"]
    keep_cols = [c for c in keep_cols if c in out.columns]
    return out.sort_values([c for c in ["scan_date","route","severity_ratio","adjusted_gap_minutes"] if c in out.columns], ascending=[True,True,False,False])[keep_cols].reset_index(drop=True)


# =============================
# Stable overrides for Pickup Route Optimizer
# =============================
APP_CONFIG = {
    "station_address": "45 Di Poce Way Vaughan, ON L4H 4J4",
    "wave_starts": {"W1": "08:15", "W2": "10:30"},
    "weekend_starts": {"Saturday": "08:15", "Sunday": "09:30"},
    "start_of_day_station_min": 30,
    "end_of_day_station_min": 15,
    "consolidation_time_tolerance_min": 5,
    "large_gap_internal_floor_min": 10.0,
    "reason_code_map": {
        910: "Cancelled",
        0: "Completed",
    },
}


def _is_route_section_start_row(row_text: str) -> bool:
    txt = clean_text(row_text) or ""
    txt_u = txt.upper()
    return ("LOC:" in txt_u) and ("DATE:" in txt_u) and ("ROUTE:" in txt_u)


def _derive_route_metrics_from_stops(gap_stops_df: pd.DataFrame) -> pd.DataFrame:
    if gap_stops_df is None or gap_stops_df.empty:
        return pd.DataFrame()

    g = gap_stops_df.copy()
    if "scan_date" in g.columns:
        g["scan_date"] = pd.to_datetime(g["scan_date"], errors="coerce").dt.normalize()
    if "route" in g.columns:
        g["route"] = pd.to_numeric(g["route"], errors="coerce").astype("Int64")
    if "fedex_id" in g.columns:
        g["fedex_id"] = g["fedex_id"].astype(str)
    else:
        g["fedex_id"] = ""

    if "is_event_row" not in g.columns:
        g["is_event_row"] = g.get("address", pd.Series("", index=g.index)).astype(str).str.startswith("*")
    if "event_type" not in g.columns:
        g["event_type"] = np.where(
            g["is_event_row"],
            g.get("address", pd.Series("", index=g.index)).astype(str).str.replace(r"^\*\d+-", "", regex=True).str.upper(),
            "",
        )

    grp_cols = ["scan_date", "route", "fedex_id"]
    rows = []
    for keys, grp in g.sort_values([c for c in grp_cols + ["activity_dt", "stop_order"] if c in g.columns]).groupby(grp_cols, dropna=False):
        grp = grp.copy().reset_index(drop=True)
        customer = grp.loc[~grp["is_event_row"].fillna(False)].copy()
        customer = customer.loc[customer.get("activity_dt", pd.Series(pd.NaT, index=customer.index)).notna()].copy()

        def _first_event_dt(pattern: str):
            s = grp.loc[grp["event_type"].astype(str).str.contains(pattern, na=False), "activity_dt"]
            return s.min() if not s.empty else pd.NaT

        def _last_event_dt(pattern: str):
            s = grp.loc[grp["event_type"].astype(str).str.contains(pattern, na=False), "activity_dt"]
            return s.max() if not s.empty else pd.NaT

        leave_building_dt = _first_event_dt(r"LEAVE BUILDING")
        return_to_bldg_dt = _last_event_dt(r"RETURN TO BUILDING")
        on_area_start_dt = customer["activity_dt"].min() if not customer.empty else pd.NaT
        on_area_end_dt = customer["activity_dt"].max() if not customer.empty else pd.NaT

        begin_breaks = grp.loc[grp["event_type"].astype(str).str.contains(r"BEGIN BREAK", na=False), "activity_dt"].sort_values().tolist()
        end_breaks = grp.loc[grp["event_type"].astype(str).str.contains(r"END BREAK", na=False), "activity_dt"].sort_values().tolist()
        break_minutes = 0.0
        remaining_end_breaks = list(end_breaks)
        for bt in begin_breaks:
            if pd.isna(bt):
                continue
            et = next((x for x in remaining_end_breaks if pd.notna(x) and x >= bt), None)
            if et is not None:
                break_minutes += max((et - bt).total_seconds() / 60.0, 0.0)
                remaining_end_breaks.remove(et)

        stop_order_numeric = pd.to_numeric(customer.get("stop_order"), errors="coerce") if not customer.empty else pd.Series(dtype=float)
        total_stops_actual = float(stop_order_numeric.notna().sum()) if not customer.empty else 0.0
        cutoff_1030 = pd.Timestamp(pd.Timestamp(keys[0]).date()).replace(hour=10, minute=30) if pd.notna(keys[0]) else pd.NaT
        st_lt_1030_actual = float((customer["activity_dt"] <= cutoff_1030).sum()) if not customer.empty and pd.notna(cutoff_1030) else np.nan
        st_gt_1030_actual = float((customer["activity_dt"] > cutoff_1030).sum()) if not customer.empty and pd.notna(cutoff_1030) else np.nan

        hr_or_min = np.nan
        if pd.notna(leave_building_dt) and pd.notna(return_to_bldg_dt):
            hr_or_min = max((return_to_bldg_dt - leave_building_dt).total_seconds() / 60.0, 0.0)

        hr_oa_min = np.nan
        if pd.notna(on_area_start_dt) and pd.notna(on_area_end_dt):
            hr_oa_min = max((on_area_end_dt - on_area_start_dt).total_seconds() / 60.0, 0.0)

        hr_to_area_min = np.nan
        if pd.notna(leave_building_dt) and pd.notna(on_area_start_dt):
            hr_to_area_min = max((on_area_start_dt - leave_building_dt).total_seconds() / 60.0, 0.0)

        hr_from_area_min = np.nan
        if pd.notna(on_area_end_dt) and pd.notna(return_to_bldg_dt):
            hr_from_area_min = max((return_to_bldg_dt - on_area_end_dt).total_seconds() / 60.0, 0.0)

        actual_sth_oa = (total_stops_actual / (hr_oa_min / 60.0)) if pd.notna(hr_oa_min) and hr_oa_min > 0 else np.nan
        actual_sth_or = (total_stops_actual / (hr_or_min / 60.0)) if pd.notna(hr_or_min) and hr_or_min > 0 else np.nan

        first_row = grp.iloc[0] if not grp.empty else pd.Series(dtype=object)
        courier_name = None
        if "courier_name" in grp.columns:
            courier_nonnull = grp["courier_name"].dropna().astype(str)
            courier_nonnull = courier_nonnull[courier_nonnull.str.strip() != ""]
            courier_name = courier_nonnull.iloc[0] if not courier_nonnull.empty else None

        rows.append({
            "scan_date": keys[0],
            "route": keys[1],
            "route_key": normalize_route_key(keys[1]),
            "fedex_id": keys[2],
            "courier_name": courier_name,
            "loc": first_row.get("loc"),
            "loc_nbr": first_row.get("loc_nbr"),
            "source_file": first_row.get("source_file"),
            "leave_building_time": leave_building_dt.strftime("%H:%M") if pd.notna(leave_building_dt) else None,
            "on_area_start_time": on_area_start_dt.strftime("%H:%M") if pd.notna(on_area_start_dt) else None,
            "on_area_end_time": on_area_end_dt.strftime("%H:%M") if pd.notna(on_area_end_dt) else None,
            "return_to_bldg_time": return_to_bldg_dt.strftime("%H:%M") if pd.notna(return_to_bldg_dt) else None,
            "leave_building_dt": leave_building_dt,
            "on_area_start_dt": on_area_start_dt,
            "on_area_end_dt": on_area_end_dt,
            "return_to_bldg_dt": return_to_bldg_dt,
            "total_paid_hours_text": None,
            "total_paid_hours_min": np.nan,
            "actual_hr_oa_text": None,
            "actual_hr_or_text": None,
            "hr_to_area_text": None,
            "hr_from_area_text": None,
            "hr_break_or_text": None,
            "actual_hr_oa_min": hr_oa_min,
            "actual_hr_or_min": hr_or_min,
            "hr_to_area_min": hr_to_area_min,
            "hr_from_area_min": hr_from_area_min,
            "hr_break_or_min": break_minutes,
            "actual_sth_oa": actual_sth_oa,
            "actual_sth_or": actual_sth_or,
            "total_stops_actual": total_stops_actual,
            "st_lt_1030_actual": st_lt_1030_actual,
            "st_gt_1030_actual": st_gt_1030_actual,
        })
    return pd.DataFrame(rows)


def read_gap_html_file(file_obj):
    """Parse FWR GAP saved HTML into stop rows + route-level metrics.
    Robust to route sections that are missing some summary/header content.
    """
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
        if _is_route_section_start_row(row_text):
            route_start_idxs.append(i)

    if not route_start_idxs:
        return pd.DataFrame(), pd.DataFrame()
    route_start_idxs.append(len(trs))

    stop_rows = []
    metric_rows = []
    source_file = getattr(file_obj, "name", "gap_html")

    for s_idx, e_idx in zip(route_start_idxs[:-1], route_start_idxs[1:]):
        section_rows = trs[s_idx:e_idx]
        if not section_rows:
            continue

        header_text = " ".join(td.get_text(" ", strip=True) for td in section_rows[0].find_all("td"))
        route_info = _extract_route_header_fields(header_text)

        for tr in section_rows[:12]:
            block_text = " ".join(td.get_text(" ", strip=True) for td in tr.find_all("td"))
            route_info.update({k: v for k, v in _extract_identity_block(block_text).items() if v})
            if not route_info.get("total_paid_hours_text"):
                m = re.search(r"TOTAL PAID HRS:\s*([0-9:]+)", block_text, flags=re.I)
                if m:
                    route_info["total_paid_hours_text"] = m.group(1)
            if not route_info.get("fedex_id"):
                m = re.search(r"FEDEX ID:\s*(\d+)", block_text, flags=re.I)
                if m:
                    route_info["fedex_id"] = m.group(1)

        stop_header_idx = None
        summary_map = {}
        last_map = {}

        for local_idx, tr in enumerate(section_rows):
            cells = [clean_text(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
            if cells[:5] == STOP_HEADERS[:5]:
                stop_header_idx = local_idx
                break
            if len(cells) == 5 and cells[0] not in {"Element", "Plan"} and not str(cells[0]).startswith("LOC:"):
                summary_map[cells[0]] = cells[1]
            elif len(cells) == 2 and cells[0] not in {"Element", "Plan"}:
                if cells[0] and cells[1]:
                    last_map[cells[0]] = cells[1]

        if stop_header_idx is None:
            continue

        for tr in section_rows[stop_header_idx + 1:]:
            cells = [clean_text(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
            if len(cells) != len(STOP_HEADERS):
                continue
            if cells[0] == "Date":
                continue
            row = dict(zip(STOP_HEADERS, cells))
            row["source_file"] = source_file
            row["courier_name"] = route_info.get("courier_name") or route_info.get("courier_short_name")
            row["loc"] = route_info.get("loc")
            stop_rows.append(row)

        metric_rows.append({
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
            "source_file": source_file,
        })

    stop_df = pd.DataFrame(stop_rows)
    metrics_df = pd.DataFrame(metric_rows)
    return stop_df, metrics_df


def enrich_gap_route_metrics_from_stops(gap_route_metrics_df, gap_stops_df):
    derived = _derive_route_metrics_from_stops(gap_stops_df)
    if (gap_route_metrics_df is None or gap_route_metrics_df.empty) and derived.empty:
        return pd.DataFrame()
    if gap_route_metrics_df is None or gap_route_metrics_df.empty:
        out = derived.copy()
    else:
        out = standardize_gap_route_metrics(gap_route_metrics_df)
        if not derived.empty:
            merge_cols = ["scan_date", "route", "fedex_id"]
            derived_std = standardize_gap_route_metrics(derived)
            out = out.merge(derived_std, on=merge_cols, how="outer", suffixes=("", "_drv"))
            prefer_fill_cols = [
                "courier_name", "loc", "loc_nbr", "source_file",
                "leave_building_time", "on_area_start_time", "on_area_end_time", "return_to_bldg_time",
                "leave_building_dt", "on_area_start_dt", "on_area_end_dt", "return_to_bldg_dt",
                "total_paid_hours_text", "total_paid_hours_min",
                "actual_hr_oa_text", "actual_hr_or_text", "hr_to_area_text", "hr_from_area_text", "hr_break_or_text",
                "hr_to_area_min", "hr_from_area_min", "hr_break_or_min",
                "st_lt_1030_actual", "st_gt_1030_actual",
            ]
            always_prefer_derived_cols = [
                "actual_hr_oa_min", "actual_hr_or_min", "actual_sth_oa", "actual_sth_or", "total_stops_actual"
            ]
            for col in prefer_fill_cols:
                drv_col = f"{col}_drv"
                if drv_col in out.columns:
                    if col not in out.columns:
                        out[col] = out[drv_col]
                    else:
                        out[col] = out[col].where(out[col].notna(), out[drv_col])
                    out = out.drop(columns=[drv_col])

            for col in always_prefer_derived_cols:
                drv_col = f"{col}_drv"
                if drv_col in out.columns:
                    out[col] = out[drv_col]
                    out = out.drop(columns=[drv_col])
            extra_drv_cols = [c for c in out.columns if c.endswith("_drv")]
            if extra_drv_cols:
                out = out.drop(columns=extra_drv_cols)

    if out.empty:
        return out

    g = gap_stops_df.copy() if gap_stops_df is not None else pd.DataFrame()
    if not g.empty:
        if "scan_date" in g.columns:
            g["scan_date"] = pd.to_datetime(g["scan_date"], errors="coerce").dt.normalize()
        if "route" in g.columns:
            g["route"] = pd.to_numeric(g["route"], errors="coerce").astype("Int64")
        if "fedex_id" in g.columns:
            g["fedex_id"] = g["fedex_id"].astype(str)
        if "is_event_row" not in g.columns:
            g["is_event_row"] = g.get("address", pd.Series("", index=g.index)).astype(str).str.startswith("*")

        agg = (
            g.groupby(["scan_date", "route", "fedex_id"], dropna=False)
            .agg(
                gap_sum_minutes_all=("gap_minutes", "sum"),
                gap_avg_minutes_all=("gap_minutes", "mean"),
                gap_sum_minutes_customer=("gap_minutes", lambda s: s[g.loc[s.index, "is_event_row"] == False].sum()),
                customer_stop_count=("is_event_row", lambda s: (~s).sum()),
                event_row_count=("is_event_row", "sum"),
                first_activity_dt=("activity_dt", "min"),
                last_activity_dt=("activity_dt", "max"),
                delivery_stop_count=("is_delivery_like", "sum"),
                pickup_stop_count=("is_pickup_like", "sum"),
            )
            .reset_index()
        )
        out = out.merge(agg, on=["scan_date", "route", "fedex_id"], how="left")

    if "route" in out.columns:
        out["route_key"] = out["route"].apply(normalize_route_key)
    return out


def build_large_gap_exceptions(gap_df, gap_history=None, floor_minutes=None, exclude_first_stop_gap=True):
    if floor_minutes is None:
        floor_minutes = float(APP_CONFIG.get("large_gap_internal_floor_min", 10.0))
    return globals()["__builtins__"] and _build_large_gap_exceptions_impl(gap_df, gap_history, floor_minutes, exclude_first_stop_gap)


def _build_large_gap_exceptions_impl(gap_df, gap_history=None, floor_minutes=10.0, exclude_first_stop_gap=True):
    if gap_df is None or gap_df.empty:
        return pd.DataFrame()

    def _prep(df):
        d = df.copy()
        if "scan_date" in d.columns:
            d["scan_date"] = pd.to_datetime(d["scan_date"], errors="coerce").dt.normalize()
        if "route" in d.columns:
            d["route"] = pd.to_numeric(d["route"], errors="coerce").astype("Int64")
        d["fedex_id"] = d.get("fedex_id", pd.Series("", index=d.index)).astype(str)
        d["courier_name"] = d.get("courier_name", pd.Series("", index=d.index)).astype(str)
        d["activity_dt"] = pd.to_datetime(d.get("activity_dt"), errors="coerce")
        d["stop_order"] = pd.to_numeric(d.get("stop_order"), errors="coerce")
        d["gap_minutes"] = pd.to_numeric(d.get("gap_minutes"), errors="coerce")
        d["address"] = d.get("address", pd.Series("", index=d.index)).astype(str)
        d["stop_type"] = d.get("stop_type", pd.Series("", index=d.index)).astype(str)
        if "is_event_row" not in d.columns:
            d["is_event_row"] = d["address"].str.startswith("*")
        d["event_type"] = np.where(d["is_event_row"], d["address"].str.replace(r"^\*\d+-", "", regex=True).str.upper(), "")
        d["is_customer_stop"] = ~d["is_event_row"].fillna(False)
        return d

    def _customer_legs(d):
        d = _prep(d)
        grp_cols = [c for c in ["scan_date", "route", "fedex_id"] if c in d.columns]
        rows = []
        if not grp_cols:
            return pd.DataFrame()
        sort_cols = [c for c in grp_cols + ["activity_dt", "stop_order"] if c in d.columns]
        for _, grp in d.sort_values(sort_cols).groupby(grp_cols, dropna=False):
            grp = grp.reset_index(drop=True)
            cust_idx = grp.index[grp["is_customer_stop"] & grp["activity_dt"].notna()].tolist()
            if exclude_first_stop_gap and cust_idx:
                cust_idx = cust_idx[1:]
            for cur_idx in cust_idx:
                prev_candidates = grp.index[(grp.index < cur_idx) & grp["is_customer_stop"] & grp["activity_dt"].notna()].tolist()
                if not prev_candidates:
                    continue
                prev_idx = prev_candidates[-1]
                prev = grp.loc[prev_idx]
                cur_row = grp.loc[cur_idx]
                between = grp.loc[(grp.index > prev_idx) & (grp.index < cur_idx)].copy()
                begin_times = between.loc[between["event_type"].str.contains("BEGIN BREAK", na=False), "activity_dt"].sort_values().tolist()
                end_times = between.loc[between["event_type"].str.contains("END BREAK", na=False), "activity_dt"].sort_values().tolist()
                break_mins = 0.0
                remaining_end_times = list(end_times)
                for bt in begin_times:
                    et = next((x for x in remaining_end_times if pd.notna(x) and pd.notna(bt) and x >= bt), None)
                    if et is not None:
                        break_mins += max((et - bt).total_seconds() / 60.0, 0.0)
                        remaining_end_times.remove(et)
                leg_elapsed = (cur_row["activity_dt"] - prev["activity_dt"]).total_seconds() / 60.0 if pd.notna(cur_row["activity_dt"]) and pd.notna(prev["activity_dt"]) else np.nan
                adjusted = leg_elapsed - break_mins if pd.notna(leg_elapsed) else np.nan
                rows.append({
                    "scan_date": cur_row.get("scan_date"),
                    "route": cur_row.get("route"),
                    "fedex_id": cur_row.get("fedex_id"),
                    "courier_name": cur_row.get("courier_name"),
                    "stop_order": cur_row.get("stop_order"),
                    "stop_type": cur_row.get("stop_type"),
                    "before_address": prev.get("address"),
                    "after_address": cur_row.get("address"),
                    "before_time": prev.get("activity_dt").strftime("%H:%M") if pd.notna(prev.get("activity_dt")) else "",
                    "after_time": cur_row.get("activity_dt").strftime("%H:%M") if pd.notna(cur_row.get("activity_dt")) else "",
                    "leg_elapsed_minutes": leg_elapsed,
                    "break_minutes_between": break_mins,
                    "adjusted_gap_minutes": adjusted,
                    "gap_from_to": f"{prev.get('address','')} -> {cur_row.get('address','')}",
                    "gap_time_window": f"{prev.get('activity_dt').strftime('%H:%M') if pd.notna(prev.get('activity_dt')) else ''} -> {cur_row.get('activity_dt').strftime('%H:%M') if pd.notna(cur_row.get('activity_dt')) else ''}",
                    "source_file": cur_row.get("source_file"),
                })
        return pd.DataFrame(rows)

    cur_legs = _customer_legs(gap_df)
    hist_legs = _customer_legs(gap_history if gap_history is not None and not gap_history.empty else gap_df)
    if cur_legs.empty or hist_legs.empty:
        return pd.DataFrame()

    route_stats = hist_legs.groupby("route", dropna=False)["adjusted_gap_minutes"].agg(route_gap_median="median", route_gap_mean="mean", route_gap_count="count").reset_index()
    p90 = hist_legs.groupby("route", dropna=False)["adjusted_gap_minutes"].quantile(0.90).reset_index(name="route_gap_p90")
    route_stats = route_stats.merge(p90, on="route", how="left")
    route_stats["route_gap_threshold"] = route_stats.apply(
        lambda r: max(
            float(r["route_gap_p90"]) if pd.notna(r["route_gap_p90"]) else 0.0,
            float(r["route_gap_median"]) * 2.0 if pd.notna(r["route_gap_median"]) else 0.0,
            float(floor_minutes),
        ),
        axis=1,
    )

    out = cur_legs.merge(route_stats, on="route", how="left")
    out["route_gap_threshold"] = out["route_gap_threshold"].fillna(float(floor_minutes))
    out = out[out["adjusted_gap_minutes"] >= out["route_gap_threshold"]].copy()
    if out.empty:
        return pd.DataFrame()
    out["severity_ratio"] = np.where(out["route_gap_threshold"] > 0, out["adjusted_gap_minutes"] / out["route_gap_threshold"], np.nan)
    out["severity_band"] = np.select([out["severity_ratio"] >= 2.0, out["severity_ratio"] >= 1.5], ["High", "Medium"], default="Watch")
    keep_cols = ["scan_date","route","courier_name","fedex_id","before_address","after_address","before_time","after_time","leg_elapsed_minutes","break_minutes_between","adjusted_gap_minutes","stop_type","route_gap_median","route_gap_p90","route_gap_threshold","severity_ratio","severity_band","gap_from_to","gap_time_window","source_file"]
    keep_cols = [c for c in keep_cols if c in out.columns]
    return out.sort_values([c for c in ["scan_date","route","severity_ratio","adjusted_gap_minutes"] if c in out.columns], ascending=[True,True,False,False])[keep_cols].reset_index(drop=True)

# Final override patches

def _is_route_section_start_row(row_text: str) -> bool:
    txt = clean_text(row_text) or ""
    return txt.startswith("LOC:") and ("DATE:" in txt) and ("ROUTE:" in txt)


def build_large_gap_exceptions(gap_df, gap_history=None, floor_minutes=None, exclude_first_stop_gap=True):
    if floor_minutes is None:
        floor_minutes = float(APP_CONFIG.get("large_gap_internal_floor_min", 10.0))
    return _build_large_gap_exceptions_impl(gap_df, gap_history, floor_minutes, exclude_first_stop_gap)


# -----------------------------
# Cut Run Optimizer
# -----------------------------
def _minute_of_day_from_ts(ts):
    ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts):
        return np.nan
    return ts.hour * 60 + ts.minute + ts.second / 60.0


def _mode_or_first(series):
    s = pd.Series(series).dropna()
    if s.empty:
        return None
    try:
        m = s.mode(dropna=True)
        if not m.empty:
            return m.iloc[0]
    except Exception:
        pass
    return s.iloc[0]


def get_anchor_catalog(anchor_refs=None, pickup_stops_df=None):
    rows = []
    if pickup_stops_df is not None and not pickup_stops_df.empty:
        p = pickup_stops_df.copy()
        if "work_area_no" in p.columns:
            for key, grp in p.groupby("work_area_no", dropna=True):
                key_norm = normalize_work_area_key(key)
                if key_norm is None:
                    continue
                rows.append({
                    "anchor_key": key_norm,
                    "anchor_display": format_work_area_display(key),
                    "wave_label": _mode_or_first(grp.get("wave_label", pd.Series(dtype=object))),
                    "anchor_source": "pickup_history",
                })
    if anchor_refs is not None and not anchor_refs.empty:
        a = anchor_refs.copy()
        if "work_area_key" in a.columns:
            for _, r in a.iterrows():
                key_norm = normalize_work_area_key(r.get("work_area_key"))
                if key_norm is None:
                    continue
                rows.append({
                    "anchor_key": key_norm,
                    "anchor_display": format_work_area_display(key_norm),
                    "wave_label": clean_text(r.get("wave")) if pd.notna(r.get("wave")) else None,
                    "anchor_source": "anchor_reference",
                })
    if not rows:
        return pd.DataFrame(columns=["anchor_key", "anchor_display", "wave_label", "anchor_source"])
    out = pd.DataFrame(rows)
    out = out.sort_values([c for c in ["anchor_key", "anchor_source"] if c in out.columns]).drop_duplicates(subset=["anchor_key"], keep="first")
    return out.reset_index(drop=True)


def build_pickup_anchor_history(pickup_stops_df, day_name=None):
    if pickup_stops_df is None or pickup_stops_df.empty:
        empty_daily = pd.DataFrame(columns=[
            "pickup_date", "weekday", "route", "anchor_key", "anchor_display", "wave_label",
            "pickup_stop_count", "package_count", "ready_min", "close_min", "pickup_min",
            "account_count", "city", "postal_fsa"
        ])
        empty_summary = pd.DataFrame(columns=[
            "route", "anchor_key", "anchor_display", "wave_label", "hist_days", "avg_pickup_stops",
            "avg_packages", "median_close_min", "median_ready_min", "receiver_overlap_share", "cities"
        ])
        return empty_daily, empty_summary

    p = pickup_stops_df.copy()
    if "pickup_date" in p.columns:
        p["pickup_date"] = pd.to_datetime(p["pickup_date"], errors="coerce")
        p["weekday"] = p["pickup_date"].dt.day_name()
    else:
        p["weekday"] = None
    if "wave_label" not in p.columns:
        if "work_area" in p.columns and "pickup_date" in p.columns:
            inferred = p.apply(lambda r: infer_wave_start(r.get("work_area"), r.get("pickup_date")), axis=1)
            p["wave_label"] = [x[1] for x in inferred]
        else:
            p["wave_label"] = None
    if day_name:
        p = p[p["weekday"].astype(str).str.lower() == str(day_name).lower()].copy()
    if p.empty:
        return build_pickup_anchor_history(pd.DataFrame(columns=p.columns), None)

    p["route"] = pd.to_numeric(p.get("route"), errors="coerce").astype("Int64")
    p["anchor_key"] = p.get("work_area_no").apply(normalize_work_area_key) if "work_area_no" in p.columns else None
    if "anchor_key" not in p.columns:
        p["anchor_key"] = None
    p["anchor_display"] = p["anchor_key"].apply(format_work_area_display)
    p["ready_min"] = p.get("ready_pickup_dt", pd.Series(pd.NaT, index=p.index)).apply(_minute_of_day_from_ts)
    p["close_min"] = p.get("close_pickup_dt", pd.Series(pd.NaT, index=p.index)).apply(_minute_of_day_from_ts)
    p["pickup_min"] = p.get("pickup_dt", pd.Series(pd.NaT, index=p.index)).apply(_minute_of_day_from_ts)
    p["package_count"] = pd.to_numeric(p.get("packages"), errors="coerce").fillna(0.0)
    p["city"] = p.get("city", pd.Series("", index=p.index)).fillna("").astype(str)
    p["postal_fsa"] = p.get("postal_code", pd.Series("", index=p.index)).fillna("").astype(str).str.replace(" ", "").str[:3]

    daily = (
        p.groupby(["pickup_date", "weekday", "route", "anchor_key", "anchor_display"], dropna=False)
        .agg(
            wave_label=("wave_label", _mode_or_first),
            pickup_stop_count=("pickup_key", "nunique") if "pickup_key" in p.columns else ("address", "count"),
            package_count=("package_count", "sum"),
            ready_min=("ready_min", "median"),
            close_min=("close_min", "median"),
            pickup_min=("pickup_min", "median"),
            account_count=("account_name", pd.Series.nunique) if "account_name" in p.columns else ("address", pd.Series.nunique),
            city=("city", _mode_or_first),
            postal_fsa=("postal_fsa", _mode_or_first),
        )
        .reset_index()
    )

    if daily.empty:
        return daily, pd.DataFrame()

    anchor_summary = (
        daily.groupby(["route", "anchor_key", "anchor_display"], dropna=False)
        .agg(
            wave_label=("wave_label", _mode_or_first),
            hist_days=("pickup_date", "nunique"),
            avg_pickup_stops=("pickup_stop_count", "mean"),
            avg_packages=("package_count", "mean"),
            median_close_min=("close_min", "median"),
            median_ready_min=("ready_min", "median"),
            cities=("city", lambda s: " | ".join(sorted(set([x for x in s.dropna().astype(str) if x]))[:5])),
            postal_fsa=("postal_fsa", _mode_or_first),
        )
        .reset_index()
    )

    anchor_totals = anchor_summary.groupby("anchor_key", dropna=False)["hist_days"].sum().reset_index(name="anchor_hist_days")
    anchor_summary = anchor_summary.merge(anchor_totals, on="anchor_key", how="left")
    anchor_summary["receiver_overlap_share"] = np.where(anchor_summary["anchor_hist_days"] > 0, anchor_summary["hist_days"] / anchor_summary["anchor_hist_days"], 0.0)
    return daily, anchor_summary


def build_route_baseline_profiles(gap_route_metrics_df, pickup_stops_df, day_name=None):
    metrics = gap_route_metrics_df.copy() if gap_route_metrics_df is not None else pd.DataFrame()
    if not metrics.empty:
        metrics["scan_date"] = pd.to_datetime(metrics.get("scan_date"), errors="coerce")
        metrics["weekday"] = metrics["scan_date"].dt.day_name()
        if day_name:
            metrics = metrics[metrics["weekday"].astype(str).str.lower() == str(day_name).lower()].copy()
        metrics["route"] = pd.to_numeric(metrics.get("route"), errors="coerce").astype("Int64")
        metrics["total_stops_actual"] = pd.to_numeric(metrics.get("total_stops_actual"), errors="coerce")
        metrics["actual_hr_or_min"] = pd.to_numeric(metrics.get("actual_hr_or_min"), errors="coerce")
        metrics["actual_hr_oa_min"] = pd.to_numeric(metrics.get("actual_hr_oa_min"), errors="coerce")
        metrics["actual_sth_or"] = pd.to_numeric(metrics.get("actual_sth_or"), errors="coerce")
        metrics["actual_sth_oa"] = pd.to_numeric(metrics.get("actual_sth_oa"), errors="coerce")
        metrics["packages_proxy"] = pd.to_numeric(metrics.get("delivery_stop_count"), errors="coerce")
        base_metrics = (
            metrics.groupby("route", dropna=False)
            .agg(
                hist_days=("scan_date", "nunique"),
                avg_total_stops=("total_stops_actual", "mean"),
                avg_hr_or_min=("actual_hr_or_min", "mean"),
                avg_hr_oa_min=("actual_hr_oa_min", "mean"),
                avg_sth_or=("actual_sth_or", "mean"),
                avg_sth_oa=("actual_sth_oa", "mean"),
                avg_gap_customer=("gap_sum_minutes_customer", "mean") if "gap_sum_minutes_customer" in metrics.columns else ("actual_hr_or_min", lambda s: np.nan),
                median_return_min=("return_to_bldg_dt", lambda s: np.nanmedian([_minute_of_day_from_ts(x) for x in s]) if len(s) else np.nan),
                median_on_area_end_min=("on_area_end_dt", lambda s: np.nanmedian([_minute_of_day_from_ts(x) for x in s]) if len(s) else np.nan),
            )
            .reset_index()
        )
    else:
        base_metrics = pd.DataFrame(columns=["route", "hist_days", "avg_total_stops", "avg_hr_or_min", "avg_hr_oa_min", "avg_sth_or", "avg_sth_oa", "avg_gap_customer", "median_return_min", "median_on_area_end_min"])

    daily_anchor, anchor_summary = build_pickup_anchor_history(pickup_stops_df, day_name=day_name)
    if not daily_anchor.empty:
        pickup_route = (
            daily_anchor.groupby("route", dropna=False)
            .agg(
                avg_pickup_stops=("pickup_stop_count", "mean"),
                avg_pickup_packages=("package_count", "mean"),
                avg_anchor_count=("anchor_key", pd.Series.nunique),
                dominant_wave=("wave_label", _mode_or_first),
                median_close_min=("close_min", "median"),
            )
            .reset_index()
        )
    else:
        pickup_route = pd.DataFrame(columns=["route", "avg_pickup_stops", "avg_pickup_packages", "avg_anchor_count", "dominant_wave", "median_close_min"])

    if base_metrics.empty and pickup_route.empty:
        return pd.DataFrame(), daily_anchor, anchor_summary

    profiles = pd.merge(base_metrics, pickup_route, on="route", how="outer")
    profiles["route"] = pd.to_numeric(profiles.get("route"), errors="coerce").astype("Int64")
    profiles = filter_to_our_workgroup(profiles, "route") if not profiles.empty else profiles
    profiles["avg_minutes_per_stop"] = np.where(
        profiles.get("avg_total_stops", pd.Series(dtype=float)).fillna(0) > 0,
        profiles.get("avg_hr_or_min", pd.Series(dtype=float)) / profiles.get("avg_total_stops", pd.Series(dtype=float)),
        np.nan,
    )
    profiles["avg_minutes_per_pickup_stop"] = np.where(
        profiles.get("avg_pickup_stops", pd.Series(dtype=float)).fillna(0) > 0,
        profiles.get("avg_hr_or_min", pd.Series(dtype=float)) / profiles.get("avg_pickup_stops", pd.Series(dtype=float)),
        profiles.get("avg_minutes_per_stop", pd.Series(dtype=float)),
    )
    return profiles.reset_index(drop=True), daily_anchor, anchor_summary


def get_strategy_weights(preset="Balanced", overrides=None):
    presets = {
        "Balanced": {
            "pickup_safety": 0.30,
            "added_time": 0.23,
            "proximity": 0.18,
            "stop_density": 0.12,
            "package_burden": 0.10,
            "fragmentation": 0.07,
        },
        "Pickup Safe": {
            "pickup_safety": 0.40,
            "added_time": 0.18,
            "proximity": 0.16,
            "stop_density": 0.10,
            "package_burden": 0.10,
            "fragmentation": 0.06,
        },
        "Shortest Added Drive": {
            "pickup_safety": 0.24,
            "added_time": 0.34,
            "proximity": 0.22,
            "stop_density": 0.08,
            "package_burden": 0.07,
            "fragmentation": 0.05,
        },
        "Highest Density": {
            "pickup_safety": 0.24,
            "added_time": 0.18,
            "proximity": 0.15,
            "stop_density": 0.24,
            "package_burden": 0.11,
            "fragmentation": 0.08,
        },
    }
    weights = presets.get(preset, presets["Balanced"]).copy()
    if overrides:
        for k, v in overrides.items():
            if k in weights:
                weights[k] = float(v)
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        return presets["Balanced"]
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def _normalize_scores(arr, higher_is_better=True):
    s = pd.Series(arr, dtype="float64")
    if s.empty:
        return s
    valid = s.dropna()
    if valid.empty or valid.nunique() <= 1:
        out = pd.Series(0.5, index=s.index, dtype="float64")
        out[s.isna()] = np.nan
        return out
    lo, hi = valid.min(), valid.max()
    scaled = (s - lo) / (hi - lo)
    if not higher_is_better:
        scaled = 1 - scaled
    return scaled.clip(0, 1)


def _close_time_risk(anchor_close_min, receiver_wave, overlap_share, receiver_end_min):
    risk = 0.0
    disqualify = False
    if pd.notna(anchor_close_min):
        if str(receiver_wave).upper() == "W2":
            if anchor_close_min < 13 * 60:
                risk += 1.0
                disqualify = True
            elif anchor_close_min < 14 * 60 and (pd.isna(overlap_share) or overlap_share < 0.20):
                risk += 0.8
        if pd.notna(receiver_end_min) and receiver_end_min > anchor_close_min and (pd.isna(overlap_share) or overlap_share < 0.20):
            risk += 0.7
            if receiver_end_min - anchor_close_min > 60:
                disqualify = True
    return min(risk, 1.5), disqualify


def simulate_cut_route_plan(
    cut_route,
    route_profiles,
    anchor_summary,
    day_name=None,
    candidate_routes=None,
    include_anchors=None,
    prioritize_anchors=None,
    exclude_anchors=None,
    weights=None,
    package_soft_warning=400.0,
):
    cut_route = int(cut_route)
    include_set = {normalize_work_area_key(x) for x in (include_anchors or []) if normalize_work_area_key(x) is not None}
    prioritize_set = {normalize_work_area_key(x) for x in (prioritize_anchors or []) if normalize_work_area_key(x) is not None}
    exclude_set = {normalize_work_area_key(x) for x in (exclude_anchors or []) if normalize_work_area_key(x) is not None}
    if weights is None:
        weights = get_strategy_weights()

    rp = route_profiles.copy() if route_profiles is not None else pd.DataFrame()
    if rp.empty or anchor_summary is None or anchor_summary.empty:
        return {
            "cut_route": cut_route,
            "feasible": False,
            "overall_score": np.nan,
            "message": "Not enough route profile or pickup anchor history to simulate a cut plan.",
            "assignment_plan": pd.DataFrame(),
            "receiver_summary": pd.DataFrame(),
            "cut_route_summary": pd.DataFrame(),
            "scorecard": pd.DataFrame(),
        }

    rp["route"] = pd.to_numeric(rp.get("route"), errors="coerce").astype("Int64")
    scope_routes = sorted({int(r) for r in rp["route"].dropna().astype(int).tolist() if int(r) != cut_route})
    if candidate_routes:
        scope_routes = [int(r) for r in scope_routes if int(r) in {int(x) for x in candidate_routes} and int(r) != cut_route]
    if not scope_routes:
        return {
            "cut_route": cut_route,
            "feasible": False,
            "overall_score": np.nan,
            "message": "No receiving routes available inside the selected scope.",
            "assignment_plan": pd.DataFrame(),
            "receiver_summary": pd.DataFrame(),
            "cut_route_summary": pd.DataFrame(),
            "scorecard": pd.DataFrame(),
        }

    cut_profile = rp[rp["route"] == cut_route].copy()
    cut_prof = cut_profile.iloc[0].to_dict() if not cut_profile.empty else {"route": cut_route}

    anchor_df = anchor_summary.copy()
    anchor_df["route"] = pd.to_numeric(anchor_df.get("route"), errors="coerce").astype("Int64")
    anchor_df = anchor_df[anchor_df["route"] == cut_route].copy()
    if include_set:
        anchor_df = anchor_df[anchor_df["anchor_key"].isin(include_set)].copy()
    if exclude_set:
        anchor_df = anchor_df[~anchor_df["anchor_key"].isin(exclude_set)].copy()
    if anchor_df.empty:
        return {
            "cut_route": cut_route,
            "feasible": False,
            "overall_score": np.nan,
            "message": "No matching anchor slices found for the selected cut route and filters.",
            "assignment_plan": pd.DataFrame(),
            "receiver_summary": pd.DataFrame(),
            "cut_route_summary": pd.DataFrame([cut_prof]),
            "scorecard": pd.DataFrame(),
        }

    route_lookup = rp.set_index("route", drop=False).to_dict("index")
    assignments = []
    receiver_loads = {r: {"added_stops": 0.0, "added_packages": 0.0, "added_minutes": 0.0, "anchor_count": 0} for r in scope_routes}

    anchor_df["_is_prioritized_anchor"] = anchor_df["anchor_key"].isin(prioritize_set)
    anchor_df = anchor_df.sort_values(["_is_prioritized_anchor", "avg_pickup_stops"], ascending=[False, False]).reset_index(drop=True)

    for _, a in anchor_df.iterrows():
        cand_rows = []
        for rr in scope_routes:
            rec = route_lookup.get(rr, {})
            hist_match = anchor_summary[(anchor_summary["anchor_key"] == a.get("anchor_key")) & (anchor_summary["route"] == rr)]
            overlap_share = float(hist_match["receiver_overlap_share"].iloc[0]) if not hist_match.empty and pd.notna(hist_match["receiver_overlap_share"].iloc[0]) else 0.0
            receiver_wave = rec.get("dominant_wave") or (hist_match["wave_label"].iloc[0] if not hist_match.empty else None)
            route_distance = abs(int(rr) - int(cut_route))
            proximity_raw = (0.65 * overlap_share) + (0.20 if str(receiver_wave) == str(a.get("wave_label")) and pd.notna(a.get("wave_label")) else 0.0) + (0.15 * (1.0 / (1.0 + route_distance)))
            mins_per_stop = rec.get("avg_minutes_per_pickup_stop")
            if pd.isna(mins_per_stop) or mins_per_stop <= 0:
                mins_per_stop = rec.get("avg_minutes_per_stop")
            if pd.isna(mins_per_stop) or mins_per_stop <= 0:
                mins_per_stop = cut_prof.get("avg_minutes_per_pickup_stop")
            if pd.isna(mins_per_stop) or mins_per_stop <= 0:
                mins_per_stop = 6.0
            added_stops = float(a.get("avg_pickup_stops") or 0.0)
            added_packages = float(a.get("avg_packages") or 0.0)
            added_minutes = added_stops * float(mins_per_stop)

            projected_stops = float(rec.get("avg_total_stops") or 0.0) + receiver_loads[rr]["added_stops"] + added_stops
            projected_minutes = float(rec.get("avg_hr_or_min") or 0.0) + receiver_loads[rr]["added_minutes"] + added_minutes
            projected_packages = float(rec.get("avg_pickup_packages") or 0.0) + receiver_loads[rr]["added_packages"] + added_packages
            projected_sth = projected_stops / (projected_minutes / 60.0) if projected_minutes > 0 else np.nan
            density_gain = projected_sth - float(rec.get("avg_sth_or") or 0.0) if pd.notna(projected_sth) else 0.0

            risk_value, disqualify = _close_time_risk(a.get("median_close_min"), receiver_wave, overlap_share, rec.get("median_on_area_end_min"))
            package_warning = projected_packages > float(package_soft_warning)

            cand_rows.append({
                "receiver_route": rr,
                "receiver_wave": receiver_wave,
                "overlap_share": overlap_share,
                "proximity_raw": proximity_raw,
                "added_stops": added_stops,
                "added_packages": added_packages,
                "added_minutes": added_minutes,
                "projected_stops": projected_stops,
                "projected_minutes": projected_minutes,
                "projected_packages": projected_packages,
                "projected_sth_or": projected_sth,
                "density_gain": density_gain,
                "pickup_risk_raw": risk_value,
                "disqualify": disqualify,
                "package_warning": package_warning,
            })

        cand = pd.DataFrame(cand_rows)
        cand["pickup_safety_score"] = _normalize_scores(1 - cand["pickup_risk_raw"], higher_is_better=True)
        cand["added_time_score"] = _normalize_scores(cand["added_minutes"], higher_is_better=False)
        cand["proximity_score"] = _normalize_scores(cand["proximity_raw"], higher_is_better=True)
        cand["stop_density_score"] = _normalize_scores(cand["density_gain"], higher_is_better=True)
        cand["package_burden_score"] = _normalize_scores(cand["projected_packages"], higher_is_better=False)
        frag_value = 1.0 / (1.0 + pd.Series(range(len(cand)), index=cand.index, dtype="float64"))
        cand["fragmentation_score"] = frag_value
        cand["weighted_score"] = (
            weights["pickup_safety"] * cand["pickup_safety_score"].fillna(0) +
            weights["added_time"] * cand["added_time_score"].fillna(0) +
            weights["proximity"] * cand["proximity_score"].fillna(0) +
            weights["stop_density"] * cand["stop_density_score"].fillna(0) +
            weights["package_burden"] * cand["package_burden_score"].fillna(0) +
            weights["fragmentation"] * cand["fragmentation_score"].fillna(0)
        )
        cand = cand.sort_values(["disqualify", "weighted_score", "added_minutes"], ascending=[True, False, True]).reset_index(drop=True)
        best = cand.iloc[0].to_dict()
        receiver_loads[int(best["receiver_route"])] ["added_stops"] += float(best["added_stops"])
        receiver_loads[int(best["receiver_route"])] ["added_packages"] += float(best["added_packages"])
        receiver_loads[int(best["receiver_route"])] ["added_minutes"] += float(best["added_minutes"])
        receiver_loads[int(best["receiver_route"])] ["anchor_count"] += 1
        assignments.append({
            "cut_route": cut_route,
            "anchor_key": a.get("anchor_key"),
            "anchor_display": a.get("anchor_display"),
            "anchor_wave": a.get("wave_label"),
            "anchor_hist_days": a.get("hist_days"),
            "avg_pickup_stops": a.get("avg_pickup_stops"),
            "avg_packages": a.get("avg_packages"),
            "median_close_min": a.get("median_close_min"),
            "prioritized_anchor": a.get("anchor_key") in prioritize_set,
            **best,
        })

    assignment_df = pd.DataFrame(assignments)
    if assignment_df.empty:
        return {
            "cut_route": cut_route,
            "feasible": False,
            "overall_score": np.nan,
            "message": "No assignment candidates could be built.",
            "assignment_plan": pd.DataFrame(),
            "receiver_summary": pd.DataFrame(),
            "cut_route_summary": pd.DataFrame([cut_prof]),
            "scorecard": pd.DataFrame(),
        }

    feasible = not assignment_df["disqualify"].fillna(False).any()
    receiver_rows = []
    for rr, load in receiver_loads.items():
        if load["anchor_count"] <= 0:
            continue
        rec = route_lookup.get(rr, {})
        base_stops = float(rec.get("avg_total_stops") or 0.0)
        base_minutes = float(rec.get("avg_hr_or_min") or 0.0)
        base_packages = float(rec.get("avg_pickup_packages") or 0.0)
        new_stops = base_stops + load["added_stops"]
        new_minutes = base_minutes + load["added_minutes"]
        new_packages = base_packages + load["added_packages"]
        new_sth = new_stops / (new_minutes / 60.0) if new_minutes > 0 else np.nan
        receiver_rows.append({
            "receiver_route": rr,
            "receiver_wave": rec.get("dominant_wave"),
            "base_avg_stops": base_stops,
            "base_avg_hr_or_min": base_minutes,
            "base_avg_packages": base_packages,
            "added_stops": load["added_stops"],
            "added_minutes": load["added_minutes"],
            "added_packages": load["added_packages"],
            "new_avg_stops": new_stops,
            "new_avg_hr_or_min": new_minutes,
            "new_avg_sth_or": new_sth,
            "new_avg_packages": new_packages,
            "package_soft_warning": new_packages > float(package_soft_warning),
            "assigned_anchor_count": load["anchor_count"],
        })
    receiver_summary = pd.DataFrame(receiver_rows).sort_values("added_minutes", ascending=False).reset_index(drop=True)

    scorecard = pd.DataFrame([{
        "cut_route": cut_route,
        "feasible": feasible,
        "overall_score": float(assignment_df["weighted_score"].mean() * 100.0),
        "anchors_reassigned": len(assignment_df),
        "receiving_routes_used": assignment_df["receiver_route"].nunique(),
        "total_added_stops": float(assignment_df["added_stops"].sum()),
        "total_added_packages": float(assignment_df["added_packages"].sum()),
        "total_added_minutes": float(assignment_df["added_minutes"].sum()),
        "pickup_risk_flags": int(assignment_df["disqualify"].fillna(False).sum()),
        "package_soft_warnings": int(receiver_summary["package_soft_warning"].fillna(False).sum()) if not receiver_summary.empty else 0,
    }])

    cut_route_summary = pd.DataFrame([{
        "cut_route": cut_route,
        "avg_total_stops": cut_prof.get("avg_total_stops"),
        "avg_hr_or_min": cut_prof.get("avg_hr_or_min"),
        "avg_hr_oa_min": cut_prof.get("avg_hr_oa_min"),
        "avg_sth_or": cut_prof.get("avg_sth_or"),
        "avg_sth_oa": cut_prof.get("avg_sth_oa"),
        "avg_pickup_stops": cut_prof.get("avg_pickup_stops"),
        "avg_pickup_packages": cut_prof.get("avg_pickup_packages"),
        "dominant_wave": cut_prof.get("dominant_wave"),
        "hist_days": cut_prof.get("hist_days"),
    }])

    return {
        "cut_route": cut_route,
        "feasible": feasible,
        "overall_score": float(scorecard["overall_score"].iloc[0]),
        "message": "Plan built successfully." if feasible else "Plan built, but at least one assigned anchor raised a pickup-safety disqualifier.",
        "assignment_plan": assignment_df.sort_values(["prioritized_anchor", "weighted_score", "added_minutes"], ascending=[False, False, True]).reset_index(drop=True),
        "receiver_summary": receiver_summary,
        "cut_route_summary": cut_route_summary,
        "scorecard": scorecard,
    }


def build_cut_run_optimizer(
    gap_route_metrics_df,
    pickup_stops_df,
    anchor_refs=None,
    day_name=None,
    mode="test_selected_route",
    cut_route=None,
    candidate_routes=None,
    include_anchors=None,
    prioritize_anchors=None,
    exclude_anchors=None,
    strategy_preset="Balanced",
    strategy_overrides=None,
    package_soft_warning=400.0,
):
    route_profiles, anchor_daily, anchor_summary = build_route_baseline_profiles(gap_route_metrics_df, pickup_stops_df, day_name=day_name)
    anchor_catalog = get_anchor_catalog(anchor_refs=anchor_refs, pickup_stops_df=pickup_stops_df)
    weights = get_strategy_weights(strategy_preset, strategy_overrides)

    route_profiles = filter_to_our_workgroup(route_profiles, "route") if not route_profiles.empty else route_profiles
    if candidate_routes:
        candidate_routes = sorted({int(r) for r in candidate_routes if is_relevant_workgroup_route(r)})
    else:
        candidate_routes = sorted({int(r) for r in route_profiles.get("route", pd.Series(dtype="Int64")).dropna().astype(int).tolist()})

    if cut_route is not None:
        plans = [simulate_cut_route_plan(
            cut_route=int(cut_route),
            route_profiles=route_profiles,
            anchor_summary=anchor_summary,
            day_name=day_name,
            candidate_routes=candidate_routes,
            include_anchors=include_anchors,
            prioritize_anchors=prioritize_anchors,
            exclude_anchors=exclude_anchors,
            weights=weights,
            package_soft_warning=package_soft_warning,
        )]
    else:
        plans = []

    suggestions = []
    if mode == "suggest_best_route" or cut_route is None:
        for r in candidate_routes:
            plan = simulate_cut_route_plan(
                cut_route=int(r),
                route_profiles=route_profiles,
                anchor_summary=anchor_summary,
                day_name=day_name,
                candidate_routes=[x for x in candidate_routes if int(x) != int(r)],
                include_anchors=include_anchors,
                prioritize_anchors=prioritize_anchors,
                exclude_anchors=exclude_anchors,
                weights=weights,
                package_soft_warning=package_soft_warning,
            )
            suggestions.append({
                "cut_route": int(r),
                "feasible": plan.get("feasible"),
                "overall_score": plan.get("overall_score"),
                "anchors_reassigned": plan.get("scorecard", pd.DataFrame()).get("anchors_reassigned", pd.Series([np.nan])).iloc[0] if isinstance(plan.get("scorecard"), pd.DataFrame) and not plan.get("scorecard").empty else np.nan,
                "receiving_routes_used": plan.get("scorecard", pd.DataFrame()).get("receiving_routes_used", pd.Series([np.nan])).iloc[0] if isinstance(plan.get("scorecard"), pd.DataFrame) and not plan.get("scorecard").empty else np.nan,
                "total_added_minutes": plan.get("scorecard", pd.DataFrame()).get("total_added_minutes", pd.Series([np.nan])).iloc[0] if isinstance(plan.get("scorecard"), pd.DataFrame) and not plan.get("scorecard").empty else np.nan,
                "pickup_risk_flags": plan.get("scorecard", pd.DataFrame()).get("pickup_risk_flags", pd.Series([np.nan])).iloc[0] if isinstance(plan.get("scorecard"), pd.DataFrame) and not plan.get("scorecard").empty else np.nan,
                "message": plan.get("message"),
            })
            if cut_route is not None and int(r) == int(cut_route):
                plans = [plan]

    suggestions_df = pd.DataFrame(suggestions)
    if not suggestions_df.empty:
        suggestions_df = suggestions_df.sort_values(["feasible", "overall_score", "total_added_minutes"], ascending=[False, False, True]).reset_index(drop=True)

    selected_plan = plans[0] if plans else (simulate_cut_route_plan(
        cut_route=int(suggestions_df.iloc[0]["cut_route"]),
        route_profiles=route_profiles,
        anchor_summary=anchor_summary,
        day_name=day_name,
        candidate_routes=[x for x in candidate_routes if int(x) != int(suggestions_df.iloc[0]["cut_route"])],
        include_anchors=include_anchors,
        prioritize_anchors=prioritize_anchors,
        exclude_anchors=exclude_anchors,
        weights=weights,
        package_soft_warning=package_soft_warning,
    ) if not suggestions_df.empty else {
        "cut_route": None,
        "feasible": False,
        "overall_score": np.nan,
        "message": "No routes available for cut analysis.",
        "assignment_plan": pd.DataFrame(),
        "receiver_summary": pd.DataFrame(),
        "cut_route_summary": pd.DataFrame(),
        "scorecard": pd.DataFrame(),
    })

    return {
        "route_profiles": route_profiles,
        "anchor_catalog": anchor_catalog,
        "anchor_daily": anchor_daily,
        "anchor_summary": anchor_summary,
        "suggestions": suggestions_df,
        "selected_plan": selected_plan,
        "weights": pd.DataFrame([weights]),
    }
