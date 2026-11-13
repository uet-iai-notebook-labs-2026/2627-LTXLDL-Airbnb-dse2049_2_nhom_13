"""QA cho ba bảng subset, giữ giao diện của khung qa.py được cung cấp.

Đầu vào trong paths.processed: listings_subset.csv, calendar_subset.csv.gz,
reviews_subset.csv. Đầu ra: listings_clean.csv, reviews_clean.csv,
calendar_listing_agg.csv, calendar_month_agg.csv; hai báo cáo nằm ở paths.reports.

Quy tắc giữ cờ trên listings/reviews, không xoá dữ liệu gốc. price_ars chỉ dùng
giá hợp lệ ở snapshot cuối theo hợp đồng; price_num giữ giá đã parse ở mọi mốc.
occ_H là tỷ lệ ngày KHÔNG KHẢ DỤNG quan sát được, không phải tỷ lệ đặt phòng.
qa_report đo tác động loại riêng từng cờ, không phải số dòng loại tích luỹ.

Chạy riêng từ gốc repo: python -m src.qa --config configs/Buenos_Aires.yaml
Hoặc gọi qa.run(cfg) từ pipeline. Chưa có subset thì báo rõ tệp cần chuẩn bị.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .common import get_logger, last_snapshot, path

log = get_logger("qa")
TABLE_LISTINGS, TABLE_CALENDAR, TABLE_REVIEWS = "listings", "calendar", "reviews"
IN_LISTINGS, IN_CALENDAR, IN_REVIEWS = "listings_subset.csv", "calendar_subset.csv.gz", "reviews_subset.csv"
OUT_LISTINGS, OUT_REVIEWS = "listings_clean.csv", "reviews_clean.csv"
OUT_CAL_LISTING, OUT_CAL_MONTH = "calendar_listing_agg.csv", "calendar_month_agg.csv"
OUT_QA_REPORT, OUT_PROFILE = "qa_report.csv", "data_profile.csv"
RULE_REPORT_COLUMNS = ["rule_id", "table", "name", "condition", "reason", "action",
                       "n_rows_before", "n_flagged", "n_rows_after", "pct_flagged",
                       "metric", "metric_before", "metric_after"]
PROFILE_COLUMNS = ["table", "column", "dtype", "n_null", "pct_null", "n_unique"]
FLAG_PREFIX = "flag_"
L_TYPE_PREFIX = "L_TYPE_"
L_PRICE_MISSING, L_PRICE_RANGE, L_PRICE_MAD = "L_PRICE_MISSING", "L_PRICE_RANGE", "L_PRICE_MAD"
L_DUP, L_PANEL, L_COORD, L_MINNIGHTS = "L_DUP", "L_PANEL", "L_COORD", "L_MINNIGHTS"
L_ROOMTYPE, L_AREA_MISSING, L_RARE_AREA = "L_ROOMTYPE", "L_AREA_MISSING", "L_RARE_AREA"
L_REVIEW_INCONSIST = "L_REVIEW_INCONSIST"
C_DATE, C_DUP, C_AVAIL, C_WINDOW, C_MINNIGHTS, C_ORPHAN = (
    "C_DATE", "C_DUP", "C_AVAIL", "C_WINDOW", "C_MINNIGHTS", "C_ORPHAN")
R_MISSING, R_DUP, R_REPEAT, R_DATE, R_ORPHAN = "R_MISSING", "R_DUP", "R_REPEAT", "R_DATE", "R_ORPHAN"
R_AUTO, R_SHORT, R_LONG, R_HTML = "R_AUTO", "R_SHORT", "R_LONG", "R_HTML"
CALENDAR_VALID_AVAILABLE = {"t", "f"}
CALENDAR_WINDOW_BEFORE_DAYS, CALENDAR_WINDOW_AFTER_DAYS = 1, 366
OCCUPANCY_HORIZONS = (30, 90, 365)
MIN_REVIEW_CHARS = 3
HTML_BR_REGEX = r"<br\s*/?>"
QA_CFG_KEYS = ["price_min_ars", "price_max_ars", "mad_z_threshold", "min_nights_cap", "dedupe_keys",
               "bbox", "allowed_room_types", "min_category_count", "numeric_cols", "max_review_chars", "auto_text_regex"]
DEFAULT_QA = {
    "price_min_ars": 1, "price_max_ars": 5_000_000, "mad_z_threshold": 3.5,
    "min_nights_cap": 365, "dedupe_keys": ["id", "snapshot"],
    "bbox": {"lat_min": -34.75, "lat_max": -34.50, "lon_min": -58.55, "lon_max": -58.30},
    "allowed_room_types": ["Entire home/apt", "Private room", "Shared room", "Hotel room"],
    "min_category_count": 5,
    "numeric_cols": ["price", "latitude", "longitude", "minimum_nights", "number_of_reviews",
                     "review_scores_rating", "calculated_host_listings_count"],
    "max_review_chars": 10_000,
    "auto_text_regex": r"(?:the host cancel(?:ed|led) this reservation|this is an automated posting)",
}
CAL_LISTING_COLUMNS = ["snapshot", "listing_id", "n_days", "unavail_days", "occ_30", "occ_90", "occ_365", "unavail_rate"]
CAL_MONTH_COLUMNS = ["snapshot", "analysis_area", "month", "n", "n_unavail", "unavail_rate"]


def _qa(cfg: dict) -> dict:
    return {**DEFAULT_QA, **(cfg.get("qa") or {})}


def _bool(mask: pd.Series, index: pd.Index) -> pd.Series:
    return mask.reindex(index).fillna(False).astype(bool)


def _text(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna("").str.strip()


def _dates(series: pd.Series) -> pd.Series:
    # Các file Inside Airbnb dùng ngày ISO, không đoán DD/MM với MM/DD.
    return pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")


def _snapshots(series: pd.Series) -> pd.Series:
    return _dates(series).dt.strftime("%Y-%m-%d")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if f"{col}_num" in df:
        return df[f"{col}_num"]
    if col not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return parse_price(df[col]) if col == "price" else pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{name} thiếu cột bắt buộc: {', '.join(missing)}")


def _read_bool(series: pd.Series) -> pd.Series:
    return _text(series).str.lower().isin({"true", "t", "1", "1.0", "x", "yes", "y"})


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    table: str
    name: str
    condition: str
    reason: str
    action: str


class Recorder:
    """Tác động từng quy tắc độc lập; pct_flagged dùng thang 0–100."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, spec: RuleSpec, df: pd.DataFrame, mask: pd.Series,
            metric_name: str, metric_fn: Callable) -> pd.Series:
        mask = _bool(mask, df.index)
        n, flagged = len(df), int(mask.sum())
        self.rows.append({**vars(spec), "n_rows_before": n, "n_flagged": flagged,
                          "n_rows_after": n - flagged, "pct_flagged": flagged / n * 100 if n else 0.0,
                          "metric": metric_name, "metric_before": metric_fn(df),
                          "metric_after": metric_fn(df.loc[~mask])})
        return mask

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=RULE_REPORT_COLUMNS)


def parse_price(s: pd.Series) -> pd.Series:
    """Parse dạng thực tế '$86,206.29'; từ chối dấu phân cách sai và số vô hạn."""
    text = _text(s)
    pattern = r"[+-]?\s*\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    valid = text.str.fullmatch(pattern)
    cleaned = text.where(valid).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    result, errors = df.copy(), {}
    for col in dict.fromkeys(cols):
        if col not in result:
            continue
        numeric = (parse_price(result[col]) if col == "price" else
                   pd.to_numeric(result[col], errors="coerce").replace([np.inf, -np.inf], np.nan))
        result[f"{col}_num"] = numeric
        errors[col] = _text(result[col]).ne("") & numeric.isna()
    return result, errors


def robust_zscore(values: pd.Series, groups: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = values.groupby(groups, dropna=False).transform("median")
    deviation = (values - median).abs()
    mad = deviation.groupby(groups, dropna=False).transform("median")
    return 0.6745 * (values - median) / mad.where(mad.ne(0))


def profile(df: pd.DataFrame, name: str) -> pd.DataFrame:
    n = len(df)
    return pd.DataFrame([{"table": name, "column": col, "dtype": str(df[col].dtype),
                          "n_null": int(df[col].isna().sum()),
                          "pct_null": float(df[col].isna().mean() * 100) if n else 0.0,
                          "n_unique": int(df[col].nunique(dropna=True))}
                         for col in df], columns=PROFILE_COLUMNS)


def metric_median_price(df: pd.DataFrame, cfg: dict) -> float:
    values = _num(df, "price").loc[_snapshots(df["snapshot"]).eq(last_snapshot(cfg))]
    return float(values.median()) if values.notna().any() else np.nan


def metric_unavailable_rate(df: pd.DataFrame) -> float:
    return float(_text(df["available"]).str.lower().eq("f").mean()) if len(df) else np.nan


def metric_row_count(df: pd.DataFrame) -> float:
    return float(len(df))


def load_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folder = path(cfg, "processed")
    files = [folder / name for name in (IN_LISTINGS, IN_CALENDAR, IN_REVIEWS)]
    missing = [str(file) for file in files if not file.is_file()]
    if missing:
        raise FileNotFoundError("Cần tạo các bảng subset trước khi chạy QA:\n" + "\n".join(missing))
    frames = []
    for file in files:
        # Đọc văn bản trước để không làm tròn ID > 2**53, kể cả có ô rỗng.
        df = pd.read_csv(file, dtype="string", keep_default_na=False, na_values=[""])
        for col in ("id", "listing_id", "reviewer_id", "host_id"):
            if col in df:
                try:
                    df[col] = df[col].astype("Int64")
                except (ValueError, TypeError, OverflowError) as exc:
                    raise ValueError(f"{file.name}: {col} phải là số nguyên int64 chính xác") from exc
        _require(df, ["snapshot"], file.name)
        snapshots = _snapshots(df["snapshot"])
        if snapshots.isna().any() or not snapshots.isin([str(s) for s in cfg["snapshots"]]).all():
            raise ValueError(f"{file.name}: snapshot không hợp lệ hoặc không thuộc cấu hình")
        df["snapshot"] = snapshots
        frames.append(df)
    listings, calendar, reviews = frames
    _require(listings, ["id", "snapshot", "analysis_area", "price", "room_type", "latitude", "longitude",
                        "neighbourhood_cleansed", "calculated_host_listings_count"], IN_LISTINGS)
    _require(calendar, ["listing_id", "snapshot", "date", "available", "minimum_nights"], IN_CALENDAR)
    _require(reviews, ["id", "listing_id", "snapshot", "date", "comments", "llm_sample", "hand_label"], IN_REVIEWS)
    for df, keys in ((listings, ["id"]), (calendar, ["listing_id"]), (reviews, ["id", "listing_id"])):
        if df[keys].isna().any().any():
            raise ValueError(f"Khoá {keys} không được thiếu")
    reviews["id"] = reviews["id"].astype("int64")
    for col in ("llm_sample", "hand_label"):
        reviews[col] = _read_bool(reviews[col])
    return listings, calendar, reviews


def save_outputs(cfg: dict, listings_clean: pd.DataFrame, reviews_clean: pd.DataFrame,
                 cal_listing_agg: pd.DataFrame, cal_month_agg: pd.DataFrame,
                 rec: Recorder, profiles: pd.DataFrame) -> None:
    processed, reports = path(cfg, "processed"), path(cfg, "reports")
    processed.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    for folder, name, frame in (
        (processed, OUT_LISTINGS, listings_clean), (processed, OUT_REVIEWS, reviews_clean),
        (processed, OUT_CAL_LISTING, cal_listing_agg), (processed, OUT_CAL_MONTH, cal_month_agg),
        (reports, OUT_QA_REPORT, rec.frame()), (reports, OUT_PROFILE, profiles),
    ):
        frame.to_csv(folder / name, index=False, encoding="utf-8", lineterminator="\n")


def rule_type_error(df: pd.DataFrame, col: str, bad_mask: pd.Series) -> pd.Series:
    return _bool(bad_mask, df.index)


def rule_price_missing(df: pd.DataFrame, cfg: dict) -> pd.Series:
    return _snapshots(df["snapshot"]).eq(last_snapshot(cfg)) & _num(df, "price").isna()


def rule_price_range(df: pd.DataFrame, cfg: dict) -> pd.Series:
    settings, values = _qa(cfg), _num(df, "price")
    return values.notna() & (values.le(0) | values.lt(settings["price_min_ars"]) | values.gt(settings["price_max_ars"]))


def rule_price_outlier_mad(df: pd.DataFrame, cfg: dict, already_out: pd.Series) -> pd.Series:
    price = _num(df, "price")
    eligible = _snapshots(df["snapshot"]).eq(last_snapshot(cfg)) & price.gt(0) & ~_bool(already_out, df.index)
    # Bản sao không được làm lệch median/MAD.
    eligible &= ~rule_listing_duplicate(df, cfg)
    zscore = robust_zscore(np.log(price.loc[eligible]), df.loc[eligible, "room_type"])
    return zscore.abs().gt(_qa(cfg)["mad_z_threshold"]).reindex(df.index, fill_value=False)


def rule_listing_duplicate(df: pd.DataFrame, cfg: dict) -> pd.Series:
    keys = list(dict.fromkeys([*_qa(cfg)["dedupe_keys"], "snapshot"]))
    _require(df, keys, TABLE_LISTINGS)
    return df.duplicated(keys, keep="first")


def rule_panel_unbalanced(df: pd.DataFrame, cfg: dict) -> pd.Series:
    snapshots = _snapshots(df["snapshot"])
    expected = {str(s) for s in cfg["snapshots"]}
    seen = snapshots.where(snapshots.isin(expected)).groupby(df["id"]).transform("nunique")
    return seen.ne(len(expected))


def rule_coord_invalid(df: pd.DataFrame, cfg: dict) -> pd.Series:
    bbox = _qa(cfg)["bbox"]
    if isinstance(bbox, dict):
        lat_min, lat_max, lon_min, lon_max = (bbox[k] for k in ("lat_min", "lat_max", "lon_min", "lon_max"))
    else:
        lat_min, lat_max, lon_min, lon_max = bbox
    if not (-90 <= lat_min <= lat_max <= 90 and -180 <= lon_min <= lon_max <= 180):
        raise ValueError("bbox cần [lat_min, lat_max, lon_min, lon_max] hợp lệ")
    valid = (_num(df, "latitude").between(lat_min, lat_max)
             & _num(df, "longitude").between(lon_min, lon_max))
    return ~valid.fillna(False)


def rule_min_nights_extreme(df: pd.DataFrame, cfg: dict) -> pd.Series:
    return _num(df, "minimum_nights").gt(_qa(cfg)["min_nights_cap"])


def rule_room_type_invalid(df: pd.DataFrame, cfg: dict) -> pd.Series:
    return ~df["room_type"].isin(_qa(cfg)["allowed_room_types"])


def rule_area_missing(df: pd.DataFrame) -> pd.Series:
    return _text(df["analysis_area"]).eq("")


def rule_area_rare(df: pd.DataFrame, cfg: dict) -> pd.Series:
    unique = df.drop_duplicates(["snapshot", "analysis_area", "id"])
    counts = unique.groupby(["snapshot", "analysis_area"], dropna=False)["id"].nunique()
    keys = pd.MultiIndex.from_frame(df[["snapshot", "analysis_area"]])
    sizes = pd.Series(counts.reindex(keys).to_numpy(), index=df.index)
    return ~rule_area_missing(df) & sizes.lt(_qa(cfg)["min_category_count"])


def rule_review_count_inconsistent(df: pd.DataFrame) -> pd.Series:
    count = _num(df, "number_of_reviews")
    scores = [c for c in df if c.startswith("review_scores_") and not c.endswith("_num")]
    has_score = pd.Series(False, index=df.index)
    for col in scores:
        has_score |= _num(df, col).notna()
    last = df.get("last_review", pd.Series(pd.NA, index=df.index, dtype="string"))
    return (count.eq(0) & has_score) | (count.gt(0) & _dates(last).isna())


def qa_listings(cfg: dict, listings: pd.DataFrame, rec: Recorder) -> pd.DataFrame:
    cols = list(dict.fromkeys([*_qa(cfg)["numeric_cols"], *DEFAULT_QA["numeric_cols"]]))
    df, errors = coerce_numeric(listings, cols)
    df["snapshot"] = _snapshots(df["snapshot"])

    def record(rule_id, name, condition, reason, action, mask):
        spec = RuleSpec(rule_id, TABLE_LISTINGS, name, condition, reason, action)
        df[FLAG_PREFIX + rule_id] = rec.add(spec, df, mask, "median_price_ars_latest",
                                           lambda frame: metric_median_price(frame, cfg))

    for col, mask in errors.items():
        record(L_TYPE_PREFIX + col, f"Sai kiểu {col}", f"{col} không rỗng nhưng không parse được thành số hữu hạn",
               "Tránh tính toán trên kiểu sai", "Giữ dòng; giá trị số lỗi = NaN", rule_type_error(df, col, mask))
    rules = [
        (L_PRICE_MISSING, "Thiếu giá", "Giá không parse được ở mốc cuối", "Không thể tính KPI giá", "Loại khỏi price_valid", rule_price_missing(df, cfg)),
        (L_PRICE_RANGE, "Giá ngoài miền", f"Giá <= 0 hoặc ngoài [{_qa(cfg)['price_min_ars']}, {_qa(cfg)['price_max_ars']}] ARS", "Giá không hợp lệ/ngoại lai cứng", "Loại khỏi price_valid", rule_price_range(df, cfg)),
        (L_DUP, "Trùng trong mốc", f"Bản sao theo {_qa(cfg)['dedupe_keys']} và snapshot", "Tránh đếm hai lần", "Giữ cờ, loại khỏi KPI", rule_listing_duplicate(df, cfg)),
        (L_PANEL, "Panel thiếu mốc", "ID không xuất hiện đủ các snapshot cấu hình", "So sánh cùng tập listing cần panel cân bằng", "Chỉ gắn cờ", rule_panel_unbalanced(df, cfg)),
        (L_COORD, "Toạ độ lỗi", f"Toạ độ thiếu/sai hoặc ngoài bbox {_qa(cfg)['bbox']}", "Tránh đặt sai vị trí trên bản đồ", "Giữ cờ, loại khỏi bản đồ", rule_coord_invalid(df, cfg)),
        (L_MINNIGHTS, "Lưu trú tối thiểu dài", f"minimum_nights > {_qa(cfg)['min_nights_cap']}", "Khác phân khúc thuê ngắn hạn", "Chỉ gắn cờ", rule_min_nights_extreme(df, cfg)),
        (L_ROOMTYPE, "Loại phòng lạ", f"room_type không thuộc {_qa(cfg)['allowed_room_types']}", "Phát hiện danh mục lệch", "Chỉ gắn cờ", rule_room_type_invalid(df, cfg)),
        (L_AREA_MISSING, "Thiếu vùng", "analysis_area rỗng", "Không thể phân nhóm khu vực", "Chỉ gắn cờ", rule_area_missing(df)),
        (L_RARE_AREA, "Vùng ít listing", f"Số ID phân biệt/vùng/mốc < {_qa(cfg)['min_category_count']}", "Cỡ mẫu nhỏ làm KPI kém ổn định", "Chỉ gắn cờ", rule_area_rare(df, cfg)),
        (L_REVIEW_INCONSIST, "Mâu thuẫn reviews", "0 reviews nhưng có điểm, hoặc >0 reviews nhưng last_review thiếu/sai", "Kiểm tra sự nhất quán", "Chỉ gắn cờ", rule_review_count_inconsistent(df)),
    ]
    for rule in rules:
        record(*rule)
    record(L_PRICE_MAD, "Ngoại lai giá MAD", f"|robust_zscore(log giá)| > {_qa(cfg)['mad_z_threshold']}, theo room_type ở mốc cuối",
           "Phát hiện giá khác biệt trong cùng phân khúc", "Loại khỏi price_valid; giữ dòng để kiểm tra",
           rule_price_outlier_mad(df, cfg, df[FLAG_PREFIX + L_PRICE_RANGE]))
    df["price_valid"] = (_snapshots(df["snapshot"]).eq(last_snapshot(cfg)) & df["price_num"].notna()
                         & ~df[[FLAG_PREFIX + r for r in (L_PRICE_RANGE, L_PRICE_MAD, L_DUP)]].any(axis=1))
    df["price_ars"] = df["price_num"].where(df["price_valid"])
    return df


def rule_cal_date_invalid(df: pd.DataFrame) -> pd.Series:
    return _dates(df["date"]).isna()


def rule_cal_duplicate(df: pd.DataFrame) -> pd.Series:
    keys = df[["listing_id", "snapshot"]].copy()
    keys["date"] = _dates(df["date"])
    return keys.duplicated(["listing_id", "date", "snapshot"])


def rule_cal_available_invalid(df: pd.DataFrame) -> pd.Series:
    return ~_text(df["available"]).str.lower().isin(CALENDAR_VALID_AVAILABLE)


def rule_cal_outside_window(df: pd.DataFrame) -> pd.Series:
    delta = (_dates(df["date"]) - _dates(df["snapshot"])).dt.days
    return delta.notna() & ~delta.between(-CALENDAR_WINDOW_BEFORE_DAYS, CALENDAR_WINDOW_AFTER_DAYS)


def rule_cal_min_nights_extreme(df: pd.DataFrame, cfg: dict) -> pd.Series:
    return _num(df, "minimum_nights").gt(_qa(cfg)["min_nights_cap"])


def rule_cal_orphan(df: pd.DataFrame, listing_ids: pd.Series) -> pd.Series:
    return ~df["listing_id"].isin(listing_ids)


def aggregate_calendar_by_listing(valid: pd.DataFrame) -> pd.DataFrame:
    if valid.empty:
        return pd.DataFrame(columns=CAL_LISTING_COLUMNS)
    work = valid[["snapshot", "listing_id"]].copy()
    work["unavailable"] = _text(valid["available"]).str.lower().eq("f").astype(int)
    delta = (_dates(valid["date"]) - _dates(valid["snapshot"])).dt.days
    for horizon in OCCUPANCY_HORIZONS:
        work[f"occ_{horizon}"] = work["unavailable"].where(delta.ge(0) & delta.lt(horizon))
    result = work.groupby(["snapshot", "listing_id"], sort=True).agg(
        n_days=("unavailable", "size"), unavail_days=("unavailable", "sum"),
        **{f"occ_{h}": (f"occ_{h}", "mean") for h in OCCUPANCY_HORIZONS}).reset_index()
    result["unavail_rate"] = result["unavail_days"] / result["n_days"]
    return result[CAL_LISTING_COLUMNS]


def aggregate_calendar_by_month(valid: pd.DataFrame, area_map: pd.Series) -> pd.DataFrame:
    if valid.empty:
        return pd.DataFrame(columns=CAL_MONTH_COLUMNS)
    work = valid[["snapshot", "listing_id"]].copy()
    if isinstance(area_map.index, pd.MultiIndex):
        keys = pd.MultiIndex.from_frame(work[["snapshot", "listing_id"]])
        work["analysis_area"] = area_map.reindex(keys).to_numpy()
    else:
        work["analysis_area"] = work["listing_id"].map(area_map)
    work["month"] = _dates(valid["date"]).dt.strftime("%Y-%m")
    work["unavailable"] = _text(valid["available"]).str.lower().eq("f").astype(int)
    result = work.groupby(["snapshot", "analysis_area", "month"], sort=True, dropna=False).agg(
        n=("unavailable", "size"), n_unavail=("unavailable", "sum")).reset_index()
    result["unavail_rate"] = result["n_unavail"] / result["n"]
    return result[CAL_MONTH_COLUMNS]


def _orphan_in_snapshot(df: pd.DataFrame, listings: pd.DataFrame) -> pd.Series:
    pairs = listings[["snapshot", "id"]].rename(columns={"id": "listing_id"}).drop_duplicates()
    return pd.Series(~pd.MultiIndex.from_frame(df[["snapshot", "listing_id"]]).isin(
        pd.MultiIndex.from_frame(pairs)), index=df.index)


def qa_calendar(cfg: dict, calendar: pd.DataFrame, listings_clean: pd.DataFrame,
                rec: Recorder) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = calendar.copy()
    df["snapshot"] = _snapshots(df["snapshot"])
    df["date"] = _dates(df["date"])
    df["available"] = _text(df["available"]).str.lower()
    rules = [
        (C_DATE, "Ngày lỗi", "date không parse được", "Không xác định được ngày", "Loại khỏi tổng hợp", rule_cal_date_invalid(df)),
        (C_DUP, "Ngày trùng", "Trùng listing_id, date, snapshot; giữ dòng đầu", "Tránh đếm hai lần", "Loại khỏi tổng hợp", rule_cal_duplicate(df)),
        (C_AVAIL, "Trạng thái lỗi", "available không thuộc {t,f}", "Không suy diễn trạng thái thiếu", "Loại khỏi tổng hợp", rule_cal_available_invalid(df)),
        (C_WINDOW, "Ngoài cửa sổ", "date ngoài [snapshot-1 ngày, snapshot+366 ngày]", "Bảo đảm cửa sổ lịch", "Loại khỏi tổng hợp", rule_cal_outside_window(df)),
        (C_MINNIGHTS, "Lưu trú tối thiểu dài", f"minimum_nights > {_qa(cfg)['min_nights_cap']}", "Khác phân khúc thuê ngắn hạn", "Chỉ gắn cờ", rule_cal_min_nights_extreme(df, cfg)),
        (C_ORPHAN, "Listing không tồn tại", "listing_id không có trong listings cùng snapshot", "Tránh tổng hợp dòng không liên kết", "Loại khỏi tổng hợp", _orphan_in_snapshot(df, listings_clean)),
    ]
    excluded = pd.Series(False, index=df.index)
    for rule_id, name, condition, reason, action, mask in rules:
        mask = rec.add(RuleSpec(rule_id, TABLE_CALENDAR, name, condition, reason, action),
                       df, mask, "unavailable_rate", metric_unavailable_rate)
        if rule_id != C_MINNIGHTS:
            excluded |= mask
    valid = df.loc[~excluded]
    area_map = (listings_clean.drop_duplicates(["snapshot", "id"])
                .rename(columns={"id": "listing_id"}).set_index(["snapshot", "listing_id"])["analysis_area"])
    return aggregate_calendar_by_listing(valid), aggregate_calendar_by_month(valid, area_map)


def rule_rev_missing(df: pd.DataFrame) -> pd.Series:
    return _text(df["comments"]).eq("")


def rule_rev_duplicate(df: pd.DataFrame) -> pd.Series:
    return df.duplicated(["id", "snapshot"], keep="first")


def rule_rev_repeat_across_snapshots(df: pd.DataFrame) -> pd.Series:
    snapshots = _snapshots(df["snapshot"])
    earliest = snapshots.groupby(df["id"]).transform("min")
    return snapshots.gt(earliest)


def rule_rev_date_invalid(df: pd.DataFrame) -> pd.Series:
    return _dates(df["date"]).isna()


def rule_rev_orphan(df: pd.DataFrame, listing_ids: pd.Series) -> pd.Series:
    return ~df["listing_id"].isin(listing_ids)


def rule_rev_auto_text(df: pd.DataFrame, cfg: dict) -> pd.Series:
    pattern = _qa(cfg)["auto_text_regex"]
    return _text(df["comments"]).str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False) if pattern else pd.Series(False, index=df.index)


def rule_rev_too_short(df: pd.DataFrame, missing: pd.Series) -> pd.Series:
    return ~_bool(missing, df.index) & clean_comment_text(df["comments"]).str.len().lt(MIN_REVIEW_CHARS)


def rule_rev_too_long(df: pd.DataFrame, cfg: dict) -> pd.Series:
    return clean_comment_text(df["comments"]).str.len().gt(_qa(cfg)["max_review_chars"])


def rule_rev_has_html(df: pd.DataFrame) -> pd.Series:
    return _text(df["comments"]).str.contains(HTML_BR_REGEX, flags=re.IGNORECASE, regex=True, na=False)


def clean_comment_text(comments: pd.Series) -> pd.Series:
    return (_text(comments).str.replace(HTML_BR_REGEX, " ", flags=re.IGNORECASE, regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def qa_reviews(cfg: dict, reviews: pd.DataFrame, listings_clean: pd.DataFrame, rec: Recorder) -> pd.DataFrame:
    df = reviews.copy()
    df["snapshot"] = _snapshots(df["snapshot"])
    df = df.sort_values(["snapshot", "id"], kind="stable").reset_index(drop=True)
    for col in ("llm_sample", "hand_label"):
        df[col] = _read_bool(df[col])
    missing = rule_rev_missing(df)
    rules = [
        (R_MISSING, "Thiếu review", "comments null hoặc toàn khoảng trắng", "Không có văn bản để phân tích", "Loại khỏi use_for_llm", missing),
        (R_DUP, "Review trùng", "Trùng id, snapshot; giữ dòng đầu", "Tránh phân tích hai lần", "Loại khỏi use_for_llm", rule_rev_duplicate(df)),
        (R_REPEAT, "Review lặp mốc", "id xuất hiện ở snapshot sớm hơn", "Một review chỉ là một quan sát văn bản", "Loại khỏi use_for_llm", rule_rev_repeat_across_snapshots(df)),
        (R_DATE, "Ngày review lỗi", "date không parse được", "Không xác định được thời gian review", "Loại khỏi use_for_llm", rule_rev_date_invalid(df)),
        (R_ORPHAN, "Review không liên kết", "listing_id không có trong listings cùng snapshot", "Không nối được KPI khu vực", "Loại khỏi use_for_llm", _orphan_in_snapshot(df, listings_clean)),
        (R_AUTO, "Review tự động", f"comments khớp regex: {_qa(cfg)['auto_text_regex']}", "Không phải trải nghiệm khách", "Loại khỏi use_for_llm", rule_rev_auto_text(df, cfg)),
        (R_SHORT, "Review quá ngắn", f"Văn bản sạch < {MIN_REVIEW_CHARS} ký tự, không kể thiếu", "Thiếu nội dung phân tích", "Loại khỏi use_for_llm", rule_rev_too_short(df, missing)),
        (R_LONG, "Review quá dài", f"Văn bản sạch > {_qa(cfg)['max_review_chars']} ký tự", "Giới hạn đầu vào và chi phí LLM", "Loại khỏi use_for_llm", rule_rev_too_long(df, cfg)),
        (R_HTML, "Có thẻ xuống dòng", f"comments chứa {HTML_BR_REGEX}", "Chuẩn hoá văn bản", "Thay br bằng khoảng trắng, giữ nội dung", rule_rev_has_html(df)),
    ]
    excluded = pd.Series(False, index=df.index)
    for rule_id, name, condition, reason, action, mask in rules:
        mask = rec.add(RuleSpec(rule_id, TABLE_REVIEWS, name, condition, reason, action),
                       df, mask, "row_count", metric_row_count)
        df[FLAG_PREFIX + rule_id] = mask
        if rule_id != R_HTML:
            excluded |= mask
    df["comments_clean"] = clean_comment_text(df["comments"])
    df["use_for_llm"] = ~excluded
    return df


def run(cfg: dict, **_) -> None:
    if cfg.get("currency", "ARS") != "ARS":
        raise ValueError("QA này tạo price_ars; currency phải là ARS (không tự đổi tiền).")
    last_snapshot(cfg)
    listings, calendar, reviews = load_inputs(cfg)
    profiles = pd.concat([profile(df, name) for df, name in (
        (listings, TABLE_LISTINGS), (calendar, TABLE_CALENDAR), (reviews, TABLE_REVIEWS))], ignore_index=True)
    rec = Recorder()
    listings_clean = qa_listings(cfg, listings, rec)
    cal_listing, cal_month = qa_calendar(cfg, calendar, listings_clean, rec)
    reviews_clean = qa_reviews(cfg, reviews, listings_clean, rec)
    save_outputs(cfg, listings_clean, reviews_clean, cal_listing, cal_month, rec, profiles)
    log.info("QA xong: %d quy tắc, báo cáo %s", len(rec.rows), path(cfg, "reports") / OUT_QA_REPORT)


def main() -> None:
    import yaml

    # Windows khi ghi vào pipe có thể mặc định cp1252, không chứa tiếng Việt.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/Buenos_Aires.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    try:
        run(cfg)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(1, f"QA: {exc}\n")


if __name__ == "__main__":
    main()
