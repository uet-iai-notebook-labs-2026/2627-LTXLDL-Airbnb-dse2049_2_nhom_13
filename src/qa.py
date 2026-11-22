"""QA cho ba bảng subset, giữ giao diện của khung qa.py được cung cấp.

Đầu vào trong paths.processed: listings_subset.csv, calendar_subset.csv.gz,
reviews_subset.csv. Đầu ra: listings_clean.csv, reviews_clean.csv,
calendar_listing_agg.csv, calendar_month_agg.csv; hai báo cáo nằm ở paths.reports.

Quy tắc giữ cờ trên listings/reviews, không xoá dữ liệu gốc. price_ars chỉ dùng
giá hợp lệ ở snapshot cuối theo hợp đồng; price_num giữ giá đã parse ở mọi mốc.
occ_H là tỷ lệ ngày KHÔNG KHẢ DỤNG quan sát được, không phải tỷ lệ đặt phòng.
qa_report đo tác động loại riêng từng cờ, không phải số dòng loại tích luỹ.

Chạy riêng từ gốc repo: python -m src.qa --config configs/Buenos_Aires.yaml
Hoặc python src/qa.py, hoặc gọi qa.run(cfg) từ pipeline. Không cần common.py.
Đường dẫn tương đối tính từ gốc repo. Chưa có subset thì báo rõ tệp cần chuẩn bị.
"""
from __future__ import annotations

import argparse
from datetime import date
import logging
from pathlib import Path
import re
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_logger(name: str) -> logging.Logger:
    """Nhận tên logger và trả logger tương ứng từ hệ thống logging."""
    return logging.getLogger(name)


def last_snapshot(cfg: dict) -> str:
    """Mốc cuối từ config; nhận cả chuỗi ISO và datetime.date do YAML đọc."""
    values = cfg.get("snapshots")
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("Config cần danh sách snapshots không rỗng")
    snapshots = []
    for value in values:
        text = str(value)
        try:
            if date.fromisoformat(text).isoformat() != text:
                raise ValueError
        except ValueError:
            raise ValueError(f"Snapshot {text!r} cần đúng định dạng YYYY-MM-DD") from None
        snapshots.append(text)
    if len(snapshots) != len(set(snapshots)):
        raise ValueError("Config snapshots bị trùng mốc")
    return max(snapshots)


def path(cfg: dict, name: str) -> Path:
    """Lấy thư mục từ paths.<name>, hoặc <name>_dir; không phụ thuộc cwd."""
    defaults = {"processed": "data/processed", "reports": "reports", "raw": "data/raw",
                "samples": "data/samples", "figures": "figures"}
    paths = cfg.get("paths") or {}
    if not isinstance(paths, dict):
        raise ValueError("Config paths phải là mapping hoặc null")
    value = paths.get(name, cfg.get(f"{name}_dir", defaults.get(name)))
    if value is None:
        raise ValueError(f"Config chưa có đường dẫn {name!r}")
    resolved = Path(value)
    return resolved if resolved.is_absolute() else REPO_ROOT / resolved

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
    """Nhận cfg và trả cấu hình QA với các giá trị mặc định được ghi đè theo cfg.qa."""
    return {**DEFAULT_QA, **(cfg.get("qa") or {})}


def _bool(mask: pd.Series, index: pd.Index) -> pd.Series:
    """Căn mask theo index và trả mặt nạ bool, coi giá trị thiếu là False."""
    return mask.reindex(index).fillna(False).astype(bool)


def _text(series: pd.Series) -> pd.Series:
    """Nhận Series và trả chuỗi đã bỏ khoảng trắng hai đầu, thay giá trị thiếu bằng rỗng."""
    return series.astype("string").fillna("").str.strip()


def _dates(series: pd.Series) -> pd.Series:
    """Nhận Series ngày ISO YYYY-MM-DD và trả datetime, giá trị sai thành NaT."""
    # Các file Inside Airbnb dùng ngày ISO, không đoán DD/MM với MM/DD.
    return pd.to_datetime(series, format="%Y-%m-%d", errors="coerce")


def _snapshots(series: pd.Series) -> pd.Series:
    """Nhận Series snapshot và trả chuỗi YYYY-MM-DD, giữ giá trị thiếu nếu ngày sai."""
    return _dates(series).dt.strftime("%Y-%m-%d")


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    """Từ df và tên cột, trả cột _num sẵn có hoặc số đã parse; thiếu cột trả Series NaN."""
    if f"{col}_num" in df:
        return df[f"{col}_num"]
    if col not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return parse_price(df[col]) if col == "price" else pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    """Kiểm tra các cột bắt buộc của df; báo ValueError kèm tên bảng nếu thiếu."""
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise ValueError(f"{name} thiếu cột bắt buộc: {', '.join(missing)}")


def _read_bool(series: pd.Series) -> pd.Series:
    """Nhận Series và trả bool True cho các biểu diễn true/t/1/1.0/x/yes/y, bỏ qua hoa thường."""
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
        """Khởi tạo danh sách rỗng để lưu các dòng báo cáo quy tắc."""
        self.rows: list[dict] = []

    def add(self, spec: RuleSpec, df: pd.DataFrame, mask: pd.Series,
            metric_name: str, metric_fn: Callable) -> pd.Series:
        """Nhận quy tắc, df, mask và hàm metric; trả mask bool đã căn theo df.
        Ghi một dòng báo cáo vào self.rows, với True là dòng bị quy tắc gắn cờ.
        Các giá trị after giả định bỏ riêng mask này; df gốc không bị xoá dòng.
        """
        mask = _bool(mask, df.index)
        n, flagged = len(df), int(mask.sum())
        self.rows.append({**vars(spec), "n_rows_before": n, "n_flagged": flagged,
                          "n_rows_after": n - flagged, "pct_flagged": flagged / n * 100 if n else 0.0,
                          "metric": metric_name, "metric_before": metric_fn(df),
                          "metric_after": metric_fn(df.loc[~mask])})
        return mask

    def frame(self) -> pd.DataFrame:
        """Trả DataFrame từ các dòng đã ghi với đúng thứ tự cột báo cáo QA."""
        return pd.DataFrame(self.rows, columns=RULE_REPORT_COLUMNS)


def parse_price(s: pd.Series) -> pd.Series:
    """Parse dạng thực tế '$86,206.29'; từ chối dấu phân cách sai và số vô hạn.
    Trả Series số thực, với chuỗi thiếu hoặc không hợp lệ được đổi thành NaN.
    """
    text = _text(s)
    pattern = r"[+-]?\s*\$?\s*(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    valid = text.str.fullmatch(pattern)
    cleaned = text.where(valid).str.replace(r"[$,\s]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Nhận df và danh sách cột, trả bản sao có cột _num cùng dict mặt nạ lỗi.
    Mỗi mask True khi ô gốc không rỗng nhưng không parse được thành số hữu hạn.
    Giữ nguyên cột gốc; bỏ qua tên cột không tồn tại trong df.
    """
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
    """Nhận giá trị và nhóm, trả z-score theo median/MAD từng nhóm; MAD bằng 0 cho NaN."""
    values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = values.groupby(groups, dropna=False).transform("median")
    deviation = (values - median).abs()
    mad = deviation.groupby(groups, dropna=False).transform("median")
    return 0.6745 * (values - median) / mad.where(mad.ne(0))


def profile(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Nhận df và tên bảng, trả thống kê kiểu dữ liệu, thiếu và số giá trị phân biệt từng cột."""
    n = len(df)
    return pd.DataFrame([{"table": name, "column": col, "dtype": str(df[col].dtype),
                          "n_null": int(df[col].isna().sum()),
                          "pct_null": float(df[col].isna().mean() * 100) if n else 0.0,
                          "n_unique": int(df[col].nunique(dropna=True))}
                         for col in df], columns=PROFILE_COLUMNS)


def metric_median_price(df: pd.DataFrame, cfg: dict) -> float:
    """Từ df và cfg, trả trung vị mọi giá parse được ở snapshot cuối; không có giá trả NaN."""
    values = _num(df, "price").loc[_snapshots(df["snapshot"]).eq(last_snapshot(cfg))]
    return float(values.median()) if values.notna().any() else np.nan


def metric_unavailable_rate(df: pd.DataFrame) -> float:
    """Từ df lịch, trả tỷ lệ dòng available='f' (không khả dụng); bảng rỗng trả NaN."""
    return float(_text(df["available"]).str.lower().eq("f").mean()) if len(df) else np.nan


def metric_row_count(df: pd.DataFrame) -> float:
    """Nhận df và trả số dòng dưới dạng float để dùng làm metric QA."""
    return float(len(df))


def load_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nhận cfg, đọc ba bảng subset từ thư mục processed và trả listings, calendar, reviews.
    Giữ ID nguyên chính xác, chuẩn hoá snapshot và hai cột bool của reviews.
    Báo lỗi nếu thiếu tệp, thiếu khoá/cột bắt buộc hoặc snapshot ngoài cấu hình.
    """
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
    """Nhận cfg, các bảng QA, Recorder và profile; ghi bốn bảng đầu ra cùng hai báo cáo CSV.
    Tạo thư mục processed và reports nếu cần, dùng UTF-8 và không ghi chỉ mục.
    Không trả giá trị; các tệp trùng tên tại đích được ghi đè.
    """
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
    """Từ df và mask lỗi kiểu của col, trả mask bool đã căn chỉ mục; True là ô parse lỗi."""
    return _bool(bad_mask, df.index)


def rule_price_missing(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Từ df và cfg, trả mask True cho giá thiếu hoặc parse lỗi tại snapshot cuối."""
    return _snapshots(df["snapshot"]).eq(last_snapshot(cfg)) & _num(df, "price").isna()


def rule_price_range(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Từ df và cfg, trả mask True cho giá có số nhưng <= 0 hoặc ngoài khoảng giá cấu hình."""
    settings, values = _qa(cfg), _num(df, "price")
    return values.notna() & (values.le(0) | values.lt(settings["price_min_ars"]) | values.gt(settings["price_max_ars"]))


def rule_price_outlier_mad(df: pd.DataFrame, cfg: dict, already_out: pd.Series) -> pd.Series:
    """Nhận df, cfg và mask đã loại; trả mask True khi |z-score log giá| vượt ngưỡng MAD.
    Chỉ xét snapshot cuối, giá dương, chưa bị loại và không phải bản sao listing.
    Tính median/MAD riêng theo room_type rồi căn kết quả về toàn bộ chỉ mục df.
    """
    price = _num(df, "price")
    eligible = _snapshots(df["snapshot"]).eq(last_snapshot(cfg)) & price.gt(0) & ~_bool(already_out, df.index)
    # Bản sao không được làm lệch median/MAD.
    eligible &= ~rule_listing_duplicate(df, cfg)
    zscore = robust_zscore(np.log(price.loc[eligible]), df.loc[eligible, "room_type"])
    return zscore.abs().gt(_qa(cfg)["mad_z_threshold"]).reindex(df.index, fill_value=False)


def rule_listing_duplicate(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Từ df và cfg, trả mask True cho bản sao sau dòng đầu theo dedupe_keys cộng snapshot."""
    keys = list(dict.fromkeys([*_qa(cfg)["dedupe_keys"], "snapshot"]))
    _require(df, keys, TABLE_LISTINGS)
    return df.duplicated(keys, keep="first")


def rule_panel_unbalanced(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Từ df và cfg, trả mask True cho ID thiếu ít nhất một snapshot thuộc cấu hình."""
    snapshots = _snapshots(df["snapshot"])
    expected = {str(s) for s in cfg["snapshots"]}
    seen = snapshots.where(snapshots.isin(expected)).groupby(df["id"]).transform("nunique")
    return seen.ne(len(expected))


def rule_coord_invalid(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Từ df và bbox trong cfg, trả mask True cho toạ độ thiếu/sai/ngoài bbox; bbox sai gây lỗi."""
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
    """Từ df và cfg, trả mask True khi minimum_nights dạng số vượt min_nights_cap."""
    return _num(df, "minimum_nights").gt(_qa(cfg)["min_nights_cap"])


def rule_room_type_invalid(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Từ df và cfg, trả mask True khi room_type không thuộc danh sách loại phòng cho phép."""
    return ~df["room_type"].isin(_qa(cfg)["allowed_room_types"])


def rule_area_missing(df: pd.DataFrame) -> pd.Series:
    """Nhận df và trả mask True cho analysis_area thiếu, rỗng hoặc chỉ có khoảng trắng."""
    return _text(df["analysis_area"]).eq("")


def rule_area_rare(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Từ df và cfg, trả mask True cho vùng không rỗng có ít ID phân biệt hơn ngưỡng ở từng snapshot."""
    unique = df.drop_duplicates(["snapshot", "analysis_area", "id"])
    counts = unique.groupby(["snapshot", "analysis_area"], dropna=False)["id"].nunique()
    keys = pd.MultiIndex.from_frame(df[["snapshot", "analysis_area"]])
    sizes = pd.Series(counts.reindex(keys).to_numpy(), index=df.index)
    return ~rule_area_missing(df) & sizes.lt(_qa(cfg)["min_category_count"])


def rule_review_count_inconsistent(df: pd.DataFrame) -> pd.Series:
    """Từ df, trả mask True khi 0 reviews vẫn có điểm, hoặc có reviews nhưng last_review thiếu/sai."""
    count = _num(df, "number_of_reviews")
    scores = [c for c in df if c.startswith("review_scores_") and not c.endswith("_num")]
    has_score = pd.Series(False, index=df.index)
    for col in scores:
        has_score |= _num(df, col).notna()
    last = df.get("last_review", pd.Series(pd.NA, index=df.index, dtype="string"))
    return (count.eq(0) & has_score) | (count.gt(0) & _dates(last).isna())


def qa_listings(cfg: dict, listings: pd.DataFrame, rec: Recorder) -> pd.DataFrame:
    """Nhận cfg, listings và Recorder; trả bản sao có cột số, cờ QA, price_valid và price_ars.
    Giữ mọi dòng và ghi tác động giả định của từng quy tắc độc lập vào rec.
    price_valid chỉ True ở snapshot cuối khi giá có số và không mắc cờ miền giá, MAD, trùng.
    price_ars chỉ giữ giá ở các dòng price_valid; price_num giữ giá parse được ở mọi mốc.
    """
    cols = list(dict.fromkeys([*_qa(cfg)["numeric_cols"], *DEFAULT_QA["numeric_cols"]]))
    df, errors = coerce_numeric(listings, cols)
    df["snapshot"] = _snapshots(df["snapshot"])

    def record(rule_id, name, condition, reason, action, mask):
        """Nhận mô tả quy tắc cùng mask và ghi cột cờ True cho dòng vi phạm vào df bao ngoài.
        Thêm báo cáo vào rec với trung vị giá snapshot cuối trước/sau giả định bỏ riêng mask.
        Không trả giá trị và không xoá dòng khỏi df.
        """
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
    """Nhận df lịch và trả mask True khi date không parse được thành ngày ISO."""
    return _dates(df["date"]).isna()


def rule_cal_duplicate(df: pd.DataFrame) -> pd.Series:
    """Nhận df lịch và trả mask True cho bản sao sau dòng đầu theo listing_id, ngày đã parse, snapshot."""
    keys = df[["listing_id", "snapshot"]].copy()
    keys["date"] = _dates(df["date"])
    return keys.duplicated(["listing_id", "date", "snapshot"])


def rule_cal_available_invalid(df: pd.DataFrame) -> pd.Series:
    """Nhận df lịch và trả mask True khi available sau chuẩn hoá không thuộc {'t', 'f'}."""
    return ~_text(df["available"]).str.lower().isin(CALENDAR_VALID_AVAILABLE)


def rule_cal_outside_window(df: pd.DataFrame) -> pd.Series:
    """Nhận df lịch và trả mask True cho ngày parse được nằm ngoài [snapshot-1, snapshot+366 ngày]."""
    delta = (_dates(df["date"]) - _dates(df["snapshot"])).dt.days
    return delta.notna() & ~delta.between(-CALENDAR_WINDOW_BEFORE_DAYS, CALENDAR_WINDOW_AFTER_DAYS)


def rule_cal_min_nights_extreme(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Từ df lịch và cfg, trả mask True khi minimum_nights dạng số vượt min_nights_cap."""
    return _num(df, "minimum_nights").gt(_qa(cfg)["min_nights_cap"])


def rule_cal_orphan(df: pd.DataFrame, listing_ids: pd.Series) -> pd.Series:
    """Từ df lịch và tập listing_ids, trả mask True khi listing_id không thuộc tập được truyền vào."""
    return ~df["listing_id"].isin(listing_ids)


def aggregate_calendar_by_listing(valid: pd.DataFrame) -> pd.DataFrame:
    """Nhận các dòng lịch hợp lệ và trả tổng hợp theo snapshot, listing_id.
    Đếm ngày quan sát, ngày không khả dụng và tỷ lệ cho các cửa sổ [0, 30/90/365) ngày.
    occ_* và unavail_rate đo không khả dụng quan sát được, không phải tỷ lệ đặt phòng.
    Đầu vào rỗng trả bảng rỗng có đủ các cột đầu ra.
    """
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
    """Nhận lịch hợp lệ và ánh xạ vùng theo listing hoặc cặp snapshot, listing_id.
    Trả số ngày, số ngày không khả dụng và tỷ lệ theo snapshot, vùng, tháng.
    Tỷ lệ phản ánh trạng thái available, không đo số đêm đặt phòng thực tế.
    Đầu vào rỗng trả bảng rỗng có đủ các cột đầu ra.
    """
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
    """Từ df và listings, trả mask True khi cặp snapshot, listing_id không có trong listings."""
    pairs = listings[["snapshot", "id"]].rename(columns={"id": "listing_id"}).drop_duplicates()
    return pd.Series(~pd.MultiIndex.from_frame(df[["snapshot", "listing_id"]]).isin(
        pd.MultiIndex.from_frame(pairs)), index=df.index)


def qa_calendar(cfg: dict, calendar: pd.DataFrame, listings_clean: pd.DataFrame,
                rec: Recorder) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nhận cfg, lịch, listings_clean và Recorder; trả hai bảng tổng hợp theo listing và tháng.
    Ghi vào rec tác động giả định của từng mask riêng trên lịch đã chuẩn hoá.
    Loại hợp các cờ lỗi khỏi tổng hợp, riêng minimum_nights dài chỉ ghi cờ báo cáo.
    Các tỷ lệ đầu ra đo ngày không khả dụng, không phải công suất đặt phòng thực tế.
    """
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
    """Nhận df reviews và trả mask True khi comments thiếu, rỗng hoặc chỉ có khoảng trắng."""
    return _text(df["comments"]).eq("")


def rule_rev_duplicate(df: pd.DataFrame) -> pd.Series:
    """Nhận df reviews và trả mask True cho bản sao sau dòng đầu theo id, snapshot."""
    return df.duplicated(["id", "snapshot"], keep="first")


def rule_rev_repeat_across_snapshots(df: pd.DataFrame) -> pd.Series:
    """Nhận df reviews và trả mask True khi id xuất hiện ở snapshot muộn hơn mốc sớm nhất của id."""
    snapshots = _snapshots(df["snapshot"])
    earliest = snapshots.groupby(df["id"]).transform("min")
    return snapshots.gt(earliest)


def rule_rev_date_invalid(df: pd.DataFrame) -> pd.Series:
    """Nhận df reviews và trả mask True khi date không parse được thành ngày ISO."""
    return _dates(df["date"]).isna()


def rule_rev_orphan(df: pd.DataFrame, listing_ids: pd.Series) -> pd.Series:
    """Từ df reviews và tập listing_ids, trả mask True khi listing_id không thuộc tập được truyền vào."""
    return ~df["listing_id"].isin(listing_ids)


def rule_rev_auto_text(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Từ df reviews và cfg, trả mask True khi comments khớp regex tự động; regex rỗng cho toàn False."""
    pattern = _qa(cfg)["auto_text_regex"]
    return _text(df["comments"]).str.contains(pattern, flags=re.IGNORECASE, regex=True, na=False) if pattern else pd.Series(False, index=df.index)


def rule_rev_too_short(df: pd.DataFrame, missing: pd.Series) -> pd.Series:
    """Từ df reviews và mask thiếu, trả mask True cho comments không thiếu nhưng văn bản sạch dưới 3 ký tự."""
    return ~_bool(missing, df.index) & clean_comment_text(df["comments"]).str.len().lt(MIN_REVIEW_CHARS)


def rule_rev_too_long(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Từ df reviews và cfg, trả mask True khi văn bản comments đã làm sạch dài hơn max_review_chars."""
    return clean_comment_text(df["comments"]).str.len().gt(_qa(cfg)["max_review_chars"])


def rule_rev_has_html(df: pd.DataFrame) -> pd.Series:
    """Nhận df reviews và trả mask True khi comments chứa thẻ br, không phân biệt hoa thường."""
    return _text(df["comments"]).str.contains(HTML_BR_REGEX, flags=re.IGNORECASE, regex=True, na=False)


def clean_comment_text(comments: pd.Series) -> pd.Series:
    """Nhận Series comments, trả văn bản đã thay thẻ br bằng khoảng trắng và gộp khoảng trắng liên tiếp."""
    return (_text(comments).str.replace(HTML_BR_REGEX, " ", flags=re.IGNORECASE, regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def qa_reviews(cfg: dict, reviews: pd.DataFrame, listings_clean: pd.DataFrame, rec: Recorder) -> pd.DataFrame:
    """Nhận cfg, reviews, listings_clean và Recorder; trả bản sao sắp theo snapshot, id có các cờ QA.
    Giữ mọi dòng, thêm comments_clean và use_for_llm; chỉ cờ HTML không làm mất quyền dùng LLM.
    Ghi vào rec số dòng trước/sau giả định bỏ riêng từng mask, không phải xoá tích luỹ.
    Chuẩn hoá llm_sample và hand_label thành bool trên bảng kết quả.
    """
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
    """Nhận cfg, kiểm tra tiền tệ ARS và snapshot rồi điều phối QA cho ba bảng subset.
    Tạo profile từ dữ liệu đầu vào, gắn cờ listings/reviews và tổng hợp lịch hợp lệ.
    Ghi các bảng đầu ra, báo cáo QA/profile và log hoàn tất; không trả giá trị.
    """
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
    """Đọc --config từ dòng lệnh, nạp YAML và gọi run với cấu hình dạng mapping.
    Cấu hình UTF-8 cho luồng xuất hỗ trợ và bật logging thông tin.
    Không trả giá trị; lỗi cấu hình hoặc đầu vào được báo qua parser với mã thoát 1.
    """
    import yaml

    # Windows khi ghi vào pipe có thể mặc định cp1252, không chứa tiếng Việt.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/Buenos_Aires.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config_file = Path(args.config)
    if not config_file.is_absolute():
        config_file = REPO_ROOT / config_file
    try:
        with config_file.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        if not isinstance(cfg, dict):
            raise ValueError("Config YAML phải là mapping")
        run(cfg)
    except (ValueError, FileNotFoundError, KeyError, yaml.YAMLError) as exc:
        parser.exit(1, f"QA: {exc}\n")


if __name__ == "__main__":
    main()
