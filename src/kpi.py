"""Seven sample-based Airbnb KPIs for hosts and property managers.

Reads QA outputs only; never downloads data, calls a model, or reads .env.
Prices require QA's price_valid mask. Calendar unavailability is not occupancy.
Review dates describe historical feedback, independently of snapshot dates.
Run: python src/kpi.py --config configs/Buenos_Aires.yaml --output-dir PATH
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import sys
import tempfile

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("kpi")
DEFAULT_ASPECTS = ["vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn"]
DEFAULT_ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
SENTIMENTS = {"positive", "negative", "mixed"}
LISTING_COLUMNS = ["snapshot", "scope", "area_col", "area", "kpi", "value", "unit",
                   "numerator", "denominator", "n_listings", "n_valid", "status"]
REVIEW_COLUMNS = ["period_type", "period", "scope", "area_col", "area", "kpi", "value",
                  "unit", "numerator", "denominator", "n_eligible_reviews",
                  "n_classified_reviews", "n_listings", "coverage", "status"]
DEFINITION_COLUMNS = ["kpi", "name", "objective", "formula", "unit", "dimensions", "notes"]
OUTPUT_NAMES = ["kpi_listings.csv", "kpi_reviews.csv", "kpi_aspects.csv",
                "kpi_definitions.csv", "kpi_summary.json"]


def _mapping(value, name):
    """Kiểm tra value là dict; None được đổi thành dict rỗng, kiểu khác gây ValueError."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping or null")
    return value


def _nonempty(value, name):
    """Kiểm tra value là chuỗi không rỗng rồi trả chuỗi đã bỏ khoảng trắng hai đầu."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _iso(value, name):
    """Chuẩn hóa value thành chuỗi ngày YYYY-MM-DD; ngày hoặc định dạng sai gây ValueError."""
    text = str(value).strip()
    try:
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", text):
            raise ValueError
        if date.fromisoformat(text).isoformat() != text:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(f"{name}: invalid ISO date {text!r}") from None
    return text


def _root_path(value):
    """Trả Path tuyệt đối của value, lấy gốc repo làm mốc cho đường dẫn tương đối."""
    result = Path(value)
    return (result if result.is_absolute() else REPO_ROOT / result).resolve()


def _settings(cfg):
    """Đọc và kiểm tra cấu hình KPI: đường dẫn, tiền tệ, mốc dữ liệu và các cột sử dụng.
    Trả dict tham số dùng chung; khi thiếu snapshots, bước đọc listings sẽ
    suy ra các mốc từ dữ liệu thay vì ghi cứng thành phố hoặc ngày.
    """
    cfg = _mapping(cfg, "config")
    k = _mapping(cfg.get("kpi"), "kpi")
    qa = _mapping(cfg.get("qa"), "qa")
    llm = _mapping(cfg.get("llm"), "llm")
    paths = _mapping(cfg.get("paths"), "paths")
    snapshots = cfg.get("snapshots")
    if snapshots is not None:
        if not isinstance(snapshots, (list, tuple)) or not snapshots:
            raise ValueError("snapshots must be a nonempty list when explicitly supplied")
        snapshots = [_iso(v, "snapshot config") for v in snapshots]
        if len(set(snapshots)) != len(snapshots):
            raise ValueError("snapshots must not contain duplicates")
    room_types = qa.get("allowed_room_types", DEFAULT_ROOM_TYPES)
    aspects = llm.get("aspects", DEFAULT_ASPECTS)
    for values, name in ((room_types, "qa.allowed_room_types"), (aspects, "llm.aspects")):
        if (not isinstance(values, list) or not values
                or any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{name} must contain unique nonempty strings")
    if any("|" in a for a in aspects):
        raise ValueError("llm.aspects must not contain '|'")
    settings = {"snapshots": sorted(snapshots) if snapshots is not None else None,
                "currency": _nonempty(cfg.get("currency"), "currency"),
                "city": str(cfg.get("city", "")), "room_types": list(room_types), "aspects": list(aspects)}
    for key, default in (("stakeholder", "host/property_manager"), ("area_col", "neighbourhood_cleansed"),
                         ("price_col", "price_num"), ("min_nights_col", "minimum_nights_num"),
                         ("entire_home_type", "Entire home/apt")):
        settings[key] = _nonempty(k.get(key, default), f"kpi.{key}")
    if settings["entire_home_type"] not in room_types:
        raise ValueError("kpi.entire_home_type must be in qa.allowed_room_types")
    processed = _root_path(paths.get("processed", cfg.get("processed_dir", "data/processed")))
    settings["processed"] = processed
    for key, default in (("listings", "listings_clean.csv"), ("calendar", "calendar_listing_agg.csv"),
                         ("reviews", "reviews_clean.csv"), ("llm", "llm_results.csv")):
        value = k.get(key + "_file")
        settings[key + "_file"] = _root_path(value) if value is not None else processed / default
    return settings


def _require(frame, columns, name):
    """Kiểm tra frame có đủ columns; báo lỗi kèm tên bảng name nếu thiếu cột."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing required columns {', '.join(missing)}")


def _read(file, required):
    """Đọc CSV thành DataFrame dạng chuỗi để giữ nguyên ID; kiểm tra tệp và các cột bắt buộc."""
    if not file.is_file():
        raise FileNotFoundError(f"Required KPI input not found: {file}")
    try:
        frame = pd.read_csv(file, dtype="string", keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise ValueError(f"{file.name}: expected a CSV header") from None
    _require(frame, required, file.name)
    return frame


def _booleans(values, name):
    """Chuẩn hóa Series cờ True/False, x/rỗng và các dạng tương đương thành Series bool."""
    text = values.astype("string").fillna("").str.strip().str.lower()
    true = {"true", "t", "1", "1.0", "x", "yes", "y"}
    false = {"false", "f", "0", "0.0", "", "no", "n"}
    if not text.isin(true | false).all():
        raise ValueError(f"{name}: invalid boolean value")
    return text.isin(true)


def _ids(values, name):
    """Chuẩn hóa Series ID thành chuỗi số nguyên int64 dương; từ chối float hoặc ID sai."""
    def convert(value):
        """Kiểm tra một value là ID số nguyên dương chính xác và trả dạng chuỗi chuẩn."""
        text = str(value).strip()
        if not re.fullmatch(r"[0-9]+", text) or not 0 < int(text) <= 2**63 - 1:
            raise ValueError(f"{name}: IDs must be positive exact int64 integers")
        return str(int(text))
    return values.map(convert).astype("string")


def _numeric(values):
    """Chuyển Series về số; dữ liệu không parse được hoặc không hữu hạn trở thành giá trị thiếu."""
    result = pd.to_numeric(values, errors="coerce")
    return result.where(result.map(lambda v: pd.notna(v) and math.isfinite(float(v))))


def _filter_snapshots(frame, settings, name):
    """Kiểm tra ngày snapshot và trả bản sao các dòng nằm trong danh sách mốc cấu hình."""
    frame = frame.copy()
    frame["snapshot"] = frame["snapshot"].map(lambda v: _iso(v, name + ".snapshot"))
    return frame.loc[frame["snapshot"].isin(settings["snapshots"])].copy()


def _listings(settings):
    """Đọc listings sạch, chọn mốc dữ liệu và loại các bản sao đã được flag_L_DUP đánh dấu.
    Trả DataFrame listing duy nhất theo (snapshot, id), có cột hỗ trợ cho vùng,
    giá, số đêm và loại phòng; thiếu mốc hoặc còn trùng khóa sẽ báo lỗi.
    """
    area, price, nights = (settings[k] for k in ("area_col", "price_col", "min_nights_col"))
    frame = _read(settings["listings_file"], ["id", "snapshot", area, price, "price_valid", "room_type", nights])
    if settings["snapshots"] is None:
        snapshots = sorted({_iso(v, "listings.snapshot") for v in frame["snapshot"]})
        if not snapshots:
            raise ValueError("listings: cannot infer snapshots from an empty table")
        settings["snapshots"] = snapshots
    frame = _filter_snapshots(frame, settings, "listings")
    frame["id"] = _ids(frame["id"], "listings.id")
    if "flag_L_DUP" in frame:
        frame = frame.loc[~_booleans(frame["flag_L_DUP"], "flag_L_DUP")].copy()
    if frame.duplicated(["snapshot", "id"]).any():
        raise ValueError("listings: duplicate (snapshot, id) after excluding flag_L_DUP")
    absent = sorted(set(settings["snapshots"]) - set(frame["snapshot"]))
    if absent:
        raise ValueError(f"listings: missing configured snapshots {', '.join(absent)}")
    frame["_area"] = frame[area].str.strip().fillna("")
    frame["_price_valid"] = _booleans(frame["price_valid"], "listings.price_valid")
    frame["_price"] = _numeric(frame[price])
    frame["_nights"] = _numeric(frame[nights])
    frame["_known_room"] = frame["room_type"].isin(settings["room_types"])
    return frame.sort_values(["snapshot", "id"], kind="stable").reset_index(drop=True)


def _calendar(settings, listings):
    """Đọc bảng tổng hợp lịch và kiểm tra số ngày hữu hạn, nguyên, không âm.
    Trả DataFrame đã nối vùng theo đúng (snapshot, listing_id); khóa trùng,
    listing không khớp hoặc số ngày không khả dụng lớn hơn số ngày quan sát gây lỗi.
    """
    frame = _read(settings["calendar_file"], ["snapshot", "listing_id", "n_days", "unavail_days"])
    frame = _filter_snapshots(frame, settings, "calendar")
    frame["listing_id"] = _ids(frame["listing_id"], "calendar.listing_id")
    if frame.duplicated(["snapshot", "listing_id"]).any():
        raise ValueError("calendar: duplicate (snapshot, listing_id)")
    for col in ("n_days", "unavail_days"):
        values = _numeric(frame[col])
        if values.isna().any() or not values.map(lambda v: v >= 0 and int(v) == v).all():
            raise ValueError(f"calendar.{col}: counts must be finite nonnegative integers")
        frame[col] = values.map(int)
    if (frame["unavail_days"] > frame["n_days"]).any():
        raise ValueError("calendar: unavail_days must not exceed n_days")
    lookup = listings[["snapshot", "id", "_area"]].rename(columns={"id": "listing_id"})
    frame = frame.merge(lookup, on=["snapshot", "listing_id"], how="left", validate="one_to_one", indicator=True)
    if frame["_merge"].ne("both").any():
        raise ValueError("calendar: listing_id missing from canonical listings in the same snapshot")
    return frame.drop(columns="_merge")


def _groups(frame):
    """Lần lượt sinh nhóm toàn mẫu rồi từng vùng của frame, kể cả vùng có tên rỗng."""
    yield "all", "", frame
    for area, group in frame.groupby("_area", sort=True, dropna=False):
        yield "area", str(area), group


def _listing_kpis(settings, listings, calendar):
    """Tính năm KPI listing/lịch theo snapshot cho toàn mẫu và từng vùng.
    Trả bảng dạng dài gồm số listing, giá hợp lệ trung vị, tỷ lệ nguyên căn,
    số đêm tối thiểu trung vị và tổng ngày không khả dụng/tổng ngày quan sát.
    """
    rows = []
    for snapshot in settings["snapshots"]:
        source = listings.loc[listings["snapshot"].eq(snapshot)]
        cal = calendar.loc[calendar["snapshot"].eq(snapshot)]
        for scope, area, group in _groups(source):
            count = len(group)
            base = {"snapshot": snapshot, "scope": scope, "area_col": settings["area_col"],
                    "area": area, "n_listings": count}

            def add(kpi, value, unit, numerator, denominator, valid, empty_status):
                """Thêm một dòng KPI vào rows, kèm mẫu số, cỡ mẫu hợp lệ và trạng thái độ phủ."""
                status = empty_status if value is None else ("ok" if valid == count else "partial_coverage")
                rows.append({**base, "kpi": kpi, "value": value, "unit": unit,
                             "numerator": numerator, "denominator": denominator,
                             "n_valid": valid, "status": status})

            add("listing_count", count, "listings", count, None, count, "no_listings")
            prices = group.loc[group["_price_valid"] & group["_price"].gt(0), "_price"].dropna()
            empty_price = ("not_validated_for_snapshot" if snapshot != settings["snapshots"][-1]
                           and not group["_price_valid"].any() else "no_valid_price")
            add("median_valid_price", float(prices.median()) if len(prices) else None,
                settings["currency"], None, len(prices), len(prices), empty_price)
            known = group.loc[group["_known_room"]]
            numerator = int(known["room_type"].eq(settings["entire_home_type"]).sum())
            add("entire_home_share", numerator / len(known) if len(known) else None,
                "share", numerator, len(known), len(known), "no_valid_room_type")
            nights = group.loc[group["_nights"].gt(0), "_nights"].dropna()
            add("median_min_nights", float(nights.median()) if len(nights) else None,
                "nights", None, len(nights), len(nights), "no_valid_minimum_nights")
            observed = cal if scope == "all" else cal.loc[cal["_area"].eq(area)]
            denominator = sum(int(v) for v in observed["n_days"])
            numerator = sum(int(v) for v in observed["unavail_days"])
            n_valid = int(observed["n_days"].gt(0).sum())
            add("pooled_unavailable_day_rate", numerator / denominator if denominator else None,
                "share", numerator, denominator, n_valid, "no_observed_calendar_days")
    return pd.DataFrame(rows, columns=LISTING_COLUMNS)


def _reviews(settings, listings):
    """Đọc review và kết quả LLM, giữ mỗi review đủ điều kiện một lần ở mốc sớm nhất.
    Nối vùng theo snapshot nguồn của review, kiểm tra ID/nhãn và độ đầy đủ của
    mẫu được đánh dấu; trả bảng ghép, giữ review chưa phân loại ở trạng thái chưa biết.
    """
    frame = _read(settings["reviews_file"], ["id", "listing_id", "date", "snapshot", "use_for_llm"])
    frame = _filter_snapshots(frame, settings, "reviews")
    for col in ("id", "listing_id"):
        frame[col] = _ids(frame[col], "reviews." + col)
    frame["date"] = frame["date"].str.strip()
    if (frame.groupby("id")[["listing_id", "date"]].nunique(dropna=False) > 1).any().any():
        raise ValueError("reviews: an ID has conflicting listing_id or date across snapshots")
    frame["_sample"] = (_booleans(frame["llm_sample"], "reviews.llm_sample")
                        if "llm_sample" in frame else False)
    frame["_sample"] = frame.groupby("id")["_sample"].transform("any")
    eligible = frame.loc[_booleans(frame["use_for_llm"], "reviews.use_for_llm")].copy()
    for col in ("comments_clean", "comments"):
        if col in eligible and eligible.groupby("id")[col].nunique(dropna=False).gt(1).any():
            raise ValueError(f"reviews: conflicting {col} for repeated eligible review IDs")
    eligible = eligible.sort_values(["snapshot", "id"], kind="stable").drop_duplicates("id")
    eligible["date"] = eligible["date"].map(lambda v: _iso(v, "eligible review date"))
    if eligible["date"].gt(eligible["snapshot"]).any():
        raise ValueError("reviews: eligible review date is later than its canonical snapshot")
    lookup = listings[["snapshot", "id", "_area"]].rename(columns={"id": "listing_id"})
    eligible = eligible.merge(lookup, on=["snapshot", "listing_id"], how="left",
                              validate="many_to_one", indicator=True)
    if eligible["_merge"].ne("both").any():
        raise ValueError("reviews: eligible review cannot join listing in its canonical snapshot")
    eligible = eligible.drop(columns="_merge").rename(columns={"id": "review_id"})
    results = _read(settings["llm_file"], ["review_id", "sentiment", "aspects_positive", "aspects_negative"])
    results["review_id"] = _ids(results["review_id"], "llm.review_id")
    if results["review_id"].duplicated().any():
        raise ValueError("LLM results: duplicate review_id")
    if not results["sentiment"].isin(SENTIMENTS).all():
        raise ValueError("LLM results: sentiment must be positive, negative, or mixed")
    allowed = set(settings["aspects"])

    def aspects(value):
        """Tách chuỗi khía cạnh thành frozenset theo token '|'; từ chối token rỗng hoặc ngoài cấu hình."""
        tokens = str(value).split("|") if value else []
        if any(token not in allowed for token in tokens):
            raise ValueError("LLM results: unknown or empty aspect token")
        return frozenset(tokens)

    for col in ("aspects_positive", "aspects_negative"):
        results[col] = results[col].map(aspects)
    known = set(eligible["review_id"])
    if not set(results["review_id"]).issubset(known):
        raise ValueError("LLM results: review_id not found among canonical eligible reviews")
    expected = set(eligible.loc[eligible["_sample"], "review_id"])
    missing = expected - set(results["review_id"])
    if missing:
        raise ValueError(f"LLM results incomplete: {len(missing)} flagged eligible review IDs are missing")
    results = results[["review_id", "sentiment", "aspects_positive", "aspects_negative"]]
    joined = eligible.merge(results, on="review_id", how="left", validate="one_to_one", indicator=True)
    joined["_classified"] = joined["_merge"].eq("both")
    joined["_month"] = joined["date"].str.slice(0, 7)
    return joined.drop(columns="_merge")


def _review_kpis(settings, reviews):
    """Tính tỷ lệ review tích cực và tỷ lệ nhắc tiêu cực từng khía cạnh.
    Trả hai DataFrame theo toàn bộ lịch sử hoặc tháng viết review, cho toàn mẫu
    và từng vùng; mẫu số chỉ gồm review đã phân loại, kèm độ phủ trên review đủ điều kiện.
    """
    review_rows, aspect_rows = [], []
    periods = [("all", "", reviews)]
    periods.extend(("review_month", str(month), group)
                   for month, group in reviews.groupby("_month", sort=True))
    for period_type, period, source in periods:
        for scope, area, group in _groups(source):
            if group.empty:
                continue
            eligible = len(group)
            classified = group.loc[group["_classified"]]
            count = len(classified)
            base = {"period_type": period_type, "period": period, "scope": scope,
                    "area_col": settings["area_col"], "area": area, "unit": "share",
                    "denominator": count, "n_eligible_reviews": eligible,
                    "n_classified_reviews": count, "n_listings": int(group["listing_id"].nunique()),
                    "coverage": count / eligible,
                    "status": "no_classified_reviews" if not count else
                    ("ok" if count == eligible else "partial_coverage")}
            positive = int(classified["sentiment"].eq("positive").sum())
            review_rows.append({**base, "kpi": "positive_review_share", "numerator": positive,
                                "value": positive / count if count else None})
            for aspect in settings["aspects"]:
                negative = sum(aspect in labels for labels in classified["aspects_negative"])
                aspect_rows.append({**base, "kpi": "negative_aspect_share", "aspect": aspect,
                                    "numerator": negative, "value": negative / count if count else None})
    return (pd.DataFrame(review_rows, columns=REVIEW_COLUMNS),
            pd.DataFrame(aspect_rows, columns=REVIEW_COLUMNS + ["aspect"]))


def _definitions(settings):
    """Tạo DataFrame mô tả bảy KPI: tên, mục tiêu, công thức, đơn vị, chiều phân tích và giới hạn."""
    geographic = "snapshot; scope; " + settings["area_col"]
    historical = "review date month or all historical dates; scope; " + settings["area_col"]
    sample = "Assigned listing/review sample only; no citywide representativeness is assumed. "
    data = [
        ("listing_count", "Observed listings", "Compare observed competitive supply",
         "count(distinct canonical listing ID)", "listings", geographic,
         "Exclude flag_L_DUP; other listing QA flags remain unless needed by the individual KPI."),
        ("median_valid_price", "Median validated advertised price", "Compare sample price positioning",
         "median(configured price_col where price_valid is true and price is finite and positive)",
         settings["currency"], geographic,
         "QA currently validates the latest snapshot only. Unsupported earlier prices remain null; no price history is inferred."),
        ("entire_home_share", "Entire home share", "Understand the observed accommodation mix",
         "count(room_type = configured entire_home_type) / count(allowed room_type)", "share", geographic,
         "Unknown room types are excluded from this denominator only; report n_valid."),
        ("median_min_nights", "Median minimum stay", "Compare advertised stay restrictions",
         "median(finite positive configured min_nights_col)", "nights", geographic,
         "Extreme minimum-night flags are retained; no short-stay cap is imposed."),
        ("pooled_unavailable_day_rate", "Observed unavailable day share", "Inspect calendar availability restrictions",
         "sum(unavail_days) / sum(n_days)", "share", geographic,
         "Observed valid listing-days only, not a mean of listing rates. Unavailability includes blocked dates and is not occupancy, bookings, or revenue. Window and exposure may differ by snapshot."),
        ("positive_review_share", "Positive historical review share", "Inspect classified guest feedback",
         "count(classified review IDs with sentiment positive) / count(classified review IDs)", "share", historical,
         "Each review ID counts once. Review months describe writing dates, not snapshot trends. LLM labels are model judgments and the labeled subset may be biased; missing labels remain unknown."),
        ("negative_aspect_share", "Negative aspect mention share", "Identify guest concerns in classified reviews",
         "count(classified review IDs negatively mentioning the aspect) / count(classified review IDs)", "share", historical + "; aspect",
         "Exact aspect tokens; empty aspects mean no detected mention. Aspects can co-occur and the same aspect can have both polarities. Shares need not sum to one; unclassified reviews are excluded."),
    ]
    return pd.DataFrame([dict(zip(DEFINITION_COLUMNS, (*row[:6], sample + row[6]))) for row in data],
                        columns=DEFINITION_COLUMNS)


def _sha256(file):
    """Đọc tệp theo từng khối và trả mã SHA-256, không cần nạp cả tệp vào bộ nhớ."""
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(settings, listings, calendar, reviews, table, inputs, skip_llm):
    """Tổng hợp số lượng, độ phủ, chính sách tính và cảnh báo thành dict cho báo cáo JSON.
    Đọc các tệp đầu vào để lấy mã SHA-256; không suy rộng mẫu phân công thành
    thống kê đại diện cho cả thành phố.
    """
    price_coverage, calendar_coverage = {}, {}
    for snapshot in settings["snapshots"]:
        rows = table.loc[table["snapshot"].eq(snapshot) & table["scope"].eq("all")].set_index("kpi")
        for metric, target in (("median_valid_price", price_coverage),
                               ("pooled_unavailable_day_rate", calendar_coverage)):
            row = rows.loc[metric]
            target[snapshot] = {"n_listings": int(row["n_listings"]), "n_valid": int(row["n_valid"]),
                                "coverage": int(row["n_valid"]) / int(row["n_listings"]), "status": row["status"]}
    warnings = ["All KPIs describe assigned samples, without an assumption of citywide representativeness.",
                "Calendar unavailability includes owner blocks and does not establish occupancy, bookings, or revenue.",
                "Snapshot calendar windows differ and can overlap; retain snapshot as a dimension when comparing them."]
    if any(v["coverage"] < 1 for v in price_coverage.values()):
        warnings.append("Price coverage is incomplete; price_valid is the sole QA eligibility mask and missing validated prices are not zero.")
    invalid_flagged_prices = int((listings["_price_valid"] &
                                  (listings["_price"].isna() | listings["_price"].le(0))).sum())
    if invalid_flagged_prices:
        warnings.append(f"{invalid_flagged_prices} rows marked price_valid have nonfinite or nonpositive prices and were excluded.")
    if any(v["coverage"] < 1 for v in calendar_coverage.values()):
        warnings.append("Some listings have no observed calendar days; unavailable-day rates use observed days only.")
    if listings["_area"].eq("").any():
        warnings.append("Missing geographic values are retained in an explicit blank-area group.")
    review_summary = {"skipped": skip_llm}
    if skip_llm:
        warnings.append("LLM review KPIs were explicitly skipped; review and aspect CSV files contain headers only.")
    else:
        classified = reviews.loc[reviews["_classified"]]
        eligible_count, classified_count = len(reviews), len(classified)
        review_summary.update({"n_eligible_reviews": eligible_count,
                               "n_flagged_sample_eligible": int(reviews["_sample"].sum()),
                               "n_classified_reviews": classified_count,
                               "n_eligible_listings": int(reviews["listing_id"].nunique()),
                               "n_classified_listings": int(classified["listing_id"].nunique()),
                               "coverage": classified_count / eligible_count if eligible_count else None,
                               "eligible_date_min": reviews["date"].min() if eligible_count else None,
                               "eligible_date_max": reviews["date"].max() if eligible_count else None,
                               "classified_date_min": classified["date"].min() if classified_count else None,
                               "classified_date_max": classified["date"].max() if classified_count else None})
        warnings.extend(["Review periods use historical writing dates, not snapshot dates or contemporary sentiment trends.",
                         "LLM shares describe classified reviews only; sample assignment, historical coverage, and model errors can bias them."])
        if eligible_count and classified_count < eligible_count:
            warnings.append("Unclassified eligible reviews remain unknown; consult each group's classification coverage.")
    return {"schema_version": 1, "city": settings["city"], "currency": settings["currency"],
            "stakeholder": settings["stakeholder"], "snapshots": settings["snapshots"],
            "counts": {"canonical_listing_rows": len(listings), "unique_listing_ids": int(listings["id"].nunique()),
                       "calendar_listing_rows": len(calendar), "n_kpi_concepts": 7,
                       "n_price_valid_but_unusable": invalid_flagged_prices,
                       "n_computed_kpi_concepts": 5 if skip_llm else 7},
            "coverage": {"price_by_snapshot": price_coverage, "calendar_by_snapshot": calendar_coverage,
                         "reviews": review_summary},
            "policies": {"area_col": settings["area_col"], "price_col": settings["price_col"],
                         "price_eligibility": "price_valid AND finite positive configured price_col; no inferred historical validation",
                         "min_nights_col": settings["min_nights_col"], "minimum_night_extremes": "retained",
                         "room_types": settings["room_types"], "entire_home_type": settings["entire_home_type"],
                         "aspects": settings["aspects"], "ratio_scale": "0..1",
                         "calendar_denominator": "sum of observed valid listing-days, separately by snapshot",
                         "review_time": "review writing month; all historical dates retained",
                         "review_area": "canonical eligible review snapshot joined to the same listing snapshot",
                         "classification_completeness": "all flagged eligible sample IDs required; otherwise any eligible subset is accepted",
                         "review_coverage": "classified canonical eligible review IDs / all canonical eligible review IDs",
                         "review_n_listings": "distinct listing IDs among canonical eligible reviews in each group",
                         "zero_denominator": "null value with explicit status; never fabricated zero",
                         "median_denominator": "number of valid observations; numerator is blank",
                         "output_checksums": "four CSV files only; summary excludes its own checksum"},
            "inputs": {key: {"path": str(file), "sha256": _sha256(file)} for key, file in inputs.items()},
            "warnings": warnings, "output_files": OUTPUT_NAMES}


def run(cfg: dict, output_dir=None, skip_llm=False, **_):
    """Chạy bước KPI theo cfg, ghi năm tệp kết quả và trả dict tóm tắt.
    Kiểm tra dữ liệu trước khi ghi; skip_llm=True chỉ tính listing/lịch và ghi
    header cho hai bảng review. output_dir mặc định là thư mục processed.
    Từng tệp được thay nguyên tử; cả năm tệp không phải một giao dịch chung.
    """
    settings = _settings(cfg)
    if not isinstance(skip_llm, bool):
        raise ValueError("skip_llm must be a boolean")
    destination = _root_path(output_dir) if output_dir is not None else settings["processed"]
    keys = ["listings", "calendar"] + ([] if skip_llm else ["reviews", "llm"])
    inputs = {key: settings[key + "_file"].resolve() for key in keys}
    all_inputs = [settings[key + "_file"].resolve() for key in ("listings", "calendar", "reviews", "llm")]
    for name in OUTPUT_NAMES:
        target = (destination / name).resolve()
        if any(target == source or (target.exists() and source.exists() and target.samefile(source))
               for source in all_inputs):
            raise ValueError(f"Output would overwrite an input file: {target}")
    listings = _listings(settings)
    calendar = _calendar(settings, listings)
    listing_table = _listing_kpis(settings, listings, calendar)
    reviews = None if skip_llm else _reviews(settings, listings)
    if skip_llm:
        review_table = pd.DataFrame(columns=REVIEW_COLUMNS)
        aspect_table = pd.DataFrame(columns=REVIEW_COLUMNS + ["aspect"])
    else:
        review_table, aspect_table = _review_kpis(settings, reviews)
    definitions = _definitions(settings)
    summary = _summary(settings, listings, calendar, reviews, listing_table, inputs, skip_llm)
    tables = dict(zip(OUTPUT_NAMES[:4], [listing_table, review_table, aspect_table, definitions]))
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".kpi-stage-", dir=destination) as temporary:
        staging = Path(temporary)
        for name, table in tables.items():
            table.to_csv(staging / name, index=False, encoding="utf-8", lineterminator="\n", na_rep="")
        summary["output_checksums"] = {name: _sha256(staging / name) for name in tables}
        (staging / OUTPUT_NAMES[-1]).write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        for name in OUTPUT_NAMES:
            (staging / name).replace(destination / name)
    LOG.info("KPI: %d listing rows, %d review rows, %d aspect rows -> %s",
             len(listing_table), len(review_table), len(aspect_table), destination)
    return summary


def main():
    """Đọc --config/--output-dir/--skip-llm, nạp YAML và gọi run; lỗi được báo ngắn gọn qua CLI."""
    import yaml

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/Buenos_Aires.yaml")
    parser.add_argument("--output-dir", help="Destination for the five KPI outputs (relative to repository root)")
    parser.add_argument("--skip-llm", action="store_true", help="Compute listing/calendar KPIs; write review headers only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        with _root_path(args.config).open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
        summary = run(cfg, output_dir=args.output_dir, skip_llm=args.skip_llm)
        LOG.info("Coverage: %s", json.dumps(summary["coverage"], ensure_ascii=False, sort_keys=True))
    except (ValueError, FileNotFoundError, KeyError, OSError, yaml.YAMLError, pd.errors.ParserError) as exc:
        parser.exit(1, f"KPI: {exc}\n")


if __name__ == "__main__":
    main()
