"""Sáu hình KPI Airbnb ngoại tuyến, có kiểm tra dữ liệu và xuất PNG/SVG.

Chỉ đọc bảng QA/KPI và báo cáo đã có; không gọi mạng, mô hình hay đọc .env.
Chạy: python src/viz.py --config configs/Buenos_Aires.yaml --skip-llm
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
STEMS = ("01_listing_map", "02_prices_by_area", "03_snapshot_trends",
         "04_review_sentiment_yearly", "05_negative_aspects", "06_model_comparison")
DEFAULTS = {"listings": "listings_clean.csv", "kpi_listings": "kpi_listings.csv",
            "kpi_reviews": "kpi_reviews.csv", "kpi_aspects": "kpi_aspects.csv",
            "evaluation": "evaluation_metrics.csv", "evaluation_summary": "evaluation_summary.json",
            "kpi_summary": "kpi_summary.json"}
BLUE, TEAL, GRAY = "#245a81", "#258979", "#a6adb4"


def _mapping(value, name):
    """Trả dict cấu hình; chấp nhận null và từ chối kiểu dữ liệu khác."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected a mapping or null")
    return value


def _text(value, name):
    """Kiểm tra và trả chuỗi cấu hình không rỗng đã bỏ khoảng trắng ở hai đầu."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name}: expected a nonempty string")
    return value.strip()


def _path(value):
    """Giải đường dẫn tương đối theo gốc dự án chứa tệp nguồn này."""
    value = Path(value)
    return (value if value.is_absolute() else REPO_ROOT / value).resolve()


def _iso(value, name):
    """Chuẩn hóa ngày ISO đầy đủ; từ chối ngày hoặc định dạng không hợp lệ."""
    value = str(value).strip()
    try:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError
        return date.fromisoformat(value).isoformat()
    except ValueError:
        raise ValueError(f"{name}: invalid ISO date {value!r}") from None


def _settings(cfg, output_dir):
    """Đọc đường dẫn, tên cột, mốc dữ liệu và tham số vẽ từ cấu hình."""
    cfg = _mapping(cfg, "config")
    v, k, paths = (_mapping(cfg.get(key), key) for key in ("viz", "kpi", "paths"))
    _mapping(cfg.get("llm"), "llm")
    settings = {"city": str(cfg.get("city", "")), "currency": _text(cfg.get("currency"), "currency")}
    for key, default in (("top_n", 12), ("min_group_n", 5), ("min_review_n", 20), ("dpi", 180)):
        value = v.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"viz.{key}: expected a positive integer")
        settings[key] = value
    formats = v.get("formats", ["png", "svg"])
    if (not isinstance(formats, list) or not formats or len(set(formats)) != len(formats)
            or any(value not in {"png", "svg"} for value in formats)):
        raise ValueError("viz.formats: expected unique png/svg formats")
    settings["formats"] = formats
    settings["font_family"] = v.get("font_family", "DejaVu Sans")
    if not (isinstance(settings["font_family"], str) and settings["font_family"].strip() or
            isinstance(settings["font_family"], list) and settings["font_family"] and
            all(isinstance(value, str) and value.strip() for value in settings["font_family"])):
        raise ValueError("viz.font_family: expected a font name or nonempty list")
    for key, default in (("price_col", "price_num"), ("area_col", "neighbourhood_cleansed")):
        settings[key] = _text(k.get(key, default), f"kpi.{key}")
    for key, default in (("latitude_col", "latitude_num"), ("longitude_col", "longitude_num")):
        settings[key] = _text(v.get(key, default), f"viz.{key}")
    for key, default in (("processed", "data/processed"), ("reports", "reports"),
                         ("raw", "data/raw"), ("figures", "figures")):
        settings[key] = _path(paths.get(key, cfg.get(key + "_dir", default)))
    settings["output"] = _path(output_dir) if output_dir is not None else settings["figures"]
    for key, filename in DEFAULTS.items():
        directory = settings["reports"] if key.startswith("evaluation") else settings["processed"]
        override = v.get(key + "_file")
        settings[key + "_file"] = _path(override) if override is not None else directory / filename
    settings["geojson_file"] = _path(v["geojson_file"]) if v.get("geojson_file") is not None else None
    snapshots = cfg.get("snapshots")
    if snapshots is not None:
        if not isinstance(snapshots, (list, tuple)) or not snapshots:
            raise ValueError("snapshots: expected a nonempty list")
        snapshots = [_iso(value, "snapshots") for value in snapshots]
        if len(set(snapshots)) != len(snapshots):
            raise ValueError("snapshots: duplicate dates")
    settings["snapshots"] = sorted(snapshots) if snapshots is not None else None
    return settings


def _sha256(path):
    """Tính SHA-256 theo khối để ghi xuất xứ mà không sửa tệp đầu vào."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _protect_inputs(settings):
    """Từ chối đầu ra trùng đường dẫn hoặc hard link với bất kỳ đầu vào đã cấu hình."""
    targets = [settings["output"] / (stem + "." + fmt) for stem in STEMS for fmt in settings["formats"]]
    targets.append(settings["output"] / "viz_summary.json")
    sources = [value for key, value in settings.items() if key.endswith("_file") and value is not None]
    sources.append(settings["raw"] / settings["latest"] / "neighbourhoods.geojson")
    for source in sources:
        for target in targets:
            if source == target or source.exists() and target.exists() and source.samefile(target):
                raise ValueError(f"output/input collision: {target}")


def _read(settings, key, inputs, required=False, json_file=False):
    """Đọc CSV giữ nguyên chuỗi hoặc JSON; ghi đường dẫn và checksum tệp được dùng."""
    path = settings[key + "_file"]
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"{key}: input not found: {path}")
        return {} if json_file else None
    inputs[key] = {"path": str(path), "sha256": _sha256(path)}
    if json_file:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return _mapping(value, path.name)
    try:
        return pd.read_csv(path, dtype="string", keep_default_na=False)
    except pd.errors.EmptyDataError:
        raise ValueError(f"{path.name}: expected a CSV header") from None


def _require(frame, columns, name):
    """Báo rõ tên bảng và các cột bắt buộc bị thiếu."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name}: missing columns: {', '.join(missing)}")


def _numbers(frame, columns, name, counts=()):
    """Đổi số KPI nghiêm ngặt; cho phép ô rỗng nhưng cấm số vô hạn và số đếm sai."""
    for col in columns:
        text = frame[col].str.strip()
        values = pd.to_numeric(text.replace("", pd.NA), errors="coerce").astype(float)
        invalid = text.ne("") & (values.isna() | ~np.isfinite(values))
        if invalid.any():
            raise ValueError(f"{name}.{col}: invalid finite numeric value")
        if col in counts and ((values.dropna() < 0).any() or (values.dropna() % 1 != 0).any()):
            raise ValueError(f"{name}.{col}: counts must be nonnegative integers")
        frame[col] = values


def _unique(frame, columns, name):
    """Từ chối khóa trùng để tránh cộng lặp hoặc tự chọn một dòng không xác định."""
    if frame.duplicated(columns).any():
        raise ValueError(f"{name}: duplicate key ({', '.join(columns)})")


def _ratios(frame, name):
    """Kiểm tra tỷ lệ 0..1 và tính nhất quán giữa giá trị, tử số và mẫu số."""
    for col in ("value", "coverage"):
        if col in frame:
            values = frame[col].dropna()
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"{name}.{col}: ratios must be in [0, 1]")
    num, den, value = (frame[col] for col in ("numerator", "denominator", "value"))
    if ((num > den) & num.notna() & den.notna()).any():
        raise ValueError(f"{name}: numerator exceeds denominator")
    if (value.notna() & (den.isna() | num.isna() | den.le(0))).any():
        raise ValueError(f"{name}: defined ratio requires a positive denominator and numerator")
    expected = num / den.where(den > 0)
    if (value.notna() & ~np.isclose(value, expected, rtol=1e-8, atol=1e-10, equal_nan=True)).any():
        raise ValueError(f"{name}: value does not match numerator/denominator")
    if (den.gt(0) & num.notna() & value.isna()).any():
        raise ValueError(f"{name}: ratio missing despite available numerator/denominator")


def _kpi_table(frame, key, settings):
    """Kiểm tra bảng KPI, khóa chiều phân tích, số đếm và các tỷ lệ trước khi vẽ."""
    if frame is None:
        return None
    dimensions = (["snapshot"] if key == "kpi_listings" else ["period_type", "period"])
    columns = dimensions + ["scope", "area_col", "area", "kpi", "value", "unit", "numerator", "denominator"]
    counts = ["numerator", "denominator", "n_listings"]
    counts += ["n_valid"] if key == "kpi_listings" else ["n_eligible_reviews", "n_classified_reviews"]
    if key == "kpi_aspects":
        columns += ["aspect"]
    _require(frame, columns + counts + ([] if key == "kpi_listings" else ["coverage"]), key)
    numeric = list(dict.fromkeys(["value"] + counts + ([] if key == "kpi_listings" else ["coverage"])))
    _numbers(frame, numeric, key, counts)
    if not frame["scope"].isin(["all", "area"]).all():
        raise ValueError(f"{key}: scope must be all or area")
    if (frame["scope"].eq("all") & frame["area"].ne("")).any():
        raise ValueError(f"{key}: all-scope area must be empty")
    if (frame["scope"].eq("area") & frame["area"].str.strip().eq("")).any():
        raise ValueError(f"{key}: area-scope area must be nonempty")
    if frame["kpi"].str.strip().eq("").any():
        raise ValueError(f"{key}: empty KPI name")
    if key == "kpi_listings":
        frame["snapshot"] = frame["snapshot"].map(lambda value: _iso(value, key + ".snapshot"))
        prices = frame.loc[frame["kpi"].eq("median_valid_price")]
        if prices["unit"].ne(settings["currency"]).any():
            raise ValueError(f"{key}: median_valid_price unit must match configured currency")
        if prices["value"].dropna().le(0).any():
            raise ValueError(f"{key}: median_valid_price must be positive when defined")
        ratios = frame.loc[frame["kpi"].isin(["entire_home_share", "pooled_unavailable_day_rate"])]
        if (frame["n_valid"] > frame["n_listings"]).any():
            raise ValueError(f"{key}: n_valid exceeds n_listings")
    else:
        if not frame["period_type"].isin(["all", "review_month"]).all():
            raise ValueError(f"{key}: unsupported period_type")
        if (frame["period_type"].eq("all") & frame["period"].ne("")).any():
            raise ValueError(f"{key}: all-history period must be empty")
        for value in frame.loc[frame["period_type"].eq("review_month"), "period"]:
            _iso(str(value) + "-01", key + ".period")
        if (frame["n_classified_reviews"] > frame["n_eligible_reviews"]).any():
            raise ValueError(f"{key}: classified count exceeds eligible count")
        if not np.allclose(frame["denominator"], frame["n_classified_reviews"], equal_nan=True):
            raise ValueError(f"{key}: denominator must equal n_classified_reviews")
        if key == "kpi_aspects" and frame["aspect"].str.strip().eq("").any():
            raise ValueError(f"{key}: empty aspect")
        ratios = frame
    _unique(frame, dimensions + ["scope", "area_col", "area", "kpi"] + (["aspect"] if key == "kpi_aspects" else []), key)
    _ratios(ratios, key)
    return frame


def _booleans(values, name):
    """Chuẩn hóa cờ QA thường gặp thành bool và từ chối giá trị mơ hồ."""
    text = values.fillna("").str.strip().str.lower()
    true, false = {"true", "t", "1", "1.0", "x", "yes", "y"}, {"false", "f", "0", "0.0", "", "no", "n"}
    if not text.isin(true | false).all():
        raise ValueError(f"{name}: invalid boolean value")
    return text.isin(true)


def _map_data(frame, settings):
    """Lấy tọa độ hợp lệ của mốc mới nhất và giá dương được QA xác nhận."""
    required = ["id", "snapshot", settings["latitude_col"], settings["longitude_col"]]
    if frame is None or not set(required).issubset(frame.columns):
        return None
    frame = frame.copy()
    frame["snapshot"] = frame["snapshot"].map(lambda value: _iso(value, "listings.snapshot"))
    frame = frame.loc[frame["snapshot"].eq(settings["latest"])].copy()
    input_count = len(frame)
    frame["id"] = frame["id"].str.strip()
    for value in frame["id"]:
        if not re.fullmatch(r"[0-9]+", value) or not 0 < int(value) <= 2**63 - 1:
            raise ValueError("listings.id: IDs must be exact positive int64 integers")
    frame["id"] = frame["id"].map(lambda value: str(int(value)))
    for flag in ("flag_L_DUP", "flag_L_COORD"):
        frame[flag] = _booleans(frame[flag], "listings." + flag) if flag in frame else False
    frame = frame.loc[~frame["flag_L_DUP"]].copy()
    _unique(frame, ["snapshot", "id"], "listings")
    for destination, source in (("_lat", settings["latitude_col"]), ("_lon", settings["longitude_col"]),
                                ("_price", settings["price_col"])):
        frame[destination] = pd.to_numeric(frame[source], errors="coerce").astype(float) if source in frame else np.nan
    valid = (np.isfinite(frame["_lat"]) & np.isfinite(frame["_lon"]) & frame["_lat"].between(-90, 90)
             & frame["_lon"].between(-180, 180) & ~frame["flag_L_COORD"])
    frame = frame.loc[valid].copy()
    frame["_price_valid"] = _booleans(frame["price_valid"], "listings.price_valid") if "price_valid" in frame else False
    frame["_price_valid"] &= np.isfinite(frame["_price"]) & frame["_price"].gt(0)
    frame.attrs["n_input_listings"] = input_count
    return frame


def _outlines(settings, inputs):
    """Đọc vòng biên GeoJSON Polygon/MultiPolygon cục bộ bằng thư viện chuẩn."""
    path = settings["geojson_file"] or settings["raw"] / settings["latest"] / "neighbourhoods.geojson"
    if not path.is_file():
        return []
    inputs["geojson"] = {"path": str(path), "sha256": _sha256(path)}
    document = _mapping(json.loads(path.read_text(encoding="utf-8-sig")), "geojson")
    features = document.get("features", []) if document.get("type") == "FeatureCollection" else [document]
    rings = []
    for feature in features:
        geometry = feature.get("geometry") if feature.get("type") == "Feature" else feature
        if not geometry:
            continue
        polygons = ([geometry.get("coordinates", [])] if geometry.get("type") == "Polygon" else
                    geometry.get("coordinates", []) if geometry.get("type") == "MultiPolygon" else [])
        for polygon in polygons:
            for ring in polygon:
                points = np.asarray([point[:2] for point in ring], dtype=float)
                if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                    raise ValueError("geojson: invalid polygon coordinates")
                if (np.abs(points[:, 0]) > 180).any() or (np.abs(points[:, 1]) > 90).any():
                    raise ValueError("geojson: coordinates outside longitude/latitude bounds")
                rings.append(points)
    return rings


def _prepare(settings, skip_llm):
    """Đọc và xác thực toàn bộ đầu vào trước khi tạo bất kỳ tệp hình nào."""
    inputs, warnings, data = {}, [], {}
    for key in ("kpi_listings", "kpi_reviews", "kpi_aspects"):
        frame = None if skip_llm and key != "kpi_listings" else _read(settings, key, inputs, key == "kpi_listings")
        data[key] = _kpi_table(frame, key, settings)
    listing = data["kpi_listings"]
    available = sorted(listing["snapshot"].unique())
    if settings["snapshots"] is None:
        settings["snapshots"] = available
    missing = set(settings["snapshots"]) - set(available)
    if missing:
        raise ValueError("kpi_listings: configured snapshots missing: " + ", ".join(sorted(missing)))
    if not settings["snapshots"]:
        raise ValueError("kpi_listings: cannot infer snapshots from an empty table")
    settings["latest"] = max(settings["snapshots"])
    _protect_inputs(settings)
    data["kpi_listings"] = listing.loc[listing["snapshot"].isin(settings["snapshots"])].copy()
    data["map"] = _map_data(_read(settings, "listings", inputs), settings)
    data["outlines"] = _outlines(settings, inputs)
    data["kpi_summary"] = _read(settings, "kpi_summary", inputs, json_file=True)
    coverage = _mapping(data["kpi_summary"].get("coverage"), "kpi_summary.coverage")
    coverage["reviews"] = _mapping(coverage.get("reviews"), "kpi_summary.coverage.reviews")
    data["kpi_summary"]["coverage"] = coverage
    data["kpi_summary"]["inputs"] = _mapping(data["kpi_summary"].get("inputs"), "kpi_summary.inputs")
    checksums = _mapping(data["kpi_summary"].get("output_checksums"), "kpi_summary.output_checksums")
    for key in ("kpi_listings", "kpi_reviews", "kpi_aspects"):
        if key in inputs:
            expected = checksums.get(settings[key + "_file"].name, checksums.get(DEFAULTS[key]))
            if expected is not None and expected != inputs[key]["sha256"]:
                raise ValueError(f"kpi_summary: checksum mismatch for {key}")
    evaluation = None if skip_llm else _read(settings, "evaluation", inputs)
    data["evaluation_summary"] = {} if skip_llm else _read(settings, "evaluation_summary", inputs, json_file=True)
    data["evaluation_note"] = ""
    if evaluation is not None:
        _require(evaluation, ["model", "task", "n_reviews", "accuracy", "f1_macro"], "evaluation")
        _unique(evaluation, ["model", "task"], "evaluation")
        _numbers(evaluation, ["n_reviews", "accuracy", "f1_macro"], "evaluation", ["n_reviews"])
        if evaluation["model"].str.strip().eq("").any():
            raise ValueError("evaluation: empty model name")
        for col in ("accuracy", "f1_macro"):
            if not evaluation[col].dropna().between(0, 1).all():
                raise ValueError(f"evaluation.{col}: ratios must be in [0, 1]")
        provenance = _mapping(data["kpi_summary"].get("inputs"), "kpi_summary.inputs")
        current = _mapping(provenance.get("llm"), "kpi_summary.inputs.llm")
        references = data["evaluation_summary"].get("inputs", [])
        references = list(references.values()) if isinstance(references, dict) else references
        if references is None:
            references = []
        if not isinstance(references, list) or any(not isinstance(item, dict) for item in references):
            raise ValueError("evaluation_summary.inputs: expected input records")
        if current.get("sha256"):
            matches = [item for item in references if isinstance(item, dict) and
                       Path(str(item.get("path", "")).replace("\\", "/")).name ==
                       Path(str(current.get("path", "llm_results.csv")).replace("\\", "/")).name]
            candidates = matches or references
            if not candidates or not any(item.get("sha256") == current["sha256"] for item in candidates):
                evaluation = None
                data["evaluation_note"] = "Không xác nhận được checksum LLM tương thích: bỏ qua báo cáo đánh giá cũ hoặc thiếu xuất xứ."
                warnings.append(data["evaluation_note"])
    data["evaluation"] = evaluation
    aspect = data["kpi_aspects"]
    if aspect is not None:
        selected = aspect.loc[aspect["scope"].eq("all") & aspect["period_type"].eq("all") &
                              aspect["kpi"].eq("negative_aspect_share") & aspect["area_col"].eq(settings["area_col"])]
        if selected["denominator"].nunique(dropna=False) > 1:
            raise ValueError("kpi_aspects: all-scope aspects must share one classified denominator")
    return data, inputs, warnings


def _figure(settings, title, subtitle="", nrows=1):
    """Tạo hình thống nhất với tiêu đề tiếng Việt và chừa khoảng cho chú thích."""
    fig, axes = plt.subplots(nrows=nrows, figsize=(11, 6.6), squeeze=False)
    fig.suptitle(f"{title} · {settings['city']}" if settings["city"] else title,
                 x=0.08, y=0.97, ha="left", fontsize=17, fontweight="bold", color="#20364a")
    fig.text(0.08, 0.913, subtitle, ha="left", va="top", fontsize=10, color="#526170")
    fig.subplots_adjust(left=0.12, right=0.94, bottom=0.20, top=0.84, hspace=0.68)
    for ax in axes.flat:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#ccd2d8")
        ax.spines["bottom"].set_color("#ccd2d8")
        ax.tick_params(labelsize=10)
        ax.set_axisbelow(True)
    return fig, list(axes.flat)


def _footnote(fig, text):
    """Đặt chú thích nguồn và giới hạn mẫu dưới vùng vẽ, tự ngắt dòng."""
    width = max(72, min(132, int(fig.get_figwidth() * 12)))
    fig.text(0.08, 0.055, textwrap.fill(text, width), fontsize=8.5, color="#526170", va="bottom", linespacing=1.45)


def _placeholder(settings, title, note, status="unavailable"):
    """Tạo hình báo thiếu dữ liệu rõ ràng, không thay quan sát thiếu bằng số không."""
    fig, (ax,) = _figure(settings, title, "Chưa có dữ liệu phù hợp để thể hiện")
    ax.set_axis_off()
    ax.text(0.5, 0.5, textwrap.fill(note, 76), transform=ax.transAxes, ha="center", va="center",
            fontsize=13, color="#526170", bbox={"boxstyle": "round,pad=1", "fc": "#f1f4f7", "ec": "none"})
    _footnote(fig, "Nguồn: các tệp QA/KPI/báo cáo cục bộ. Ô thiếu dữ liệu được giữ là chưa biết; hình không suy diễn số quan sát.")
    return fig, {"status": status, "n_observations": None, "note": note}


def _result(n, note, **extra):
    """Tạo metadata chuẩn cho một biểu đồ có dữ liệu quan sát."""
    return {"status": "ok", "n_observations": int(n), "note": note, **extra}


def _listing_map(settings, data):
    """Vẽ tọa độ mẫu ở mốc mới nhất; tô màu chỉ cho giá QA hợp lệ."""
    frame = data["map"]
    if frame is None or frame.empty:
        return _placeholder(settings, "Phân bố listing trong mẫu", "Thiếu bảng/cột tọa độ hoặc không có tọa độ hợp lệ ở mốc mới nhất.")
    valid = frame["_price_valid"]
    fig, (ax,) = _figure(settings, "Phân bố listing trong mẫu",
                         f"Mốc {settings['latest']} · {len(frame):,} tọa độ hợp lệ · {int(valid.sum()):,} giá QA hợp lệ")
    fig.set_size_inches(8.8, 8)
    fig.subplots_adjust(right=0.80)
    for ring in data["outlines"]:
        ax.plot(ring[:, 0], ring[:, 1], color="#b2bcc5", linewidth=0.65, zorder=1)
    if (~valid).any():
        ax.scatter(frame.loc[~valid, "_lon"], frame.loc[~valid, "_lat"], s=14, color=GRAY, alpha=0.7,
                   label=f"Chưa có giá hợp lệ (n={int((~valid).sum()):,})", zorder=2)
    if valid.any():
        price = frame.loc[valid, "_price"]
        lo, hi = float(price.min()), float(price.max())
        norm = LogNorm(lo, hi) if hi > lo else Normalize(max(0, lo * 0.95), lo * 1.05)
        points = ax.scatter(frame.loc[valid, "_lon"], frame.loc[valid, "_lat"], c=price, cmap="viridis",
                            norm=norm, s=17, alpha=0.8, edgecolors="none", zorder=3)
        colorbar = fig.colorbar(points, ax=ax, pad=0.03, fraction=0.04)
        colorbar.set_label(f"Giá chào mỗi đêm ({settings['currency']})", fontsize=10)
        if hi == lo:
            colorbar.set_ticks([lo])
            colorbar.set_ticklabels([f"{lo:,.0f}"])
    for column, setter, bounds in (("_lon", ax.set_xlim, (-180, 180)), ("_lat", ax.set_ylim, (-90, 90))):
        lo, hi = frame[column].min(), frame[column].max()
        pad = max((hi - lo) * 0.07, 0.005)
        setter(max(bounds[0], lo - pad), min(bounds[1], hi + pad))
    ax.set_aspect(1 / max(abs(math.cos(math.radians(frame["_lat"].mean()))), 0.01))
    ax.set_anchor("C")
    ax.set_xlabel("Kinh độ")
    ax.set_ylabel("Vĩ độ")
    ax.grid(alpha=0.15)
    if (~valid).any():
        ax.legend(loc="best", fontsize=8)
    _footnote(fig, "Nguồn: listings_clean (QA), biên GeoJSON cục bộ nếu có. Chỉ là mẫu listing; loại cờ trùng/tọa độ sai. Màu giá dùng thang log khi giá khác nhau; bản đồ không thể hiện công suất đặt phòng.")
    return fig, _result(len(frame), "Tọa độ ở mốc mới nhất; giá thiếu hiển thị xám.", n_input_listings=frame.attrs["n_input_listings"], n_valid_prices=int(valid.sum()), snapshot=settings["latest"], n_outline_rings=len(data["outlines"]))


def _prices(settings, data):
    """Chọn vùng đủ mẫu ở mốc cuối theo số giá hợp lệ rồi xếp theo giá trung vị."""
    frame = data["kpi_listings"]
    selected = frame.loc[frame["snapshot"].eq(settings["latest"]) & frame["scope"].eq("area") &
                         frame["area_col"].eq(settings["area_col"]) & frame["kpi"].eq("median_valid_price") &
                         frame["n_valid"].ge(settings["min_group_n"]) & frame["value"].gt(0)].copy()
    selected = selected.sort_values(["n_valid", "area"], ascending=[False, True], kind="stable").head(settings["top_n"])
    selected = selected.sort_values(["value", "area"], kind="stable")
    if selected.empty:
        return _placeholder(settings, "Giá trung vị theo khu vực", "Không có vùng đủ giá QA hợp lệ ở mốc mới nhất; không dùng giá từ mốc cũ.")
    fig, (ax,) = _figure(settings, "Giá trung vị theo khu vực",
                         f"Mốc {settings['latest']} · tối đa {settings['top_n']} vùng nhiều giá hợp lệ nhất · tối thiểu n={settings['min_group_n']}")
    y = np.arange(len(selected))
    labels = [textwrap.fill(str(value), 24) for value in selected["area"]]
    row_height = max(0.34, max(label.count("\n") + 1 for label in labels) * 0.17 + 0.12)
    fig.set_size_inches(11, max(6.6, len(selected) * row_height / 0.64))
    ax.barh(y, selected["value"], color=BLUE, height=0.66)
    ax.set_yticks(y, labels)
    ax.set_xlabel(f"Giá chào mỗi đêm trung vị ({settings['currency']})")
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_xlim(0, selected["value"].max() * 1.43)
    for index, row in enumerate(selected.itertuples()):
        ax.text(row.value + selected["value"].max() * 0.015, index,
                f"{row.value:,.0f}  ·  n={int(row.n_valid):,}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.26)
    _footnote(fig, "Nguồn: kpi_listings, scope=area, median_valid_price. n là số giá QA hợp lệ; vùng thiếu giá bị loại. Mẫu được giao không bảo đảm đại diện cho toàn thành phố.")
    return fig, _result(selected["n_valid"].sum(), "Chọn vùng theo n_valid rồi sắp theo trung vị.", snapshot=settings["latest"], areas=selected[["area", "value", "n_valid"]].to_dict("records"))


def _trends(settings, data):
    """Giữ chiều snapshot và khoảng trống khi vẽ số listing cùng tỷ lệ ngày không khả dụng."""
    frame = data["kpi_listings"]
    frame = frame.loc[frame["scope"].eq("all") & frame["area_col"].eq(settings["area_col"])]
    snapshots = settings["snapshots"]
    series = {}
    for key in ("listing_count", "pooled_unavailable_day_rate"):
        series[key] = frame.loc[frame["kpi"].eq(key)].set_index("snapshot").reindex(snapshots)
    counts, rates = (series[key] for key in ("listing_count", "pooled_unavailable_day_rate"))
    if counts["value"].notna().sum() + rates["value"].notna().sum() == 0:
        return _placeholder(settings, "So sánh các mốc dữ liệu", "Không có KPI toàn mẫu hợp lệ ở các mốc cấu hình.")
    fig, axes = _figure(settings, "So sánh các mốc dữ liệu", "Hai chỉ tiêu trên mẫu · mỗi mốc giữ cửa sổ lịch riêng", nrows=2)
    x = np.arange(len(snapshots))
    for ax, values, color in ((axes[0], counts["value"], BLUE), (axes[1], rates["value"], TEAL)):
        ax.plot(x, values, marker="o", color=color, linewidth=2)
        ax.set_xticks(x, snapshots)
        ax.grid(axis="y", alpha=0.18)
        ax.set_xlim(-0.25, max(len(x) - 0.75, 0.25))
    axes[0].set_ylabel("Số listing")
    axes[0].set_ylim(bottom=0, top=max(counts["value"].max() * 1.3, 1) if counts["value"].notna().any() else 1)
    axes[1].set_ylabel("Ngày không khả dụng")
    axes[1].set_ylim(0, 1)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    for index, value in enumerate(counts["value"]):
        if pd.notna(value):
            axes[0].annotate(f"n={int(value):,}", (index, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    for index, value in enumerate(rates["value"]):
        if pd.notna(value):
            axes[1].annotate(f"{value:.2%}", (index, value), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    denominators = [f"{snap}: {int(row.denominator):,}" if pd.notna(row.denominator) else f"{snap}: thiếu"
                    for snap, row in rates.iterrows()]
    _footnote(fig, "Nguồn: kpi_listings, scope=all. Mẫu số tỷ lệ = tổng ngày lịch quan sát: " + "; ".join(denominators) + ". Ngày không khả dụng gồm ngày chủ nhà chặn, không đồng nghĩa lượt đặt hay công suất; các cửa sổ lịch có thể chồng lấn.")
    return fig, _result(counts["value"].dropna().sum(), "Số listing cộng theo mốc, không phải số ID duy nhất; giá trị thiếu giữ khoảng trống.", snapshots=snapshots, calendar_denominators=[None if pd.isna(v) else int(v) for v in rates["denominator"]])


def _sentiment(settings, data, skip_llm):
    """Gộp tử và mẫu số theo năm viết review, không lấy trung bình tỷ lệ tháng."""
    frame = data["kpi_reviews"]
    if skip_llm or frame is None:
        return _placeholder(settings, "Cảm xúc review theo năm viết", "Bỏ qua phần LLM theo yêu cầu." if skip_llm else "Chưa có bảng KPI review.", "skipped" if skip_llm else "unavailable")
    frame = frame.loc[frame["scope"].eq("all") & frame["area_col"].eq(settings["area_col"]) &
                      frame["period_type"].eq("review_month") & frame["kpi"].eq("positive_review_share")].copy()
    if frame.empty:
        return _placeholder(settings, "Cảm xúc review theo năm viết", "Không có KPI tháng viết review ở phạm vi toàn mẫu.")
    frame["year"] = frame["period"].str[:4].astype(int)
    yearly = frame.groupby("year")[["numerator", "denominator", "n_eligible_reviews"]].sum(min_count=1)
    yearly = yearly.reindex(range(int(yearly.index.min()), int(yearly.index.max()) + 1))
    shares = (yearly["numerator"] / yearly["denominator"]).where(yearly["denominator"].ge(settings["min_review_n"]))
    if shares.notna().sum() == 0:
        return _placeholder(settings, "Cảm xúc review theo năm viết", f"Không có năm đủ {settings['min_review_n']} review đã phân loại; không hiển thị tỷ lệ thiếu cơ sở.")
    fig, axes = _figure(settings, "Cảm xúc review theo năm viết", f"Lịch sử review · tỷ lệ chỉ hiện khi n phân loại ≥ {settings['min_review_n']}", nrows=2)
    x = np.arange(len(yearly))
    axes[0].bar(x, shares, color=TEAL, width=0.68)
    axes[0].set_ylim(0, 1.15)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Review tích cực")
    axes[1].bar(x, yearly["n_eligible_reviews"], color="#dce3e9", label="Review đủ điều kiện")
    axes[1].bar(x, yearly["denominator"], color=BLUE, width=0.52, label="Đã phân loại (n)")
    axes[1].set_ylabel("Số review")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=8, loc="upper left", ncols=2)
    for index, share in enumerate(shares):
        if pd.notna(share):
            axes[0].text(index, share + 0.025, f"n={int(yearly.iloc[index]['denominator'])}", ha="center", fontsize=8)
    for ax in axes:
        ax.set_xticks(x, yearly.index.astype(str), rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.18)
    cutoff = data["kpi_summary"].get("coverage", {}).get("reviews", {}).get("eligible_date_max")
    cutoff_note = f" Ngày viết cuối được quan sát: {cutoff}; năm cuối có thể chưa trọn năm." if cutoff else ""
    _footnote(fig, "Nguồn: kpi_reviews, scope=all, review_month. Tỷ lệ năm = tổng tích cực / tổng đã phân loại; khoảng trống là năm thiếu/ít mẫu. Năm viết lịch sử không phải năm snapshot; phân loại mô hình có sai số." + cutoff_note)
    return fig, _result(yearly["denominator"].sum(), "Gộp tử số/mẫu số tháng theo năm viết; giữ khoảng trống khi thiếu mẫu.", years=[int(v) for v in yearly.index], numerators=[None if pd.isna(v) else int(v) for v in yearly["numerator"]], denominators=[None if pd.isna(v) else int(v) for v in yearly["denominator"]], review_date_max=cutoff)


def _aspects(settings, data, skip_llm):
    """Vẽ tỷ lệ nhắc tiêu cực từng khía cạnh trên cùng tập review đã phân loại."""
    frame = data["kpi_aspects"]
    if skip_llm or frame is None:
        return _placeholder(settings, "Khía cạnh bị nhắc tiêu cực", "Bỏ qua phần LLM theo yêu cầu." if skip_llm else "Chưa có bảng KPI khía cạnh.", "skipped" if skip_llm else "unavailable")
    selected = frame.loc[frame["scope"].eq("all") & frame["area_col"].eq(settings["area_col"]) &
                         frame["period_type"].eq("all") & frame["kpi"].eq("negative_aspect_share") & frame["value"].notna()].copy()
    if selected.empty:
        return _placeholder(settings, "Khía cạnh bị nhắc tiêu cực", "Chưa có review đã phân loại cho các khía cạnh toàn lịch sử.")
    selected = selected.sort_values(["value", "aspect"], kind="stable")
    denominator = int(selected.iloc[0]["denominator"])
    fig, (ax,) = _figure(settings, "Khía cạnh bị nhắc tiêu cực", f"Toàn bộ lịch sử review · chung mẫu số n={denominator:,} review đã phân loại")
    y = np.arange(len(selected))
    labels = [textwrap.fill(value, 25) for value in selected["aspect"]]
    row_height = max(0.36, max(label.count("\n") + 1 for label in labels) * 0.17 + 0.12)
    fig.set_size_inches(11, max(6.6, len(selected) * row_height / 0.64))
    ax.barh(y, selected["value"], color="#b16b51", height=0.63)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Tỷ lệ review đã phân loại có nhắc tiêu cực")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlim(0, min(1, max(selected["value"].max() * 1.65, 0.04)))
    for index, row in enumerate(selected.itertuples()):
        inside = row.value > ax.get_xlim()[1] * 0.76
        ax.text(row.value + ax.get_xlim()[1] * (-0.015 if inside else 0.015), index,
                f"{row.value:.1%}  ·  {int(row.numerator):,}/{denominator:,}", va="center", fontsize=10,
                ha="right" if inside else "left", color="white" if inside else "#20364a")
    ax.grid(axis="x", alpha=0.18)
    fig.subplots_adjust(left=0.24)
    _footnote(fig, "Nguồn: kpi_aspects, period_type=all, scope=all. Một review có thể nhắc nhiều khía cạnh; các tỷ lệ không cộng thành 100%. Đây là nhận định mô hình trên mẫu đã phân loại, không phải kết luận về mọi khách lưu trú.")
    return fig, _result(denominator, "Các khía cạnh dùng chung mẫu số và có thể chồng lấn.", aspects=selected[["aspect", "value", "numerator", "denominator"]].to_dict("records"))


def _comparison(settings, data, skip_llm):
    """So sánh accuracy và macro F1 cảm xúc; nêu rõ trạng thái nhãn tham chiếu."""
    frame = data["evaluation"]
    if skip_llm or frame is None:
        note = "Bỏ qua phần LLM theo yêu cầu." if skip_llm else data["evaluation_note"] or "Chưa có báo cáo đánh giá tương thích."
        return _placeholder(settings, "So sánh mô hình phân loại cảm xúc", note, "skipped" if skip_llm else "unavailable")
    selected = frame.loc[frame["task"].eq("sentiment")].copy()
    selected = selected.loc[selected["n_reviews"].gt(0) & selected[["accuracy", "f1_macro"]].notna().all(axis=1)]
    if selected.empty:
        return _placeholder(settings, "So sánh mô hình phân loại cảm xúc", "Không có điểm cảm xúc và cỡ mẫu đánh giá hợp lệ.")
    status = str(data["evaluation_summary"].get("label_status", "unknown"))
    provisional = status.lower() not in {"verified", "human_verified", "human_reviewed", "verified_human"}
    subtitle = "Nhãn tham chiếu tạm thời, chưa xác nhận rà soát bởi con người" if provisional else "Đối chiếu nhãn tham chiếu đã được xác nhận rà soát"
    fig, (ax,) = _figure(settings, "So sánh mô hình phân loại cảm xúc", subtitle)
    x, width = np.arange(len(selected)), 0.30
    for offset, metric, label, color in ((-width / 2, "accuracy", "Accuracy", BLUE), (width / 2, "f1_macro", "Macro F1", TEAL)):
        ax.bar(x + offset, selected[metric], width, color=color, label=label)
        for index, value in enumerate(selected[metric]):
            ax.text(index + offset, value + 0.02, f"{value:.1%}", ha="center", fontsize=10)
    ax.set_xticks(x, [textwrap.fill(str(row.model), 30) + f"\nn={int(row.n_reviews):,}" for row in selected.itertuples()])
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylabel("Điểm đối chiếu cảm xúc")
    ax.legend(loc="upper center", ncols=2, fontsize=10, bbox_to_anchor=(0.5, -0.13), frameon=False)
    ax.grid(axis="y", alpha=0.18)
    fig.subplots_adjust(bottom=0.26)
    _footnote(fig, "Nguồn: evaluation_metrics, task=sentiment; n là số review đối chiếu của từng mô hình. Điểm mô tả tập nhãn hiện tại, không suy rộng ra toàn dữ liệu. Trạng thái nhãn: " + status + ".")
    return fig, _result(selected["n_reviews"].max(), "Số quan sát là cỡ mẫu lớn nhất của một mô hình; các mô hình có thể cùng review.", label_status=status, reference_provisional=provisional, models=selected[["model", "n_reviews", "accuracy", "f1_macro"]].to_dict("records"))


def run(cfg, output_dir=None, skip_llm=False, **_):
    """Kiểm tra đầu vào rồi xuất sáu hình và JSON; chỉ thay tệp đầu ra do module sở hữu.

    Trả dict gồm checksum nguồn, cấu hình, phạm vi, cỡ mẫu, cảnh báo và trạng thái
    từng hình. skip_llm chỉ bỏ phần review/đánh giá; không gọi API trong mọi chế độ.
    """
    settings = _settings(cfg, output_dir)
    data, inputs, warnings = _prepare(settings, bool(skip_llm))
    warnings.extend(["Các hình chỉ mô tả mẫu được giao, không mặc nhiên đại diện toàn thành phố.",
                     "Ngày lịch không khả dụng không xác lập lượt đặt phòng, công suất hay doanh thu.",
                     "Review dùng năm viết lịch sử; mô hình và việc chọn mẫu có thể gây thiên lệch."])
    summary = {"schema_version": 1, "city": settings["city"], "currency": settings["currency"],
               "snapshots": settings["snapshots"], "latest_snapshot": settings["latest"], "skip_llm": bool(skip_llm),
               "inputs": inputs, "warnings": warnings, "charts": {}, "output_files": [],
               "settings": {key: settings[key] for key in ("top_n", "min_group_n", "min_review_n", "dpi", "formats", "price_col", "area_col", "latitude_col", "longitude_col", "font_family")}}
    settings["output"].mkdir(parents=True, exist_ok=True)
    builders = (_listing_map, _prices, _trends, _sentiment, _aspects, _comparison)
    style = {"font.family": settings["font_family"], "axes.unicode_minus": False, "svg.hashsalt": "airbnb-viz-v1",
             "savefig.facecolor": "white", "figure.facecolor": "white", "font.size": 10}
    with tempfile.TemporaryDirectory(prefix=".viz-stage-", dir=settings["output"]) as temporary, plt.rc_context(style):
        staging = Path(temporary)
        for stem, builder in zip(STEMS, builders):
            fig = None
            try:
                fig, chart = builder(settings, data, bool(skip_llm)) if builder in (_sentiment, _aspects, _comparison) else builder(settings, data)
                chart["files"] = [stem + "." + fmt for fmt in settings["formats"]]
                chart["sources"] = (["listings", "geojson"] if stem == STEMS[0] else ["kpi_listings"] if stem in STEMS[1:3] else ["kpi_reviews"] if stem == STEMS[3] else ["kpi_aspects"] if stem == STEMS[4] else ["evaluation", "evaluation_summary"])
                chart["sources"] = [key for key in chart["sources"] if key in inputs]
                chart["scope"] = "area" if stem == STEMS[1] else "all"
                summary["charts"][stem] = chart
                for fmt, name in zip(settings["formats"], chart["files"]):
                    metadata = {"Date": None, "Creator": "Airbnb KPI visualization"} if fmt == "svg" else {"Software": "Airbnb KPI visualization"}
                    fig.savefig(staging / name, dpi=settings["dpi"], metadata=metadata)
                    summary["output_files"].append(name)
            finally:
                if fig is not None:
                    plt.close(fig)
        summary["output_files"].append("viz_summary.json")
        (staging / "viz_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        for name in summary["output_files"]:
            os.replace(staging / name, settings["output"] / name)
    return summary


def main(argv=None):
    """Đọc YAML từ CLI, chạy ngoại tuyến và in vị trí cùng trạng thái sáu hình."""
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/Buenos_Aires.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args(argv)
    try:
        cfg = yaml.safe_load(_path(args.config).read_text(encoding="utf-8-sig"))
        summary = run(cfg, output_dir=args.output_dir, skip_llm=args.skip_llm)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(1, f"viz: {exc}\n")
    print(json.dumps({"charts": {key: value["status"] for key, value in summary["charts"].items()},
                      "output_files": summary["output_files"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
