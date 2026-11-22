"""Tải dữ liệu Inside Airbnb và tạo subset theo ID được phân công cho nhóm.

Cách chạy (từ thư mục gốc của repo):
    python src/download.py                          # đọc cấu hình và CSV phân công local
    python src/download.py --config configs/Buenos_Aires.yaml
    python src/download.py --force                  # tải lại kể cả khi tệp đã có
    python src/download.py --check-only             # chỉ kiểm tra tệp đã tải, không tải mới
    python src/download.py --filter-only            # chỉ lọc dữ liệu thô đã tải
    python src/download.py --local-samples          # dùng CSV local nếu config chọn Sheet
    python src/download.py --group-code <ma-du-lieu> --listings-file <file-listing-dung-nhom>

Config theo configs/Buenos_Aires.yaml:
    city: Buenos Aires
    data_root: argentina/ciudad-autónoma-de-buenos-aires/buenos-aires
    currency: ARS
    snapshots: [2025-09-27, 2025-12-29, 2026-03-30, 2026-06-29]
    raw_dir: data/raw                         # mặc định data/raw
    samples_dir: data/samples                 # chứa group_listings.csv, group_reviews.csv
    group_code: <ma-du-lieu>                 # khớp chính xác cột Mã dữ liệu
    paths: {processed: data/processed}

Với mỗi mốc, tải về <raw_dir>/<mốc>/:
    listings.csv.gz, calendar.csv.gz, reviews.csv.gz, neighbourhoods.geojson

Mỗi tệp được kiểm tra: tồn tại, không rỗng, đọc được, có đủ cột khoá.
Sau khi tải, luôn lọc lại (kể cả raw đã có), ghi ba bảng subset và city_review_stats.json.
Dữ liệu thô được giữ nguyên. ID được đọc dạng chuỗi để không làm tròn số lớn.
Nhận CSV có header hoặc bản xuất không header. Không tự trộn dữ liệu khác mã nhóm.
Chỉ tải từ Inside Airbnb (CC BY 4.0, ghi nguồn trong báo cáo).
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import date
import gzip
import io
from itertools import chain
import json
import math
import re
import sys
import time
import zlib
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://data.insideairbnb.com"
DEFAULT_CONFIG = "configs/Buenos_Aires.yaml"

# (thư mục trên server, tên tệp)
REMOTE_FILES = (
    ("data", "listings.csv.gz"),
    ("data", "calendar.csv.gz"),
    ("data", "reviews.csv.gz"),
    ("visualisations", "neighbourhoods.geojson"),
)

# Cột khoá tối thiểu mà các bước sau (QA, KPI, LLM) cần dùng.
REQUIRED_COLUMNS = {
    "listings.csv.gz": [
        "id", "last_scraped", "host_id", "price", "room_type",
        "neighbourhood_cleansed", "latitude", "longitude",
        "number_of_reviews", "calculated_host_listings_count",
    ],
    "calendar.csv.gz": ["listing_id", "date", "available", "minimum_nights"],
    "reviews.csv.gz": ["listing_id", "id", "date", "comments"],
}

USER_AGENT = "UET-DSE2049-student-project/1.0 (educational use)"
RETRIES = 3
TIMEOUT = 60        # giây
CHUNK = 1 << 20     # 1 MiB
# Mô tả listing và nội dung review có thể vượt giới hạn mặc định của csv.
csv.field_size_limit(10 * CHUNK)

SUBSET_FILES = {
    "listings.csv.gz": "listings_subset.csv",
    "calendar.csv.gz": "calendar_subset.csv.gz",
    "reviews.csv.gz": "reviews_subset.csv",
}


def load_config(path: Path) -> dict:
    """Đọc config YAML và kiểm tra các khoá bắt buộc.
    Nhận đường dẫn YAML, trả dict đã bổ sung thư mục mặc định và chuẩn hoá snapshots.
    Thoát chương trình nếu thiếu tệp/khoá hoặc danh sách mốc rỗng, sai kiểu;
    cấu hình sai nội dung có thể gây ValueError, lỗi đọc/YAML được truyền ra ngoài.
    """
    if not path.exists():
        sys.exit(f"Không thấy tệp config: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML phải là mapping")
    for key in ("data_root", "snapshots"):
        if key not in cfg:
            sys.exit(f"Config thiếu khoá bắt buộc: '{key}' ({path})")
    cfg.setdefault("raw_dir", "data/raw")
    cfg.setdefault("samples_dir", "data/samples")
    if not isinstance(cfg["snapshots"], list) or not cfg["snapshots"]:
        sys.exit("Config snapshots phải là danh sách mốc dữ liệu không rỗng")
    cfg["snapshots"] = _snapshots(cfg)
    if not isinstance(cfg["data_root"], str) or any(part in {"", ".", ".."} for part in cfg["data_root"].strip("/").split("/")):
        raise ValueError("data_root phải là đường dẫn country/region/city hợp lệ")
    return cfg


def _snapshots(cfg: dict) -> list[str]:
    """Trả danh sách ngày YYYY-MM-DD từ cfg; báo ValueError nếu rỗng, trùng hoặc sai định dạng."""
    snapshots = [str(value) for value in cfg.get("snapshots", [])]
    if not snapshots or len(snapshots) != len(set(snapshots)):
        raise ValueError("snapshots phải không rỗng và không được trùng mốc")
    for snapshot in snapshots:
        try:
            if date.fromisoformat(snapshot).isoformat() != snapshot:
                raise ValueError
        except ValueError:
            raise ValueError(f"Mốc {snapshot!r} cần đúng định dạng ngày YYYY-MM-DD") from None
    return snapshots


def _group_rows(path: Path, group_code: str, columns: list[str], source_group_code: str | None = None) -> list[dict[str, str]]:
    """Đọc header Việt/Anh hoặc CSV không header, không bỏ mất dòng ID đầu tiên.
    Nhận đường dẫn CSV, mã nhóm đích/nguồn tuỳ chọn và các cột cần lấy.
    Trả danh sách dict của các dòng thuộc mã được chấp nhận, chỉ giữ cột yêu cầu.
    Báo ValueError nếu bảng rỗng, sai bố cục/cột hoặc không có dòng đúng nhóm.
    """
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        first = next((row for row in reader if row and any(value.strip() for value in row)), None)
        if first is None:
            raise ValueError(f"{path.name}: file phân công rỗng")
        aliases = {"group_code": "Mã dữ liệu", "data_code": "Mã dữ liệu", "llm_sample": "Mẫu LLM", "hand_label": "Nhãn tay"}
        header = [aliases.get(value.strip(), value.strip()) for value in first]
        needed = ["Mã dữ liệu", *columns]
        if set(needed).issubset(header):
            if len(header) != len(set(header)):
                raise ValueError(f"{path.name}: tên cột bị trùng")
            records = reader
        else:
            # Chỉ đoán bố cục của một dòng DỮ LIỆU có mã nhóm và ID nguyên,
            # không coi một header sai chính tả là dữ liệu.
            layouts = (["Mã dữ liệu", "listing_id", "analysis_area"],
                       ["Lớp", "Nhóm", "Mã dữ liệu", "listing_id", "analysis_area"]) if "analysis_area" in columns else (
                       ["Mã dữ liệu", "review_id", "listing_id", "Mẫu LLM", "Nhãn tay"],
                       ["Lớp", "Nhóm", "Mã dữ liệu", "review_id", "listing_id", "Mẫu LLM", "Nhãn tay"])
            layout = next((layout for layout in layouts if len(layout) == len(first)
                           and re.fullmatch(r"DSE2049-[A-Za-z0-9_-]+", first[layout.index("Mã dữ liệu")].strip())
                           and first[layout.index("listing_id")].strip().isascii()
                           and first[layout.index("listing_id")].strip().isdecimal()), None)
            if layout is None:
                raise ValueError(f"{path.name}: thiếu cột {', '.join(sorted(set(needed) - set(header)))}; hoặc CSV không header sai bố cục")
            header, records = layout, chain([first], reader)
        rows = []
        available = set()
        for values in records:
            if not values or not any(value.strip() for value in values):
                continue
            if len(values) != len(header):
                raise ValueError(f"{path.name}: dòng {reader.line_num} sai số cột")
            row = dict(zip(header, (value.strip() for value in values)))
            available.add(row["Mã dữ liệu"])
            if row["Mã dữ liệu"] in {group_code, source_group_code}:
                rows.append({col: row[col] for col in columns})
    if not rows:
        raise ValueError(f"{path.name}: không có ID cho Mã dữ liệu '{group_code}'. Mã hiện có: {', '.join(sorted(available))}. Cần file đúng nhóm; không tự đổi mã.")
    return rows


def _sample_id(value: str, name: str) -> str:
    """Chuẩn hoá chuỗi ID thành số nguyên dương int64 dạng chuỗi; dữ liệu sai gây ValueError kèm tên nguồn."""
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"{name}: ID không hợp lệ '{value}' (cần chuỗi chữ số, không dùng float)")
    number = int(value)
    if not 0 < number <= 2**63 - 1:
        raise ValueError(f"{name}: ID phải là số nguyên dương trong miền int64")
    return str(number)


def _sample_flag(value: str) -> str:
    """Đổi cờ chữ/bool dạng chuỗi thành "x" hoặc chuỗi rỗng; giá trị khác gây ValueError."""
    text = value.strip().lower()
    if text in {"x", "true", "t", "1", "yes", "y"}:
        return "x"
    if text in {"", "false", "f", "0", "no", "n"}:
        return ""
    raise ValueError("Mẫu LLM/Nhãn tay chỉ nhận x/rỗng hoặc giá trị bool hợp lệ")


def _load_regions(samples_dir: Path, city: str | None = None) -> dict[str, tuple[float, float, float, float]]:
    """Đọc bảng vùng trong samples_dir và chọn các vùng của city nếu được chỉ định.
    Trả ánh xạ analysis_area sang (lon_min, lat_min, lon_max, lat_max), hoặc {} nếu không có tệp.
    Ưu tiên group_vung.csv rồi group_areas.csv; chỉ đọc dữ liệu nguồn.
    Báo ValueError khi thiếu/sai dữ liệu, trùng vùng hoặc không xác định được thành phố.
    """
    source = next((samples_dir / name for name in ("group_vung.csv", "group_areas.csv") if (samples_dir / name).is_file()), None)
    if source is None:
        return {}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Thành phố", "analysis_area", "lon_min", "lat_min", "lon_max", "lat_max"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{source.name}: thiếu cột vùng {sorted(required - set(reader.fieldnames or []))}")
        rows = []
        for row in reader:
            if None in row or any(row.get(key) is None or not row[key].strip() for key in required):
                raise ValueError(f"{source.name}: dòng {reader.line_num} thiếu ô bắt buộc hoặc sai số cột")
            rows.append(row)
    cities = {row["Thành phố"].strip().casefold() for row in rows}
    if city is None and len(cities) > 1:
        raise ValueError("Bảng vùng có nhiều thành phố: cần cấu hình city")
    selected_city = city.strip().casefold() if city is not None else next(iter(cities), "")
    regions = {}
    for row in rows:
        if row["Thành phố"].strip().casefold() != selected_city:
            continue
        area = row["analysis_area"].strip()
        try:
            bounds = tuple(float(row[key].strip().replace(",", ".")) for key in ("lon_min", "lat_min", "lon_max", "lat_max"))
        except (AttributeError, TypeError, ValueError):
            raise ValueError(f"{source.name}: toạ độ vùng không phải số") from None
        lon_min, lat_min, lon_max, lat_max = bounds
        if (not area or not all(math.isfinite(value) for value in bounds)
                or not (-180 <= lon_min < lon_max <= 180 and -90 <= lat_min < lat_max <= 90)):
            raise ValueError(f"{source.name}: biên vùng {area!r} không hợp lệ")
        if area in regions:
            raise ValueError(f"{source.name}: vùng {area!r} bị trùng")
        regions[area] = bounds
    if not regions:
        raise ValueError(f"{source.name}: không có vùng của thành phố {city!r}")
    return regions




def load_group_samples(samples_dir: Path, group_code: str | None, *,
                       listings_file: Path | None = None, reviews_file: Path | None = None,
                       city: str | None = None, source_group_code: str | None = None,
                       _listing_rows: list[dict[str, str]] | None = None,
                       _review_rows: list[dict[str, str]] | None = None) -> tuple[
    dict[str, str], dict[str, dict[str, str]]
]:
    """Đọc ID, vùng phân tích và cờ review của đúng nhóm từ các bảng phân công.
    Nhận thư mục/tệp CSV hoặc các dòng có sẵn, cùng mã nhóm đích và mã nguồn tuỳ chọn.
    Trả hai dict: listing_id -> analysis_area và review_id -> listing_id/cờ chuẩn hoá.
    Đối chiếu bảng vùng nếu có; báo ValueError khi ID/cờ sai, thông tin mâu thuẫn
    hoặc review tham chiếu listing không nằm trong bảng phân công.
    """
    group_code = str(group_code or "").strip()
    if not group_code:
        raise ValueError("Cần đặt group_code trong config hoặc truyền --group-code theo cột Mã dữ liệu")
    if source_group_code is not None:
        source_group_code = source_group_code.strip()
        if not source_group_code:
            raise ValueError("source_group_code không được rỗng")
    listings: dict[str, str] = {}
    listing_rows = _listing_rows if _listing_rows is not None else _group_rows(
        listings_file or samples_dir / "group_listings.csv", group_code,
        ["listing_id", "analysis_area"], source_group_code)
    if not listing_rows:
        raise ValueError("Danh sách listing phân công rỗng")
    for row in listing_rows:
        listing_id = _sample_id(row["listing_id"], "group_listings.csv")
        area = row["analysis_area"]
        if not area:
            raise ValueError(f"Listing {listing_id}: thiếu analysis_area")
        if listing_id in listings and listings[listing_id] != area:
            raise ValueError(f"Listing {listing_id}: analysis_area mâu thuẫn trong bảng phân công")
        listings[listing_id] = area

    regions = _load_regions(samples_dir, city)
    unknown = set(listings.values()) - set(regions) if regions else set()
    if unknown:
        raise ValueError(f"analysis_area không có trong bảng vùng: {', '.join(sorted(unknown))}")

    reviews: dict[str, dict[str, str]] = {}
    orphan_reviews, orphan_listings = 0, set()
    review_rows = _review_rows if _review_rows is not None else _group_rows(
        reviews_file or samples_dir / "group_reviews.csv", group_code,
        ["review_id", "listing_id", "Mẫu LLM", "Nhãn tay"], source_group_code)
    if not review_rows:
        raise ValueError("Danh sách review phân công rỗng")
    for row in review_rows:
        review_id = _sample_id(row["review_id"], "group_reviews.csv")
        listing_id = _sample_id(row["listing_id"], "group_reviews.csv")
        if listing_id not in listings:
            orphan_reviews += 1
            orphan_listings.add(listing_id)
            continue
        metadata = {"listing_id": listing_id, "llm_sample": _sample_flag(row["Mẫu LLM"]),
                    "hand_label": _sample_flag(row["Nhãn tay"])}
        if review_id in reviews and reviews[review_id] != metadata:
            raise ValueError(f"Review {review_id}: thông tin mâu thuẫn trong bảng phân công")
        reviews[review_id] = metadata
    if orphan_reviews:
        raise ValueError(f"Có {orphan_reviews:,} review tham chiếu {len(orphan_listings):,} listing ID không có trong bảng listing. "
                         "Đổi mã nhóm không khắc phục lệch ID; cần danh sách phân công đầy đủ và khớp nhau.")
    return listings, reviews


def load_sheet_samples(cfg: dict, sheet: str, *, source_group_code: str | None = None,
                       listing_rows: tuple[int, int] | None = None,
                       review_rows: tuple[int, int] | None = None) -> tuple[dict, dict]:
    """Đọc khoảng dòng 1-based trên Sheet, giữ nguyên ID/cờ, không sửa nguồn.
    Nhận cfg, URL/ID Sheet và khoảng dòng/tab từ tham số hoặc cấu hình.
    Gửi HTTP lấy CSV rồi trả cặp dict listing/review đã kiểm tra bằng load_group_samples.
    In số dòng đã đọc; lỗi HTTP, quyền truy cập, khoảng dòng hoặc mã nhóm được báo ra ngoài.
    """
    match = re.fullmatch(r"https://docs\.google\.com/spreadsheets/d/([A-Za-z0-9_-]+)(?:[/?#].*)?", sheet)
    sheet_id = match.group(1) if match else sheet
    if not re.fullmatch(r"[A-Za-z0-9_-]+", sheet_id):
        raise ValueError("assignment-sheet phải là URL docs.google.com/spreadsheets/d/<id> hoặc ID bảng tính")
    group_code = str(cfg.get("group_code") or "").strip()
    if not group_code:
        raise ValueError("Cần group_code của nhóm đích")
    ranges = cfg.get("assignment_ranges") or {}
    tabs = cfg.get("assignment_tabs") or {}
    if not isinstance(ranges, dict) or not isinstance(tabs, dict):
        raise ValueError("assignment_ranges/assignment_tabs phải là mapping")
    listing_rows = listing_rows if listing_rows is not None else ranges.get("listing")
    review_rows = review_rows if review_rows is not None else ranges.get("review")
    if any(not isinstance(bounds, (tuple, list)) or len(bounds) != 2 for bounds in (listing_rows, review_rows)):
        raise ValueError("Dùng Sheet cần --listing-rows START END và --review-rows START END, hoặc assignment_ranges trong config")
    tables = {}
    specs = [("listing", listing_rows, ["Lớp", "Nhóm", "Mã dữ liệu", "listing_id", "analysis_area"], "E"),
             ("review", review_rows, ["Lớp", "Nhóm", "Mã dữ liệu", "review_id", "listing_id", "Mẫu LLM", "Nhãn tay"], "G")]
    with requests.Session() as session:
        session.headers.update({"User-Agent": USER_AGENT})
        for kind, bounds, columns, last_col in specs:
            tab = tabs.get(kind, kind.title())
            if not isinstance(tab, str) or not tab.strip():
                raise ValueError("Tên tab trong assignment_tabs cần chuỗi không rỗng")
            start, end = bounds
            if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int) or start < 1 or end < start:
                raise ValueError("Khoảng dòng Sheet cần số nguyên 1-based: đầu >= 1, cuối >= đầu")
            response = session.get(f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq",
                                   params={"tqx": "out:csv", "sheet": tab, "range": f"A{start}:{last_col}{end}", "headers": "0"},
                                   timeout=TIMEOUT)
            response.raise_for_status()
            response.encoding = "utf-8"
            text = response.text.lstrip("\ufeff")
            if text.lstrip().startswith("<"):
                raise ValueError(f"Không đọc được tab {tab}; kiểm tra quyền chia sẻ/đăng nhập")
            rows = list(csv.reader(io.StringIO(text)))
            if len(rows) != end - start + 1 or any(len(row) != len(columns) for row in rows):
                raise ValueError(f"Tab {tab}: số dòng/cột không đúng khoảng đã yêu cầu")
            records = [dict(zip(columns, (value.strip() for value in row))) for row in rows]
            if any(row["Mã dữ liệu"] not in {group_code, source_group_code} for row in records):
                raise ValueError(f"Tab {tab}: khoảng dòng chứa mã nhóm khác; cần --source-group-code đúng dữ liệu nguồn")
            tables[kind] = records
            print(f"  Google Sheet {tab}: dòng {start}–{end}, {len(records):,} dòng")
    return load_group_samples(REPO_ROOT / cfg.get("samples_dir", "data/samples"), group_code,
                              city=cfg.get("city"), source_group_code=source_group_code,
                              _listing_rows=tables["listing"], _review_rows=tables["review"])


@contextmanager
def _subset_output(destination: Path, compressed: bool):
    """Mở destination để ghi UTF-8 và yield luồng văn bản, tự đóng khi rời context.
    compressed chọn gzip hoặc văn bản thường; gzip có mtime=0 và không ghi tên .part
    để chạy lại ra cùng byte.
    """
    if compressed:
        with destination.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                with io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text:
                    yield text
    else:
        with destination.open("w", encoding="utf-8", newline="") as text:
            yield text


def build_subsets(cfg: dict, listing_areas: dict[str, str],
                  review_samples: dict[str, dict[str, str]]) -> dict[str, int]:
    """Lọc streaming theo ID/snapshot; tạo xong các bảng rồi mới thay từng file đích.
    Nhận cfg, ánh xạ vùng của listing và metadata review; trả số dòng theo tên tệp subset.
    Ghi ba bảng subset cùng city_review_stats.json vào thư mục processed và in số ID khớp.
    Kiểm tra cột/dòng nguồn, báo lỗi nếu một loại bảng không có dòng khớp ở bất kỳ mốc nào;
    luôn dọn các tệp .part còn lại, kể cả khi phát sinh lỗi đọc/ghi.
    """
    raw_dir = REPO_ROOT / cfg.get("raw_dir", "data/raw")
    processed_dir = REPO_ROOT / (cfg.get("paths") or {}).get("processed", "data/processed")
    snapshots = _snapshots(cfg)
    processed_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    counts: dict[str, int] = {}
    city_stats = {}
    try:
        for source_name, output_name in SUBSET_FILES.items():
            extras = ["snapshot"]
            if source_name == "listings.csv.gz":
                extras += ["analysis_area"]
            elif source_name == "reviews.csv.gz":
                extras += ["llm_sample", "hand_label"]

            # Inside Airbnb có thể thêm/bớt cột giữa các mốc: giữ hợp các cột.
            fields: list[str] = []
            for snap in snapshots:
                source = raw_dir / snap / source_name
                with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as f:
                    header = next(csv.reader(f), [])
                missing = set(REQUIRED_COLUMNS[source_name]) - set(header)
                if missing:
                    raise ValueError(f"{source}: thiếu cột {', '.join(sorted(missing))}")
                if len(header) != len(set(header)):
                    raise ValueError(f"{source}: cột nguồn bị trùng")
                fields.extend(c for c in header if c not in fields and c not in extras)
            fields += extras

            dest = processed_dir / output_name
            tmp = dest.with_suffix(dest.suffix + ".part")
            staged.append((tmp, dest))
            total = 0
            expected_ids = set(review_samples if source_name == "reviews.csv.gz" else listing_areas)
            with _subset_output(tmp, output_name.endswith(".gz")) as out:
                writer = csv.DictWriter(out, fieldnames=fields)
                writer.writeheader()
                for snap in snapshots:
                    matched: set[str] = set()
                    kept = 0
                    n_city, chars_city = 0, 0
                    source = raw_dir / snap / source_name
                    with gzip.open(source, "rt", encoding="utf-8-sig", newline="") as f:
                        reader = csv.reader(f)
                        header = next(reader)
                        key = "listing_id" if source_name == "calendar.csv.gz" else "id"
                        key_index = header.index(key)
                        comments_index = header.index("comments") if source_name == "reviews.csv.gz" else None
                        for values in reader:
                            if not values:
                                continue
                            if len(values) != len(header):
                                raise ValueError(f"{source}: dòng {reader.line_num} sai số cột")
                            if comments_index is not None:
                                n_city += 1
                                chars_city += len(values[comments_index])
                            raw_id = values[key_index].strip()
                            row_id = str(int(raw_id)) if raw_id.isascii() and raw_id.isdecimal() else raw_id
                            if row_id not in expected_ids:
                                continue
                            row = dict(zip(header, values))
                            # Các cột phân công được tái tạo, không tin cờ có sẵn ở nguồn.
                            for col in extras:
                                row.pop(col, None)
                            row[key] = row_id
                            if source_name == "listings.csv.gz":
                                row["analysis_area"] = listing_areas[row_id]
                            elif source_name == "reviews.csv.gz":
                                meta = review_samples[row_id]
                                source_listing = row["listing_id"].strip()
                                source_listing = str(int(source_listing)) if source_listing.isascii() and source_listing.isdecimal() else source_listing
                                if source_listing != meta["listing_id"]:
                                    continue
                                row.update(meta)
                            row["snapshot"] = snap
                            writer.writerow(row)
                            matched.add(row_id)
                            kept += 1
                    total += kept
                    if comments_index is not None:
                        city_stats[snap] = {"n_reviews_city": n_city, "total_chars_city": chars_city}
                    print(f"  lọc {snap}/{source_name}: {kept:,} dòng, "
                          f"{len(matched):,}/{len(expected_ids):,} ID")
                    if expected_ids - matched:
                        print(f"    lưu ý: {len(expected_ids - matched):,} ID phân công không có ở mốc này")
            if total == 0:
                raise ValueError(f"{source_name}: không có dòng khớp ID nhóm ở bất kỳ snapshot nào")
            counts[output_name] = total

        stats_dest = processed_dir / "city_review_stats.json"
        stats_tmp = stats_dest.with_suffix(".json.part")
        staged.append((stats_tmp, stats_dest))
        stats_tmp.write_text(json.dumps(city_stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

        for tmp, dest in staged:
            tmp.replace(dest)
    finally:
        for tmp, _ in staged:
            tmp.unlink(missing_ok=True)
    return counts


def validate(path: Path, name: str) -> str | None:
    """Kiểm tra 1 tệp. Trả về None nếu hợp lệ, ngược lại trả về mô tả lỗi.

    `name` là tên tệp "thật" (vì khi đang tải, đường dẫn có đuôi .part).
    Đọc hết CSV gzip để kiểm CRC/footer và cột bắt buộc; GeoJSON cần có features.
    """
    if not path.exists():
        return "tệp không tồn tại"
    if path.stat().st_size == 0:
        return "tệp rỗng"
    try:
        if name.endswith(".csv.gz"):
            with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as f:
                header = next(csv.reader(f), [])
                # Đọc tới cuối để kiểm CRC/footer gzip, không chỉ kiểm header.
                while f.read(CHUNK):
                    pass
            missing = [c for c in REQUIRED_COLUMNS.get(name, []) if c not in header]
            if missing:
                return "thiếu cột: " + ", ".join(missing)
        elif name.endswith(".geojson"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("features"):
                return "geojson không có features"
    except (OSError, EOFError, zlib.error, ValueError, csv.Error) as e:
        # ValueError bao gồm UnicodeDecodeError và json.JSONDecodeError
        return f"không đọc được: {e}"
    return None


def download_file(session: requests.Session, url: str, dest: Path) -> None:
    """Tải 1 tệp an toàn: ghi ra .part, kiểm tra, rồi mới đổi tên thành tệp chính.
    Nhận session HTTP, URL nguồn và đường dẫn dest; trả None sau khi ghi thành công.
    Tạo thư mục đích, kiểm số byte/cấu trúc tệp và thử lại tối đa RETRIES lần.
    In lỗi mỗi lần thử; xoá tệp tạm rồi báo RuntimeError nếu hết lượt tải.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: str = ""

    for attempt in range(1, RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                expected = int(r.headers.get("Content-Length", 0))
                written = 0
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK):
                        f.write(chunk)
                        written += len(chunk)
            if expected and written != expected:
                raise OSError(f"tải thiếu: nhận {written} / {expected} byte")
            problem = validate(tmp, dest.name)
            if problem:
                raise ValueError(problem)
            tmp.replace(dest)
            return
        except (requests.RequestException, OSError, ValueError) as e:
            last_err = str(e)
            print(f"    lần {attempt}/{RETRIES} lỗi: {last_err}")
            if attempt < RETRIES:
                time.sleep(2 * attempt)

    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Không tải được {url}: {last_err}")


def main() -> None:
    """Đọc tham số CLI và cấu hình để tải, kiểm tra hoặc lọc dữ liệu theo chế độ chọn.
    Lấy phân công từ CSV/Sheet, kiểm tra tệp thô và tạo subset trừ chế độ --check-only.
    Các cờ --filter-only/--check-only giới hạn việc tải/ghi; hàm trả None khi hoàn tất.
    Có thể gửi HTTP, ghi tệp và in tiến độ; lỗi cấu hình, phân công hoặc dữ liệu
    được báo qua parser/sys.exit với thông báo cho người chạy.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                     epilog="Mặc định đọc CSV local theo config. Sheet chỉ được đọc khi chỉ định URL và khoảng dòng.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--group-code", help="mã nhóm khớp cột Mã dữ liệu trong các CSV phân công")
    parser.add_argument("--samples-dir", help="thư mục chứa group_listings.csv và group_reviews.csv")
    parser.add_argument("--listings-file", help="CSV phân công listing đúng nhóm (không sửa file samples hiện tại)")
    parser.add_argument("--reviews-file", help="CSV phân công review đúng nhóm")
    parser.add_argument("--source-group-code", help="chấp nhận thêm mã nguồn dùng chung đã xác nhận; vẫn kiểm tra toàn bộ liên kết ID")
    parser.add_argument("--assignment-sheet", help="URL hoặc ID Google Sheet phân công; đọc Listing/Review trực tiếp, giữ nguyên CSV local")
    parser.add_argument("--local-samples", action="store_true", help="đọc CSV local, bỏ assignment_sheet trong config")
    parser.add_argument("--listing-rows", nargs=2, type=int, metavar=("START", "END"), help="khoảng dòng Listing trên Sheet, hoặc đọc assignment_ranges.listing")
    parser.add_argument("--review-rows", nargs=2, type=int, metavar=("START", "END"), help="khoảng dòng Review trên Sheet, hoặc đọc assignment_ranges.review")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="tải lại tệp đã tồn tại rồi lọc")
    mode.add_argument("--check-only", action="store_true",
                      help="chỉ kiểm tra dữ liệu thô đã có, không tải hoặc lọc")
    mode.add_argument("--filter-only", action="store_true",
                      help="chỉ tạo lại subset từ dữ liệu thô đã có, không tải")
    args = parser.parse_args()

    config_path = (REPO_ROOT / args.config).resolve()
    try:
        cfg = load_config(config_path)
    except (OSError, ValueError, yaml.YAMLError) as e:
        sys.exit(f"Lỗi cấu hình: {e}")
    if args.local_samples and args.assignment_sheet:
        parser.error("--local-samples không kết hợp --assignment-sheet")
    if args.assignment_sheet is None and not args.local_samples:
        args.assignment_sheet = cfg.get("assignment_sheet")
    if args.source_group_code is None:
        args.source_group_code = cfg.get("source_group_code")
    city_url = f"{BASE_URL}/{cfg['data_root'].strip('/')}"
    raw_dir = REPO_ROOT / cfg["raw_dir"]
    if args.group_code is not None:
        cfg["group_code"] = args.group_code
    if args.samples_dir is not None:
        cfg["samples_dir"] = args.samples_dir

    if not args.check_only:
        try:
            if args.assignment_sheet:
                if args.listings_file or args.reviews_file:
                    parser.error("--assignment-sheet không kết hợp --listings-file/--reviews-file")
                listing_areas, review_samples = load_sheet_samples(
                    cfg, args.assignment_sheet, source_group_code=args.source_group_code,
                    listing_rows=tuple(args.listing_rows) if args.listing_rows is not None else None,
                    review_rows=tuple(args.review_rows) if args.review_rows is not None else None)
            else:
                listing_areas, review_samples = load_group_samples(
                    REPO_ROOT / cfg["samples_dir"], cfg.get("group_code"),
                    listings_file=REPO_ROOT / args.listings_file if args.listings_file else None,
                    reviews_file=REPO_ROOT / args.reviews_file if args.reviews_file else None,
                    city=cfg.get("city"), source_group_code=args.source_group_code)
        except (OSError, EOFError, zlib.error, ValueError, csv.Error, requests.RequestException) as e:
            sys.exit(f"Lỗi danh sách ID nhóm: {e}")
        print(f"Nhóm {cfg['group_code']}: {len(listing_areas):,} listings, "
              f"{len(review_samples):,} reviews được phân công")

    if not args.filter_only:
        failed: list[str] = []
        with requests.Session() as session:
            # identity: không cho server nén thêm, để so sánh Content-Length chính xác
            session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})
            for snap in cfg["snapshots"]:
                snap = str(snap)
                print(f"[{snap}]")
                for folder, name in REMOTE_FILES:
                    dest = raw_dir / snap / name
                    label = f"{snap}/{name}"

                    if args.check_only:
                        problem = validate(dest, name)
                        print(f"  {'OK ' if problem is None else 'LỖI'} {name}"
                              + (f": {problem}" if problem else ""))
                        if problem:
                            failed.append(label)
                        continue

                    if dest.exists() and not args.force and validate(dest, name) is None:
                        print(f"  bỏ qua tải (đã có, hợp lệ): {name}")
                        continue

                    url = f"{city_url}/{snap}/{folder}/{name}"
                    print(f"  tải: {url}")
                    try:
                        download_file(session, url, dest)
                        print(f"  OK  {name} ({dest.stat().st_size / 1e6:.1f} MB)")
                    except RuntimeError as e:
                        print(f"  LỖI: {e}")
                        failed.append(label)

        if failed:
            sys.exit("Các tệp lỗi:\n  " + "\n  ".join(failed))

    if args.check_only:
        print("Xong: dữ liệu thô đầy đủ và hợp lệ.")
        return
    try:
        counts = build_subsets(cfg, listing_areas, review_samples)
    except (OSError, EOFError, zlib.error, ValueError, csv.Error) as e:
        sys.exit(f"Lỗi lọc dữ liệu nhóm: {e}")
    print("Xong: đã tạo subset theo ID nhóm trong "
          + str(REPO_ROOT / (cfg.get("paths") or {}).get("processed", "data/processed")))
    for name, count in counts.items():
        print(f"  {name}: {count:,} dòng")


if __name__ == "__main__":
    main()
