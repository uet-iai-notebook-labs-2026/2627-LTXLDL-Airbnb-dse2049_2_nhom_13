"""Tải và kiểm tra dữ liệu thô từ Inside Airbnb theo config của nhóm.

Cách chạy (từ thư mục gốc của repo):
    python src/download.py                          # dùng configs/paris.yaml
    python src/download.py --config configs/paris.yaml
    python src/download.py --force                  # tải lại kể cả khi tệp đã có
    python src/download.py --check-only             # chỉ kiểm tra tệp đã tải, không tải mới

Config theo configs/paris.yaml:
    city: paris
    data_root: france/ile-de-france/paris    # đường dẫn trên data.insideairbnb.com
    currency: EUR
    snapshots: [2025-09-12, 2025-12-13, 2026-03-21, 2026-06-16]
    raw_dir: data/raw                         # mặc định data/raw

Với mỗi mốc, tải về <raw_dir>/<mốc>/:
    listings.csv.gz, calendar.csv.gz, reviews.csv.gz, neighbourhoods.geojson

Mỗi tệp được kiểm tra: tồn tại, không rỗng, đọc được, có đủ cột khoá.
Chỉ tải từ Inside Airbnb (CC BY 4.0, ghi nguồn trong báo cáo).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import time
import zlib
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://data.insideairbnb.com"

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
    "calendar.csv.gz": ["listing_id", "date", "available"],
    "reviews.csv.gz": ["listing_id", "id", "date", "comments"],
}

USER_AGENT = "UET-DSE2049-student-project/1.0 (educational use)"
RETRIES = 3
TIMEOUT = 60        # giây
CHUNK = 1 << 20     # 1 MiB


def load_config(path: Path) -> dict:
    """Đọc config YAML và kiểm tra các khoá bắt buộc."""
    if not path.exists():
        sys.exit(f"Không thấy tệp config: {path}")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for key in ("data_root", "snapshots"):
        if key not in cfg:
            sys.exit(f"Config thiếu khoá bắt buộc: '{key}' ({path})")
    cfg.setdefault("raw_dir", "data/raw")
    return cfg


def validate(path: Path, name: str) -> str | None:
    """Kiểm tra 1 tệp. Trả về None nếu hợp lệ, ngược lại trả về mô tả lỗi.

    `name` là tên tệp "thật" (vì khi đang tải, đường dẫn có đuôi .part).
    """
    if not path.exists():
        return "tệp không tồn tại"
    if path.stat().st_size == 0:
        return "tệp rỗng"
    try:
        if name.endswith(".csv.gz"):
            with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), [])
            missing = [c for c in REQUIRED_COLUMNS.get(name, []) if c not in header]
            if missing:
                return "thiếu cột: " + ", ".join(missing)
        elif name.endswith(".geojson"):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not data.get("features"):
                return "geojson không có features"
    except (OSError, EOFError, zlib.error, ValueError) as e:
        # ValueError bao gồm UnicodeDecodeError và json.JSONDecodeError
        return f"không đọc được: {e}"
    return None


def download_file(session: requests.Session, url: str, dest: Path) -> None:
    """Tải 1 tệp an toàn: ghi ra .part, kiểm tra, rồi mới đổi tên thành tệp chính."""
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
            time.sleep(2 * attempt)

    tmp.unlink(missing_ok=True)
    raise RuntimeError(f"Không tải được {url}: {last_err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/paris.yaml")
    parser.add_argument("--force", action="store_true", help="tải lại tệp đã tồn tại")
    parser.add_argument("--check-only", action="store_true",
                        help="chỉ kiểm tra tệp đã có, không tải")
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config)
    city_url = f"{BASE_URL}/{cfg['data_root'].strip('/')}"
    raw_dir = REPO_ROOT / cfg["raw_dir"]

    session = requests.Session()
    # identity: không cho server nén thêm, để so sánh Content-Length chính xác
    session.headers.update({"User-Agent": USER_AGENT, "Accept-Encoding": "identity"})

    failed: list[str] = []
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
                print(f"  bỏ qua (đã có, hợp lệ): {name}")
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
    print("Xong: dữ liệu thô đầy đủ và hợp lệ.")


if __name__ == "__main__":
    main()