"""Chạy toàn bộ quy trình Airbnb bằng một lệnh, mặc định dùng kết quả LLM trong cache.

Ví dụ:
    python run_pipeline.py --config configs/Buenos_Aires.yaml
    python run_pipeline.py --offline
    python run_pipeline.py --from-processed
    python run_pipeline.py --offline --data-only

--skip-llm là cách ghi tường minh chế độ cache mặc định, vẫn tính KPI review.
--data-only bỏ LLM, baseline và đánh giá; phù hợp khi dữ liệu mới chưa có cache/nhãn.
--call-llm cho phép module LLM gọi API nếu cần; không dùng cờ này khi chỉ chấm lại.
Tệp cấu hình chỉ được đọc, không ghi lại; mặc định không đọc .env hay sửa nhãn tay.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
from datetime import date
import importlib
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO_ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("pipeline")
STEP_NAMES = ("download", "qa", "llm", "baseline", "evaluate", "kpi", "viz")
DIRECTORIES = {"raw": "data/raw", "samples": "data/samples", "processed": "data/processed",
               "reports": "reports", "figures": "figures"}


def _mapping(value, name):
    """Trả dict cấu hình, chấp nhận null và báo ValueError khi sai kiểu."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} phải là mapping hoặc null")
    return value


def _root_path(value):
    """Trả Path tuyệt đối; đường dẫn tương đối được tính từ gốc repo."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Đường dẫn phải là chuỗi không rỗng")
    path = Path(value)
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def _normalize_config(cfg):
    """Sao chép cấu hình và đồng bộ các đường dẫn để mọi module dùng cùng dữ liệu.

    paths.<name> được ưu tiên hơn <name>_dir; cả hai được đổi thành cùng đường
    dẫn tuyệt đối. Không sửa dict của người gọi hoặc tệp YAML gốc.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML phải là mapping")
    result = deepcopy(cfg)
    snapshots = result.get("snapshots")
    if not isinstance(snapshots, (list, tuple)) or not snapshots:
        raise ValueError("Config cần danh sách snapshots không rỗng")
    normalized = []
    for raw in snapshots:
        value = str(raw)
        try:
            if date.fromisoformat(value).isoformat() != value:
                raise ValueError
        except ValueError:
            raise ValueError(f"Snapshot không hợp lệ: {value!r}; cần YYYY-MM-DD") from None
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("Config snapshots có mốc trùng")
    result["snapshots"] = sorted(normalized)
    currency = result.get("currency")
    if not isinstance(currency, str) or not currency.strip():
        raise ValueError("Config cần currency không rỗng")
    result["currency"] = currency.strip()
    paths = _mapping(result.get("paths"), "paths").copy()
    for name, default in DIRECTORIES.items():
        value = paths.get(name, result.get(name + "_dir", default))
        resolved = str(_root_path(value))
        paths[name] = result[name + "_dir"] = resolved
    result["paths"] = paths
    for name in ("qa", "llm", "evaluate", "kpi", "viz"):
        _mapping(result.get(name), name)
    llm = result.get("llm")
    if isinstance(llm, dict) and llm.get("cache") is not None:
        llm["cache"] = str(_root_path(llm["cache"]))
    return result


def load_config(config_file):
    """Đọc YAML UTF-8/BOM rồi trả bản cấu hình đã chuẩn hóa, không ghi lại tệp."""
    import yaml

    with _root_path(config_file).open(encoding="utf-8-sig") as handle:
        return _normalize_config(yaml.safe_load(handle))


def _module(name):
    """Nạp module dưới namespace src để giữ đúng các import tương đối của baseline."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return importlib.import_module("src." + name)


def _download(cfg, filter_only=False, local_samples=False):
    """Gọi CLI download bằng Python hiện tại và cấu hình tạm đã đồng bộ đường dẫn.

    Truyền danh sách đối số không qua shell. --filter-only dùng raw đã có;
    --local-samples ngăn truy cập Sheet trong chế độ offline.
    """
    import yaml

    source = REPO_ROOT / "src" / "download.py"
    if not source.is_file():
        raise FileNotFoundError(f"Thiếu module download: {source}")
    with tempfile.TemporaryDirectory(prefix="airbnb-pipeline-") as directory:
        effective = Path(directory) / "config.yaml"
        effective.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        command = [sys.executable, "-B", str(source), "--config", str(effective)]
        if filter_only:
            command.append("--filter-only")
        if local_samples:
            command.append("--local-samples")
        subprocess.run(command, cwd=str(REPO_ROOT), check=True)


def _validate_reference(cfg, inputs):
    """Kiểm tra nhãn tham chiếu thuộc mẫu hiện tại trước khi xuất lại dự đoán.

    Trả số review tham chiếu. Dữ liệu mới chưa có nhãn phù hợp phải được gán
    nhãn tương ứng, hoặc dùng --data-only để chỉ chạy phân tích dữ liệu.
    """
    evaluate = _module("evaluate")
    options = evaluate.settings(cfg)
    gold = evaluate.load_annotations(options["labels_file"], options["aspects"], options["min_labels"])
    selected = {str(int(value)) for value in inputs["id"]}
    missing = set(gold) - selected
    if missing:
        raise ValueError(
            f"Có {len(missing)} ID nhãn tham chiếu không thuộc mẫu LLM hiện tại. "
            "Hãy dùng nhãn đúng bộ dữ liệu, hoặc --data-only khi chỉ kiểm tra dữ liệu mới."
        )
    evaluate.load_texts(cfg, gold)
    return len(gold)


def _llm_inputs(cfg, call_llm=False):
    """Đọc mẫu và cache theo đúng selector/signature của llm.py, không đọc khóa API.

    Trả (module LLM, DataFrame mẫu). Chế độ cache dừng khi có review thiếu
    cache hợp lệ, trước khi llm.run có thể ghi kết quả thiếu hoặc rỗng.
    """
    llm = _module("llm")
    reviews = llm.load_inputs(cfg)
    inputs = llm.select_inputs(cfg, reviews)
    if inputs.empty:
        raise ValueError("Không có review đủ điều kiện trong mẫu LLM; dùng --data-only nếu phù hợp.")
    llm.check_hand_labels_covered(inputs, reviews)
    cache_file = llm._cache_path(cfg)
    cache = llm.load_cache(cache_file)
    pending = llm.find_pending(inputs, cache, cfg)
    if pending and not call_llm:
        raise RuntimeError(
            f"Cache LLM thiếu hoặc không còn phù hợp cho {len(pending)}/{len(inputs)} review: {cache_file}. "
            "Cần cache đúng bộ dữ liệu. Dùng --data-only để bỏ phân tích LLM; "
            "--call-llm chỉ khi bạn chủ động muốn gọi API."
        )
    return llm, inputs


def _validate_llm_results(cfg, llm, inputs):
    """Xác nhận CSV LLM vừa tạo có đủ đúng ID đã chọn, không trùng hoặc bị làm tròn."""
    source = Path(cfg["paths"]["processed"]) / llm.OUT_RESULTS
    expected = {str(int(value)) for value in inputs["id"]}
    actual = set()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(llm.RESULT_COLUMNS):
            raise ValueError(f"{source.name}: schema khác RESULT_COLUMNS")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{source.name}: số cột không khớp")
            number = llm.normalize_item_id(row["review_id"])
            if number is None or str(number) in actual:
                raise ValueError(f"{source.name}: review_id sai hoặc trùng")
            actual.add(str(number))
    if actual != expected:
        raise RuntimeError(
            f"Kết quả LLM chưa đủ: thiếu {len(expected - actual)} ID, "
            f"thừa {len(actual - expected)} ID. Dừng trước khi tính KPI và đánh giá."
        )


def run(cfg, *, from_processed=False, filter_only=False, offline=False,
        call_llm=False, data_only=False, **_):
    """Chạy các bước theo thứ tự, trả tóm tắt trạng thái và thư mục đầu ra.

    Mặc định dùng cache LLM, không gọi API hay đọc khóa. --data-only bỏ LLM,
    baseline và đánh giá; --offline còn buộc dùng raw và CSV phân công local.
    from_processed bỏ download/QA, dùng dữ liệu sạch đã có trong cấu hình.
    Các module ghi kết quả theo chức năng của chúng; lỗi sẽ dừng các bước sau.
    """
    flags = {"from_processed": from_processed, "filter_only": filter_only,
             "offline": offline, "call_llm": call_llm, "data_only": data_only}
    if any(not isinstance(value, bool) for value in flags.values()):
        raise ValueError("Các tùy chọn pipeline phải là bool")
    if call_llm and (offline or data_only):
        raise ValueError("--call-llm không kết hợp với --offline hoặc --data-only")
    config = _normalize_config(cfg)
    if not from_processed and not isinstance(config.get("data_root"), str):
        raise ValueError("Bước download cần data_root trong cấu hình")
    mode = "data-only" if data_only else "call-llm" if call_llm else "cached"
    stages = []
    started = time.perf_counter()

    def step(name, action=None):
        """Ghi tiến độ và thời gian của một bước; bỏ qua rõ ràng khi không có action."""
        index = STEP_NAMES.index(name) + 1
        if action is None:
            LOG.info("[%d/%d] %s: bỏ qua", index, len(STEP_NAMES), name)
            stages.append({"name": name, "status": "skipped", "seconds": 0.0})
            return None
        LOG.info("[%d/%d] %s: bắt đầu", index, len(STEP_NAMES), name)
        before = time.perf_counter()
        try:
            result = action()
        except Exception as exc:
            LOG.error("Dừng tại bước %s: %s", name, exc)
            raise
        duration = time.perf_counter() - before
        stages.append({"name": name, "status": "done", "seconds": round(duration, 3)})
        LOG.info("[%d/%d] %s: xong (%.2fs)", index, len(STEP_NAMES), name, duration)
        return result

    step("download", None if from_processed else
         lambda: _download(config, filter_only=filter_only or offline, local_samples=offline))
    step("qa", None if from_processed else lambda: _module("qa").run(config))
    if data_only:
        for name in ("llm", "baseline", "evaluate"):
            step(name)
    else:
        def run_llm():
            """Kiểm tra cache/nhãn, gọi đúng chế độ rồi kiểm tra CSV LLM vừa sinh."""
            llm, inputs = _llm_inputs(config, call_llm=call_llm)
            references = _validate_reference(config, inputs)
            llm.run(config, skip_llm=not call_llm)
            _validate_llm_results(config, llm, inputs)
            LOG.info("LLM: đủ %d dự đoán, %d nhãn tham chiếu; chế độ=%s",
                     len(inputs), references, mode)
        step("llm", run_llm)
        step("baseline", lambda: _module("baseline").run(config))
        step("evaluate", lambda: _module("evaluate").run(config))
    step("kpi", lambda: _module("kpi").run(config, skip_llm=data_only))
    step("viz", lambda: _module("viz").run(config, skip_llm=data_only))
    summary = {"mode": mode, "stages": stages, "outputs": config["paths"].copy(),
               "seconds": round(time.perf_counter() - started, 3)}
    LOG.info("Pipeline hoàn tất: %d bước chạy, %d bước bỏ qua (%.2fs).",
             sum(row["status"] == "done" for row in stages),
             sum(row["status"] == "skipped" for row in stages), summary["seconds"])
    LOG.info("Kết quả: processed=%s; reports=%s; figures=%s",
             config["paths"]["processed"], config["paths"]["reports"], config["paths"]["figures"])
    return summary


def main(argv=None):
    """Đọc CLI/YAML rồi chạy pipeline; thông báo lỗi ngắn gọn và trả mã thoát phù hợp."""
    import yaml

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/Buenos_Aires.yaml", help="YAML, tương đối từ gốc repo")
    parser.add_argument("--from-processed", action="store_true", help="bỏ download/QA, dùng dữ liệu sạch đã có")
    parser.add_argument("--filter-only", action="store_true", help="dùng raw đã có; Sheet vẫn theo cấu hình")
    parser.add_argument("--offline", action="store_true", help="không tải raw/Sheet; dùng cache LLM, không API")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--call-llm", action="store_true", help="cho phép gọi Gemini khi thiếu cache")
    group.add_argument("--skip-llm", action="store_true", help="dùng cache LLM; đây cũng là mặc định")
    group.add_argument("--data-only", action="store_true", help="bỏ LLM/baseline/evaluate, tính KPI và hình dữ liệu")
    args = parser.parse_args(argv)
    if args.offline and args.call_llm:
        parser.error("--offline không kết hợp --call-llm")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config = load_config(args.config)
        run(config, from_processed=args.from_processed, filter_only=args.filter_only,
            offline=args.offline, call_llm=args.call_llm, data_only=args.data_only)
    except KeyboardInterrupt:
        parser.exit(130, "Pipeline đã dừng theo yêu cầu.\n")
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, csv.Error,
            subprocess.CalledProcessError, ImportError, yaml.YAMLError) as exc:
        parser.exit(1, f"Pipeline: {exc}\n")


if __name__ == "__main__":
    main()

