"""Đánh giá kết quả LLM và baseline trên cùng bộ review có nhãn tham chiếu.

Chạy từ repo: python src/evaluate.py --config configs/Buenos_Aires.yaml
Hoặc: python -m src.evaluate --config configs/Buenos_Aires.yaml

Chỉ đọc CSV đã có, không gọi API, không đọc .env và không sửa nhãn.
Sentiment: positive/mixed/negative; khía cạnh là cặp (tên, khen/chê).
Macro-F1 dùng đủ 3 lớp cảm xúc và 2 * số khía cạnh, kể cả lớp vắng.
Quy ước mẫu số bằng 0: precision/recall/F1 = 0; bộ khía cạnh rỗng
vẫn khớp chính xác với bộ rỗng. Điểm số phản ánh file nhãn đã cung cấp,
không tự xác nhận file đó là nhãn tay đã được con người kiểm chứng.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import logging
from pathlib import Path
import re
import sys
import tempfile
import os

REPO_ROOT = Path(__file__).resolve().parents[1]
SENTIMENTS = ("positive", "mixed", "negative")
DEFAULT_ASPECTS = ("vị trí", "sạch sẽ", "chủ nhà", "giá trị", "tiện nghi", "độ ồn", "an toàn")
RESULT_COLUMNS = ("review_id", "sentiment", "language", "aspects_positive", "aspects_negative")
LANGUAGE_CODES = frozenset("""aa ab ae af ak am an ar as av ay az ba be bg bh bi bm bn bo br bs ca ce ch co cr cs cu cv cy da de dv dz ee el en eo es et eu fa ff fi fj fo fr fy ga gd gl gn gu gv ha he hi ho hr ht hu hy hz ia id ie ig ii ik io is it iu ja jv ka kg ki kj kk kl km kn ko kr ks ku kv kw ky la lb lg li ln lo lt lu lv mg mh mi mk ml mn mr ms mt my na nb nd ne ng nl nn no nr nv ny oc oj om or os pa pi pl ps pt qu rm rn ro ru rw sa sc sd se sg si sk sl sm sn so sq sr ss st su sv sw ta te tg th ti tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo wa wo xh yi yo za zh zu und""".split())
OUTPUTS = {
    "metrics": "evaluation_metrics.csv",
    "by_class": "evaluation_by_class.csv",
    "confusion": "evaluation_confusion.csv",
    "errors": "evaluation_errors.csv",
    "summary": "evaluation_summary.json",
}
METRIC_COLUMNS = ("model", "task", "n_reviews", "accuracy", "precision_macro", "recall_macro",
                  "f1_macro", "precision_micro", "recall_micro", "f1_micro", "exact_match")
CLASS_COLUMNS = ("model", "task", "label", "support", "predicted_support", "tp", "fp", "fn",
                 "precision", "recall", "f1")
CONFUSION_COLUMNS = ("model", "task", "true_label", "predicted_label", "count")
ERROR_COLUMNS = ("model", "review_id", "gold_sentiment", "predicted_sentiment", "sentiment_correct",
                 "gold_language", "predicted_language", "language_correct", "gold_aspects_positive",
                 "predicted_aspects_positive", "gold_aspects_negative", "predicted_aspects_negative",
                 "aspects_exact_match", "missing_aspects", "extra_aspects", "comments_clean")
log = logging.getLogger("evaluate")


def _mapping(value, name):
    """Kiểm tra value là dict; trả dict rỗng khi value=None, báo lỗi kèm name nếu sai kiểu."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} phải là mapping hoặc null")
    return value


def _resolve(value):
    """Chuyển value thành Path; đường dẫn tương đối được tính từ gốc repo."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("Đường dẫn phải là chuỗi không rỗng")
    result = Path(value)
    return result if result.is_absolute() else REPO_ROOT / result


def path(cfg, name):
    """Lấy thư mục name từ cfg['paths'] hoặc giá trị mặc định và trả về Path tuyệt đối."""
    defaults = {"processed": "data/processed", "samples": "data/samples", "reports": "reports"}
    paths = _mapping(cfg.get("paths"), "paths")
    return _resolve(paths.get(name, cfg.get(f"{name}_dir", defaults[name])))


def reviews_source(cfg):
    """Trả (đường dẫn reviews_clean, bắt buộc hay không) theo cấu hình evaluate.reviews_file."""
    evaluate = _mapping(cfg.get("evaluate"), "evaluate")
    if "reviews_file" in evaluate:
        return _resolve(evaluate["reviews_file"]), True
    return path(cfg, "processed") / "reviews_clean.csv", False


def settings(cfg):
    """Đọc và kiểm tra cấu hình đánh giá từ cfg.
    Trả dict chứa bộ khía cạnh, số nhãn tối thiểu, trạng thái nhãn và đường dẫn
    file tham chiếu cùng hai bộ dự đoán; cấu hình sai gây ValueError.
    """
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML phải là mapping")
    evaluate = _mapping(cfg.get("evaluate"), "evaluate")
    llm = _mapping(cfg.get("llm"), "llm")
    aspects = llm.get("aspects", list(DEFAULT_ASPECTS))
    if (not isinstance(aspects, list) or not aspects
            or any(not isinstance(a, str) or not a.strip() or "|" in a for a in aspects)
            or len(set(aspects)) != len(aspects)):
        raise ValueError("llm.aspects cần danh sách tên không rỗng, không trùng, không chứa '|'")
    minimum = evaluate.get("min_labels", 100)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1:
        raise ValueError("evaluate.min_labels phải là số nguyên >= 1")
    label_status = evaluate.get("label_status", "unverified")
    if label_status not in {"unverified", "ai_assisted", "human_reviewed"}:
        raise ValueError("evaluate.label_status: unverified, ai_assisted hoặc human_reviewed")
    labels = (_resolve(evaluate["labels_file"]) if "labels_file" in evaluate
              else path(cfg, "samples") / "hand_labels.csv")
    models = {
        "llm": (_resolve(evaluate["llm_file"]) if "llm_file" in evaluate
                else path(cfg, "processed") / "llm_results.csv"),
        "baseline": (_resolve(evaluate["baseline_file"]) if "baseline_file" in evaluate
                     else path(cfg, "processed") / "baseline_results.csv"),
    }
    return {"aspects": tuple(aspects), "min_labels": minimum, "label_status": label_status,
            "labels_file": labels, "models": models}


def _exact_id(value, context):
    """Kiểm tra value là chuỗi ID int64 dương chính xác; trả chuỗi đã bỏ khoảng trắng."""
    text = value.strip()
    if not re.fullmatch(r"[1-9][0-9]*", text) or len(text) > 19 or int(text) > 2**63 - 1:
        raise ValueError(f"{context}: review_id phải là chuỗi số nguyên dương int64, không dùng float")
    return text


def _aspect_set(value, allowed, context):
    """Tách chuỗi khía cạnh bằng '|', trả frozenset; báo lỗi khi có nhãn lạ hoặc bị lặp."""
    if not value.strip():
        return frozenset()
    names = [part if part in allowed else part.strip() for part in value.split("|")]
    if any(not name or name not in allowed for name in names):
        raise ValueError(f"{context}: khía cạnh rỗng hoặc ngoài danh sách cho phép")
    if len(set(names)) != len(names):
        raise ValueError(f"{context}: khía cạnh bị lặp trong cùng một cột")
    return frozenset(names)


def load_annotations(source, aspects, minimum=1):
    """Đọc CSV nhãn hoặc dự đoán, giữ review_id dưới dạng chuỗi để bảo toàn ID lớn.
    Trả dict theo review_id, trong đó các khía cạnh là frozenset. Báo lỗi khi
    thiếu cột, trùng ID, sai nhãn hoặc số review nhỏ hơn minimum.
    """
    source = Path(source)
    allowed = set(aspects)
    records = {}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise ValueError(f"{source.name}: header bị trùng")
        missing = set(RESULT_COLUMNS) - set(headers)
        if missing:
            raise ValueError(f"{source.name}: thiếu cột {', '.join(sorted(missing))}")
        for line, raw in enumerate(reader, start=2):
            context = f"{source.name}, bản ghi {line}"
            if None in raw or any(v is None for v in raw.values()):
                raise ValueError(f"{context}: số cột không khớp header")
            review_id = _exact_id(raw["review_id"], context)
            if review_id in records:
                raise ValueError(f"{context}: review_id bị trùng {review_id}")
            sentiment, language = raw["sentiment"].strip(), raw["language"].strip()
            if sentiment not in SENTIMENTS:
                raise ValueError(f"{context}: sentiment không hợp lệ {sentiment!r}")
            if language not in LANGUAGE_CODES:
                raise ValueError(f"{context}: language không hợp lệ {language!r}")
            records[review_id] = {
                "review_id": review_id, "sentiment": sentiment, "language": language,
                "aspects_positive": _aspect_set(raw["aspects_positive"], allowed, context),
                "aspects_negative": _aspect_set(raw["aspects_negative"], allowed, context),
                "comments_clean": raw.get("comments_clean", ""),
            }
    if len(records) < minimum:
        raise ValueError(f"{source.name}: cần ít nhất {minimum} review, hiện có {len(records)}")
    return records


def aspect_pairs(record):
    """Đổi hai tập khía cạnh khen/chê của record thành tập nhãn có tiền tố positive:/negative:."""
    return frozenset([f"positive:{a}" for a in record["aspects_positive"]]
                     + [f"negative:{a}" for a in record["aspects_negative"]])


def _divide(numerator, denominator):
    """Trả numerator/denominator; quy ước trả 0.0 khi mẫu số bằng 0."""
    return numerator / denominator if denominator else 0.0


def _scores(counts):
    """Nhận bộ (TP, FP, FN), trả precision, recall và F1 với mẫu số bằng 0 được quy về 0."""
    tp, fp, fn = counts
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    f1 = _divide(2 * tp, 2 * tp + fp + fn)
    return precision, recall, f1


def _summarize(counts, labels, n):
    """Gộp số TP/FP/FN theo labels thành điểm macro, micro và bảng điểm từng lớp.
    Trả (dict chỉ số, danh sách dòng từng lớp); n là số review được chấm.
    """
    if not labels or n < 1:
        raise ValueError("Cần ít nhất một review và một lớp để chấm")
    per_label = []
    for label in labels:
        tp, fp, fn = counts[label]
        precision, recall, f1 = _scores((tp, fp, fn))
        per_label.append({"label": label, "support": tp + fn, "predicted_support": tp + fp,
                          "tp": tp, "fp": fp, "fn": fn,
                          "precision": precision, "recall": recall, "f1": f1})
    totals = tuple(sum(counts[label][i] for label in labels) for i in range(3))
    micro = _scores(totals)
    metrics = {
        "n_reviews": n, "accuracy": None,
        "precision_macro": sum(r["precision"] for r in per_label) / len(labels),
        "recall_macro": sum(r["recall"] for r in per_label) / len(labels),
        "f1_macro": sum(r["f1"] for r in per_label) / len(labels),
        "precision_micro": micro[0], "recall_micro": micro[1], "f1_micro": micro[2],
        "exact_match": None,
    }
    return metrics, per_label


def classification_metrics(truth, prediction, labels):
    """Chấm một nhãn cho mỗi review từ truth, prediction và danh sách labels.
    Trả chỉ số tổng hợp, điểm từng lớp và ma trận nhầm lẫn dạng danh sách.
    Trong ma trận, hàng là nhãn chuẩn và cột là nhãn dự đoán.
    """
    if not truth or len(truth) != len(prediction):
        raise ValueError("Hai danh sách nhãn cần cùng độ dài và không rỗng")
    if len(labels) != len(set(labels)):
        raise ValueError("Danh sách lớp bị trùng")
    allowed = set(labels)
    if any(label not in allowed for label in [*truth, *prediction]):
        raise ValueError("Có nhãn ngoài danh sách lớp")
    matrix = Counter(zip(truth, prediction))
    counts = {}
    for label in labels:
        tp = matrix[(label, label)]
        fp = sum(matrix[(other, label)] for other in labels if other != label)
        fn = sum(matrix[(label, other)] for other in labels if other != label)
        counts[label] = (tp, fp, fn)
    metrics, per_label = _summarize(counts, labels, len(truth))
    metrics["accuracy"] = sum(a == b for a, b in zip(truth, prediction)) / len(truth)
    confusion = [{"true_label": a, "predicted_label": b, "count": matrix[(a, b)]}
                 for a in labels for b in labels]
    return metrics, per_label, confusion


def multilabel_metrics(truth, prediction, labels):
    """Chấm các tập nhãn khía cạnh/khen-chê trên cùng danh sách review.
    Trả chỉ số macro/micro, tỷ lệ khớp toàn bộ tập nhãn và điểm từng nhãn.
    Hai tập rỗng vẫn được coi là khớp chính xác.
    """
    if not truth or len(truth) != len(prediction):
        raise ValueError("Hai danh sách tập nhãn cần cùng độ dài và không rỗng")
    if len(labels) != len(set(labels)):
        raise ValueError("Danh sách lớp bị trùng")
    allowed = set(labels)
    gold = [frozenset(values) for values in truth]
    predicted = [frozenset(values) for values in prediction]
    if any(not values.issubset(allowed) for values in [*gold, *predicted]):
        raise ValueError("Có nhãn khía cạnh ngoài danh sách lớp")
    counts = {
        label: (sum(label in g and label in p for g, p in zip(gold, predicted)),
                sum(label not in g and label in p for g, p in zip(gold, predicted)),
                sum(label in g and label not in p for g, p in zip(gold, predicted)))
        for label in labels
    }
    metrics, per_label = _summarize(counts, labels, len(truth))
    metrics["exact_match"] = sum(g == p for g, p in zip(gold, predicted)) / len(truth)
    return metrics, per_label


def load_texts(cfg, gold):
    """Trả dict review_id -> nội dung để đưa vào bảng lỗi, không nhân dòng theo snapshot.
    Nếu có reviews_clean.csv, kiểm tra nội dung giữa các snapshot và file nhãn
    phải khớp. Nếu nguồn không bắt buộc và không tồn tại thì dùng text trong gold.
    """
    source, required = reviews_source(cfg)
    if not source.is_file():
        if required:
            raise FileNotFoundError(f"Không tìm thấy evaluate.reviews_file: {source}")
        return {rid: row["comments_clean"] for rid, row in gold.items()}
    texts = {}
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"id", "comments_clean"}.issubset(reader.fieldnames or []):
            raise ValueError("reviews_clean.csv thiếu id/comments_clean")
        for line, raw in enumerate(reader, start=2):
            if None in raw or any(v is None for v in raw.values()):
                raise ValueError(f"reviews_clean.csv, bản ghi {line}: số cột không khớp")
            rid = raw["id"].strip()
            if rid not in gold:
                continue
            text = raw["comments_clean"]
            if rid in texts and texts[rid] != text:
                raise ValueError(f"Review {rid}: comments_clean khác nhau giữa các snapshot")
            texts[rid] = text
    missing = set(gold) - set(texts)
    if missing:
        raise ValueError(f"reviews_clean.csv thiếu {len(missing)} ID có nhãn")
    for rid, row in gold.items():
        if row["comments_clean"] and row["comments_clean"] != texts[rid]:
            raise ValueError(f"Review {rid}: nội dung file nhãn khác reviews_clean.csv")
    return texts


def evaluate_predictions(gold, predictions, aspects, texts):
    """Đối chiếu từng mô hình trong predictions với toàn bộ ID có trong gold.
    Trả các bảng chỉ số, điểm từng lớp, nhầm lẫn, lỗi và độ phủ; thiếu dự đoán
    sẽ báo lỗi, còn ID dự đoán ngoài bộ tham chiếu được thống kê nhưng không chấm.
    """
    ids = list(gold)
    pair_labels = tuple(f"{polarity}:{aspect}" for polarity in ("positive", "negative") for aspect in aspects)
    output = {"metrics": [], "by_class": [], "confusion": [], "errors": [], "coverage": {}}
    for model, rows in predictions.items():
        missing = set(ids) - set(rows)
        if missing:
            examples = ", ".join(sorted(missing, key=int)[:5])
            raise ValueError(f"{model}: thiếu dự đoán cho {len(missing)} review có nhãn; ví dụ {examples}")
        output["coverage"][model] = {
            "reference_reviews": len(ids), "matched_reviews": len(ids),
            "prediction_reviews": len(rows), "extra_predictions": len(set(rows) - set(ids)),
            "coverage": 1.0,
        }
        for task in ("sentiment", "language"):
            truth = [gold[rid][task] for rid in ids]
            predicted = [rows[rid][task] for rid in ids]
            labels = SENTIMENTS if task == "sentiment" else tuple(sorted(set(truth) | set(predicted)))
            metrics, per_label, confusion = classification_metrics(truth, predicted, labels)
            output["metrics"].append({"model": model, "task": task, **metrics})
            output["by_class"].extend({"model": model, "task": task, **r} for r in per_label)
            output["confusion"].extend({"model": model, "task": task, **r} for r in confusion)
        metrics, per_label = multilabel_metrics(
            [aspect_pairs(gold[rid]) for rid in ids], [aspect_pairs(rows[rid]) for rid in ids], pair_labels)
        output["metrics"].append({"model": model, "task": "aspects", **metrics})
        output["by_class"].extend({"model": model, "task": "aspects", **r} for r in per_label)
        for rid in ids:
            reference, predicted = gold[rid], rows[rid]
            g, p = aspect_pairs(reference), aspect_pairs(predicted)
            sentiment_ok = reference["sentiment"] == predicted["sentiment"]
            language_ok = reference["language"] == predicted["language"]
            aspects_ok = g == p
            if sentiment_ok and language_ok and aspects_ok:
                continue
            output["errors"].append({
                "model": model, "review_id": rid, "gold_sentiment": reference["sentiment"],
                "predicted_sentiment": predicted["sentiment"], "sentiment_correct": sentiment_ok,
                "gold_language": reference["language"], "predicted_language": predicted["language"],
                "language_correct": language_ok,
                "gold_aspects_positive": "|".join(sorted(reference["aspects_positive"])),
                "predicted_aspects_positive": "|".join(sorted(predicted["aspects_positive"])),
                "gold_aspects_negative": "|".join(sorted(reference["aspects_negative"])),
                "predicted_aspects_negative": "|".join(sorted(predicted["aspects_negative"])),
                "aspects_exact_match": aspects_ok, "missing_aspects": "|".join(sorted(g - p)),
                "extra_aspects": "|".join(sorted(p - g)), "comments_clean": texts.get(rid, ""),
            })
    return output


def _csv_text(rows, columns):
    """Chuyển rows thành chuỗi CSV theo columns, có dòng tiêu đề và ký tự xuống dòng LF."""
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _sha256(source):
    """Đọc tệp source và trả mã SHA-256 để nhận diện chính xác dữ liệu đầu vào."""
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def save_reports(destination, result, summary, inputs):
    """Ghi bốn CSV và một JSON đánh giá vào destination, không ghi đè tệp đầu vào.
    Chuẩn bị các tệp tạm rồi thay từng tệp đích bằng os.replace; thao tác thay
    là nguyên tử cho từng tệp, không phải một giao dịch cho cả năm tệp.
    """
    destination = Path(destination).resolve()
    targets = [destination / name for name in OUTPUTS.values()]
    source_paths = {Path(source).resolve() for source in inputs}
    if source_paths.intersection(targets):
        raise ValueError("Đường dẫn báo cáo trùng file đầu vào")
    contents = {
        "metrics": _csv_text(result["metrics"], METRIC_COLUMNS),
        "by_class": _csv_text(result["by_class"], CLASS_COLUMNS),
        "confusion": _csv_text(result["confusion"], CONFUSION_COLUMNS),
        "errors": _csv_text(result["errors"], ERROR_COLUMNS),
        "summary": json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    }
    destination.mkdir(parents=True, exist_ok=True)
    pending = []
    try:
        for key, text in contents.items():
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8-sig" if key != "summary" else "utf-8",
                                             newline="", dir=destination, prefix=".evaluate-", delete=False) as handle:
                handle.write(text)
                pending.append((Path(handle.name), destination / OUTPUTS[key]))
        for temporary, target in pending:
            os.replace(temporary, target)
    finally:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)


def run(cfg, output_dir=None, **_):
    """Chạy bước đánh giá từ cfg và trả dict tóm tắt kết quả.
    Đọc nhãn, hai bộ dự đoán và nội dung review; kiểm tra độ phủ, tính các chỉ số
    rồi ghi năm báo cáo vào output_dir hoặc thư mục reports trong cấu hình.
    Không gọi API và không sửa nhãn; trạng thái nhãn được ghi rõ trong báo cáo.
    """
    options = settings(cfg)
    gold = load_annotations(options["labels_file"], options["aspects"], options["min_labels"])
    predictions = {name: load_annotations(source, options["aspects"])
                   for name, source in options["models"].items()}
    texts = load_texts(cfg, gold)
    result = evaluate_predictions(gold, predictions, options["aspects"], texts)
    warnings = []
    if options["label_status"] != "human_reviewed":
        warnings.append("Nhãn tham chiếu chưa được xác nhận đã có người rà soát; điểm số chỉ là đối chiếu với file nhãn hiện tại.")
    language_counts = dict(Counter(r["language"] for r in gold.values()))
    sentiment_counts = dict(Counter(r["sentiment"] for r in gold.values()))
    if len(language_counts) == 1:
        warnings.append("Bộ nhãn chỉ có một ngôn ngữ; không suy rộng điểm language thành năng lực đa ngôn ngữ.")
    if any(sentiment_counts.get(label, 0) < 5 for label in SENTIMENTS):
        warnings.append("Có lớp cảm xúc dưới 5 mẫu; F1 của lớp đó rất nhạy với từng nhãn.")
    inputs = [options["labels_file"], *options["models"].values()]
    reviews_path, _ = reviews_source(cfg)
    if reviews_path.is_file():
        inputs.append(reviews_path)
    summary = {
        "n_reference_reviews": len(gold), "minimum_reference_reviews": options["min_labels"],
        "label_status": options["label_status"], "reference_sentiment_counts": sentiment_counts,
        "reference_language_counts": language_counts, "aspects": list(options["aspects"]),
        "coverage": result["coverage"], "metrics": result["metrics"],
        "error_reviews": dict(Counter(r["model"] for r in result["errors"])),
        "method": {"sentiment_macro_labels": list(SENTIMENTS),
                   "aspect_macro_labels": [f"{p}:{a}" for p in ("positive", "negative") for a in options["aspects"]],
                   "language_macro_labels": "observed reference and prediction labels on matched reference IDs",
                   "zero_division": 0, "macro_includes_zero_support_labels": True,
                   "confusion_orientation": "rows=true_label, columns=predicted_label",
                   "aspect_exact_match": "all aspect/polarity pairs must match; empty equals empty",
                   "extra_predictions": "reported but not scored",
                   "scores": "fractions in [0, 1], not percentages"},
        "inputs": [{"path": str(source.resolve()), "sha256": _sha256(source)} for source in inputs],
        "warnings": warnings,
    }
    for warning in warnings:
        log.warning(warning)
    destination = _resolve(output_dir) if output_dir is not None else path(cfg, "reports")
    save_reports(destination, result, summary, inputs)
    for row in result["metrics"]:
        if row["task"] == "sentiment":
            log.info("%s: n=%d accuracy=%.4f macro-F1=%.4f", row["model"], row["n_reviews"],
                     row["accuracy"], row["f1_macro"])
    log.info("Đã ghi 5 báo cáo đánh giá vào %s", destination)
    return summary


def main():
    """Đọc tham số --config/--output-dir, nạp YAML và gọi run; lỗi dữ liệu được báo qua CLI."""
    import yaml

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/Buenos_Aires.yaml", help="YAML; đường dẫn tương đối từ gốc repo")
    parser.add_argument("--output-dir", help="Thư mục báo cáo; mặc định paths.reports")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        config_file = _resolve(args.config)
        with config_file.open(encoding="utf-8-sig") as handle:
            cfg = yaml.safe_load(handle)
        run(cfg, output_dir=args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, csv.Error, yaml.YAMLError) as exc:
        parser.exit(1, f"Evaluate: {exc}\n")


if __name__ == "__main__":
    main()

