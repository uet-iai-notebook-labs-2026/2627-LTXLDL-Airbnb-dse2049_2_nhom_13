# Phân tích dữ liệu Airbnb — Buenos Aires

Bài tập lớn môn **Lập trình xử lý dữ liệu — UET.DSE2049**, năm học 2026–2027.
Nhóm thực hiện: **nhóm 13, lớp UET.DSE2049 2**.

Phân tích phục vụ **chủ nhà và đơn vị quản lý chỗ ở**: khảo sát giá niêm yết,
cơ cấu listing, mức độ không khả dụng của lịch và phản hồi của khách trong mẫu được giao.

| Họ tên | MSSV | GitHub |
|LÊ VĂN VIỆT|25023069|thocuong354-cloud|


Nguồn: [Inside Airbnb](https://insideairbnb.com/), giấy phép CC BY 4.0.
[Đề bài môn học](https://courses.iaidev.com/programming-for-data-processing/2627-1/projects/project_airbnb.html)
và thông báo trên Canvas là căn cứ cho yêu cầu và lịch nộp.

## 1. Cài môi trường

Phiên bản Python đã kiểm thử được ghi trong [.python-version](.python-version): **3.13.7**.
Các thư viện được ghim phiên bản trong [requirements.txt](requirements.txt), gồm
pandas, numpy, PyYAML, requests và Matplotlib.

Các lệnh dưới đây dùng **Windows CMD**, chạy từ thư mục gốc repo:

```cmd
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu đã có môi trường `.venv` đúng phiên bản thì dùng lại, không cần tạo mới.
Trên Linux/macOS, thay `.venv\Scripts\python.exe` bằng `.venv/bin/python`.

## 2. Chạy toàn bộ bằng một lệnh

Trước khi chạy đầy đủ, cần có các CSV phân công, ít nhất 100 nhãn tham chiếu phù hợp
và cache LLM đúng mẫu; xem mục **Lưu trữ và bàn giao** bên dưới.

```cmd
.venv\Scripts\python.exe run_pipeline.py --config configs/Buenos_Aires.yaml
```

Thứ tự chạy: **download → QA → LLM từ cache → baseline → evaluate → KPI → viz**.
Raw còn thiếu được tải từ Inside Airbnb; raw đã hợp lệ được tái sử dụng.
**Mặc định không gọi Gemini, không cần API key.** Cache thiếu, sai cấu hình hoặc
không khớp văn bản sẽ làm luồng đầy đủ dừng rõ ràng, không coi kết quả thiếu là thành công.

| Tùy chọn của `run_pipeline.py` | Ý nghĩa |
|---|---|
| `--offline` | Dùng raw và CSV phân công local đã có, không đọc Sheet hay gọi API. |
| `--from-processed` | Bỏ download và QA, dùng các bảng sạch đã có; các bước sau vẫn chạy. |
| `--filter-only` | Không tải raw; vẫn có thể đọc Sheet nếu cấu hình chọn Sheet. |
| `--skip-llm` | Ghi tường minh chế độ cache mặc định; vẫn đánh giá và tính KPI review. |
| `--data-only` | Bỏ LLM, baseline và evaluate; tính KPI listing/lịch và vẽ phần dữ liệu. |
| `--call-llm` | Chủ động cho phép Gemini xử lý review còn thiếu cache; cần khóa API. |

Ví dụ khi đã tải raw và có đủ cache/nhãn:

```cmd
.venv\Scripts\python.exe run_pipeline.py --offline
```

Khi chỉ muốn tính lại từ dữ liệu sạch đã có:

```cmd
.venv\Scripts\python.exe run_pipeline.py --from-processed
```

`--from-processed` trong chế độ đầy đủ cần `listings_clean.csv`, `reviews_clean.csv`,
`calendar_listing_agg.csv`, cache và nhãn tham chiếu tương ứng.
`city_review_stats.json` bổ sung thống kê cho báo cáo chi phí; GeoJSON cục bộ bổ sung đường biên cho bản đồ.

`--data-only` tạo header cho hai bảng KPI review/khía cạnh và hình thông báo bỏ qua
cho các hình 04–06. Chạy chế độ này trong cùng thư mục đầu ra sẽ thay các kết quả tương ứng trước đó.
`--call-llm` không kết hợp với `--offline` hoặc `--data-only`.

## 3. Cấu hình và mẫu dữ liệu

Cấu hình hiện tại: [configs/Buenos_Aires.yaml](configs/Buenos_Aires.yaml).

- `data_root`: `argentina/ciudad-autónoma-de-buenos-aires/buenos-aires`.
- `currency`: `ARS`; giá giữ nguyên tiền tệ nguồn, không tự quy đổi sang USD.
- `snapshots`: `2025-09-27`, `2025-12-29`, `2026-03-30`, `2026-06-29`.
- Thư mục mặc định: `data/raw`, `data/samples`, `data/processed`, `reports`, `figures`.
- Có thể đổi thư mục bằng `paths.raw`, `paths.samples`, `paths.processed`, `paths.reports`, `paths.figures`.
  Pipeline ưu tiên `paths.*`, đồng bộ với các khóa cũ như `raw_dir` và `samples_dir`.
- Đường dẫn tương đối được tính từ gốc repo, không phụ thuộc thư mục terminal hiện tại.

| File trong `data/samples` | Vai trò |
|---|---|
| `group_listings.csv` | Chọn listing và vùng phân tích cho listings/calendar. |
| `group_reviews.csv` | Chọn cặp review/listing, cờ mẫu LLM và cờ chọn nhãn tay. |
| `group_vung.csv` | Biên tọa độ các vùng phân công. |
| `hand_labels.csv` | Nhãn tham chiếu dùng để đối chiếu LLM và baseline. |

Bộ hiện tại gồm **1.000 listing**, **5.000 review**, **1.000 review chọn cho LLM**
và **100 review chọn để gán nhãn tham chiếu**. Đây là cỡ mẫu hiện tại, không phải
số dòng được ghi cứng trong các hàm đánh giá hay KPI.

CSV phân công hiện khớp các khoảng sau trên
[Google Sheet nguồn](https://docs.google.com/spreadsheets/d/1xaMiF78Ta90IWm2oHESKD35IHCGEcOMbGiYfUbmiwxA/edit):

| Tab | Khoảng ô | Số bản ghi |
|---|---|---:|
| Listing | `A5002:E6001` | 1.000 |
| Review | `A25002:G30001` | 5.000 |

Mã bộ dữ liệu nguồn là `DSE2049-C01-G06-buenos-aires` (nhóm 6, DSE 1).
Nhóm thực hiện dự án là nhóm 13, DSE 2 và sử dụng bộ phân công này theo lựa chọn đã xác nhận.
`group_code` dùng để lọc dữ liệu nguồn; không thay các chữ số trong listing/review ID để đổi tên nhóm.
`analysis_area` A01–A08 là vùng phân công, khác với `neighbourhood_cleansed` dùng để tính KPI khu vực.

Downloader mặc định đọc CSV local. Nếu cần đọc Sheet, dùng các tùy chọn
`--assignment-sheet`, `--listing-rows`, `--review-rows` của `src/download.py`
hoặc các khóa `assignment_sheet`, `assignment_ranges`, `assignment_tabs` trong YAML.
ID được xử lý dưới dạng số nguyên chính xác/chuỗi, không chuyển qua float.
Cờ `Nhãn tay=x` chỉ chọn review để gán nhãn, không phải nhãn cảm xúc.

## 4. Xử lý, kiểm tra và đánh giá

- **Download:** tải ba bảng CSV gzip và GeoJSON cho từng snapshot; kiểm tra cấu trúc
  và tính toàn vẹn, rồi lọc theo các ID được phân công.
- **QA:** gắn cờ thiếu/sai kiểu, ngoại lai, trùng, danh mục, tọa độ, ngày và liên kết.
  `qa_report.csv` ghi điều kiện, lý do, hành động và tác động khi loại riêng từng cờ.
  Cột `n_rows_after` là phép đối chiếu theo từng quy tắc, không phải số dòng thực tế
  bị xóa tích lũy. Review chứa HTML được chuẩn hóa và vẫn giữ nội dung.
- **LLM:** Gemini trích xuất cảm xúc `positive/mixed/negative`, ngôn ngữ và khía cạnh khen/chê.
  Cache gắn ID, cấu hình model/prompt/khía cạnh/độ dài và hash văn bản khi có.
- **Baseline:** từ điển và quy tắc không dùng LLM, dùng cùng bộ chọn mẫu và schema kết quả.
- **Evaluate:** đối chiếu hai phương pháp trên cùng toàn bộ ID có nhãn; thiếu dự đoán
  hoặc trùng ID sẽ báo lỗi. Báo cáo accuracy, precision/recall/F1 macro/micro,
  ma trận nhầm lẫn, tỷ lệ khớp khía cạnh và các trường hợp khác nhãn tham chiếu.

Macro cảm xúc dùng đủ ba lớp; khía cạnh được xem là các cặp tên/khen-chê.
Precision/recall/F1 có mẫu số bằng 0 được quy về 0; hai tập khía cạnh rỗng vẫn khớp chính xác.
Điểm số được lưu ở thang **0–1**. File lỗi chỉ ghi review có khác biệt và có thể có
hai dòng cho cùng review nếu cả LLM lẫn baseline đều khác nhãn tham chiếu.

`hand_labels.csv` hiện có 100 review được đề xuất nhãn ban đầu với AI hỗ trợ.
Theo xác nhận của người thực hiện, các nhãn đã được kiểm tra tay và chốt.
Cấu hình `evaluate.label_status: human_reviewed` ghi nhận trạng thái này;
`evaluation_summary.json` và biểu đồ so sánh mô hình được tạo lại từ bộ nhãn đã chốt.
Khi thay bộ nhãn mới chưa rà soát, cần đặt lại trạng thái thành `unverified`.

### Bảy KPI cho chủ nhà/đơn vị quản lý

| KPI | Công thức hoặc cách tính | Đơn vị |
|---|---|---|
| Số listing trong mẫu | Số ID listing duy nhất theo snapshot/vùng | listing |
| Giá trung vị hợp lệ | Trung vị giá hữu hạn, dương, có `price_valid=True` | ARS |
| Tỷ lệ nguyên căn | Số nguyên căn / số listing có loại phòng hợp lệ | 0–1 |
| Số đêm tối thiểu trung vị | Trung vị số đêm tối thiểu hữu hạn, dương | đêm |
| Tỷ lệ ngày không khả dụng | Tổng ngày không khả dụng / tổng ngày lịch quan sát | 0–1 |
| Tỷ lệ review tích cực | Review tích cực / review đã được LLM phân loại | 0–1 |
| Tỷ lệ phàn nàn theo khía cạnh | Review nhắc tiêu cực khía cạnh / review đã phân loại | 0–1 |

Các bảng ghi rõ tử số, mẫu số, cỡ mẫu và độ phủ. Khía cạnh có thể cùng xuất hiện
trong một review nên các tỷ lệ khía cạnh không nhất thiết cộng thành 100%.
KPI mô tả mẫu phân công, không mặc định đại diện cho toàn thành phố.

### Các giới hạn cần giữ khi diễn giải

- QA hiện yêu cầu `currency: ARS` và tạo `price_ars`; `price_valid` chỉ áp dụng cho
  snapshot cuối. Giá `price_num` ở mốc trước mới là giá đã parse, chưa phải giá đã qua đầy đủ QA.
- Ngày không khả dụng có thể là ngày chủ nhà chặn, không chứng minh lượt đặt phòng,
  công suất lưu trú hay doanh thu. Mẫu số lịch quan sát có thể khác nhau giữa các mốc.
- Review lặp qua các snapshot chỉ được tính một lần. Thời gian review là ngày khách viết;
  không dùng cùng mẫu lịch sử để tạo xu hướng cảm xúc giả qua các snapshot.
- Chi phí toàn thành phố được ngoại suy từ token trung bình trong cache và số review
  ở snapshot mới nhất. Cấu hình hiện dùng biểu giá Standard trả phí cho văn bản của
  `gemini-3.1-flash-lite`: 0,25 USD/triệu token input và 1,50 USD/triệu token output
  (gồm thinking), đối chiếu ngày 22/11/2026 tại
  [bảng giá Gemini](https://aistudio.google.com/docs/pricing#gemini-3.1-flash-lite).
  Với bộ hiện tại, 1.182.517 review có chi phí ước tính khoảng **227,37 USD**.
  Đây là kịch bản trả phí, không phải hóa đơn hay cam kết xử lý toàn thành phố trong
  hạn mức miễn phí; chưa gồm retry/lô lỗi và các chi phí phụ. Khi đổi model, cập nhật
  đơn giá trong YAML. Thiếu đơn giá thì `estimated_cost_usd` để trống, không coi là 0.

## 5. Kết quả đầu ra

| Vị trí | Nội dung chính |
|---|---|
| `data/processed` | `listings_subset.csv`, `calendar_subset.csv.gz`, `reviews_subset.csv`, `city_review_stats.json`; hai bảng sạch và hai bảng tổng hợp lịch; `llm_results.csv`, `baseline_results.csv`; bốn CSV KPI và `kpi_summary.json`. |
| `reports` | `qa_report.csv`, `data_profile.csv`, `llm_cost_estimate.csv` khi có thống kê; `evaluation_metrics.csv`, `evaluation_by_class.csv`, `evaluation_confusion.csv`, `evaluation_errors.csv`, `evaluation_summary.json`. |
| `figures` | Sáu hình, mặc định mỗi hình có PNG và SVG, cùng `viz_summary.json`. |

Bốn CSV KPI là `kpi_listings.csv`, `kpi_reviews.csv`, `kpi_aspects.csv`, `kpi_definitions.csv`.
Hai bảng sạch là `listings_clean.csv`, `reviews_clean.csv`; hai bảng lịch là
`calendar_listing_agg.csv`, `calendar_month_agg.csv`.

Từ điển dữ liệu nằm tại [reports/data_dictionary.csv](reports/data_dictionary.csv).
Mỗi dòng mô tả một cột của một bảng CSV: `file`, `column`, `dtype` (kiểu dữ liệu logic),
`description`, `unit`, `rule_or_notes`. Tệp bao phủ các CSV phân công, tập con, dữ liệu
sạch, lịch tổng hợp, nhãn tham chiếu, kết quả mô hình, KPI và báo cáo dạng bảng hiện có.
Khi thay dữ liệu có thêm hoặc đổi cột, cần cập nhật từ điển theo schema mới.

| Tên hình | Nội dung |
|---|---|
| `01_listing_map` | Tọa độ listing ở mốc cuối, màu theo giá hợp lệ; giá thiếu được thể hiện riêng. |
| `02_prices_by_area` | Giá trung vị của các khu vực đủ mẫu. |
| `03_snapshot_trends` | Số listing và tỷ lệ ngày không khả dụng qua snapshot. |
| `04_review_sentiment_yearly` | Review tích cực theo năm viết, có cỡ mẫu và ngưỡng hiển thị. |
| `05_negative_aspects` | Tỷ lệ nhắc tiêu cực từng khía cạnh. |
| `06_model_comparison` | Accuracy và macro-F1 cảm xúc của hai phương pháp, kèm trạng thái nhãn. |

`viz_summary.json` ghi nguồn, checksum, cỡ mẫu và trạng thái từng hình.
Mặc định hình giá chọn tối đa 12 khu vực có nhiều giá hợp lệ nhất, tối thiểu 5 giá/vùng;
hình tỷ lệ review theo năm cần ít nhất 20 review được phân loại trong năm.
Có thể đổi `viz.top_n`, `viz.min_group_n`, `viz.min_review_n`, `viz.formats`, `viz.dpi` trong YAML.
Thiếu dữ liệu hợp lệ được thể hiện bằng hình thông báo hoặc khoảng trống, không dựng số 0 giả.

## 6. Khóa API và file môi trường

File [.env.example](.env.example) chỉ là mẫu tên biến, không chứa khóa thật.
Chỉ cần khóa khi chủ động gọi Gemini, ví dụ `--call-llm`.

Nếu chưa có `.env`, có thể tạo từ mẫu bằng CMD:

```cmd
if not exist .env copy .env.example .env
```

Sau đó tự điền `GEMINI_API_KEY` trong `.env` local. Giữ nguyên `.env` đang có nếu đã cấu hình.
Biến môi trường được ưu tiên hơn `.env`; `.env` không được commit hay chia sẻ.
Tên biến có thể đổi bằng `llm.api_key_env`; model và các tham số LLM được đặt trong YAML.

Lệnh chỉ đọc cache khi chạy riêng bước LLM:

```cmd
.venv\Scripts\python.exe src/llm.py --skip-llm
```

Chạy riêng `src/llm.py` mà không có `--skip-llm` có thể gọi API khi cache thiếu.
Ở `src/kpi.py` và `src/viz.py`, `--skip-llm` mang nghĩa bỏ phần review;
điều này khác với chế độ đọc cache của `run_pipeline.py --skip-llm`.

## 7. Lưu trữ và bàn giao

Raw, môi trường ảo và `.env` được loại khỏi Git. `.gitignore` hiện cũng bỏ qua
toàn bộ `data/processed`; vì vậy cache có trên máy chưa đồng nghĩa đã có trong bản clone.

Bản nộp chạy đầy đủ không API cần có trong Git:

- Các CSV phân công và vùng được dùng trong `data/samples`.
- `data/samples/hand_labels.csv` phù hợp với bộ dữ liệu và được khai báo đúng trạng thái.
- `data/processed/llm_cache.jsonl` hợp lệ, hoặc cache ở đường dẫn cấu hình tương ứng.
- Mã nguồn, YAML, `.python-version`, `requirements.txt`, README và `.env.example`.

Ví dụ thêm mẫu và cache mặc định một cách tường minh, không thêm raw hay `.env`:

```cmd
git add data/samples/group_listings.csv data/samples/group_reviews.csv data/samples/group_vung.csv data/samples/hand_labels.csv
git add -f data/processed/llm_cache.jsonl
```

Kiểm tra và commit các tệp trên trước khi thử bằng một bản clone mới.
Không đưa khóa thật vào bất kỳ tệp được commit nào. Chỉ commit dữ liệu processed
phù hợp giới hạn kích thước của đề bài; không cần ép thêm toàn bộ thư mục này.

Trước khi nộp còn cần hoàn thiện báo cáo PDF kỹ thuật (tối đa 10 trang), đồng thời
kiểm tra thông tin người thực hiện và [AI_USAGE.md](AI_USAGE.md).
Ghi commit SHA vào báo cáo và tạo/push tag `final` theo hướng dẫn môn học.

## 8. Khi thay dữ liệu hoặc snapshot

Đổi `--config` sang YAML phù hợp và thay các CSV phân công tương ứng; giữ schema
đầu vào được các module yêu cầu. Mốc mới chưa có cache/nhãn có thể kiểm tra phần dữ liệu bằng:

```cmd
.venv\Scripts\python.exe run_pipeline.py --config configs/Buenos_Aires.yaml --data-only
```

Thay đường dẫn YAML ở lệnh trên bằng file cấu hình mới khi cần.
Muốn đánh giá đầy đủ phải có dự đoán/cache và nhãn khớp ID của chính bộ dữ liệu đó.
Cache cũ không được coi là phù hợp chỉ vì số dòng bằng nhau.
KPI/viz đọc tiền tệ từ cấu hình, nhưng QA hiện dành cho ARS; nếu đổi sang tiền tệ
khác cần điều chỉnh hợp đồng QA trước khi chạy toàn bộ, không chỉ đổi chữ `currency`.

