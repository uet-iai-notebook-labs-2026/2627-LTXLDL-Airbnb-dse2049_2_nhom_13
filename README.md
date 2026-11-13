# Bài tập lớn cuối kỳ — Inside Airbnb

**Môn học:** Lập trình xử lý dữ liệu (UET.DSE2049), học kỳ 1, năm học 2026–2027.

**Đề bài chính thức:** [Bài tập lớn Inside Airbnb](https://courses.iaidev.com/programming-for-data-processing/2627-1/projects/project_airbnb.html).

Đọc đầy đủ đề bài, phân công nhóm/thành phố và thông báo trên Canvas trước khi bắt đầu. Yêu cầu, thời hạn và rubric áp dụng theo thông báo chính thức mới nhất.

Repo template ban đầu chỉ có README này. Nhóm tự xây dựng mã nguồn, cấu hình và các sản phẩm theo đề bài, đồng thời cập nhật các mục bên dưới trong quá trình làm.

## Tạo repo của nhóm

1. Một thành viên mở [airbnb-project-template](https://github.com/tiendat-aops/airbnb-project-template), chọn **Use this template → Create a new repository**.
2. Chọn **Owner: `uet-iai-notebook-labs-2026`** và **Private**. Chỉ tạo một repo cho cả nhóm.
3. Tên gợi ý: `2627-LTXLDL-Airbnb-<TenNhom>`. Ví dụ: `2627-LTXLDL-Airbnb-dse2049_7-nhom01`.
4. Cấp quyền làm việc cho các thành viên trong nhóm; bảo đảm giảng viên phụ trách có quyền truy cập.
5. Điền URL repo vào bảng đăng ký do giảng viên cung cấp.

Nếu chưa có quyền đọc template hoặc tạo repo trong org, liên hệ giảng viên.

## Thông tin nhóm

- **Mã lớp:** TODO
- **Tên/mã nhóm:** TODO
- **Thành phố được phân công:** TODO
- **Bên đặt hàng phân tích:** TODO
- **Bốn mốc dữ liệu:** TODO — theo thành phố trong đề bài.

| Họ tên | MSSV | GitHub username | Công việc phụ trách |
|---|---|---|---|
| TODO | TODO | TODO | TODO |

Thêm dòng cho đầy đủ thành viên. Mỗi thành viên cần hiểu và giải thích được toàn bộ sản phẩm của nhóm.

## Cài đặt và chạy

Nhóm thay các mục TODO bằng hướng dẫn đã kiểm thử trên bản clone mới:

- **Phiên bản Python:** TODO — ghi phiên bản cụ thể trong `.python-version`.
- **Cài môi trường và thư viện:** TODO — cung cấp lệnh cài và `requirements.txt` ghim phiên bản hoặc lockfile tương đương.
- **Cấu hình thành phố, snapshot và tham số:** TODO — ghi đường dẫn và cách điều chỉnh.
- **Biến môi trường chứa API key:** TODO — ghi tên biến và nhà cung cấp; không ghi khóa thật.
- **Một lệnh chạy toàn bộ pipeline, bao gồm LLM:** TODO.
- **Các tệp kết quả để đối chiếu:** TODO — dữ liệu xử lý, QA, KPI, kết quả LLM, hình và báo cáo.

## Lưu ý khi làm bài

- Commit và push thường xuyên từ đầu kỳ.
- Tạo `.gitignore` trước khi thêm dữ liệu hoặc khóa API: loại `data/raw/`, `.venv/`, `.env` và mọi tệp chứa khóa thật.
- Tạo và cập nhật `AI_USAGE.md` ở gốc repo theo yêu cầu minh bạch AI trong đề bài.
- Mã nguồn phải đọc khóa API từ biến môi trường để giảng viên có thể chạy lại bằng khóa riêng.
- Tự kiểm tra bằng cách clone repo sang thư mục mới, cài môi trường theo README và chạy lại toàn bộ pipeline.

## Nộp bài

Sau khi hoàn tất sản phẩm, cập nhật báo cáo và đẩy commit cuối lên GitHub, tạo tag theo đề bài:

```bash
git tag final
git push origin final
```

Ghi commit SHA được nộp vào báo cáo. Thời hạn, cách đăng ký và lịch vấn đáp xem tại đề bài và Canvas.

## Bước QA đã triển khai

`src/qa.py` hoàn thiện khung QA được cung cấp; `src/common.py` cung cấp đường dẫn
và logger. Phần này đã kiểm thử với Python 3.13.7. Các mô-đun KPI/LLM/viz và
`run_pipeline.py` hiện chưa được triển khai; lệnh dưới đây chỉ chạy QA.

Từ thư mục gốc repo, cài môi trường (nếu đã có `.venv` thì dùng lại):

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

QA yêu cầu ba bảng subset trong `data/processed/`:

- `listings_subset.csv`: `id`, `snapshot`, `analysis_area`, `price`, `room_type`,
  `latitude`, `longitude`, `neighbourhood_cleansed`, `calculated_host_listings_count`.
  Nên giữ thêm `minimum_nights`, `number_of_reviews`, `review_scores_rating`,
  `last_review` để các quy tắc tương ứng có dữ liệu kiểm tra.
- `calendar_subset.csv.gz`: `listing_id`, `snapshot`, `date`, `available`,
  `minimum_nights`. Không bắt buộc giá vì calendar thực tế có mốc không chứa cột giá.
- `reviews_subset.csv`: `id`, `listing_id`, `snapshot`, `date`, `comments`,
  `llm_sample`, `hand_label`; giữ thêm `reviewer_id` nếu có.

Repo chưa có `src/subset.py` và các file subset chính thức. Khi chuẩn bị subset,
lọc theo `data/samples/listing_ids.csv`, nối `analysis_area` theo `id`, thêm cột
`snapshot` từ tên thư mục dữ liệu thô. ID phải đọc trực tiếp dạng int64/Int64 hoặc
chuỗi, không đi qua float. Ngày dùng `YYYY-MM-DD`.

Khi đã có subset, chạy:

```powershell
.venv\Scripts\python.exe -m src.qa --config configs/Buenos_Aires.yaml
```

Kết quả gồm bốn bảng trong `data/processed/`: `listings_clean.csv`,
`reviews_clean.csv`, `calendar_listing_agg.csv`, `calendar_month_agg.csv`;
và `reports/qa_report.csv`, `reports/data_profile.csv`.

Quy ước cần hiểu khi dùng đầu ra:

- Listings/reviews giữ nguyên số dòng và thêm cờ; calendar loại các dòng sai ngày,
  trùng, trạng thái không hợp lệ, ngoài cửa sổ hoặc không nối được listing cùng mốc.
- `price_valid` và `price_ars` chỉ dành cho snapshot cuối theo khung; muốn phân tích
  giá nhiều mốc phải dùng `price_num` và áp cờ phù hợp ở từng mốc.
- `occ_H` dùng các ngày quan sát hợp lệ trong `[snapshot, snapshot + H ngày)`;
  thiếu cả cửa sổ thì trả NaN. Ngày không khả dụng có thể do chủ nhà chặn lịch.
- `use_for_llm` là điều kiện QA; bước LLM còn cần lọc theo `llm_sample`/`hand_label`.
  Review trùng ID qua các mốc chỉ giữ lần đầu cho điều kiện này.
- `pct_flagged`/`pct_null` dùng thang 0–100; các tỷ lệ calendar dùng thang 0–1.
  `n_rows_after` và `metric_after` mô phỏng loại riêng từng quy tắc, không phải
  số dòng thực bị xoá tích luỹ. Profile ghi dtype sau khi đọc CSV (ưu tiên chuỗi
  để không làm tròn ID), trước khi ép số và thêm cờ.
- Các ngưỡng QA và bbox trong YAML là giá trị khởi đầu, cần nhóm giải thích và
  điều chỉnh dựa trên khảo sát; không phải kết luận về thị trường.
