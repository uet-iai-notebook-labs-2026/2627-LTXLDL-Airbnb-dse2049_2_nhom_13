# AI_USAGE.md

Dự án: Phân tích dữ liệu Airbnb tại Buenos Aires — nhóm 13, lớp UET.DSE2049 2.

## Công cụ đã dùng (vào việc gì)

AI được sử dụng để hỗ trợ định hướng, phát triển và kiểm tra dự án. Phạm vi hỗ trợ
bao gồm đề xuất cách xử lý, bổ sung/sửa mã theo yêu cầu, giải thích hàm, viết tài liệu,
thiết kế kiểm thử và gợi ý nhãn review. Dữ liệu nguồn lấy từ Inside Airbnb và bảng
phân công của môn học.

| Công cụ | Công việc đã sử dụng | Tệp hoặc đầu ra liên quan |
|---|---|---|
| ChatGPT/Codex | Hỗ trợ hoàn thiện mã, xử lý lỗi, thêm giải thích chức năng, thiết kế/chạy kiểm thử tạm; đề xuất nhãn ban đầu; hỗ trợ tài liệu và đối chiếu đầu ra. | Các module trong `src/`, `run_pipeline.py`, `data/samples/hand_labels.csv`, `README.md`, `.env.example`, `AI_USAGE.md`, `reports/data_dictionary.csv`. |
| Gemini API | Trích xuất cảm xúc, ngôn ngữ và các khía cạnh khen/chê từ mẫu review; lưu kết quả để tái sử dụng và đánh giá. | `src/llm.py`, `data/processed/llm_cache.jsonl`, `data/processed/llm_results.csv`. |

Model đang cấu hình là `gemini-3.1-flash-lite`. Pipeline mặc định đọc cache đã lưu;
chỉ gọi API khi chọn tường minh `--call-llm`. Khóa API thật được giữ ở môi trường
local và không ghi trong tài liệu hoặc các tệp dùng để bàn giao.

## Prompt then chốt (5-10): bước nào, vào tệp nào

Bảng sau tóm tắt 10 nhóm yêu cầu chính đã trao đổi, không phải trích nguyên văn
hoặc nhật ký đầy đủ của mọi tin nhắn.

| STT | Nội dung yêu cầu hỗ trợ | Bước thực hiện | Tệp liên quan |
|---|---|---|---|
| 1 | Đọc repo và đề bài, chỉ ra phần còn thiếu và thứ tự nên thực hiện. | Rà soát yêu cầu, xác định hướng phân tích. | `README.md`, cấu trúc repo. |
| 2 | Hoàn thiện tải/lọc dữ liệu theo CSV phân công và các mốc cấu hình; kiểm tra dữ liệu và quy tắc QA. | Download và QA. | `src/download.py`, `src/qa.py`, `configs/Buenos_Aires.yaml`. |
| 3 | Hoàn thiện phương pháp đối chứng không dùng LLM, dùng cùng mẫu review và schema kết quả. | Baseline. | `src/baseline.py`. |
| 4 | Hoàn thiện trích xuất bằng Gemini, đọc khóa từ môi trường, kiểm tra đầu ra và dùng cache. | LLM. | `src/llm.py`, `.env.example`, `data/processed/llm_cache.jsonl`. |
| 5 | Gợi ý nhãn cho 100 review được chọn và điền nhãn đề xuất vào file tham chiếu. | Chuẩn bị nhãn để người thực hiện rà soát. | `data/samples/hand_labels.csv`. |
| 6 | Đánh giá LLM và baseline trên cùng bộ nhãn; kiểm tra công thức, ID lớn và các trường hợp thiếu/trùng dự đoán. | Evaluate. | `src/evaluate.py`, `reports/evaluation_*.csv`, `reports/evaluation_summary.json`. |
| 7 | Đề xuất bảy KPI phục vụ chủ nhà/đơn vị quản lý, tính đúng mẫu số, khu vực, thời gian và độ phủ. | KPI. | `src/kpi.py`, `data/processed/kpi_*.csv`. |
| 8 | Vẽ hình không gian, thời gian, phản hồi khách và so sánh mô hình; ghép các module thành một lệnh chạy. | Viz và pipeline. | `src/viz.py`, `figures/`, `run_pipeline.py`. |
| 9 | Thêm giải thích công dụng hàm, hướng dẫn cài/chạy và viết khai báo AI theo tiêu chí môn học. | Hoàn thiện tài liệu. | Docstring trong mã, `README.md`, `.env.example`, `AI_USAGE.md`. |
| 10 | Hoàn thiện từ điển dữ liệu và ước tính chi phí toàn thành phố bằng đơn giá chính thức, kiểm tra từ cache. | Mô tả dữ liệu và chi phí LLM. | `reports/data_dictionary.csv`, `configs/Buenos_Aires.yaml`, `reports/llm_cost_estimate.csv`. |

## AI sai ở đâu (ví dụ thật)

Các lỗi sau được phát hiện khi kiểm tra bản nháp hoặc rà soát nội dung trước khi chốt:

| Sai sót | Cách phát hiện | Cách sửa |
|---|---|---|
| Bản nháp viz chưa từ chối trường hợp đơn vị giá trong KPI khác tiền tệ cấu hình, có thể ghi sai nhãn đơn vị trên hình. | Thay tiền tệ/đơn vị trong fixture và đối chiếu dữ liệu với nhãn hình. | Bổ sung kiểm tra nhất quán đơn vị trước khi xuất hình. |
| Bản nháp viz chưa chặn đường dẫn đầu ra trùng tệp đầu vào khi dùng đường dẫn tùy chỉnh. | Kiểm thử cấu hình có đường dẫn va chạm, đối chiếu nội dung tệp trước/sau. | Kiểm tra va chạm đường dẫn trước khi ghi, bảo toàn tệp nguồn. |
| Bản nháp từ điển mô tả KPI giá dùng trực tiếp `price_ars`. Mã KPI/viz mặc định đọc `price_num` và lọc `price_valid`. | Đối chiếu dòng mô tả với cách chọn cột giá trong `src/kpi.py` và `src/viz.py`. | Sửa mô tả: KPI mặc định đọc `price_num` và lọc `price_valid`; `price_ars` giữ cùng giá ở các dòng hợp lệ. |

## Nhóm kiểm chứng thế nào

Nhóm hiện có một người thực hiện. Người thực hiện cung cấp repo, file khung và
dữ liệu phân công; xác nhận bộ dữ liệu, chọn hướng phục vụ chủ nhà/đơn vị quản lý,
duyệt phạm vi sửa, chạy các module trên máy và phản hồi lỗi hoặc nội dung chưa hiểu.

Các kiểm thử kỹ thuật được thực hiện trong môi trường hỗ trợ bằng fixture hoặc
bản sao dữ liệu. Theo yêu cầu, file test được giữ ngoài repo. Những con số dưới đây
ghi nhận kiểm tra đã chạy trong các phiên hỗ trợ, không phải bộ test đã commit.

| Phần | Nội dung kiểm chứng đã thực hiện |
|---|---|
| Evaluate | 26 trường hợp: công thức tính điểm, ID lớn, thiếu/trùng dự đoán, nhãn sai, khía cạnh khen/chê và dữ liệu/đường dẫn khác. |
| KPI | 25 trường hợp: mẫu số không đều, dữ liệu thiếu, review lặp qua snapshot, nối khu vực theo mốc nguồn và cấu hình khác. |
| Viz | 17 trường hợp: dữ liệu hình, tỷ lệ gộp theo năm, thiếu nguồn, checksum, đơn vị và đường dẫn. Các hình PNG/SVG trên dữ liệu thật được mở kiểm tra. |
| Pipeline | 19 trường hợp: thứ tự bước, chế độ cache/offline/data-only, cache thiếu/sai và dữ liệu giả lập chạy đủ bảy bước; chặn lời gọi API trong các kiểm thử liên quan. |
| Chi phí LLM | 11 trường hợp: công thức, đơn giá thiếu hoặc bằng 0, thống kê thiếu/sai, ước tính token dự phòng và token thinking. Đối chiếu bằng phép tính Decimal độc lập; thay đơn giá không làm mất hiệu lực cache. |
| Từ điển dữ liệu | Đối chiếu đủ 421 cột của 25 CSV hiện tại, không thiếu/trùng cặp tệp-cột; đọc lại CSV sau xuất để kiểm tra nội dung. |

Đã đối chiếu đầu ra: LLM và baseline cùng có 1.000 review; cả 100 ID tham chiếu
đều có dự đoán của hai phương pháp. KPI được kiểm tra bằng phép tính độc lập về
số listing, giá trung vị, số ngày lịch và tỷ lệ review tích cực. Khi thêm docstring,
đã kiểm tra cú pháp và so sánh cây cú pháp sau khi bỏ docstring để kiểm tra logic.

100 nhãn tham chiếu được đề xuất ban đầu với AI hỗ trợ. Theo xác nhận của người
thực hiện, các nhãn sau đó đã được kiểm tra tay và chốt. Cấu hình và báo cáo ghi
`label_status: human_reviewed`; nguồn gợi ý ban đầu vẫn được khai báo. Bộ nhãn hiện
chỉ có tiếng Anh và có ít mẫu negative, nên cần giữ giới hạn này khi diễn giải điểm.

Người thực hiện chịu trách nhiệm đọc hiểu, kiểm chứng và giải thích mã nguồn, nhãn,
công thức cùng số liệu đã nộp. Các kiểm tra có hỗ trợ công cụ nêu trên không đồng
nghĩa mọi phần đã được người thực hiện kiểm chứng độc lập.
