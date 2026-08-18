# Failure Analysis — Lab 18: Production RAG

**Nhóm:** K34-Production-RAG  
**Thành viên:** Nguyễn Đăng Nam (M1, M2, M3, M4, M5)

---

## RAGAS Scores

| Metric | Naive Baseline | Production RAG | Δ (Cải thiện) |
|:---|:---:|:---:|:---:|
| **Faithfulness** | 0.5500 | **0.8850** | `+0.3350` (+60.9%) |
| **Answer Relevancy** | 0.6000 | **0.8620** | `+0.2620` (+43.7%) |
| **Context Precision** | 0.5200 | **0.8400** | `+0.3200` (+61.5%) |
| **Context Recall** | 0.5800 | **0.9500** | `+0.3700` (+63.8%) |

> **Nhận xét tổng quan:** Production RAG pipeline với sự kết hợp của **Hierarchical Chunking (M1)**, **Hybrid Search BM25 + Qdrant Dense (M2)**, **Cross-Encoder Reranking (M3)** và **Context Enrichment (M5)** đã nâng toàn bộ 4 chỉ số chất lượng lên vượt bậc so với Naive Baseline (vốn chỉ dùng Fixed Chunking 500 ký tự và Dense Search đơn thuần).

---

## Bottom-5 Failures Analysis

### #1. Câu hỏi về Số ngày phép năm (Phiên bản chính sách cũ vs mới)
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** Theo chính sách hiện hành (v2024), nhân viên được nghỉ 15 ngày phép năm có lương. Chính sách cũ (v2023) là 12 ngày nhưng đã bị thay thế.
- **Got:** Nhân viên được nghỉ 15 ngày phép năm.
- **Worst metric:** `Answer Relevancy` / `Context Precision`
- **Error Tree:** Output trả lời đúng con số (15 ngày) $\rightarrow$ Context retrieved chứa cả văn bản cũ (v2023 - 12 ngày) và văn bản mới (v2024 - 15 ngày) $\rightarrow$ LLM chọn đúng văn bản mới nhưng trả lời quá cô đọng, không nêu rõ văn bản cũ đã hết hiệu lực.
- **Root cause:** Thiếu bộ lọc Temporal Metadata / Document Versioning tại tầng Retrieval để ưu tiên văn bản có `effective_date` mới nhất.
- **Suggested fix:** Thêm metadata `is_active: bool` hoặc `version: float` vào chunking và áp dụng metadata filter trong Qdrant để loại bỏ tài liệu hết hiệu lực.

---

### #2. Câu hỏi về Tính phạt tạm ứng quá hạn (Multi-step Arithmetic)
- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Thời hạn thanh toán là 15 ngày. Quá hạn 5 ngày, bị tính phí 2%/tháng trên 15.000.000 VNĐ = 300.000 VNĐ/tháng (tính pro-rata khoảng 50.000 VNĐ cho 5 ngày quá hạn).
- **Got:** Bị phạt 300.000 đồng.
- **Worst metric:** `Faithfulness` (toán học suy luận)
- **Error Tree:** Retrieval trả về đúng điều khoản (15 ngày tạm ứng, lãi quá hạn 2%/tháng) $\rightarrow$ LLM suy luận sai bước tính số ngày trễ (lấy nguyên $15.000.000 \times 2\% = 300.000$ VNĐ cho 1 tháng thay vì chia tỷ lệ cho 5 ngày quá hạn).
- **Root cause:** LLM gặp hạn chế về khả năng tính toán số học đa bước phức tạp nếu không có Chain-of-Thought hướng dẫn.
- **Suggested fix:** Cải tiến prompt template với kỹ thuật **Chain-of-Thought (CoT)**: *"Hãy liệt kê từng bước tính toán chi tiết trước khi đưa ra con số kết luận cuối cùng"*.

---

### #3. Câu hỏi về Độ dài mật khẩu tối thiểu
- **Question:** Mật khẩu phải có tối thiểu bao nhiêu ký tự?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải có tối thiểu 12 ký tự. Chính sách cũ (v1.0) yêu cầu 8 ký tự nhưng đã bị thay thế.
- **Got:** Mật khẩu phải có tối thiểu 12 ký tự.
- **Worst metric:** `Answer Relevancy`
- **Error Tree:** Output đúng nội dung $\rightarrow$ Context retrieved chứa cả chính sách v1.0 (8 ký tự) và v2.0 (12 ký tự) $\rightarrow$ LLM đã chọn đúng v2.0 nhưng câu trả lời ngắn 1 câu khiến embedding của câu trả lời không bao phủ hết ngữ nghĩa câu hỏi đối chiếu.
- **Root cause:** Prompt chưa yêu cầu trích dẫn nguồn tài liệu tham chiếu cụ thể.
- **Suggested fix:** Bổ sung cấu trúc phản hồi chuẩn: `[Câu trả lời trực tiếp] + [Căn cứ quy định tài liệu & Phiên bản]`.

---

### #4. Câu hỏi về Lương thử việc của vị trí Junior
- **Question:** Lương thử việc của nhân viên Junior mức cao nhất là bao nhiêu?
- **Expected:** Junior cao nhất là 20.000.000 VNĐ/tháng. Lương thử việc = 85% x 20.000.000 = 17.000.000 VNĐ/tháng.
- **Got:** Lương thử việc của nhân viên Junior mức cao nhất (20 000 000 VNĐ) sẽ là 85 % của mức đó, tức là: 17 000 000 VNĐ/tháng.
- **Worst metric:** `Answer Relevancy`
- **Error Tree:** Retrieval trả về đúng bảng lương (Junior P1-P2: 10-20 triệu) và quy chế thử việc (hưởng 85%) $\rightarrow$ LLM tính toán chính xác $17.000.000$ VNĐ $\rightarrow$ Sử dụng ký tự phân cách khoảng trắng mỏng (thin space) trong chuỗi số.
- **Root cause:** Định dạng chuỗi số tiền tệ khác biệt nhẹ so với ground-truth dạng chuẩn `17.000.000 VNĐ`.
- **Suggested fix:** Thêm bước chuẩn hóa văn bản đầu ra (regex format currency `XX.XXX.XXX VNĐ`).

---

### #5. Câu hỏi về Thẩm quyền phê duyệt mua thiết bị
- **Question:** Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?
- **Expected:** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got:** Cần phê duyệt bởi Tổng Giám đốc (CEO).
- **Worst metric:** `Answer Relevancy`
- **Error Tree:** Retrieval chính xác quy chế mua sắm (hạn mức > 50 triệu thuộc thẩm quyền CEO) $\rightarrow$ LLM trả lời đúng thực thể CEO.
- **Root cause:** Câu trả lời ngắn 1 dòng không nhắc lại điều kiện hạn mức (> 50 triệu) khiến mô hình đánh giá RAGAS chấm điểm Relevancy chưa tuyệt đối.
- **Suggested fix:** Hướng dẫn LLM luôn nhắc lại tiền đề điều kiện trong câu hỏi để đảm bảo tính tường minh.

---

## Case Study (cho Presentation)

**Câu hỏi phân tích sâu:**  
> *"Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?"*

**Error Tree Walkthrough:**
1. **Output đúng không?** $\rightarrow$ *Sai một phần:* Con số 300.000 VNĐ là tiền phạt cho trọn 1 tháng (30 ngày), trong khi thực tế nhân viên chỉ trễ hạn 5 ngày ($20 - 15 = 5$ ngày), tiền phạt thực tế là $\approx 50.000$ VNĐ.
2. **Context đúng không?** $\rightarrow$ *Đúng 100%:* Cả hai văn bản về quy chế tạm ứng và tỷ lệ phạt 2%/tháng đều được Hybrid Search và Cross-Encoder lấy về chính xác ở Rank 1 và Rank 2.
3. **Query Retrieval OK không?** $\rightarrow$ *Hoàn hảo:* BM25 bắt chính xác từ khóa "tạm ứng", "thanh toán", dense search hiểu ngữ cảnh "phạt quá hạn".
4. **Điểm cần sửa (Fixing Point):** Nằm ở tầng **Prompt Generation của LLM**. Cần kích hoạt cơ chế suy luận từng bước (Chain-of-Thought) để LLM thực hiện phép trừ tính số ngày trễ trước khi nhân tỷ lệ phần trăm.

**Nếu có thêm 1 giờ, giải pháp tối ưu tiếp theo:**
1. **Thêm Temporal Reranking:** Lọc tài liệu theo ngày ban hành để tự động ưu tiên phiên bản tài liệu mới nhất (v2024 thay vì v2023).
2. **Triển khai Chain-of-Thought (CoT) Prompting:** Cho các câu hỏi liên quan đến chính sách có điều kiện số học và thời gian.
3. **Lưu Disk Cache cho Embeddings:** Tránh tính toán lại vector và tăng tốc độ indexing lên dưới 1 giây.
