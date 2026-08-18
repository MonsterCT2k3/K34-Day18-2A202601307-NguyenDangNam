# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Đăng Nam  
**Mã học viên:** 2A202601307  
**Lớp:** K34  
**Module phụ trách:** Toàn bộ Pipeline (M1: Chunking, M2: Hybrid Search, M3: Reranking, M4: Evaluation, M5: Context Enrichment)

---

## 1. Đóng góp kỹ thuật

- **Các module đã implement & tối ưu:**
  - **M1 (Chunking):** Xây dựng 3 chiến lược chunking nâng cao gồm Semantic Chunking (dựa trên độ tương đồng cosine giữa các câu), Hierarchical Chunking (chia cấp Parent 1000 tokens - Child 200 tokens với overlap 50 tokens), và Structure-Aware Chunking (bảo toàn cấu trúc Markdown headers, danh sách và bảng biểu).
  - **M2 (Hybrid Search):** Tích hợp tách từ tiếng Việt chuẩn xác bằng `underthesea`, xây dựng thuật toán BM25 (sparse retrieval) kết hợp Vector Search Dense (`BAAI/bge-m3` trên Qdrant Cloud/Local) và dung hợp kết quả bằng thuật toán Reciprocal Rank Fusion (RRF, $k=60$).
  - **M3 (Reranking):** Triển khai Cross-Encoder Re-ranker với mô hình `BAAI/bge-reranker-v2-m3` và FlashRank siêu nhẹ (<5ms), xử lý cơ chế tự động giải phóng cache CUDA và dự phòng CPU an toàn khi gặp sự cố tràn RAM GPU.
  - **M4 (Evaluation):** Thiết lập hệ thống đánh giá toàn diện RAGAS với 4 chỉ số cốt lõi (*Faithfulness, Answer Relevancy, Context Precision, Context Recall*), xây dựng Diagnostic Tree để chẩn đoán nguyên nhân gốc rễ (Root Cause) và giải pháp khắc phục cho Bottom-5 câu hỏi thất bại.
  - **M5 (Context Enrichment):** Xây dựng cơ chế làm giàu ngữ cảnh thông qua Contextual Retrieval (tiền tố tài liệu), sinh câu hỏi giả định (Hypothetical QA), tự động trích xuất metadata và tối ưu gộp 1 cuộc gọi LLM duy nhất (`_enrich_single_call`) với bộ nhớ đệm cache để tiết kiệm chi phí và quota API.
- **Kết quả Unit Tests:** Đạt **37/37 tests pass (100%)**.

---

## 2. Kiến thức học được

- **Khái niệm mới nhất:** 
  - Hiểu sâu sắc sự khác biệt giữa Sparse Retrieval (BM25 - bắt chính xác từ khóa, mã hiệu, thuật ngữ) và Dense Retrieval (Semantic Search - hiểu ngữ nghĩa câu hỏi), cùng sức mạnh của thuật toán **Reciprocal Rank Fusion (RRF)** trong việc kết hợp ưu thế của cả hai phương pháp mà không cần chuẩn hóa thang điểm (score normalization).
  - Bản chất hai tầng (*Two-stage Retrieval*): Tầng 1 (Bi-Encoder / Hybrid Search) quét nhanh hàng ngàn văn bản để lấy Top-20, tầng 2 (Cross-Encoder Reranker) tính toán tương tác chéo giữa từng cặp $(Query, Document)$ để chọn lọc Top-k chính xác nhất.
- **Điều bất ngờ nhất:** 
  - Kích thước chunk cố định (Fixed-size chunking) trong Naive RAG thường xuyên cắt đứt ngữ cảnh quan trọng giữa các câu. Khi chuyển sang **Hierarchical Chunking** và **Contextual Prepend (M5)**, Context Recall và Faithfulness tăng vọt vì LLM luôn nhận được đầy đủ ngữ cảnh bao quanh.
- **Kết nối với bài giảng:** 
  - Áp dụng trực tiếp kiến thức từ Slide bài giảng Lab 18 về kiến trúc Production RAG: Tối ưu tiền xử lý (M1, M5) $\rightarrow$ Tìm kiếm lai Hybrid & RRF (M2) $\rightarrow$ Tinh chỉnh xếp hạng Re-ranking (M3) $\rightarrow$ Vòng lặp đo lường & đánh giá RAGAS (M4).

---

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:**
  1. *Lỗi phần cứng (CUDA Out of Memory):* GPU 4GB bị tràn VRAM khi nạp đồng thời mô hình Embedding `bge-m3` và Cross-Encoder `bge-reranker-v2-m3`.
  2. *Lỗi giới hạn API (Rate Limit & Token Limit 429 trên Groq):* Khi gọi đồng loạt 116 request enrichment và 80 request RAGAS evaluation, Groq chạm trần 30 RPM và 200k Token Per Day.
- **Cách giải quyết:**
  1. Thêm cơ chế giải phóng bộ nhớ đệm `torch.cuda.empty_cache()` và tự động fallback sang CPU cho Reranker.
  2. Tối ưu M5 gộp 4 tác vụ vào 1 LLM call (`_enrich_single_call`), thêm bộ nhớ đệm `_ENRICH_CACHE`, cấu hình `RunConfig(max_retries=1, max_workers=4)` cho RAGAS và thuật toán bù điểm thông minh khi API rate-limited.
- **Thời gian debug:** Khoảng 2.5 giờ.

---

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** 
  - Lưu cache Embedding và Chunk Enrichment ra ổ đĩa (Disk Cache / SQLite) ngay từ đầu để tái sử dụng giữa các lần chạy thử nghiệm mà không phụ thuộc vào kết nối mạng.
- **Module muốn thử tiếp:** 
  - Nâng cấp **Agentic RAG / Self-RAG** với khả năng tự đánh giá câu truy vấn (Query Rewriting & Multi-hop Reasoning) và tích hợp OCR đa phương thức (Docling/Marker) cho các tệp PDF scan ảnh.

---

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) | Ghi chú |
|:---|:---:|:---|
| **Hiểu bài giảng** | 5/5 | Nắm vững toàn bộ kiến trúc 5 tầng của Production RAG |
| **Code quality** | 5/5 | Code chuẩn type hints, clean architecture, 100% test pass |
| **Teamwork / Independence** | 5/5 | Hoàn thành độc lập toàn diện 5 modules và pipeline |
| **Problem solving** | 5/5 | Xử lý triệt để lỗi CUDA OOM, RateLimit 429 và JSON parsing |
