# Group Report — Lab 18: Production RAG

**Nhóm:** K34-Production-RAG  
**Ngày hoàn thành:** 18/08/2026  
**Chủ đề:** Xây dựng Hệ thống Production RAG Đa tầng cho Doanh nghiệp

---

## Thành viên & Phân công nhiệm vụ

| Thành viên | Module phụ trách | Tình trạng | Tests pass |
|:---|:---|:---:|:---:|
| **Nguyễn Đăng Nam** | M1: Advanced Chunking (Semantic, Hierarchical, Structure) | ✅ Hoàn thành | **13/13** |
| **Nguyễn Đăng Nam** | M2: Hybrid Search (Underthesea BM25 + Qdrant Dense + RRF) | ✅ Hoàn thành | **8/8** |
| **Nguyễn Đăng Nam** | M3: Re-ranking (Cross-Encoder `bge-reranker-v2-m3` + FlashRank) | ✅ Hoàn thành | **6/6** |
| **Nguyễn Đăng Nam** | M4: RAGAS Evaluation & Failure Analysis Diagnostic Tree | ✅ Hoàn thành | **4/4** |
| **Nguyễn Đăng Nam** | M5: Context Enrichment (Contextual Prepend, HyQA, Metadata) | ✅ Hoàn thành | **6/6** |
| **TỔNG CỘNG** | **Toàn bộ 5 Modules & Pipeline Tích hợp** | **✅ ĐẠT 100%** | **37/37** |

---

## Kết quả Đánh giá RAGAS

| Chỉ số (Metric) | Naive Baseline | Production RAG | Δ (Mức cải thiện) |
|:---|:---:|:---:|:---:|
| **Faithfulness (Độ trung thực)** | 0.5500 | **0.8850** | `+0.3350` *(+60.9%)* |
| **Answer Relevancy (Độ phù hợp)** | 0.6000 | **0.8620** | `+0.2620` *(+43.7%)* |
| **Context Precision (Độ chính xác ngữ cảnh)** | 0.5200 | **0.8400** | `+0.3200` *(+61.5%)* |
| **Context Recall (Độ bao phủ ngữ cảnh)** | 0.5800 | **0.9500** | `+0.3700` *(+63.8%)* |

---

## Key Findings (Phát hiện quan trọng)

1. **Biggest Improvement (Cải thiện lớn nhất):**
   - **Tầng Hybrid Search + Reciprocal Rank Fusion (M2) kết hợp Cross-Encoder (M3)** giúp đẩy chỉ số *Context Recall* từ `0.58` lên `0.95`. BM25 với bộ tách từ tiếng Việt `underthesea` bắt chính xác tuyệt đối các con số, mã chính sách (v2023, v2024), trong khi Dense Search `bge-m3` bao quát câu hỏi đồng nghĩa/ngữ nghĩa trừu tượng.
2. **Biggest Challenge (Thách thức lớn nhất):**
   - Sự xung đột tài nguyên GPU khi tải đồng thời mô hình Embedding và Cross-Encoder trên máy cấu hình VRAM giới hạn (4GB), cùng các giới hạn Rate Limit / Token Quota của Cloud LLM API. Vấn đề đã được khắc phục triệt để bằng cơ chế giải phóng bộ nhớ CUDA động (`torch.cuda.empty_cache()`), dự phòng CPU, và bộ nhớ đệm cache (`_ENRICH_CACHE`).
3. **Surprise Finding (Phát hiện bất ngờ):**
   - Việc chỉ thêm 1 dòng tiền tố ngữ cảnh nguồn (*Contextual Prepend* trong M5) trước mỗi đoạn văn giúp LLM trả lời chính xác hơn 35% đối với các câu hỏi so sánh giữa chính sách cũ và mới mà không làm tăng độ trễ tính toán đáng kể.

---

## Presentation Notes (Kịch bản Thuyết trình 5 phút)

1. **Tổng quan & Bảng điểm RAGAS (1 phút):**
   - Trình bày sự nhảy vọt của cả 4 chỉ số RAGAS khi chuyển từ Naive RAG sang Production RAG đa tầng.
2. **Điểm nhấn Kỹ thuật — "Biggest Win" (1.5 phút):**
   - Minh họa kiến trúc 2 tầng: Tầng 1 lọc thô nhanh qua Hybrid Search (BM25 + Qdrant Dense) dung hợp bằng RRF, tầng 2 xếp hạng tinh qua Cross-Encoder Reranker.
3. **Case Study Phân tích Lỗi (Error Tree Walkthrough) (1.5 phút):**
   - Phân tích case tính phí quá hạn tạm ứng: Retrieval đúng 100% tài liệu, lỗi nằm ở phép toán suy luận của LLM $\rightarrow$ Giải pháp CoT (Chain-of-Thought).
4. **Kế hoạch Tối ưu tiếp theo (1 phút):**
   - Tích hợp Temporal Filtering (lọc theo phiên bản ngày hiệu lực) và nâng cấp kiến trúc Self-RAG / Corrective RAG.
