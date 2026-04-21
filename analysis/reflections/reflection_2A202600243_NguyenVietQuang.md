# Reflection — Nguyễn Việt Quang

## 1. Vai trò & Đóng góp

- **Role:** Agent Developer (Regression Testing & Release Gate)
- **Module phụ trách:**
  - `agent/main_agent.py` — Xây dựng 2 phiên bản RAG Agent (V1 Base + V2 Optimized)
  - `main.py` — Tích hợp Agent vào Benchmark pipeline, so sánh V1 vs V2, Release Gate
  - branch `agent`: `commit agent` -> `main`

### Các đóng góp cụ thể:

1. **Thiết kế kiến trúc Agent V1/V2:**
   - V1 (Base): Keyword-based retrieval top-1, chỉ trích câu đầu tiên từ context → cố tình đơn giản để làm baseline
   - V2 (Optimized): Retrieval top-3 với bigram matching, tổng hợp nhiều context, xử lý adversarial/out-of-scope
   - Agent load knowledge base động từ `golden_set.jsonl` thay vì hardcode

2. **Tích hợp pipeline trong main.py:**
   - Tạo `ExpertEvaluator` bridge để tính Retrieval metrics thật (Hit Rate, MRR) từ `RetrievalEvaluator`
   - Chạy benchmark V1 vs V2, so sánh regression, Release Gate tự động (APPROVE/BLOCK)

3. **Kết quả benchmark đạt được:**
   - V1: Hit Rate 64%, V2: Hit Rate 92% (+28%)
   - Release Gate: APPROVE — tất cả quality gates đều pass

## 2. Kiến thức kỹ thuật

### Giải thích các khái niệm:

**MRR (Mean Reciprocal Rank):**
MRR đo lường khả năng tìm đúng tài liệu của Retriever. Công thức: MRR = 1/vị_trí_đúng_đầu_tiên.
- Nếu tài liệu đúng nằm ở vị trí 1 → MRR = 1.0 (tốt nhất)
- Vị trí 2 → MRR = 0.5
- Vị trí 3 → MRR = 0.33
- Không tìm thấy → MRR = 0.0

Trong bài lab, V1 chỉ retrieve 1 doc nên MRR = 0 hoặc 1. V2 retrieve 3 docs nên MRR mịn hơn (0.33, 0.5, 1.0). MRR trung bình V2 = 0.84 > V1 = 0.64, chứng minh V2 tìm đúng tài liệu nhanh hơn.

**Cohen's Kappa:**
Là thước đo mức độ đồng thuận giữa 2 người chấm (hoặc 2 model Judge) sau khi loại bỏ yếu tố ngẫu nhiên.
- Kappa = 1.0: đồng thuận hoàn toàn
- Kappa > 0.8: rất tốt (almost perfect agreement)
- Kappa 0.6-0.8: tốt (substantial agreement)
- Kappa < 0.4: kém

Khác với Agreement Rate đơn giản (chỉ đếm % trùng khớp), Cohen's Kappa trừ đi xác suất trùng ngẫu nhiên. Ví dụ: 2 Judge đều cho 5 điểm mọi câu → Agreement = 100% nhưng Kappa có thể thấp vì đó có thể là ngẫu nhiên.

**Position Bias:**
Là hiện tượng LLM Judge luôn ưu tiên đánh giá cao câu trả lời ở vị trí đầu tiên (khi so sánh 2 responses). Để phát hiện, ta đổi vị trí 2 responses rồi chấm lại — nếu điểm thay đổi đáng kể (delta > 1.0) thì Judge bị bias.

Cách khắc phục:
- Chạy đánh giá 2 lần với thứ tự đảo ngược
- Lấy trung bình kết quả
- Hoặc yêu cầu Judge đánh giá từng response riêng lẻ thay vì so sánh

### Trade-off Chi phí vs Chất lượng:
- **Retrieval:** V1 (1 chunk) nhanh + rẻ nhưng Hit Rate thấp (64%). V2 (3 chunks) tốn thêm tokens nhưng Hit Rate cao hơn (92%). → Trade-off hợp lý vì chất lượng tăng 28% trong khi chi phí chỉ tăng ~3x tokens cho retrieval.
- **Multi-Judge:** Dùng 2 model (gpt-4o-mini + gpt-4o) tốn gấp đôi chi phí nhưng đáng tin cậy hơn. Có thể tối ưu bằng cách: chỉ dùng multi-judge cho hard cases, single judge (gpt-4o-mini) cho easy cases → giảm ~30% chi phí.
- **Generation:** Dùng gpt-4o-mini thay vì gpt-4o cho Agent generation → rẻ hơn 17x mà chất lượng vẫn đủ tốt.

## 3. Vấn đề gặp phải & Cách giải quyết

### Vấn đề 1: Knowledge Base hardcode quá cồng kềnh
- **Mô tả:** Ban đầu hardcode toàn bộ 25 chunks (150+ dòng code) trực tiếp trong agent — khó bảo trì, không linh hoạt khi golden_set thay đổi.
- **Giải pháp:** Refactor để load KB động từ `golden_set.jsonl`. Hàm `load_knowledge_base()` tự động đọc file, tách keywords từ content, và build dict. Khi golden_set thay đổi, Agent tự động cập nhật mà không cần sửa code.

## 4. Bài học rút ra

1. **Thiết kế interface trước, implement sau:** Khi làm nhóm, cần thống nhất interface (input/output format) giữa các module trước khi code. Điều này tránh conflict khi merge code.

2. **Baseline quan trọng:** V1 cố tình yếu (top-1, đơn giản) để V2 có cái để so sánh. Nếu V1 đã quá tốt thì không thể demo regression testing hiệu quả.

3. **Tách biệt concerns:** Agent chỉ lo retrieve + generate, không can thiệp vào Judge hay Runner. Khi teammate thay đổi Judge, code Agent không cần sửa.

4. **Đo lường trước khi tối ưu:** Hit Rate 64% → 92% cho biết chính xác retrieval là bottleneck của V1, không phải generation. Nhờ metrics rõ ràng mới biết cần fix ở đâu.
