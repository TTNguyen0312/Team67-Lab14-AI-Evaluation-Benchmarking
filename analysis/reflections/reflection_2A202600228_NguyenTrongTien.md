# Reflection — Nguyễn Trọng Tiến

## 1. Vai trò & Đóng góp

- **Role:** Pipeline Developer (SDG + Retrieval Evaluation + Async Runner)
- **Module phụ trách:**
  - `engine/retrieval_eval.py`: Xây dựng toàn bộ Retrieval & RAG Quality Evaluation Engine
  - `engine/runner.py`: Hỗ trợ Minh xây dựng BenchmarkRunner với async batch execution
  - `main.py`: Tích hợp pipeline, chạy V1/V2 song song với asyncio
  - `data/golden_set.jsonl`: Thiết kế và tạo Golden Dataset (50+ test cases)
  - `analysis/failure_analysis.md`: Hỗ trợ Quang phân tích lỗi và báo cáo nhóm

### Các đóng góp cụ thể:

1. **Xây dựng Retrieval Evaluation Engine (`retrieval_eval.py`):**
   - Implement đầy đủ 4 metrics: Hit Rate, MRR, Faithfulness, Relevancy
   - `calculate_hit_rate()`: kiểm tra xem ít nhất 1 chunk đúng có trong top-k retrieved không
   - `calculate_mrr()`: tính 1/rank của chunk đúng đầu tiên
   - `calculate_faithfulness()`: đo % token trong answer có xuất hiện trong context → phát hiện hallucination
   - `calculate_relevancy()`: đo % keyword trong question được answer đề cập đến
   - `evaluate_batch()`: tổng hợp metrics trên toàn bộ dataset bất đồng bộ

2. **Xây dựng BenchmarkRunner (`runner.py`):**
   - Class `BenchmarkRunner` tích hợp Agent + Judge + RetrievalEvaluator thành pipeline hoàn chỉnh
   - `run_single_test()`: đo latency, gọi agent, tính RAGAS metrics, gọi multi-judge evaluation
   - `run_all()` với batch_size=5: chạy song song theo batch để tránh rate limit API

3. **Parallel Async Pipeline (`main.py`):**
   - Chạy V1 và V2 agent song song bằng `asyncio.gather()` thay vì tuần tự → tiết kiệm 50% thời gian
   - Token usage tracking và cost reporting chi tiết theo từng model
   - Tích hợp token usage từ judge vào mỗi test result

4. **Golden Dataset:**
   - Tạo 50+ test cases với `ground_truth_ids`từ chunking chroma database chuẩn để tính Hit Rate/MRR chính xác
   - Đảm bảo dataset có đủ adversarial cases và edge cases

5. **Kết quả benchmark đạt được:**
   - V1: Hit Rate 50%, MRR 0.50, avg_score 3.35, pass_rate 66%
   - V2: Hit Rate 78%, MRR 0.70, avg_score 3.58, pass_rate 70%
   - Delta: Hit Rate +28%, MRR +0.20, Score +0.23

## 2. Kiến thức kỹ thuật

### Giải thích các khái niệm:

**Hit Rate:**
Hit Rate đo xem trong top-k kết quả truy vấn có chứa ít nhất 1 chunk đúng không. Công thức đơn giản: 1.0 nếu tìm thấy, 0.0 nếu không. Khác với Recall (đếm tỷ lệ chunk đúng tìm được), Hit Rate chỉ quan tâm "có tìm được hay không", phù hợp cho RAG vì chỉ cần 1 chunk liên quan là đủ để generate câu trả lời đúng.

**MRR (Mean Reciprocal Rank):**
MRR đo vị trí của chunk đúng đầu tiên trong kết quả truy vấn. Công thức: MRR = 1/rank_đúng_đầu_tiên.
- Vị trí 1 → MRR = 1.0 (tốt nhất)
- Vị trí 2 → MRR = 0.5
- Vị trí 3 → MRR = 0.33
- Không tìm thấy → MRR = 0.0

MRR quan trọng hơn Hit Rate vì nó phản ánh mức độ "tự tin" của retrieve khi tìm đúng ở vị trí 1 tốt hơn nhiều so với vị trí 3.

**Faithfulness:**
Faithfulness đo tỉ lệ token trong câu trả lời của Agent có xuất hiện trong context đã retrieve hay không. Score cao → Agent đang dùng thông tin từ context, không tự bịa (hallucinate). Score thấp → Agent đang "hallucinate" và tự generate ra thông tin không có trong tài liệu.

**Async Batch Runner:**
Thay vì gọi API tuần tự (mỗi request đợi request trước xong mới gọi), async batch runner dùng `asyncio.gather()` để chạy nhiều API call song song. Với 50 test cases và batch_size=5, pipeline chạy 10 batches × 5 cases song song thay vì 50 lần tuần tự.

### Trade-off Chi phí vs Chất lượng:
- **Retrieval:** V1 (top-1, keyword) nhanh và rẻ nhưng Hit Rate chỉ 50%. V2 (top-3, bigram matching) cải thiện Hit Rate lên 78% với chi phí token tăng nhẹ. Trade-off xứng đáng.
- **Parallel Execution:** Chạy V1 và V2 song song trong main.py tăng chi phí tính toán nhưng giảm latency 50%. Với CI/CD pipeline cần chạy nhanh, đây là trade-off đúng đắn.
- **Multi-Judge (gpt-4o + gpt-4.1):** Chi phí ~$0.55/run (V1) và $0.60/run (V2), nhưng Agreement Rate đạt 0.87-0.89. Đủ tin cậy cho evaluation.

## 3. Vấn đề gặp phải & Cách giải quyết

### Vấn đề 1: Token usage không được track qua các layer
- **Mô tả:** Ban đầu `run_single_test()` chỉ trả về kết quả đánh giá, không có token usage từ judge. Khó tính tổng chi phí toàn pipeline.
- **Giải pháp:** Thêm `agent_token_usage` và `judge_token_usage` vào mỗi result dict, sau đó aggregate trong `main.py` khi tổng hợp báo cáo cuối. Mỗi layer tự báo cáo usage của mình, layer trên tổng hợp.

### Vấn đề 2: Rate limit khi gọi API song song
- **Mô tả:** Chạy 50 test cases cùng lúc với `asyncio.gather()` bị rate limit từ OpenAI API.
- **Giải pháp:** Thêm `batch_size` parameter vào `run_all()`, chạy theo từng batch nhỏ (default 5), sau đó mới chạy batch tiếp theo. Cân bằng giữa tốc độ và rate limit.

### Vấn đề 3: Golden Dataset thiếu `ground_truth_ids`
- **Mô tả:** Các test case ban đầu không có `ground_truth_ids` → không thể tính Hit Rate/MRR.
- **Giải pháp:** Refactor toàn bộ golden_set.jsonl để mỗi case đều có `ground_truth_ids` mapping đến chunk ID trong knowledge base của agent.

## 4. Bài học rút ra

1. **Đo Retrieval trước, đo Generation sau:** Nếu Retrieval đã sai (Hit Rate thấp), thì dù Generation có tốt đến đâu, câu trả lời vẫn không chính xác. Phải fix bottleneck đúng chỗ.

2. **Async không phải luôn nhanh và hiệu quả:** `asyncio.gather()` tăng tốc đáng kể nhưng phải kết hợp với batch_size để tránh rate limit. Cần hiểu constraint của external API trước khi optimize.

3. **Chất lượng data:** Golden Dataset thiếu `ground_truth_ids` làm cho Hit Rate/MRR vô nghĩa. Đầu tư thời gian vào data quality là đầu tư đúng chỗ nhất trong toàn bộ pipeline.

4. **Interface contract giữa các module:** `run_single_test()` trả về dict với format chuẩn, các component khác (reporter, gate) chỉ cần biết format đó. Khi teammate thay đổi Judge hay Agent, Runner không cần sửa miễn là interface không đổi.

5. **Benchmark infrastructure là sản phẩm:** Runner, evaluator, dataset không chỉ là thí nghiệm. Chất lượng của hạ tầng đo lường quyết định chất lượng của mọi quyết định kỹ thuật sau đó.
