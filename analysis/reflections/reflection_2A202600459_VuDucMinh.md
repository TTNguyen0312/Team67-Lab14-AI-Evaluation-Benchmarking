# Individual Reflection - Vũ Đức Minh

##  Mục tiêu cá nhân trong project

- Hiểu và xây dựng được một hệ thống benchmark AI hoàn chỉnh, không chỉ chạy được mà còn có ý nghĩa về mặt đánh giá.
- Nắm được cách đo lường chất lượng AI Agent qua:
  - Retrieval (Hit Rate, MRR)
  - Generation (correctness, hallucination)
- Áp dụng kiến thức từ chương trình AI thực chiến vào một pipeline gần với production thực tế.

---

##  Những đóng góp chính

### 1. Module đã phát triển

- **Module:** Benchmark Runner & LLM Judge Integration
- **Kết quả:**

  - Xây dựng `BenchmarkRunner`:

    - chạy async batch với `asyncio.gather`
    - gọi Agent → Judge → Retrieval Eval
    - lưu kết quả vào `benchmark_results.json`
  - Tích hợp Multi-Judge (gpt-4o và gpt-4.1):

    - xây dựng consensus logic
    - tính agreement_rate
  - Chuẩn hóa output:

    - `final_score`, `is_correct`, `hallucination`
    - `status` (pass/fail)
  - Xây dựng logic `summary.json`:

    - accuracy, avg_score, hit_rate, mrr, latency

---

### 2. Kỹ thuật đã học

- **Async Programming:** Áp dụng asyncio để chạy benchmark song song và tối ưu performance
- **Multi-Judge Consensus:** Xây dựng hệ thống đánh giá đáng tin cậy bằng gpt-4o và gpt-4.1
- **RAGAS Metrics:** Hiểu và triển khai các metric retrieval như Hit Rate và MRR

---

##  Kết quả đạt được

- **Performance:** Benchmark chạy thành công với ~50 test cases
- **Quality:**

  - Ban đầu: kết quả “đẹp giả” (100% pass, score cao)
  - Sau khi sửa: xuất hiện fail cases, phản ánh đúng chất lượng agent
- **Agreement Rate:** ~0.86–0.89 giữa gpt-4o và gpt-4.1

---

##  Những khó khăn gặp phải

### 1. Benchmark cho kết quả “đẹp giả”

- **Mô tả:** Score cao bất thường, không phản ánh đúng sai
- **Giải pháp:**

  - Chỉ giữ correctness + hallucination
  - Chuẩn hóa score về thang 0–1
  - Dùng `is_correct` thay vì threshold cứng

### 2. Tích hợp Multi-Judge

- **Mô tả:**
  - Mismatch method (`evaluate` vs `evaluate_multi_judge`)
- **Giải pháp:**
  - Chuẩn hóa interface giữa runner và judge

---

##  Bài học kinh nghiệm

### Technical

- Benchmark AI không đơn giản là chạy model và in score
- Nếu metric sai → toàn bộ hệ thống đánh giá sai

### Teamwork

- Quan trọng nhất là contract giữa các module:

  - dataset
  - agent
  - judge

### Process

- Không chờ module khác hoàn thành mới bắt đầu
- Nên build pipeline trước bằng mock → rồi tích hợp sau

---

##  Đánh giá chung

- **Điểm tự đánh giá:** 8/10

- **Điều làm được:**

  - Xây dựng được pipeline benchmark hoàn chỉnh
  - Hiểu rõ sự khác biệt giữa “chạy được” và “đánh giá đúng”

- **Điều cần cải thiện:**

  - Thiết kế hệ thống tốt hơn ngay từ đầu
  - Phát hiện lỗi metric sớm hơn trong pipeline
