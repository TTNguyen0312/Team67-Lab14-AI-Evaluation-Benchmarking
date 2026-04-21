# Failure Analysis Report

## 1. Tổng quan Benchmark (Dối với v2)

- **Tổng số cases:** 50
- **Pass/Fail:** 35 pass / 15 fail (pass rate: 70%)
- **Điểm LLM-Judge trung bình:** 3.575 / 5.0
- **Multi-Judge:** 2 models (gpt-4o + gpt-4.1), Agreement Rate trung bình: 86.8%
- **Retrieval Metrics:**
    - Hit Rate: 78% (11/50 cases bị miss retrieval)
    - MRR: 0.697
- **RAGAS Metrics:**
    - Faithfulness trung bình: 0.607
    - Relevancy trung bình: 0.697
- **Cost & Token Usage (V2 run):**
    - Tổng tokens: 157,416 (input: 120,270 / output: 37,146)
    - gpt-4o: 79,925 tokens — $0.3372
    - gpt-4.1: 77,491 tokens — $0.2680
    - **Tổng chi phí: $0.6051 USD**

### Regression V1 vs V2:
| Metric | V1 | V2 | Delta |
|--------|----|----|-------|
| avg_score | 3.347 | 3.575 | +0.228 |
| hit_rate | 0.50 | 0.78 | +0.28 |
| mrr | 0.50 | 0.697 | +0.197 |
| pass_rate | 0.66 | 0.70 | +0.04 |
| agreement_rate | 0.894 | 0.868 | -0.026 |

**Release Gate Decision: APPROVE** ✅

---

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng | Tỉ lệ | Nguyên nhân dự kiến |
| :--- | :---: | :---: | :--- |
| **Fail (score < 3)** | 15/50 | 30% | Retrieval miss + hallucination kết hợp trên câu hỏi phức tạp |
| **Hallucination Risk** (faithfulness < 0.7) | 32/50 | 64% | Agent suy luận ngoài context, đặc biệt câu hỏi dạng "Tại sao?" |
| **Low Faithfulness** (faithfulness < 0.5) | 21/50 | 42% | Agent thêm thông tin không có trong context gốc |
| **Retrieval Miss** (hit_rate = 0) | 11/50 | 22% | Keyword retrieval không tìm đúng chunk cho câu hỏi adversarial/cross-document |
| **Tone/Bias** | 0/50 | 0% | Không phát hiện vấn đề bias hay tone |

---

## 3. Phân tích 5 Whys (3 case tệ nhất)

### Case #1: "Hai đoạn văn trên đã đề cập đến quy trình hoàn tiền..." (Score: 1.625 - FAIL)
1. **Symptom:** Agent không tổng hợp được thông tin từ nhiều đoạn văn về refund process.
2. **Why 1:** Retriever không tìm được chunk phù hợp (Hit Rate = 0.0, Faithfulness = 0.431).
3. **Why 2:** Câu hỏi dạng cross-document comparison, retriever không hiểu cần 2 chunks khác nhau.
4. **Why 3:** Retrieval chỉ tìm theo keyword đơn lẻ, không hỗ trợ reasoning multi-hop.
5. **Why 4:** Golden dataset có câu hỏi yêu cầu so sánh nhiều nguồn, vượt quá khả năng keyword retrieval.
6. **Root Cause:** **Chunking strategy + Retrieval** cần retrieval multi-hop, tăng top_k hoặc dùng reranking.

### Case #2: "Hệ thống ticket được sử dụng là gì?" (Score: 1.750 - FAIL)
1. **Symptom:** Agent trả lời sai hoặc thiếu — không đề cập đến Jira (IT-ACCESS, IT-SUPPORT, IT-SOFTWARE).
2. **Why 1:** Retriever không tìm được chunk chứa thông tin ticket system (Hit Rate = 0.0, Faithfulness = 0.583).
3. **Why 2:** Thông tin về ticket systems nằm rải rác ở nhiều chunks khác nhau (access_control_sop_7, it_helpdesk_faq_5...).
4. **Why 3:** Câu hỏi ngắn, ít keyword → matching score thấp → sai chunk được retrieve.
5. **Why 4:** Không có semantic search → không hiểu "hệ thống ticket" liên quan đến Jira.
6. **Root Cause:** **Retrieval strategy**, Keyword-based retrieval yếu với câu hỏi ngắn/general. Cần ứng dụng vector search.

### Case #3: "Tại sao tôi không nhận được email từ bên ngoài mà thư mục Spam cũng không có gì?" (Score: 2.000 — FAIL)
1. **Symptom:** Agent không đưa ra hướng dẫn cụ thể (tạo ticket P2), thay vào đó tự suy luận nguyên nhân.
2. **Why 1:** Faithfulness = 0.621, nhiều nội dung không có trong context gốc.
3. **Why 2:** Retriever không tìm được chunk `it_helpdesk_faq_4` (Hit Rate = 0.0).
4. **Why 3:** Câu hỏi dạng "Tại sao", retriever ưu tiên chunks có keyword match, bỏ qua chunk chứa hướng giải quyết.
5. **Why 4:** Agent LLM khi thiếu context phù hợp, tự suy luận nguyên nhân thay vì nói "không tìm thấy".
6. **Root Cause:** **Prompting + Retrieval**. Cần prompt constraint mạnh hơn + semantic retrieval để xử lý câu "Tại sao".

---

## 4. Phân tích mẫu lỗi chung

### Pattern 1: Câu hỏi "Tại sao?" gây hallucination cao
- **32/50 cases** có faithfulness < 0.7, hầu hết là câu hỏi dạng "Tại sao...?"
- Context thường chỉ chứa WHAT (cái gì) chứ không chứa WHY (tại sao)
- Agent (LLM) tự suy luận lý do → tạo ra thông tin không có trong context
- **Giải pháp:** Prompt instruction: "Nếu context không giải thích lý do, nói rõ thay vì suy luận"

### Pattern 2: Retrieval miss cho câu hỏi phức tạp
- **11/50 cases** fail retrieval (Hit Rate = 0.0)
- Các câu hỏi failed: adversarial, cross-document, reasoning, câu ngắn/general
- Keyword retrieval hoạt động tốt cho câu hỏi factoid nhưng yếu cho câu hỏi phức tạp
- **Giải pháp:** Chuyển sang ChromaDB vector search hoặc hybrid (keyword + semantic)

### Pattern 3: Faithfulness thấp khi dùng real API
- **Faithfulness trung bình chỉ 0.607** do agent thêm giải thích ngoài context
- LLM có xu hướng mở rộng câu trả lời dù prompt yêu cầu "chỉ trả lời dựa trên context"
- **Giải pháp:** Thêm constraint trong prompt: "KHÔNG thêm thông tin ngoài context" + giảm temperature xuống 0.1

### Pattern 4: Agreement Rate giảm từ V1 sang V2 (-0.026)
- V2 có agreement rate thấp hơn V1 (86.8% vs 89.4%), 2 judges bất đồng nhiều hơn trên câu phức tạp
- V2 cố gắng trả lời nhiều hơn → câu trả lời phức tạp hơn → judges khó đánh giá nhất quán hơn
- **Giải pháp:** Xem xét dùng tiebreaker model (gpt-4o-mini) để phân xử khi agreement thấp

---

## 5. Kế hoạch cải tiến (Action Plan)

- [ ] **Retrieval:** Chuyển từ keyword-based sang ChromaDB vector search (hybrid search) để xử lý câu hỏi semantic
- [ ] **Prompting:** Thêm instruction "Chỉ trả lời dựa trên context. Nếu không tìm thấy, nói 'Tôi không tìm thấy thông tin'" để giảm hallucination
- [ ] **Reranking:** Thêm bước reranking sau retrieval để filter chunks không liên quan
- [ ] **Multi-hop:** Tăng top_k lên 5 và thêm logic tổng hợp cross-document cho câu hỏi so sánh
- [ ] **Faithfulness monitoring:** Thêm RAGAS faithfulness check vào pipeline, block response nếu faithfulness < 0.4
- [ ] **Cost optimization:** Dùng gpt-4o-mini cho single judge trên easy cases (score > 4.5), chỉ dùng multi-judge cho hard cases → giảm ~30% chi phí eval ($0.60 → ~$0.42/run)
