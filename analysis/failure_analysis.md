# Failure Analysis Report

## 1. Tổng quan Benchmark

- **Tổng số cases:** 50
- **Pass/Fail:** 49 pass / 1 fail (pass rate: 98%)
- **Điểm LLM-Judge trung bình:** 4.28 / 5.0
- **Multi-Judge:** 2 models (gpt-4o + gpt-4.1), Agreement Rate trung bình: 94.9%
- **Retrieval Metrics:**
    - Hit Rate: 78% (11/50 cases bị miss retrieval)
    - MRR: 0.697
- **RAGAS Metrics:**
    - Faithfulness trung bình: 0.60
    - Relevancy trung bình: 0.80

### Regression V1 vs V2:
| Metric | V1 | V2 | Delta |
|--------|----|----|-------|
| avg_score | 4.11 | 4.28 | +0.17 |
| hit_rate | 0.50 | 0.78 | +0.28 |
| mrr | 0.50 | 0.70 | +0.20 |
| pass_rate | 0.96 | 0.98 | +0.02 |
| agreement_rate | 0.93 | 0.95 | +0.02 |

**Release Gate Decision: APPROVE** ✅

---

## 2. Phân nhóm lỗi (Failure Clustering)

| Nhóm lỗi | Số lượng | Tỉ lệ | Nguyên nhân dự kiến |
| :--- | :---: | :---: | :--- |
| **Hallucination** | 19/50 | 38% | Agent tự suy luận ngoài context khi câu hỏi dạng "Tại sao?" |
| **Retrieval Miss** | 11/50 | 22% | Keyword retrieval không tìm đúng chunk cho câu hỏi adversarial/cross-document |
| **Low Faithfulness** | 19/50 | 38% | Agent thêm giải thích không có trong context gốc |
| **Fail (score < 3)** | 1/50 | 2% | Câu hỏi về Okta — context không có thông tin cụ thể |
| **Tone/Bias** | 0/50 | 0% | Không phát hiện vấn đề bias hay tone |

---

## 3. Phân tích 5 Whys (3 case tệ nhất)

### Case #1: "Tại sao có thể sử dụng Okta trong quá trình quản lý quyền truy cập?" (Score: 2.88 — FAIL)
1. **Symptom:** Agent trả lời "tài liệu không cung cấp thông tin về Okta" → sai so với context.
2. **Why 1:** Retriever không tìm được chunk chứa thông tin Okta (Hit Rate = 0.0).
3. **Why 2:** Từ khóa "Okta" chỉ xuất hiện trong chunk `access_control_sop_7` nhưng chunk này có ít keyword matching với câu hỏi dạng "tại sao".
4. **Why 3:** Keyword-based retrieval yếu với câu hỏi "reasoning" — câu hỏi hỏi lý do (why) nhưng retriever tìm theo keyword, không hiểu ngữ nghĩa.
5. **Why 4:** Không có semantic search (vector similarity) → chỉ dựa vào exact keyword match.
6. **Root Cause:** **Retrieval strategy** — Keyword-based retrieval không xử lý được câu hỏi dạng "Tại sao" cần semantic understanding. Cần chuyển sang vector-based retrieval (ChromaDB) hoặc hybrid search.

### Case #2: "Hai đoạn văn trên đã đề cập đến quy trình hoàn tiền. Đoạn 1 chỉ đề cập đến thông tin liên lạc..." (Score: 3.25)
1. **Symptom:** Agent nói "không có thông tin liên lạc" → nhưng context CÓ chứa email cs-refund@company.internal.
2. **Why 1:** Retriever không tìm đúng chunk chứa contact info (Hit Rate = 0.0).
3. **Why 2:** Câu hỏi dạng cross-document comparison ("hai đoạn văn") — retriever không hiểu cần 2 chunks khác nhau.
4. **Why 3:** Retrieval chỉ tìm theo keyword đơn lẻ, không hỗ trợ multi-hop reasoning.
5. **Why 4:** Golden dataset có câu hỏi yêu cầu so sánh nhiều nguồn, vượt quá khả năng của keyword retrieval.
6. **Root Cause:** **Chunking strategy + Retrieval** — Agent cần khả năng multi-hop retrieval để tổng hợp thông tin từ nhiều chunks. Cần tăng top_k hoặc dùng reranking.

### Case #3: "Điều gì được coi là bất thường trong access review?" (Score: 3.38)
1. **Symptom:** Agent tự bịa ra danh sách "bất thường" (quyền không phù hợp, tài khoản bất hoạt...) — đây là hallucination.
2. **Why 1:** Faithfulness chỉ đạt 0.327 — hầu hết nội dung trả lời không có trong context.
3. **Why 2:** Retriever không tìm được chunk `access_control_sop_6` chứa thông tin access review (Hit Rate = 0.0).
4. **Why 3:** Chunk `access_control_sop_6` rất ngắn, ít keyword → điểm matching thấp.
5. **Why 4:** Agent (dùng LLM API) khi không có context phù hợp, tự suy luận từ kiến thức nội tại thay vì nói "không tìm thấy".
6. **Root Cause:** **Prompting + Retrieval** — Agent cần prompt mạnh hơn để từ chối trả lời khi context không đủ. Retriever cần cải thiện recall cho các chunks ngắn.

---

## 4. Phân tích mẫu lỗi chung

### Pattern 1: Câu hỏi "Tại sao?" gây hallucination cao
- **19/50 cases** bị hallucination, hầu hết là câu hỏi dạng "Tại sao...?"
- Context thường chỉ chứa WHAT (cái gì) chứ không chứa WHY (tại sao)
- Agent (LLM) tự suy luận lý do → tạo ra thông tin không có trong context
- **Giải pháp:** Prompt instruction: "Nếu context không giải thích lý do, nói rõ thay vì suy luận"

### Pattern 2: Retrieval miss cho câu hỏi phức tạp
- **11/50 cases** fail retrieval (Hit Rate = 0.0)
- Các câu hỏi failed: adversarial, cross-document, reasoning
- Keyword retrieval hoạt động tốt cho câu hỏi factoid (HR=1.0) nhưng yếu cho câu hỏi phức tạp
- **Giải pháp:** Chuyển sang ChromaDB vector search hoặc hybrid (keyword + semantic)

### Pattern 3: Faithfulness thấp khi dùng real API
- **Faithfulness trung bình chỉ 0.60** — Agent thêm giải thích ngoài context
- LLM có xu hướng mở rộng câu trả lời dù prompt yêu cầu "chỉ trả lời dựa trên context"
- **Giải pháp:** Thêm constraint trong prompt: "KHÔNG thêm thông tin ngoài context" + giảm temperature xuống 0.1

---

## 5. Kế hoạch cải tiến (Action Plan)

- [ ] **Retrieval:** Chuyển từ keyword-based sang ChromaDB vector search (hybrid search) để xử lý câu hỏi semantic
- [ ] **Prompting:** Thêm instruction "Chỉ trả lời dựa trên context. Nếu không tìm thấy, nói 'Tôi không tìm thấy thông tin'" để giảm hallucination
- [ ] **Reranking:** Thêm bước reranking sau retrieval để filter chunks không liên quan
- [ ] **Multi-hop:** Tăng top_k lên 5 và thêm logic tổng hợp cross-document cho câu hỏi so sánh
- [ ] **Faithfulness monitoring:** Thêm RAGAS faithfulness check vào pipeline, block response nếu faithfulness < 0.4
- [ ] **Cost optimization:** Dùng gpt-4o-mini cho single judge trên easy cases (score > 4.5), chỉ dùng multi-judge cho hard cases → giảm ~30% chi phí eval
