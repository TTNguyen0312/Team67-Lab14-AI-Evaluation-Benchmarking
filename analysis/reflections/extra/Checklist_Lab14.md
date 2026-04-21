# Checklist Lab14 - Các B steps Cân Làm theo huong dân cúa giáo viên

## Muc tiêu cuôi cùng:
Xây dîng hê thóng Benchmark dê chûng minh  
Version 2 tôt hôn Version 1

---

## PHASE 1 - DATASET (Quan trong nhât)

### Buoc 1: Chuân bi source data
Cân có:
- document gôc
- knowledge base
- vector DB (nêu dâ có)
- chunk text
- chunk ID

Nêu dâ có vector DB:
- export chunk ra luôn.

### Buoc 2: Chunk dî liêu
Môi doan phâi có:
- chunk_id
- chunk_text
- source_document

Ví du:
```
chunk_001
- nôi dung van bân
- file policy.pdf
```

### Buoc 3: Thiêt kê prompt tao dataset
Prompt phâi yêu câu rô:
- generate question
- expected answer
- correct chunk ID
- difficulty
- category
- metadata dû dû

Phâi có:
- Good Example + Hard Case Example
- dê LLM hoc theo.

### Buoc 4: Dùng LLM tao Golden Dataset
Tao khoâng:
- 30-50 câu hôi chuân
- bao gôm:
  - easy
  - medium
  - hard
  - multi-hop reasoning
  - retrieval dê sai
  - hallucination dê xãy ra

### Buoc 5: Manual Review Dataset
Phâi kiêm tra:
- câu hôi dûng chua
- answer dûng chua
- chunk ID dûng chua
- source dûng chua

**Dây là buoc bât buôc**  
vì LLM có thê tao sai.

---

## PHASE 2 - AGENT VERSION

### Buoc 6: Tao Version 1
Ví du:
- retrieval yêu hôn
- logic cu hôn 
- prompt chua tôi uu

### Buoc 7: Tao Version 2
Ví du:
- retrieval tôt hôn
- reranking tôt hôn
- prompt tôt hôn
- final answer tôt hôn

**Muc tiêu: V2 > V1**

---

## PHASE 3 - TRUST / JUDGE

### Buoc 8: Tao LLM Judge
Dùng LLM dê dánh giá:
- output dûng hay sai
- partial correct
- hallucination
- bias
- fairness
- consistency

Ví du:
```
Question
Expected Answer
System Output
- Judge Result
```

### Buoc 9: Verify lai Judge
Vì:
- Judge LLM cung có thê sai
- nên phâi:
  - manual spot check
  - dê tránh dánh giá sai.

---

## PHASE 4 - BENCHMARK

### Buoc 10: Chây benchmark cho V1
Chây toàn bô dataset vôi:
- Agent Version 1
- luu kê t quâ.

### Buoc 11: Chây benchmark cho V2
Chây cung dataset vôi:
- Agent Version 2
- so sánh công bâng.

### Buoc 12: Tinh metric
Ví du:
- Retrieval Accuracy
- Hit Rate
- Average Hit Rate
- Final Answer Accuracy
- Hallucination Rate 
- Average Score
- Latency
- Cost
- User Satisfaction Score

---

## PHASE 5 - ANALYSIS

### Buoc 13: Phân tích nguyên nhân
Không chi ghi:
- V2 tôt hôn

mà phâi giâi thích:
- Tai sao tôt hôn
- Tôt ô dâu
- Rui ro còn gì

Ví du:
- retrieval improved
- hallucination giâm
- latency tang nhê
- cost tang nhung acceptable

---

## PHASE 6 - REPORT

### Buoc 14: Làm Final Report
Bao gôm:
- Executive Summary
- Benchmark Comparison
- Metric Table
- Trust Analysis
- Risk Analysis
- Recommendation
- Next Action

---

## FINAL DELIVERABLE

Không phâi chi là code  
mà là:
- Dataset
- + Agent V1
- + Agent V2
- + LLM Judge
- + Benchmark Result
- + Final Report
