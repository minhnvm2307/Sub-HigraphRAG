# 03. Protocol đánh giá

*Cập nhật: 2026-10-05 (thêm §0).* Mọi con số trong paper phải theo đúng file này. Nếu phải lệch khỏi protocol, ghi rõ trong `04_EXPERIMENT_LOG.md`.

## 0. Cổng retrieval trước downstream (từ 2026-10-05, nới lại cùng ngày)

Prompt, LLM và cách chấm của SubgraphRAG và REL-RAG là cố định. Vì vậy downstream chỉ khác khi retrieval khác. **Trước khi đề xuất chạy reasoning, phải lập luận được vì sao Hit/F1 có thể cao hơn**, dựa trên độ lớn của chênh lệch retrieval.

1. **Điều kiện để một phương pháp được coi là đóng góp:**
   - *In-distribution:* answer recall xấp xỉ bằng cả hai baseline, tức không kém có ý nghĩa ở budget nào (@50 / @100 / @200).
   - *Tổng quát hóa:* trong ít nhất một setting (WebQSP → CWQ, CWQ → WebQSP, hoặc dataset khác), answer recall **vượt cả SubgraphRAG (ưu tiên) lẫn REL-RAG** ở mọi budget @50 / @100 / @200. "Vượt" nghĩa là paired bootstrap 95% có cận dưới > 0.
2. **Metric hỗ trợ:** SP / GPT / GPTp (recall, precision, F1) luôn được in ra, và dùng làm bằng chứng khi chúng có ích cho việc đánh giá. Ví dụ: đáp án lấy thêm được có đi kèm bằng chứng hay không. Chúng **không bắt buộc** phải thắng. Không kết luận chỉ dựa trên một metric.
3. **Downstream:**
   - In-distribution ngang nhau thì không chạy downstream in-distribution.
   - Với setting tổng quát hóa đã qua cổng, xem bảng *hiệu quả budget* (candidate@k so với baseline@2k) và độ lớn chênh lệch để quyết định có đáng chạy không, và nêu rõ lập luận trước khi xin chạy.
   - Ngoại lệ: phương pháp thay đổi chính phần reasoning.
4. **Công cụ:** `retrieve/eval.py -d <webqsp|cwq> -p <file> ... --subset true` (từ 10-20; trước đó là `/relrag/gate_report.py`, có CI bootstrap; eval.py không có bootstrap). Verdict dùng ansR @50/100/200; mọi metric khác và @500 chỉ để tham khảo. Output lưu ở `/relrag/runs/gate/` và được tóm tắt trong `docs/REPORT.md` §3.

## 1. Dữ liệu

| Dataset | Train | Val | Test | Nguồn subgraph |
|---|---|---|---|---|
| WebQSP | 2,848 (2,717 có embedding; 2,710 có path) | 250 | 1,639 (1,559 có đáp án trong graph) | `retrieve/data_files/webqsp/processed` (subgraph của SubgraphRAG) |
| CWQ | ~27.6k (21,943 dùng được) | ~3.5k | 3,531 (2,831 có path từ topic tới đáp án) | `retrieve/data_files/cwq/processed` |

- Embedding: GTE-large-en-v1.5 đóng băng, lưu fp16 memmap.
- Nhãn yếu: triple nằm trên shortest path (`triple_scores`), hoặc rational path đã được LLM lọc (REL-RAG).
- **Subset nhỏ để thử nhanh** (seed 0, `retrieve/exp_qrdde/data/`; đã xóa 10-05, tạo lại bằng `prepare_subsets.py` nếu cần): WebQSP 800/150/500, CWQ 1500/200/600 (train/val/test).

## 2. Metric tầng retrieval

| Metric | Định nghĩa | Ghi chú |
|---|---|---|
| **Ans recall@K** (chính) | Trung bình trên các câu hỏi: \|A ∩ entity(top-K triple)\| / \|A\|, với A là đáp án có trong graph | Công thức của `retrieve/eval.py` |
| Ans hit@K | Có ít nhất một đáp án trong top-K | Là trần của Hit sau LLM |
| Triple recall@K | Tỷ lệ triple nhãn nằm trong top-K | Dùng để chọn model (val) |
| id-match và text-match | id: so theo id entity; text: so theo tên, với `a_entity_in_graph` của SubgraphRAG | **Bảng so sánh giữa các retriever dùng text-match**; id-match chỉ dùng nội bộ |

- **Budget:** báo cáo K ∈ {20, 50, 100, 500}.
- Retriever path-based phải lấp **đủ K triple** (decode best-first). Nếu retriever lấy ít hơn K, ghi số triple trung bình thực tế.
- Câu không có path vẫn được tính (recall = 0 nếu không lấy được đáp án). Chỉ loại những câu không có đáp án trong graph.

## 3. Metric downstream

- **Hit và Macro-F1 theo eval gốc** (`reason/metrics/evaluate_results.py`). Đây là protocol mà REL-RAG/SubgraphRAG dùng: kiểm chứng bằng việc dự đoán chính thức của SubgraphRAG cho ra 86.63 Hit, khớp số 86.6 mà REL-RAG trích.
- Báo cáo thêm Hit@1 / F1 theo eval corrected (`evaluate_results_corrected.py`) trong phụ lục.
- **Reasoner mặc định:** llama3.1 8B Q4_K_M qua ollama (GPU server riêng, cổng 11435), prompt `sys_icl_dc` của SubgraphRAG, temperature 0, `num_ctx` 8192 (dùng 24576 nếu budget 500).
- **Tập câu:** 1,608 câu WebQSP test đã có cho mọi run; so sánh luôn trên cùng tập câu.
- **Ý nghĩa thống kê:** paired bootstrap 2,000 lần trên F1/Hit từng câu, báo khoảng tin cậy 95%. Với ~1,600 câu, mức chênh phát hiện được vào khoảng 1.7 F1.

## 4. Tổng quát hóa (transfer)

- **WebQSP → CWQ:** train trên WebQSP, eval trên toàn bộ CWQ test. Không train hay tinh chỉnh gì trên CWQ.
- Mốc tham chiếu in-distribution: SubgraphRAG chính thức train trên CWQ (`reason/scored_triples/cwq_240907_unidir_test.pth`).
- Tách theo kiểu shift (M3): ID / compositional / OOD, định nghĩa theo tập relation và relation-path của WebQSP train (Definition 1–3 của paper).

## 5. Hiệu quả

- **Latency retriever/query:** batch 1, GPU 1080 Ti, tính cả đọc embedding, dựng graph và decode; báo median và p95 trên 200 câu.
- **End-to-end:** retriever + một lời gọi LLM (cùng reasoner), báo median trên 100 câu và số token prompt.

## 6. Quy tắc báo cáo

1. Mỗi so sánh ghi rõ: tập câu hỏi (n), budget, reasoner, loại eval.
2. Kết quả trên subset không được so với kết quả full data trong cùng một bảng mà không chú thích.
3. Thí nghiệm của phương pháp đề xuất: một seed duy nhất giống SubgraphRAG (`seed 42`, deterministic, qua `src/setup.set_seed`), cùng schedule `num_epochs 10000`, `patience 10` (quyết định 09-30). Khác biệt retrieval giữa hai phương pháp được đánh giá bằng paired bootstrap trên câu hỏi test (2,000 lần lấy mẫu, CI 95%), giống cách làm ở downstream.
4. Luôn báo cáo cả số trong paper REL-RAG lẫn số tái hiện.
