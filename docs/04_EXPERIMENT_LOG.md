# 04. Nhật ký thí nghiệm

Nhật ký từ khi đổi baseline sang REL-RAG (09/2026). Mỗi mục có mã `EXP-xx`, được tham chiếu trong [`archive/relrag_phase/01_RESEARCH_PLAN.md`](archive/relrag_phase/01_RESEARCH_PLAN.md). Các thí nghiệm giai đoạn SubgraphRAG nằm ở [`archive/subgraphrag/`](archive/subgraphrag/README.md).
Mẫu ghi: *mục tiêu → setup → kết quả → kết luận → artifact*. Số liệu retrieval dùng text-match trừ khi ghi khác ([`03_EVALUATION_PROTOCOL.md`](03_EVALUATION_PROTOCOL.md)).

| Mã | Ngày | Nội dung | Milestone |
|---|---|---|---|
| EXP-00 | 09-27 | Hủy kết quả RAPL cũ (Phase 16) | M0 |
| EXP-01 | 09-27 | Dữ liệu line graph + lọc rational path bằng gemma3 | M0 |
| EXP-02 | 09-27/28 | Train REL-RAG path/triple, nhãn SP/LLM | M0 |
| EXP-03 | 09-28 | Retrieval WebQSP (decoder lấy mẫu) + ablation | M0 |
| EXP-04 | 09-28 | Cải tiến thử: residual, joint, hybrid | M0/M2 |
| EXP-05 | 09-28 | Audit cài đặt, decoder best-first | M0 |
| EXP-06 | 09-29 | Reasoning path-based @50 | M0/M4 |
| EXP-07 | 09-30 | Hiệu chuẩn reasoner và protocol eval | M0/M4 |
| EXP-08 | 09-29 | E0: kích thước line graph | M1 |
| EXP-09 | 09-29 | E5: latency retriever | M1/M5 |
| EXP-10 | 09-30 | Latency end-to-end kiểu Table 7 | M1/M5 |
| EXP-11 | 09-29 | QR-DDE so với DDE trên subset | M2 |
| EXP-12 | 09-30 | Transfer WebQSP → CWQ (retrieval) | M3 |
| EXP-13 | 09-30 | Kiểm chứng: entity nối bằng nhiều relation | M1 / R |
| EXP-14 | 10-01 | QR-DDE so với DDE trên WebQSP full data + ablation | M2 |
| EXP-14b | 10-01 | QR-DDE so với DDE trên CWQ full data (QR-DDE bị ngắt ở epoch 10) | M2 |
| EXP-15 | 10-01 | Nhãn rational path GPT-4o (quy trình tác giả) so với gemma, 901 câu WebQSP train | M0 |
| EXP-16 | 10-01 | REL-RAG cấu hình paper (seed 42, không self-loop, nhãn GPT-4o+DeepSeek), P và T | M0 |

---

## EXP-00 — Hủy kết quả RAPL cũ
Kết quả "RAPL Recall@100 = 36.94%" của Phase 16 sai do lỗi cài đặt: chấm điểm bằng `node_emb·q` mà bỏ `r_mlp`, feature `[q|r|h]` thay vì `[h|r|t]`, nhãn SP bị gộp vào một cột, `edge_cap=30` cắt 93% số cạnh, metric khác `retrieve/eval.py`. **Không được trích dẫn.**

## EXP-01 — Dữ liệu và lọc rational path
- **Setup:** `/relrag/prepare_data.py`: node là cạnh của DiGraph (như code gốc), forward shortest path q→a (tối đa 50 path/cặp, 256 path/câu). `llm_label.py`: gemma3:12b Q4_K_M, temperature 0, prompt theo khung Figure 12 + ví dụ John Noble từ code gốc.
- **Kết quả:** gán nhãn 2,944 câu train+val (~3 giờ). Với câu có >1 relation path ứng viên (828 câu train), LLM chọn trung bình 1.01 trên 4.16. Số label path/câu: 9.6 (SP) → 5.1 (LLM). 41 câu trả lời rỗng nên dùng lại nhãn SP.
- **Chất lượng nhãn** (đối chiếu bằng chứng GPT trên 141 câu test nhiều ứng viên): path được giữ có 72.2% được GPT hỗ trợ, path bị loại chỉ 10.3%; 83.2% câu giữ được ít nhất một path được GPT hỗ trợ.
- **Artifact:** `/relrag/data/labels_gemma3_12b/*.jsonl`, `/relrag/logs/rational_path_filtering_{train,val}.log`, `eval_label_quality.py`.

## EXP-02 — Train REL-RAG
- **Setup:** `train.py`, cấu hình theo F.1 (bi-GCN 2 lớp, hidden 512, Adam 1e-3, batch 10 bằng gradient accumulation, 15 epoch, dropout 0.2, chọn model theo val loss); L_q với τ = 0.95; full line graph.
- **Run:** `path_llm`, `path_sp`, `path_llm_noselfloop`, `path_sp_noselfloop`, `triple_llm`, `triple_sp`, `triple_llm_res`, `path_llm_res`, `joint_llm` (`/relrag/runs/`). Mỗi run khoảng 50 phút.
- **Sự cố 09-28:** server sập vì cạn RAM do nhiều job chạy chồng (embedding nằm trong RAM dạng dict, DataLoader worker fork, cùng lúc với ollama). Đã khắc phục bằng memmap, `num_workers=0`, chạy tuần tự trong cgroup có giới hạn RAM (xem [`05_INFRASTRUCTURE.md`](05_INFRASTRUCTURE.md)).

## EXP-03 — Retrieval WebQSP với decoder lấy mẫu (đã được thay bằng EXP-05)

| Method | R@20 | R@50 | R@100 | R@500 |
|---|---|---|---|---|
| SubgraphRAG | 82.58 | 90.38 | 94.42 | 98.93 |
| path, nhãn gemma | 75.14 | 85.26 | 89.70 (57 triple) | – |
| triple, nhãn gemma | 68.65 | 80.43 | 87.80 | 97.56 |
| path, nhãn SP | 75.71 | 85.43 | 89.79 (66 triple) | – |
| triple, nhãn SP | 67.65 | 80.70 | 88.15 | 97.71 |
| path, không self-loop | 78.35 | 87.01 | 91.21 (53 triple) | – |

**Kết luận:** nhãn LLM ≈ nhãn SP về recall (khớp Table 8 của paper). Decoder lấy mẫu không lấp đủ budget (lỗi của mình, sửa ở EXP-05).

## EXP-04 — Cải tiến thử (không có trong paper)

| Biến thể | R@50 | R@100 | R@500 |
|---|---|---|---|
| triple + residual (skip connection z += W·x) | 89.92 | 94.54 | 99.10 |
| path + residual (best-first) | 87.79 | 93.84 | 98.50 |
| joint (L_q + L_path + L_triple) + decode hybrid | 88.57 | 91.35 | – |

Residual tăng triple-based +9.5 R@50: vì GCN normalization, feature của chính triple chỉ còn trọng số khoảng 1/(deg+1). Đây là tinh chỉnh kỹ thuật, **không đủ làm đóng góp chính**.

## EXP-05 — Audit cài đặt và decoder best-first
- **Khớp paper:** config F.1; GCN tự viết khớp PyG (sai số 3.6e-7); fp16 không mất thông tin; epoch được chọn theo val loss ≈ epoch có recall tốt nhất.
- **Nguyên nhân chính của chênh lệch là decoder:** lấy mẫu chỉ được 37.6 triple ở budget 50. Decoder **best-first** (`infer.py --mode path_bf`) lấp đủ budget.

| Method (best-first) | R@20 | R@50 | R@100 | R@500 |
|---|---|---|---|---|
| path, nhãn gemma | 75.90 | 87.06 | 92.38 | 98.29 |
| path, nhãn SP | 76.13 | 86.50 | 91.51 | 98.61 |
| path, không self-loop | 78.91 | 88.77 | 93.80 | 98.78 |
| path + residual | 77.81 | 87.79 | 93.84 | 98.50 |

- **Graph format:** DiGraph mất 9% triple và 6.4% bằng chứng GPT, ảnh hưởng 262/1411 câu. → Thử MultiDiGraph (M0).
- **Artifact:** `/relrag/runs/retrieval_compare_bestfirst.txt`.

## EXP-06 — Reasoning path-based @50
- **Setup:** `path_llm_noselfloop` + best-first 50 triple; reasoner llama3.1 Q4 (ollama), prompt `scored_50 / sys_icl_dc`. So với SubgraphRAG@50 trên cùng 1,608 câu.
- **Kết quả (eval corrected):** SubgraphRAG@50 Hit@1 86.57 / F1 72.78; REL-RAG 87.56 / 74.43. Paired bootstrap: ΔF1 +1.65 [−0.07, +3.33], ΔHit@1 +1.00 [−0.81, +2.74].

## EXP-07 — Hiệu chuẩn reasoner và protocol eval
Cùng 1,608 câu, cùng prompt.

| Run | Hit (gốc) | F1 (gốc) | Hit@1 (corrected) | F1 (corrected) |
|---|---|---|---|---|
| SubgraphRAG@100 + Llama3.1-8B đầy đủ (dự đoán chính thức) | 86.63 | (lỗi định dạng) | 88.87 | 74.54 |
| SubgraphRAG@100 + llama3.1 Q4 | 85.57 | 70.88 | 88.06 | 73.67 |
| SubgraphRAG@50 + llama3.1 Q4 | 84.27 | 70.82 | 86.57 | 72.78 |
| REL-RAG(P)@50 + llama3.1 Q4 | 83.77 | 72.33 | 87.56 | 74.43 |

**Kết luận:** paper dùng **eval gốc** (86.63 khớp 86.6 mà REL-RAG trích). Nén Q4 làm mất ~1 Hit. Theo protocol của paper, REL-RAG tái hiện được 83.8 / 72.3, so với 88.8 / 77.3 trong paper. Retrieval không phải trần (ans_hit@50 = 92.8). Giả thuyết còn mở: prompt và cách trích đáp án của tác giả.
**Artifact:** `/relrag/compare_reasoners.py`.

## EXP-08 — E0: kích thước line graph (subset test)

| | WebQSP (500) | CWQ (600) |
|---|---|---|
| \|E\| trung bình (p95) | 4,384 (7,983) | 4,718 (9,117) |
| \|E'\| trung bình / p95 / max | 413k / 1.56M / 2.67M | 481k / 1.64M / 2.67M |
| \|E'\|/\|E\| trung bình (p95) | 83× (295×) | 107× (376×) |
| Phần \|E'\| đi qua topic entity / CVT | 76.9% / 0.3% | 49.3% / 0.2% |

Giả thuyết "CVT gây bùng nổ" bị bác bỏ. **Artifact:** `retrieve/exp_qrdde/e0_linegraph_stats.py`, `logs/e0.txt`.

## EXP-09 — E5: latency retriever (200 câu WebQSP, median / p95 ms)
MLP + DDE 10.1 / 21.3; MLP + QR-DDE 11.0 / 29.8; REL-RAG 69.5 / 171.4 (dựng line graph và feature 26.2, bi-GCN 20.3, decode 19.1). **Artifact:** `retrieve/exp_qrdde/e5_latency.py`.

## EXP-10 — Latency end-to-end kiểu Table 7 (100 câu, llama3.1 Q4)

| Pipeline | Retriever | LLM median / mean | Token prompt | Tổng |
|---|---|---|---|---|
| REL-RAG path @50 | 69.5 ms | 4.06 / 9.65 s | 978 | ~4.13 s |
| SubgraphRAG @50 | 10.1 ms | 4.34 / 9.58 s | 980 | ~4.35 s |
| SubgraphRAG @100 | 10.1 ms | 5.64 / 11.53 s | 1,950 | ~5.65 s |

Trên 98% thời gian là của LLM. Table 7 của paper nhanh hơn vì so 50 triple với 100 triple. **Artifact:** `/relrag/e2e_latency.py`.

## EXP-11 — QR-DDE so với DDE trên subset (E1)
- **Setup:** `retrieve/exp_qrdde/qrdde.py`. Cả ba biến thể dùng cùng MLP scorer của SubgraphRAG, cùng BCE, cùng nhãn SP, chọn theo val triple_recall@100, 25 epoch, 2 seed. QR-DDE: h⁰ = 1[topic]·Linear(z_q); h^k = h^{k−1} + ReLU(LN(W_k(h⁰ + mean(h_u ⊙ w_q(r))))), w_q(r) = MLP([z_r ; z_q]) (tách chiều xuôi và ngược), K = 3, d = 64. **Học end-to-end chỉ trên đúng subset**, dùng embedding GTE đóng băng; không dùng BGE-M3, Path + Memory hay kNN.

| Dataset | Feature | R@20 | R@50 | R@100 | Triple R@100 |
|---|---|---|---|---|---|
| WebQSP (800 train) | DDE | 65.0 | 75.0 | 81.9 | 77.5 |
| | **QR-DDE** | **73.0** | **83.1** | **88.5** | **84.5** |
| | DDE + QR-DDE | 72.8 | 83.1 | 88.0 | 84.3 |
| CWQ (1500 train) | DDE | 59.6 | 73.7 | 82.5 | 66.2 |
| | **QR-DDE** | 65.3 | **79.5** | **86.9** | **79.6** |
| | DDE + QR-DDE | 67.0 | 78.5 | 86.4 | 78.9 |

- id-match, trung bình 2 seed; chênh lệch giữa seed ≤ 2 điểm. Text-match cho WebQSP gần như trùng (R@50 trung bình 2 seed: DDE 75.0, QR-DDE 83.1).
- CWQ, nhóm câu có ≥ 2 topic entity (308 câu) tăng +4.1 R@50, nhóm 1 topic entity tăng +7.6: **không ủng hộ** giả thuyết "yếu riêng ở câu ràng buộc/giao".
- **Chưa kết luận được ở full data:** DDE train 800 câu kém bản full 15 điểm, nên lợi thế có thể là hiệu ứng ít dữ liệu. → M2.
- **Lưu ý (phát hiện 09-30):** schedule 25 epoch cố định **khác** SubgraphRAG (`retrieve/configs/retriever/*.yaml`: `num_epochs 10000`, `patience 10`). Cả 12 run của EXP-11 đạt best ở epoch 19–24/25, tức vẫn đang cải thiện khi bị cắt. Một phần mức chênh QR-DDE – DDE có thể là **tốc độ hội tụ** chứ không phải chất lượng hội tụ. M2 chạy lại với đúng schedule của SubgraphRAG; nếu kết luận subset còn dùng trong paper thì phải chạy lại EXP-11 với cùng schedule.

## EXP-12 — Transfer WebQSP → CWQ (retrieval)
- **Setup:** model train trên WebQSP, eval trên toàn bộ CWQ test (3,531 câu), không train thêm. REL-RAG: path best-first (có và không self-loop), triple, triple + residual. DDE/QR-DDE: checkpoint subset 800 câu (EXP-11), 2 seed. Mốc tham chiếu: SubgraphRAG chính thức train trên CWQ (in-distribution). Mốc lịch sử (Phase 1): SubgraphRAG train WebQSP → CWQ đạt R@50 68.5, R@100 78.7, R@400 92.9 (id-match, cách đo cũ).
- **Sửa lỗi:** CUDA "illegal memory access" với subgraph 1 triple khi có self-loop (cuSPARSE, ma trận 1×1) → dùng ma trận dense cho graph dưới 32 node (`model.py`).
- **Kết quả** (text-match, 2,848 câu CWQ test có đáp án trong graph và có mặt ở mọi run):

| Method (train trên WebQSP trừ dòng đầu) | R@20 | R@50 | R@100 | R@500 |
|---|---|---|---|---|
| *SubgraphRAG train trên CWQ (mốc in-distribution)* | *77.10* | *86.20* | *91.43* | *98.36* |
| REL-RAG path best-first, có self-loop (đúng paper) | 38.42 | 53.64 | 62.99 | 81.01 (417 triple) |
| REL-RAG path best-first, không self-loop | 37.71 | 53.42 | 63.83 | 80.34 (420 triple) |
| REL-RAG triple (đúng paper) | 36.64 | 51.81 | 65.76 | 90.12 |
| **REL-RAG triple + residual (cải tiến của mình)** | **59.10** | **74.62** | **83.72** | **97.17** |
| DDE, 800 câu WebQSP, seed 0 / 1 | 42.19 / 43.41 | 55.88 / 57.30 | 65.04 / 66.35 | 87.84 / 89.50 |
| QR-DDE, 800 câu WebQSP, seed 0 / 1 | 42.62 / 47.23 | 58.18 / 64.02 | 71.09 / 75.15 | 92.27 / 93.06 |

- **Kết luận:**
  1. REL-RAG theo đúng cấu hình paper **tổng quát hóa kém ở tầng retrieval**: R@50 ~53 so với 86 in-distribution. Nó còn thấp hơn DDE/QR-DDE chỉ train 800 câu (56–64). Mốc lịch sử SubgraphRAG WebQSP→CWQ (Phase 1, id-match): R@50 68.5.
  2. **Residual là yếu tố quyết định cho transfer**: triple-based tăng từ 51.8 lên 74.6 R@50 (+22.8). Giả thuyết: không có skip connection, biểu diễn triple bị cấu trúc lân cận chi phối (đặc thù WebQSP) nên không chuyển được sang CWQ.
  3. QR-DDE > DDE ở cả 2 seed (+2.3 / +6.7 R@50) nhưng dao động giữa seed lớn; cần full data và 3 seed (M2).
  4. Path-based không lấp đủ budget 500 (~420 triple, giới hạn số bước của best-first); không ảnh hưởng tới @50/@100.
- **Lưu ý:** paper so line graph với *raw graph cùng kiến trúc*, và đo downstream (GPT-4o-mini, 500 triple), không đo recall. Cần một baseline SubgraphRAG/DDE train full WebQSP để so sánh chuẩn (M2).
- **Artifact:** `/relrag/runs/transfer_webqsp_to_cwq.txt`, `/relrag/run_transfer.sh`, `retrieve/exp_qrdde/transfer_eval.py`.

## EXP-13 — Kiểm chứng: entity nối với nhau bằng nhiều relation
- **Câu hỏi:** REL-RAG có xử lý được cặp entity nối bằng nhiều relation không (giả thuyết H4)?
- **Setup:** đếm trên toàn bộ WebQSP test và CWQ test (subgraph của SubgraphRAG); đối chiếu với bằng chứng GPT (`retrieve/data_files/*/gpt_triples.pth`).

| | WebQSP | CWQ |
|---|---|---|
| Triple thuộc cặp entity có ≥2 relation **cùng chiều** (h,r₁,t), (h,r₂,t) | 12.7% | 14.3% |
| Triple chỉ có thêm bản **ngược chiều** (t,r⁻¹,h) (dư thừa của Freebase) | 42.9% | 46.2% |
| Bằng chứng GPT thuộc cặp multi-relation cùng chiều | 12.5% | 11.5% |
| Số câu có bằng chứng GPT thuộc cặp multi-relation | 22.1% (312/1411) | 28.6% (909/3183) |
| Bằng chứng GPT **bị loại** do code REL-RAG gộp cạnh bằng `nx.DiGraph` | 6.5% | 6.1% |

- **Kết luận:**
  1. Multi-relation thật (cùng chiều) chiếm ~13–14% triple và ảnh hưởng 22–29% câu hỏi. Code gốc của REL-RAG chỉ giữ 1 relation cho mỗi cặp (h, t), nên mất ~6% bằng chứng. Đây là **lỗi cài đặt, sửa được bằng MultiDiGraph**, chưa đủ làm đóng góp riêng.
  2. Ở mức phương pháp, ngay cả với MultiDiGraph: hai triple song song (h,r₁,t) và (h,r₂,t) có cùng tập láng giềng trong line graph, nhưng **không có cạnh nối với nhau** (line graph chỉ có chuyển tiếp tail→head). Mô hình vì vậy không biểu diễn được một cặp entity như một *tập relation*, cũng không so sánh hay kết hợp được các relation song song.
  3. ~45% triple là bản ngược chiều dư thừa. Line graph nhân đôi số node cho chúng, góp phần vào mức bùng nổ 83–107× (EXP-08).
  4. **Cảnh báo cho hướng HGT:** HGT gốc dùng ma trận riêng cho từng loại node/relation. Freebase có hàng nghìn relation, và khi transfer sẽ gặp relation chưa thấy. Chính REL-RAG (Theorem 3.1) lập luận rằng mô hình có W_r riêng cho mỗi relation tổng quát hóa kém hơn √R lần. Muốn dùng hướng heterogeneous/graph transformer thì **phải điều kiện hóa theo embedding văn bản của relation và theo câu hỏi**, không dùng tham số riêng cho từng loại relation.


## EXP-14 — QR-DDE so với DDE trên WebQSP full data + ablation (M2)
- **Mục tiêu:** kiểm tra mức +8.1 R@50 của QR-DDE ở EXP-11 (subset 800 câu, 25 epoch) có còn khi train trên toàn bộ WebQSP với đúng protocol của SubgraphRAG không; tìm nguồn gốc lợi thế (RQ2).
- **Setup:** `retrieve/exp_qrdde/qrdde.py --full`, `run_full.sh`. Train 2,710 câu, val 234, test 1,559 (có đáp án trong graph). Cùng MLP scorer, BCE, nhãn SP, Adam 1e-3, batch 1 như `retrieve/train.py`; seed 42 deterministic, `num_epochs 10000`, `patience 10`, chọn theo val triple_recall@100; log wandb project `REL-RAG`. Một seed. Text-match theo `a_entity_in_graph` của file SubgraphRAG chính thức; khác biệt đánh giá bằng paired bootstrap trên 1,559 câu (2,000 lần, CI 95%). Embedding GTE-large đóng băng.

| Method | R@20 | R@50 | R@100 | R@200 |
|---|---|---|---|---|
| SubgraphRAG chính thức | 82.58 | 90.38 | 94.42 | 96.94 |
| REL-RAG path best-first, có self-loop (đúng paper) | 75.90 | 87.06 | 92.38 | 96.11 |
| REL-RAG path best-first, không self-loop | 78.91 | 88.77 | 93.80 | 96.76 |
| REL-RAG triple + residual | 81.54 | 89.92 | 94.54 | 97.41 |
| DDE (tái hiện SubgraphRAG) | 82.17 | 90.07 | 94.41 | 97.18 |
| QR-DDE (K=3, d=64) | 80.77 | 89.20 | 93.98 | 97.16 |
| QR-DDE K=2 | 82.08 | 89.71 | 94.01 | 96.87 |
| QR-DDE K=4 | 81.23 | 89.03 | 93.85 | 96.88 |
| QR-DDE d=32 | 81.88 | 90.01 | 94.18 | 97.24 |
| QR-DDE d=128 | 81.86 | 89.99 | 94.20 | 97.04 |
| QR-DDE bỏ query-conditioning (`qrdde_noq`) | 81.17 | 89.09 | 93.11 | 96.60 |
| RT-DDE (DDE trọng số cos(z_q, z_r), floor 0.1) | 80.17 | 88.49 | 93.13 | 96.31 |

Paired bootstrap, chênh lệch theo điểm [CI 95%]:

| So sánh | ΔR@50 | ΔR@100 |
|---|---|---|
| QR-DDE − DDE | −0.87 [−2.02, +0.26] | −0.43 [−1.34, +0.50] |
| DDE − SubgraphRAG chính thức | −0.31 [−1.31, +0.67] | −0.01 [−0.73, +0.75] |
| QR-DDE − REL-RAG path không self-loop | +0.43 [−1.01, +1.86] | +0.18 [−0.94, +1.33] |
| QR-DDE − REL-RAG triple + residual | −0.72 [−2.16, +0.65] | −0.55 [−1.70, +0.51] |
| RT-DDE − DDE | **−1.58 [−2.62, −0.56]** | **−1.28 [−2.13, −0.46]** |
| QR-DDE bỏ conditioning − QR-DDE | −0.11 [−1.22, +0.99] | −0.88 [−1.87, +0.08] |

Thông tin train (id-match từ `result.json`): DDE dừng ở epoch 39 (best 28), val triple_recall@100 87.4, test triple_recall@100 88.5, 4.22M tham số, 10.8 ms/câu. QR-DDE dừng ở epoch 30 (best 19), val 91.3, test triple_recall@100 90.7, 4.97M tham số, 11.7 ms/câu. Các biến thể QR-DDE có val 91.6–92.8.

- **Kết luận:**
  1. **Tái hiện SubgraphRAG đạt:** DDE của mình bằng bản chính thức (ΔR@50 −0.31, CI chứa 0). Baseline đúng.
  2. **Lợi thế của QR-DDE ở EXP-11 biến mất ở full data:** R@50 89.20 so với 90.07 của DDE, CI chứa 0, điểm trung bình còn thấp hơn. Kết hợp với việc mọi run EXP-11 bị cắt ở 25 epoch, mức +8 điểm ở subset là hiệu ứng ít dữ liệu / tốc độ hội tụ, không phải chất lượng mô hình. **Theo go/no-go của M2, QR-DDE không đạt.**
  3. QR-DDE khớp nhãn SP tốt hơn (triple_recall@100 cao hơn ~2 điểm ở cả val và test) nhưng không chuyển thành answer recall: DDE đã phủ gần hết đáp án ở các budget này.
  4. Ablation (RQ2): query-conditioning trong w(r) không mang lại gì đo được (`qrdde_noq` ≈ QR-DDE); K và d không tạo khác biệt vượt nhiễu (dao động giữa các cấu hình ~1 điểm R@50, trong khi chỉ có 1 seed); RT-DDE **kém DDE có ý nghĩa**.
     **Đính chính (10-01):** "RT-DDE" ở đây chỉ là DDE có trọng số cos(z_q, z_r) thô từ GTE, floor 0.1, không có relation selector, mask hay calibration. Nó **không phải** "RT combined" (weighted DDE với relation-topic score từ selector) của giai đoạn SubgraphRAG, vốn tăng Ans R@50 từ 0.906 lên 0.918 (GTE) và lên 0.935 khi kết hợp weighted loss + BGE-M3 fine-tuned (`archive/subgraphrag/TOPIC_RELATION_SYNTHESIS.md` §9). Kết quả RT-DDE ở đây không bác bỏ hướng relation-topic.
  5. In-distribution WebQSP gần như bão hòa: SubgraphRAG, DDE, mọi biến thể QR-DDE, REL-RAG triple + residual đều trong khoảng 89–90.4 R@50. Chỉ REL-RAG đúng paper (có self-loop) thấp hơn rõ (87.06). Benchmark này không phân biệt được các phương pháp; phép thử có ý nghĩa là transfer WebQSP→CWQ (EXP-12: REL-RAG đúng paper 53.6 R@50, DDE subset 56–57).
  6. Nhóm câu nhiều topic entity chỉ có 26 câu trên WebQSP test, quá ít để kết luận.
- **Còn thiếu:** transfer sang CWQ test của các checkpoint full data (`transfer_eval.py --runs_dir runs`), chưa chạy.
- **Artifact:** `retrieve/exp_qrdde/runs/webqsp_*/{result.json,best.pt,retrieval_test.pth}`, `logs/full_webqsp_*.log`, `logs/run_full.log`. Các run 25 epoch cũ (không dùng) ở `runs_ep25/`.
- **Bổ sung (10-01): evidence recall theo bằng chứng GPT** (`retrieve/data_files/webqsp/gpt_triples.pth`, 1,357 câu, trung bình 5.2 triple/câu). Answer recall không phân biệt được path đúng với path sai chạm tới đáp án, nên đo thêm tỷ lệ triple bằng chứng nằm trong top-K.

| Method | ER@50 (khớp đúng triple) | ER@100 | PR@50 (khớp cặp entity, bỏ chiều) | PR@100 |
|---|---|---|---|---|
| SubgraphRAG chính thức | 81.03 | 87.42 | 83.63 | 89.17 |
| REL-RAG path không self-loop | 59.79 | 64.07 | 81.92 | 86.47 |
| REL-RAG triple + residual | 63.91 | 69.26 | 85.06 | 89.79 |
| DDE | 81.02 | 86.93 | 83.44 | 88.85 |
| QR-DDE | 82.15 | 87.23 | 83.32 | 88.15 |
| QR-DDE bỏ conditioning | 80.20 | 85.67 | 82.06 | 86.76 |

  - QR-DDE − DDE: ER@50 +1.13 [−0.05, +2.30], PR@50 −0.12 [−1.37, +1.11] → không khác biệt; phần hơn nhỏ ở ER đến từ việc lấy thêm bản ngược chiều/relation dư thừa của cùng cặp entity.
  - Bỏ query-conditioning làm giảm có ý nghĩa: ER@50 −1.96 [−3.06, −0.86], PR@50 −1.26 [−2.42, −0.16]. Conditioning có tác dụng nhỏ với việc lấy đúng bằng chứng, dù không thấy ở answer recall.
  - **Cảnh báo:** ER của REL-RAG thấp ~20 điểm là do lệch hướng (REL-RAG duyệt xuôi từ topic entity, bỏ các bản ngược chiều dư thừa mà GPT cũng gắn nhãn), **không phải** do lấy sai bằng chứng. Ở mức cặp entity, REL-RAG triple + residual còn nhỉnh hơn QR-DDE (PR@50 −1.74 [−3.28, −0.26] cho QR-DDE). Không được dùng ER để kết luận REL-RAG kém.
  - Script phân tích: tạm trong scratchpad của session, logic ghi ở trên (tập câu: có bằng chứng GPT, có đáp án trong graph, có mặt trong mọi run).

## EXP-15 — Nhãn rational path GPT-4o (đúng quy trình tác giả) so với gemma, 901 câu WebQSP train (M0)
- **Mục tiêu:** tái hiện bước gán nhãn của REL-RAG bằng GPT-4o và xem nhãn có tốt hơn nhãn gemma (EXP-01) không.
- **Setup:** `/relrag/gpt_label.py`, bám theo code gốc: ứng viên theo `retriever_v2.cut_off_paths` (path xuôi dài d..d+2, top 10 theo độ dài), prompt nguyên văn `FindRationalPath.py`, `openai/gpt-4o` qua OpenRouter (provider OpenAI), temperature 0. Batch không dùng được (OpenRouter gỡ các endpoint `:batch` của OpenAI từ 26/08/2026), nên gọi từng request. **Điểm lệch:** bỏ ràng buộc cuối dòng trong regex parse của tác giả, vì GPT-4o hiện tại viết giải thích ngay sau `[...]` trên cùng dòng (nếu giữ regex gốc, 5/20 câu thử mất nhãn). Chỉ chạy được 901/2,710 câu train (sample_0…, hết credit), tốn $4.02 ($0.0045/câu, không có cache).
- **Kết quả** (703 câu có ≥ 2 relation path khác nhau; 198 câu chỉ có 1, GPT loại 4):

| GPT-4o so với gemma | Số câu |
|---|---|
| Chọn giống hệt | 332 (47%) |
| GPT chọn thêm path (bao trùm gemma) | 306 (44%) |
| Chọn khác hẳn | 47 (7%) |
| GPT không có nhãn | 16 (2%) |

  - Số relation path được chọn mỗi câu: GPT 1.72, gemma 0.98; 33% path GPT chọn dài hơn shortest path.
  - 20 câu không có nhãn: khoảng 11 câu GPT chủ động loại hết, phần lớn hợp lý (ví dụ hỏi sách nhưng ứng viên chỉ có path phim); 9 câu GPT chép nguyên dòng path. Trong 9 câu này có câu GPT chọn thật (627, 284) và có câu GPT kết luận "không path nào hợp lý" (49, 550), nên giữ là không có nhãn thay vì cố parse.
  - **Đọc tay 25 câu ngẫu nhiên trong 47 câu chọn khác hẳn** (Claude chấm, không có đáp án chuẩn cho train): GPT tốt hơn 21, gemma tốt hơn 3, hòa 1. Cả 3 câu gemma thắng đều do path đúng (`adjoin_s > adjoins`, `places_lived`, `compositions`) không có trong top-10 ứng viên của tác giả. Lỗi điển hình của gemma: chọn path ngắn nhất dù sai nghĩa (`place_of_birth` cho câu "sống ở đâu", `appointed_by` cho câu "ai là thống đốc", `namesakes`, `image`, `works_written`).
- **Kết luận:**
  1. Nhãn theo quy trình tác giả tốt hơn rõ rệt ở những câu hai bên khác nhau.
  2. Phần lớn chênh lệch đến từ **tập ứng viên**: gemma (EXP-01) chỉ được chọn trong các shortest path, nên câu "sống ở đâu" không có `places_lived` (dài 2) để chọn khi `place_of_birth` (dài 1) đã tới được đáp án. Chưa tách được bao nhiêu phần do ứng viên, bao nhiêu do GPT-4o.
  3. Top-10 của tác giả đôi khi cắt mất path đúng (1.5% câu theo nhãn gemma); đây là hành vi của bản gốc nên giữ nguyên.
- **Artifact:** `/relrag/data/labels_gpt4o/webqsp_train/{candidates,requests,labels,usage}.jsonl`, `responses/`.

### EXP-15b — DeepSeek-V4.1-Flash (API chính chủ) gán nhãn toàn bộ 2,710 câu, so với GPT-4o trên 901 câu
- **Setup:** `gpt_label.py sync --api deepseek --model deepseek-flash --no_reasoning`: cùng request (prompt nguyên văn của tác giả, ứng viên d..d+2 top 10), `thinking: disabled`, temperature 0, gọi thẳng api.deepseek.com. Hai tin nhắn `user` liền nhau của prompt gốc được chấp nhận, nên không phải sửa prompt. Chi phí ≤ $1.28 (ước tính theo giá giờ cao điểm), cache hit 58% số token prompt, output trung bình khoảng 240 token (GPT-4o khoảng 128). 2,693/2,710 câu ghép được ứng viên.
- **So với GPT-4o (901 câu):**

| | Câu nhiều relation path (703) | Câu 1 relation path (198) |
|---|---|---|
| Giống hệt | 392 (55.8%) | 192 (97.0%) |
| DeepSeek chọn thêm path (bao trùm GPT) | 249 (35.4%) | – |
| DeepSeek chọn ít hơn (tập con của GPT) | 36 (5.1%) | – |
| Trùng một phần | 8 (1.1%) | – |
| Khác hẳn | 2 (0.3%) | – |
| GPT loại hết, DeepSeek có nhãn | 15 (2.1%) | 4 (2.0%) |
| DeepSeek loại hết | 1 (0.1%) | 2 (1.0%) |

  - Số relation path chọn trên mỗi câu nhiều path: DeepSeek 2.47, GPT-4o 1.72, gemma 0.98 (EXP-15). 84% path DeepSeek chọn thêm dài hơn shortest path.
  - **Đọc tay 25 câu ngẫu nhiên DeepSeek chọn thêm:** 15 câu thêm path vòng vèo rõ ràng (ngôn ngữ của Úc/Iceland/Ghana lấy qua phim, báo, quốc ca; trường của Cassie Bernall lấy qua vụ thảm sát; `sport > team_coaches`; `award_nominations` cho câu hỏi phim), 10 câu thêm path hợp lý hoặc tương đương (CVT `partially_contained_by`, `salary > team`, `near_travel_destination > containedby`).
  - **19 câu GPT loại hết nhưng DeepSeek có nhãn:** phần lớn GPT loại đúng mà DeepSeek vẫn gán (Mali Empire so với nước Mali hiện tại, Nissan "ai sở hữu" lại chọn `founders`, LBJ "quê ở đâu" lại chọn `place_of_death`, nước giáp Pháp qua lưu vực sông). Một số câu DeepSeek đúng còn GPT loại nhầm (`adjoin_s > adjoins` cho South Dakota, `instruments_played` cho Béla Fleck).
- **Kết luận:** DeepSeek giữ gần như toàn bộ path GPT-4o chọn (trùng ≥ 1 path ở 99.7% câu cả hai đều có nhãn) nhưng lọc lỏng hơn rõ: khoảng 1/5 số câu nhiều path có thêm path vòng vèo, và hiếm khi loại hết. Độ chặt của nhãn DeepSeek nằm giữa GPT-4o và nhãn shortest path, tức bước lọc mà paper nhấn mạnh (Table 8) bị yếu đi một phần.
- **Artifact:** `/relrag/data/labels_deepseek-flash_deepseekapi_noreason/webqsp_train/{labels,usage,compare_disjoint}.jsonl`, `responses/`.

## EXP-14b — QR-DDE so với DDE trên CWQ full data (M2, QR-DDE chưa chạy xong)
- **Setup:** như EXP-14 (`qrdde.py --full`, seed 42 deterministic, patience 10, chọn theo val triple_recall@100, wandb project `REL-RAG`). Train 21,943 câu, đọc embedding qua `ShardEmbAdapter`; val 2,837; test 2,848. Mỗi epoch khoảng 11.7 phút (DDE) và 16 phút (QR-DDE).
- **DDE:** chạy xong, dừng ở epoch 44, best epoch 33 (val triple_recall@100 0.823). Test (id-match): R@20 76.4, **R@50 86.7**, R@100 91.3, R@200 95.1; triple_recall@100 80.7. Mốc so: SubgraphRAG chính thức train trên CWQ đạt R@50 86.2 (text-match, EXP-12), nên bản tái hiện DDE trên CWQ ngang bản gốc.
- **QR-DDE:** người dùng ngắt ở epoch 10 (log `FAILED cwq_qrdde` lúc 10-01 16:58). Best epoch 8, khi đó vẫn đang cải thiện. `best.pt` (epoch 8) còn ở `runs/cwq_qrdde/`; chưa có eval test.
- **So sánh trên val** (wandb chỉ log val ở epoch cải thiện, nên đây là số của epoch tốt nhất):

| Val CWQ (2,837 câu) | ans R@20 | ans R@50 | ans R@100 | ans R@200 | triple R@50 | triple R@100 |
|---|---|---|---|---|---|---|
| DDE, epoch 8 (cùng mức train) | – | 87.0 | – | – | – | 78.9 |
| DDE, best epoch 33 | 80 | 88.7 | 94 | 96 | 73 | 82.3 |
| QR-DDE, best epoch 8 (chưa hội tụ) | 80 | **91** | 95 | 98 | **82** | **89** |

  Số trong wandb làm tròn 2 chữ số; cột DDE epoch 8 lấy từ `result.json`.
- **Nhận xét:**
  1. **Khác với WebQSP, trên CWQ QR-DDE có tín hiệu tốt.** Chỉ sau 8 epoch, QR-DDE đã hơn DDE ở epoch tốt nhất: khoảng +2 ans R@50 và +7 triple R@100 trên val. Ở cùng epoch 8, khoảng cách ans R@50 là +4.
  2. Điều này hợp với giả thuyết rằng **WebQSP đã bão hòa** (EXP-14: mọi phương pháp 89–90 R@50), còn CWQ (câu nhiều hop, nhiều ràng buộc) mới là nơi lan truyền có điều kiện theo câu hỏi có đất. Nhóm câu nhiều topic entity cũng lớn hơn hẳn (1,460 câu val, so với 26 câu test ở WebQSP), nên đo được khác biệt ở câu ràng buộc.
  3. **Chưa đủ để kết luận:**
     - Mới có val, chưa có test, chưa text-match, chưa bootstrap.
     - QR-DDE chưa hội tụ.
     - Chênh lệch triple recall lớn hơn chênh lệch answer recall. Ở WebQSP, triple recall +2 không chuyển thành answer recall (EXP-14), nên phải kiểm tra thêm bằng evidence recall: CWQ có bằng chứng GPT ở `reason/gpt_labeled/gpt_labeled_cwq_cleaned.pt`.
- **Việc tiếp theo** (người dùng quyết định và chạy):
  - (a) Eval nhanh `best.pt` epoch 8 trên CWQ test. `qrdde.py` chưa có chế độ chỉ eval, cần thêm.
  - (b) Hoặc chạy lại `cwq_qrdde` tới khi dừng theo patience, ước tính 8–10 giờ.
  - Sau đó so test text-match với `--ref reason/scored_triples/cwq_240907_unidir_test.pth`, paired bootstrap, và evidence recall như EXP-14.
- **Artifact:** `retrieve/exp_qrdde/runs/cwq_dde/{result.json,best.pt,retrieval_test.pth}`, `runs/cwq_qrdde/best.pt` (epoch 8), `logs/full_cwq_*.log`, wandb project `REL-RAG`.

## EXP-16 — REL-RAG đúng cấu hình paper, nhãn GPT-4o + DeepSeek (M0)
- **Setup:** `/relrag/run_paper.sh`. App. F.1 (2-layer bi-GCN 512, Adam 1e-3, batch 10, 15 epoch, dropout 0.2), seed 42, không self-loop (theo code gốc). Nhãn train: `labels_hybrid_gpt4o_deepseek` (EXP-15b). Nhãn val: SP. Chọn checkpoint theo **val loss** (như paper viết). Path P mất khoảng 47 phút, triple T khoảng 43 phút.
- **Kết quả retrieval (text-match):**

| | R@20 | R@50 | R@100 | R@500 |
|---|---|---|---|---|
| WebQSP: SubgraphRAG chính thức | 82.58 | 90.38 | 94.42 | 98.93 |
| WebQSP: **REL-RAG(P)**, path best-first | 67.40 | 80.40 | 89.47 | 98.01 |
| WebQSP: **REL-RAG(T)**, triple | 59.75 | 74.18 | 83.08 | 96.48 |
| *(cũ, gemma, đã xóa: P không self-loop / T)* | *78.35 / 68.65* | *88.77 / 80.43* | *93.80 / 87.80* | – |
| CWQ transfer: SubgraphRAG train trên CWQ | 77.10 | 86.20 | 91.43 | 98.36 |
| CWQ transfer: **REL-RAG(P)** WebQSP→CWQ | 46.96 | 61.24 | 69.25 | 85.03 |
| CWQ transfer: **REL-RAG(T)** WebQSP→CWQ | 36.04 | 52.09 | 64.63 | 92.48 |
| *(cũ, gemma: P không self-loop / T)* | *37.71 / 36.64* | *53.42 / 51.81* | *63.83 / 65.76* | – |

  - **Bằng chứng GPT ở budget 50** (WebQSP, 1,357 câu): P đạt evidence theo cặp entity 75.0, so với 83.6 của SubgraphRAG (−8.6, CI [−10.4, −6.9]); ans R@50 trên tập câu này −9.7 (CI [−11.6, −7.7]). T còn thấp hơn P (−5.0 evidence, −6.3 ans R@50).
  - Diễn biến train P: val loss (theo nhãn SP) dao động 0.62–0.85 và chọn **epoch 7** (val ans R@50 0.766), trong khi epoch 12–14 đạt 0.82–0.83 và label_path_hit 0.876. T chọn epoch 13, hợp với val ans R@500 (0.96).
- **Nhận xét:**
  1. **In-distribution kém hơn hẳn** cả SubgraphRAG lẫn các run gemma trước: P −8.4, T −6.3 R@50. **Transfer của P lại tốt hơn** run gemma cũ (+7.8 R@50), còn T không đổi.
  2. **Nguyên nhân 1, chọn checkpoint (P):** paper viết "chọn theo validation loss", nhưng **code gốc** (`RAPL/src/model/Trainerv3.py::fit`) chỉ validate từ epoch 6 trở đi và giữ top-3 checkpoint theo **F1 giữa điểm kết thúc của path được sample và điểm kết thúc của path nhãn trên val**. Mình chọn theo val loss với nhãn SP, nên lấy nhầm epoch 7 chưa đủ train. Có thể kiểm tra nhanh bằng cách infer `epoch_12.pt`/`epoch_14.pt`.
  3. **Nguyên nhân 2, nhãn (cả P và T):** path nhãn dài hơn (d..d+2) và DeepSeek lọc lỏng hơn (EXP-15b), nên có thêm triple dương không chứa đáp án. T đã chọn đúng epoch mà vẫn −6 R@50, nên nhiều khả năng là do nhãn. Còn lẫn yếu tố self-loop: T cũ có self-loop, T mới thì không.
  4. Chưa đủ cơ sở so với số downstream của paper. Paper không báo cáo recall retrieval của REL-RAG.
- **Artifact:** `/relrag/runs/{path_paper,triple_paper}/` (`log.json`, `epoch_*.pt`, `best.pt`, `retrieval_*`), `runs/compare_webqsp.txt`, `runs/compare_cwq_transfer.txt`, `logs/run_paper.log`.

## EXP-17 — REL-RAG theo đúng code tác giả: retrieval (recall, precision) và reasoning dạng path (M0)
- **Setup retrieval ("code-faithful"):** `train.py --regime path --labels paths:data/labels_hybrid_gpt4o_deepseek/train_minlen.jsonl --val_labels paths:.../val_minlen.jsonl --seed 1 --no_self_loops --combine sum --gnn_dropout 0 --topic_loss sample --batch_size 4`. Chọn checkpoint theo path_f1 (validate từ epoch 6), ra epoch 15. T dùng cùng cờ với `--regime triple`, chọn theo val loss, ra epoch 14.
  - Nhãn minlen: chỉ giữ path ngắn nhất trong các path LLM đã chọn.
  - Decode: `path_bf` (best-first tới đủ budget), `path_free` (đúng `Trainerv3.decoding(way='normal')`, K=800, N=3, M=1200, depth 6, xác suất ≥ 1e-4, **bỏ** topic label ground-truth của test mà code tác giả dùng), `triple` (top-k).
  - Eval: `eval_triples.py`, text-match, 1,559 câu.
- **Answer recall WebQSP:**

| | R@20 | R@50 | R@100 | R@500 |
|---|---|---|---|---|
| SubgraphRAG chính thức | 82.58 | 90.38 | 94.42 | 98.93 |
| **P code-faithful, path_bf** | **82.83** | **90.60** | **94.84** | 98.83 |
| P code-faithful, path_free (42.8 triple/câu) | – | 90.93 tại 42.8 | – | – |
| T code-faithful | 62.95 | 75.60 | 84.82 | 96.29 |

- **CWQ transfer (2,848 câu):** SubgraphRAG train trên CWQ 86.20 R@50; P code-faithful 61.90; T code-faithful 47.76.
- **Triple R/P/F1 so ở cùng kích thước** (cắt SubgraphRAG về đúng số triple của P free trên từng câu, khoảng 42.8). Các số là P free trừ SubgraphRAG, kèm CI bootstrap 95%:
  - Nhãn SP: R +12.9, P +8.1, F1 +8.7. Nhãn này thiên vị P.
  - Nhãn GPT (exact): R −12.2, P −3.4, F1 −4.3.
  - Nhãn GPT theo cặp entity: R +3.5, P −5.6, F1 −3.0.
  - Kết luận: không có đột phá precision khi đo bằng nhãn độc lập.
  - Trần của REL-RAG với nhãn GPT: chỉ 92.2% triple bằng chứng còn trong DiGraph; ngoài ra path chỉ đi xuôi nên bỏ sót bằng chứng chiều ngược.
- **Reasoning WebQSP** (`reason_paths.py`):
  - Input dạng path và prompt lấy nguyên văn từ code tác giả (`write_question_paths_to_txt` cộng `post_str` của `generate_answers_llama31_8b.py`).
  - Reasoner: llama3.1 8B Q4 trên ollama, greedy. Tác giả dùng bf16 với beam 5, nhưng GPU 1080 Ti không chạy được.
  - Parse phần sau "The answer is:" cuối cùng.
  - Chấm: Hit của eval gốc; Macro-F1 của eval corrected nhưng **dùng match gốc** (chuỗi con), vì `match` trong repo đã bị nới thành trùng 3-gram ký tự ở commit 84aae95. Cột "3-gram" trong bảng là để đối chiếu với các số cũ.

| Run (1,6xx câu test) | Hit | Hit@1 | Macro-F1 | Macro-F1 (3-gram) |
|---|---|---|---|---|
| Paper: Ours(P)+Llama3.1-8B @50 | 88.8 | – | 77.3 | – |
| Paper: SubgraphRAG+Llama3.1-8B @100 | 86.6 | – | 70.5 | – |
| SubgraphRAG@100 + Q4, prompt SubgraphRAG | 85.05 | 80.96 | 68.83 | 73.26 |
| SubgraphRAG@50 + Q4, prompt SubgraphRAG | 84.27 | 80.60 | 69.10 | 72.78 |
| REL-RAG(P) cũ (gemma) @50 + Q4, prompt SubgraphRAG | 83.21 | 79.12 | 69.45 | 73.98 |
| **REL-RAG(P) code-faithful free + Q4, prompt tác giả** | **68.03** | 57.17 | **44.70** | 53.62 |
| REL-RAG(P) code-faithful free (~43 triple) + Q4, **prompt SubgraphRAG** (`run_reason.sh … 1000 16384`, 1,638 câu) | 83.03 | 78.39 | 68.46 | 72.43 |

  - Với cùng retrieval tự do nhưng dùng prompt SubgraphRAG, REL-RAG đạt 83.03 / 68.46. Con số này ngang SubgraphRAG@50 (84.27 / 69.10) và @100 (85.05 / 68.83), nên phần thấp ở dòng trên là do prompt / parse của tác giả.
  - Chẩn đoán run prompt tác giả:
    - 16% câu không parse được: 216 câu không theo template "The answer is".
    - Đáp án đúng có trong toàn bộ response ở 86.5% số câu, nhưng chỉ có trong phần đáp án được parse ở 60.4%.
    - Trong số câu mà đáp án nằm trong các triple đã retrieve (90.1%), chỉ 65.3% được trả lời đúng.
    - Chấm trên text thô theo eval gốc: Hit 91.95 nhưng F1 16.0, nên paper không thể chấm F1 trên text thô.
  - Nguyên nhân thấp: prompt "think step by step" với template tự do, cộng greedy Q4 và repetition_penalty 1.25, làm llama trả lời bằng câu văn hoặc bỏ template. Đây không phải lỗi retrieval.
  - Paper không công bố parser đáp án, nên chưa tái hiện được 77.3/88.8.
- **Artifact:**
  - `/relrag/runs/{path_codefaithful,triple_codefaithful}/`, `runs/eval_triples_webqsp.txt`.
  - `runs/path_codefaithful/retrieval_test_path_free/{retrieval_result_free.pth,reason/}`.
  - `logs/reason_path_codefaithful_free.log`.

## EXP-14c — QR-DDE CWQ full, train tiếp từ epoch 8 tới khi dừng (M2)
- **Setup:** `qrdde.py --dataset cwq --feat qrdde --full --out runs/cwq_qrdde_resume --resume runs/cwq_qrdde/best.pt --start_epoch 9`.
  - Chỉ còn trọng số epoch 8, nên Adam khởi động lại. Các epoch 9–10 trước đây đã bị ngắt, nên được chạy lại.
  - Patience 10 tính từ val của epoch 8. Seed 42, deterministic.
  - Best epoch 17 (val triple_recall@100 0.9025), dừng ở epoch 27. Mỗi epoch khoảng 16 phút. Chạy 10-03 03:16–07:52.
- **Test (id-match, `result.json`):** ans R@20 81.6, **R@50 89.7**, R@100 93.2, R@200 96.6; triple_recall@100 89.4. DDE ở cùng điều kiện: R@50 86.7, triple_recall@100 80.7. Latency 13.4 ms/query.
- **Text-match, 2,848 câu, paired bootstrap** (`docs/REPORT.md` §2):

| QR-DDE trừ | ansR@50 | GPT-R@50 | ansR@100 | GPT-R@100 |
|---|---|---|---|---|
| DDE | +3.02 [+1.87, +4.13] | +6.71 [+5.67, +7.80] | +1.86 [+0.86, +2.87] | +4.36 [+3.40, +5.31] |
| SubgraphRAG chính thức | +3.49 [+2.35, +4.66] | +4.06 [+3.02, +5.09] | +1.74 [+0.87, +2.65] | +0.82 [−0.10, +1.75] |

- **Nhận xét:**
  - **M2 đạt go/no-go trên CWQ:** QR-DDE hơn DDE và SubgraphRAG có ý nghĩa thống kê ở R@50.
  - Trái ngược với WebQSP (EXP-14, không khác biệt). Điều này hợp với giả thuyết: điều kiện hóa theo câu hỏi trong lúc lan truyền có tác dụng ở câu nhiều ràng buộc / nhiều anchor (CWQ), còn WebQSP thì đã bão hòa.
  - Còn thiếu: transfer WebQSP→CWQ cho checkpoint full (đang xếp hàng, screen `transfer`), và ablation `qrdde_noq` trên CWQ.
- **Artifact:** `retrieve/exp_qrdde/runs/cwq_qrdde_resume/{best.pt,result.json,retrieval_test.pth}`, `logs/full_cwq_qrdde_resume.log`.

## EXP-18 — S0: ràng buộc gold có giúp LLM không? (witness, docs/08)
- **Setup:**
  - Tập câu: WebQSP test có ràng buộc entity (SPARQL parse), đáp án đã có trong top-50 của SubgraphRAG nhưng thiếu ràng buộc.
  - Reasoner: llama3.1 Q4, prompt SubgraphRAG (`scored_100`), match gốc. File ở `/relrag/runs/oracle_constraint/`.
  - **v1 (197 câu), không hợp lệ:** ràng buộc lấy theo (predicate, entity) ở bất kỳ đâu, nên chèn nhiễu (`Alicia Keys gender Female`).
  - **v2 (185 câu), lộ đáp án:** ràng buộc neo vào nút ở vị trí `SourceNodeIndex+1` của chuỗi suy luận, nhưng chỉ thêm vào nhánh đáp án. Kết quả: base 77.84, oracle 82.16, control (anh em cùng nút) 84.32 Hit.
  - **v3 (151 câu), đối xứng:** thêm cho mọi nhánh đáp án và nhánh nhiễu có trong ngữ cảnh, xáo thứ tự.
- **Kết quả v3:**

| | Hit | Hit@1 | Macro-F1 |
|---|---|---|---|
| base | 79.47 | 72.85 | 66.37 |
| sym-oracle | 85.43 | 80.79 | 73.42 |
| sym-control | 81.46 | 77.48 | 70.00 |

  - oracle − control (Hit theo substring): +2.65 [−2.65, +7.95].
- **Kết luận:**
  - Không đạt tiêu chí go (≥ +5). Hướng witness thành phân tích phụ.
  - Phân tích retrieval vẫn giữ: 27% câu WebQSP có ràng buộc; nhãn SP chứa 0/267, nhãn SubgraphRAG 22/267; recall ràng buộc @50 từ 7–31%.

## EXP-19 — Transfer WebQSP→CWQ cho checkpoint full + probe neo lại (docs/09)
- **Transfer** (`transfer_eval.py --runs_dir runs`, text-match, 2,848 câu):

| Retriever (train WebQSP) | ansR@50 | @500 |
|---|---|---|
| DDE | 70.05 | 94.81 |
| RT-DDE | 68.20 | 93.80 |
| QR-DDE-noq | 65.21 | 90.57 |
| REL-RAG P | 61.90 | 85.21 |
| QR-DDE | 60.95 | 91.70 |

  - DDE − REL-RAG P = +8.14 [+6.22, +10.06].
- **Theo loại câu, @50:**
  - composition: DDE 54.6, REL-RAG P 26.0, SubgraphRAG train trên CWQ 83.5.
  - Câu composition 1 topic: REL-RAG P 17.8, QR-DDE 26.2, DDE 52.1.
  - Comparative/superlative: REL-RAG P 85–90, DDE ~76.
- **Probe neo lại** (không train; entity cầu nối từ top-3 của lượt 1; lượt 2 với anchor = topic ∪ cầu nối; trộn max z):

| | DDE | DDE 2 lượt |
|---|---|---|
| CWQ | 70.05 | 75.94 |
| composition | 54.6 | 74.4 |
| WebQSP | 90.07 | 89.86 |

  - Neo lại vô điều kiện làm WebQSP giảm 1.86 [−2.66, −1.08]; maxz giảm còn −0.21.
  - QR-DDE 2 lượt (top3 + lượt 2): CWQ 72.84, composition 72.7 (từ 34.4).
- **Kết luận:** lỗi tổng quát hóa thành phần là do liên quan bị gắn với vị trí. Đây là hướng chính (docs/09).
  - Đang chạy: A1 (downstream CWQ @50 cho REL-RAG P / DDE / DDE 2 lượt).
  - Sau đó: A2 (`qrdde.py --infer_passes 2 --reanchor_p 0.5`, WebQSP full).

## EXP-20 — A2: AER đã train (anchor-equivariant retrieval), WebQSP full → CWQ transfer (docs/09)
- **Setup:** `qrdde.py --dataset webqsp --feat dde --full --out runs/webqsp_dde_aer --infer_passes 2 --reanchor_p 0.5`.
  - Seed 42, patience 10, chọn checkpoint theo val triple_recall@100. Best epoch 34, dừng ở epoch 45, chạy 10-03 22:54 – 10-04 00:02.
  - Suy luận 2 lượt: cầu nối lấy từ top-3, trộn max z.
  - Latency 30.3 ms/query (2 lượt), so với khoảng 10–13 ms của DDE 1 lượt.
  - Transfer CWQ: `transfer_eval.py --runs_dir runs webqsp_dde_aer`.
- **Kết quả (text-match, paired bootstrap):**

| ansR @20/@50/@100 | WebQSP (1,559 câu) | CWQ transfer (2,848 câu) |
|---|---|---|
| SubgraphRAG (in-distribution) | 82.58 / 90.38 / 94.42 | 77.10 / 86.20 / 91.43 (train trên CWQ) |
| REL-RAG P code-faithful | 82.83 / 90.60 / 94.84 | 46.00 / 61.90 / 72.26 |
| DDE | 82.17 / 90.07 / 94.41 | 55.72 / 70.05 / 79.58 |
| **AER** | 80.65 / 89.68 / 94.18 | **62.44 / 77.95 / 85.18** |

  - WebQSP, AER trừ: REL-RAG P −0.92 [−2.30, +0.42], DDE −0.38 [−1.42, +0.61], SubgraphRAG −0.70 [−1.67, +0.23]. Đều không có ý nghĩa thống kê.
  - CWQ transfer, AER trừ: **REL-RAG P +16.05 [+14.18, +18.01]**, DDE +7.90 [+6.38, +9.38], SubgraphRAG (train trên CWQ) −8.25.
  - Theo loại câu, CWQ @50:

| Loại | AER | DDE | REL-RAG P | Probe không train |
|---|---|---|---|---|
| composition | 70.0 | 54.6 | 26.0 | 75.4 |
| conjunction | 81.3 | 78.1 | 77.6 | – |
| comparative | 85.4 | 75.5 | 84.7 | – |
| superlative | 83.1 | 76.0 | 90.2 | – |

- **Nhận xét:**
  - Train có tăng cường anchor (`reanchor_p`) cải thiện **mọi loại câu**, trong khi probe không train làm conjunction giảm nhẹ. Đổi lại, composition thấp hơn probe (70.0 so với 75.4); cần ablation `reanchor_p` và B.
  - In-distribution không khác biệt có ý nghĩa; transfer +16 so với REL-RAG.

## EXP-21 — E1–E3: bằng chứng cơ chế và ablation cho AER (docs/09)
Công cụ: `aer_probe.py`, chỉ suy luận, lưu thứ hạng từng lượt. Đánh giá offline bằng `aer_eval_all.py`, text-match ansR@50.

**E1 — can thiệp dịch anchor (WebQSP test, 1,558 câu).**
- Can thiệp: giữ nguyên câu hỏi và đáp án; anchor = một entity văn bản kề topic, chọn ngẫu nhiên với seed 0, không phải topic hay đáp án. Bằng chứng vì thế cách anchor thêm 1 hop.

| Retriever | anchor = topic | Dịch, 1 lượt | Tụt (CI 95%) | Dịch, 2 lượt (maxz) | Hồi phục |
|---|---|---|---|---|---|
| DDE | 90.1 | 69.4 | −20.65 [−22.58, −18.70] | 77.0 | +7.63 |
| QR-DDE | 89.2 | 45.8 | **−43.34** [−45.78, −40.91] | 82.6 | +36.79 |
| AER (DDE) | 90.1 | 71.2 | −18.85 [−20.70, −17.05] | 78.1 | +6.92 |

- Kết luận:
  - Bằng chứng nhân quả cho liên quan bị gắn với vị trí: cùng câu hỏi, chỉ đổi vị trí anchor mà recall tụt 19–43 điểm.
  - Mô hình học càng nhiều cấu trúc lan truyền (QR-DDE) thì càng nhạy với vị trí.
  - Neo lại hồi phục phần lớn. Train neo lại một mình chỉ giảm độ nhạy chút ít (−18.9 so với −20.7); bước lặp mới là phần quyết định.

**E2 — ablation trên CWQ transfer** (ALL / composition / conjunction / comparative / superlative):

| Cấu hình | ALL | composition | conjunction | comparative | superlative |
|---|---|---|---|---|---|
| DDE, K=1 | 70.0 | 54.6 | 78.1 | 75.5 | 76.0 |
| DDE, K=2 maxz | 76.0 | 74.6 | 76.9 | 76.0 | 75.4 |
| DDE, K=2 thay hẳn | 73.6 | 73.3 | 74.0 | 71.7 | 73.2 |
| DDE, K=3 maxz | 75.7 | 74.3 | 76.6 | 75.7 | 75.4 |
| **DDE, K=3 thay hẳn** | **68.0** | 65.7 | 70.6 | 65.9 | 60.1 |
| AER, K=1 | 74.8 | 59.6 | 81.5 | 85.4 | 85.2 |
| **AER, K=2 maxz** | **78.2** | 70.3 | 81.2 | **85.9** | 84.2 |
| AER, K=3 maxz | 77.7 | 69.9 | 81.1 | 84.9 | 81.4 |
| AER, B=1 / B=10 (K=2 maxz) | 77.4 / 77.1 | – | – | – | – |
| AER, cầu nối ngẫu nhiên (K=2 maxz) | 75.5 | 62.1 | 81.5 | 84.8 | 84.7 |

- Cầu nối theo mô hình trừ cầu nối ngẫu nhiên: +2.62 [+1.85, +3.39] tính chung, +8.2 ở composition.
- Từ đó suy ra lý do cho từng lựa chọn:
  - **Trộn max đơn điệu:** thay hẳn bằng lượt cuối thì trôi khi K tăng (68.0 ở K=3).
  - **Cầu nối theo câu hỏi:** hơn ngẫu nhiên, nhất là ở composition.
  - **B = 3.**
  - **K = 2 đủ cho CWQ** (≤ 2 hop).
  - **Train neo lại** tăng mọi loại câu ngay ở 1 lượt (+4.8).

**E3 — kiểm tra lý thuyết.**
- Mệnh đề 2, câu CWQ composition 1 topic:
  - DDE (730 câu): bước chọn cầu nối đúng 91.2%. Khi trúng 82.1, khi trượt 45.2; 1 lượt 54.0, 2 lượt 78.9.
  - AER (689 câu): đúng 89.8%. Khi trúng 79.2, khi trượt 42.9.
- Mệnh đề 3, WebQSP top-50: số triple lượt 1 bị trộn đẩy ra ≤ cận ở **100%** số câu.
  - Trung bình bị đẩy ra: 11.7 (DDE), **6.5 (AER)**.
  - Cận trung bình: 24.0 (DDE), 11.6 (AER).
  - AER nhất quán theo anchor tốt hơn.

## EXP-22 — E4: neo lại áp dụng cho REL-RAG P (line graph), không train lại
- **Setup:** `infer.py --mode path_bf_reanchor --reanchor_B 3`.
  - Lượt 1: best-first từ các triple của topic.
  - Cầu nối: entity trong top-3 triple của lượt 1.
  - Lượt 2: điểm xuất phát = các triple có head là topic hoặc cầu nối.
  - Trộn: max của log-prob path đã chuẩn hóa z.
  - Checkpoint: `runs/path_codefaithful/best.pt`.
- **Kết quả @50, text-match:**

| | path_bf | + neo lại | Chênh (CI 95%) |
|---|---|---|---|
| WebQSP | 90.60 | 90.06 | −0.55 [−1.02, −0.11] |
| CWQ transfer | 61.90 | 70.45 | +8.54 [+7.33, +9.80] |

  - CWQ theo loại câu: composition 26.0 → 54.7; conjunction 77.6 → 76.3; comparative 84.7 → 84.2; superlative 90.2 → 88.0.
  - @100 / @500 với neo lại: WebQSP 94.63 / 98.72; CWQ 80.67 / 93.07 (id-match, log infer).
- **Kết luận:**
  - Line graph cũng bị gắn với vị trí và cũng được sửa bằng neo lại. Vấn đề là vị trí so với anchor, không phải cách biểu diễn đồ thị.
  - Nguyên lý dùng được cho nhiều mô hình.
  - AER trên DDE (CWQ 78.2) vẫn hơn REL-RAG + neo lại (70.45).

## EXP-23 — E1 cho REL-RAG, ablation `reanchor_p`, AER trên nền QR-DDE, A1 tạm thời
**E1 cho REL-RAG P** (`infer.py --shift_anchor`, WebQSP, 1,532 câu):
- topic 90.7 → dịch 1 hop **21.5** (tụt **−69.20** [−71.37, −67.07]) → dịch + neo lại 44.1 (+22.55).
- Thứ tự độ nhạy với vị trí: REL-RAG (−69) > QR-DDE (−43) > DDE (−21) > AER (−19). Thứ tự này **trùng với thứ tự kém khi transfer**.
- Lưu ý: neighbour e' chọn bằng seed theo chỉ số câu, khác với seed của `aer_probe.py`. Nên thống nhất e' giữa các mô hình trước khi viết paper.

**Ablation `reanchor_p`** (AER-DDE, `infer_passes 2`, CWQ transfer @50: ALL / composition / conjunction / comparative / superlative):

| p | ALL | composition | conjunction | comparative | superlative | WebQSP |
|---|---|---|---|---|---|---|
| 0.25 | 75.38 | 68.1 | 79.7 | 77.0 | 75.4 | 89.17 |
| **0.5** | **77.95** | **70.0** | 81.3 | 85.4 | 83.1 | 89.68 |
| 1.0 | 77.13 | 63.4 | 83.3 | 84.7 | 88.5 | 89.62 |

- p = 0.5 tốt nhất tính chung. p cao hơn tăng conjunction và superlative nhưng giảm composition.

**AER trên nền QR-DDE** (`webqsp_qrdde_aer`, p = 0.5):
- WebQSP 89.52. CWQ transfer 71.53 (composition 60.4; conjunction 76.2; comparative 80.8; superlative 79.8).
- Thấp hơn AER-DDE: lan truyền có điều kiện theo câu hỏi vẫn overfit theo phân bố câu hỏi WebQSP. DDE là nền phù hợp hơn cho transfer.

**A1 tạm thời** (CWQ, 1,000 câu đầu, Llama-3.1-8B-Instruct **fp16**, vLLM trên 2× T4, prompt `sys_icl_dc` @50, `max_tokens` 4000; Hit theo eval gốc, Macro-F1 theo eval corrected với match gốc):

| | Hit | Hit@1 | Macro-F1 |
|---|---|---|---|
| REL-RAG P | 42.90 | 37.80 | 34.55 |
| DDE | 43.50 | 38.90 | 33.90 |

- DDE − REL-RAG: Hit (substring) +0.80 [−2.60, +4.20]; F1 −0.73 [−3.68, +2.27]. Không có ý nghĩa thống kê.
- Theo loại câu (Hit): composition 43.7 so với 36.4; conjunction 42.8 so với 43.0; comparative 54.7 so với 65.3; superlative 22.2 so với 37.0.
- Chênh retrieval +8.1 không chuyển thành Hit tổng, vì composition lời nhưng comparative/superlative lỗ. Hit tuyệt đối thấp (~43), nên reasoner 8B là nút thắt trên CWQ.
- Còn chờ: DDE neo lại, AER.
- **Phân tích utilization A1** (`a1_utilization.py`; 1,000 câu; đáp án trong top-50 theo so tên chuẩn hóa với `a_entity`):

| | Đáp án trong ngữ cảnh | Hit | P(Hit \| có trong ngữ cảnh) | P(Hit \| không có) |
|---|---|---|---|---|
| REL-RAG | 58.3% | 42.1 | 56.8 | 21.6 |
| DDE | 59.8% | 42.9 | 59.7 | 17.9 |

  - **Trên tập 1,000 câu này, chênh lệch đáp án trong ngữ cảnh chỉ còn +1.5.** Tập con có ít composition (34% so với 44%), và so tên theo `a_entity` chặt hơn ansR theo `a_entity_in_graph`. Hit gần bằng nhau là hợp lý.
  - Đáp án chỉ có trong ngữ cảnh của DDE: 134 câu, Hit 67.9% (với REL-RAG chỉ 24.6%). Chỉ có trong ngữ cảnh của REL-RAG: 119 câu, Hit 46.2% (DDE 5.0%).

| Loại | Trong ngữ cảnh: REL / DDE | Utilization: REL / DDE |
|---|---|---|
| composition | 18.5 / 30.5 | 85.7 / 83.7 |
| conjunction | 76.0 / 74.0 | 51.6 / 56.4 |
| comparative | 88.0 / 81.3 | 74.2 / 62.3 |
| superlative | 94.4 / 75.9 | 39.2 / 26.8 |

  - Ở composition, retrieval là nút thắt: hễ có đáp án trong ngữ cảnh thì LLM đúng khoảng 85%. AER (recall composition cao hơn hẳn) nên tạo khác biệt downstream ở đây.
  - Ở conjunction và superlative, utilization thấp (27–56%), nên reasoning là nút thắt.

## EXP-24 — Benchmark ngoài: GTSQA và MetaQA-3hop (zero-shot, retriever train trên WebQSP); hiệu chuẩn bằng file chính thức của SubgraphRAG
**Dữ liệu** (`retrieve/exp_qrdde/ext_data.py`):
- HF `Graphcore/GTSQA` (bản kèm graph, test 1,622 câu, Wikidata, có subgraph đáp án gold, `n_hops`, kiểu cấu trúc, `test_type`) và HF `mhivanov123/metaqa-3hop` (shard test 0, 4,710 câu).
- Embedding: GTE-large. Bảng embedding chung (GTSQA có 1.79M entity), index theo từng câu (`GlobalEmb`).
- Đánh giá DDE-family: `ext_eval.py`. Record cho REL-RAG: `/relrag/ext_prepare.py` (chưa chạy infer).

**GTSQA** (ansR@50 / recall subgraph gold @50 / phủ đủ subgraph gold @50):

| Lát cắt | DDE | DDE 2 lượt | AER 2 lượt | QR-DDE |
|---|---|---|---|---|
| 1 hop (521) | 44.1 / 36.4 / 31.5 | 38.4 / 30.3 / 26.1 | 37.0 / 29.0 / 25.0 | **50.6 / 43.5 / 39.2** |
| 2 hop (601) | 30.3 / 33.7 / 6.2 | 34.6 / 31.0 / 9.0 | 32.0 / 29.2 / 8.0 | **41.4 / 43.1 / 17.6** |
| 3 hop (450) | 33.9 / 31.8 / 1.3 | 33.4 / 29.3 / 2.4 | 33.8 / 30.3 / 3.3 | **42.2 / 40.3 / 5.1** |
| 4 hop (50) | 20.0 / 26.5 / 0.0 | 18.7 / 25.5 / 0.0 | 18.2 / 28.0 / 2.0 | **21.3 / 35.0 / 0.0** |
| seed = 3 (158) / seed = 4 (123) | ~18 / ~3 | ~15 / ~2 | ~20 / ~4 | ~18 / ~5 |
| unseen_graph_type (767) | 27.6 / 24.9 / 1.0 | 27.3 / 22.3 / 1.8 | 26.5 / 21.0 / 1.3 | **34.7 / 30.3 / 6.3** |

- Kết luận:
  - **Neo lại không giúp trên GTSQA** (giảm ở câu 1 hop).
  - **QR-DDE tốt nhất ở mọi lát cắt** khi chuyển sang KG khác.
  - **Câu ≥ 3 seed (truy vấn giao) sụp với mọi mô hình.**
  - Phạm vi claim của AER: tổng quát hóa thành phần dạng **chuỗi qua cầu nối** (CWQ composition), không phải dạng giao (GTSQA nhiều seed). Truy vấn giao là một khoảng trống riêng.

**MetaQA-3hop** (ansR@20 / @50 / @100):

| | ansR@20 / @50 / @100 |
|---|---|
| DDE | 12.2 / 21.0 / 33.5 |
| DDE 2 lượt | 12.4 / 21.4 / 34.3 |
| AER 2 lượt | 9.3 / 18.9 / 32.1 |
| QR-DDE | 14.9 / 21.7 / 30.8 |

- Mọi mô hình đều thấp. KG khác hẳn (phim, câu hỏi theo template); câu 3 hop có nhiều đáp án nên ansR@50 bị chặn trên.
- Chỉ 2,091/4,710 câu có path *xuôi* từ topic, nên REL-RAG (chỉ đi xuôi) bị giới hạn theo thiết kế.
- Muốn dùng MetaQA đúng cách thì phải train trên MetaQA (1 hop → 3 hop).

**Hiệu chuẩn downstream bằng file chính thức** (HF `siqim311/SubgraphRAG`, `results/KGQA/*/SubgraphRAG/Meta-Llama-3.1-8B-Instruct/`):
- Hit theo eval gốc, Macro-F1 theo eval corrected với **match gốc**:

| | Hit | Macro-F1 | Paper |
|---|---|---|---|
| WebQSP @100 | 86.61 | 70.57 | 86.6 / 70.5 ✅ |
| WebQSP @500 | 85.44 | 68.72 | – |
| CWQ @100 | 56.87 | 47.06 | 56.9 / 47.2 ✅ |
| CWQ @500 | 56.70 | 45.98 | – |

  → Cách chấm của mình trùng với cách chấm của paper.
- **Phát hiện:** file chính thức lưu **câu trả lời thô đầy đủ**, mà Hit gốc tìm chuỗi trong toàn bộ văn bản. Còn `reason/main.py` từ commit `435a998` chỉ lưu dòng `ans:`.
  - Trên 994 câu WebQSP, run vLLM fp16 của mình lưu kiểu `ans`: Hit 83.40 so với 89.44 của file chính thức; F1 gần bằng nhau (68.6 so với 69.3).
  - Đã thêm `main.py --pred_format raw` (file `*-raw-predictions.jsonl`). Mọi run dùng để so với paper phải chạy `raw`.
  - Các run A1 trước đó (lưu `ans`) chỉ so được nội bộ với nhau.

## EXP-25 — Downstream WebQSP so được với paper (Llama-3.1-8B-Instruct, dự đoán thô, cách chấm của paper)
- **Setup:**
  - Kaggle 2× T4, vLLM 0.30, fp16, TP 2. Mô hình `unsloth/Meta-Llama-3.1-8B-Instruct` (cùng trọng số với bản gốc).
  - Module reasoning của SubgraphRAG (`sys_icl_dc`), `main.py --pred_format raw --workers 48`.
  - Chấm: Hit theo eval gốc, Macro-F1 theo eval corrected với match gốc (EXP-24).
  - File: `reason/results/KGQA/webqsp/SubgraphRAG/vllm:llama31-8b/*-raw-predictions.jsonl`.
- **Kết quả:**

| Tập câu | Run | Hit | Macro-F1 |
|---|---|---|---|
| Hiệu chuẩn, 300 câu | SubgraphRAG @100, file chính thức (bf16) | 88.33 | 70.76 |
| Hiệu chuẩn, 300 câu | SubgraphRAG @100, pipeline của mình (fp16, vLLM, T4) | 89.33 | 70.60 |
| Toàn bộ 1,628 câu | SubgraphRAG @100 chính thức (paper 86.6 / 70.5) | 86.61 | 70.57 |
| Toàn bộ 1,628 câu | **REL-RAG P @50**, code-faithful (paper **88.8 / 77.3**) | **84.95** | **70.15** |
| 1,559 câu chung | SubgraphRAG @100 chính thức | 88.58 | 72.10 |
| 1,559 câu chung | REL-RAG P @50 | 86.59 | 71.41 |
| 1,559 câu chung | AER @50 | 86.40 | 69.83 |

  - Hit (bootstrap), AER trừ REL-RAG P: −0.19 [−1.99, +1.48]; AER trừ SubgraphRAG @100: −2.18 [−3.72, −0.71].
- **Kết luận:**
  - Pipeline đã hiệu chuẩn: lệch file chính thức ≤ 1 điểm.
  - **Số của REL-RAG P + Llama trong paper không tái hiện được:** 84.95 / 70.15 so với 88.8 / 77.3. Bản tái hiện còn thấp hơn SubgraphRAG @100.
  - Khác biệt còn lại: nhãn GPT-4o + DeepSeek, không phải GPT-4o cho toàn bộ. *(Đính chính 10-07: checkpoint của tác giả CÓ công bố, HF `TianJun1/RAPL`, link ở cuối `RAPL/README.md`; xem EXP-31.)*
  - AER ngang REL-RAG ở in-distribution, đúng kỳ vọng.
  - Bước tiếp theo: transfer CWQ ở dạng thô cho cả 4 retriever, vì đó là nơi AER có lợi thế.

## EXP-26 — A1 hoàn chỉnh: downstream transfer WebQSP → CWQ, đúng mô hình của paper (Llama-3.1-8B-Instruct bf16)
- **Setup:**
  - vast.ai RTX 4090 (instance 54258055, đã destroy; tổng ~$1.67), vLLM 0.30, **bf16**.
  - `main.py --pred_format raw --workers 64`, prompt `sys_icl_dc`, @50.
  - CWQ test, 1,000 câu đầu (cùng tập cho mọi retriever). Retriever đều train trên WebQSP. Chấm theo cách của paper (EXP-24).
  - File: `reason/results/KGQA/cwq/SubgraphRAG/vllm-bf16-4090/`.
- **Kết quả:**

| Retriever @50 | Hit | Macro-F1 | Hit: composition / conjunction / comparative / superlative |
|---|---|---|---|
| *SubgraphRAG @100, train trên CWQ (file chính thức, trần in-distribution)* | *58.20* | *47.47* | *55.1 / 63.6 / 52.0 / 33.3* |
| REL-RAG P | 45.10 | 34.07 | 37.5 / 49.2 / 58.7 / 33.3 |
| DDE | 48.50 | 35.69 | 47.5 / 50.6 / 60.0 / 18.5 |
| DDE + neo lại (không train) | 48.90 | **37.27** | **50.1** / 48.3 / 58.7 / 33.3 |
| **AER** | **49.60** | 36.90 | 47.8 / **51.5** / **60.0** / 27.8 |

- **Bootstrap ghép cặp (Hit):**

| So sánh | Tất cả | Composition |
|---|---|---|
| **AER − REL-RAG P** | **+4.50 [+1.00, +8.00]** | **+10.26 [+4.11, +15.84]** |
| DDE + neo lại − REL-RAG P | +3.80 [+0.10, +7.50] | +12.61 [+6.74, +18.48] |
| DDE − REL-RAG P | +3.40 [−0.10, +6.90] | +9.97 |
| AER − DDE | +1.10 [−1.90, +4.20] | +0.29 |

- **Kết luận:**
  - **C11 đạt ở mức REL-RAG:** với đúng mô hình reasoner của paper, AER hơn REL-RAG có ý nghĩa thống kê khi transfer (+4.5 Hit; composition +10.3).
  - Lợi thế downstream của AER so với DDE thường **chưa có ý nghĩa**, dù retrieval hơn +7.9 ansR@50 trên toàn CWQ. Lý do một phần là tập 1,000 câu có ít composition (34%), và LLM 8B là nút thắt (utilization ~57–60%, EXP-23).
  - Khoảng cách tới trần in-distribution (58.2 Hit) vẫn lớn: AER thu hẹp ~1/3 khoảng cách giữa REL-RAG và trần.

## EXP-27 — Cổng retrieval cho AER: nhiều budget, mọi metric (`03_EVALUATION_PROTOCOL.md` §0)
- **Lý do:** người dùng (10-05) yêu cầu phải lập luận được khả năng reasoning tăng trước khi chạy downstream, và không được kết luận chỉ dựa trên answer recall.
  - Bản quy tắc đầu (thắng mọi metric ở mọi budget) bị người dùng đánh giá là quá chặt.
  - Bản đã nới: in-distribution xấp xỉ bằng nhau, cộng với ít nhất một setting tổng quát hóa mà ansR vượt cả hai baseline ở @50/100/200. SP / GPT là metric hỗ trợ.
- **Setup:** `/relrag/gate_report.py`, chỉ eval CPU trên các file export đã có.
  - Paired bootstrap 2,000 lần.
  - REL-RAG @200 lấy prefix của run @500; đã kiểm tra prefix trùng hoàn toàn với các run @100 riêng (300/300 câu).
  - Trên CWQ transfer, SubgraphRAG = bản tái hiện MLP + DDE train trên WebQSP.
- **Kết quả** (`docs/REPORT.md` §3):
  - WebQSP: ansR hòa với cả hai baseline ở mọi budget (PASS).
  - CWQ transfer: ansR thắng cả hai ở @50/100/200 (PASS).
  - Metric hỗ trợ: so với SubgraphRAG, SP-R +2.2 đến +3.5 và GPT-R +3.3 đến +3.9, nhỏ hơn nhiều so với ansR +4 đến +8.
  - Hiệu quả budget so với SubgraphRAG không đạt.
- **Kết luận:**
  - AER qua cổng.
  - Downstream so với REL-RAG có cơ sở (EXP-26 +4.50 Hit). So với SubgraphRAG thì không đáng kỳ vọng, và EXP-25 (WebQSP, in-distribution ngang nhau) lẽ ra không cần chạy.
  - Cải tiến tiếp theo nên nhắm vào recall bằng chứng.

## Dọn dẹp 2026-10-05 (người dùng duyệt)
Giải phóng khoảng 10 GB. Các file kết quả retrieval, `best.pt`, nhãn và embedding chính đều được giữ nguyên.
- **Đã xóa:**
  - Mọi `epoch_*.pt` trong `/relrag/runs/*` (giữ `best.pt`). Các mục artifact cũ có nhắc tới `epoch_*.pt` không còn file tương ứng.
  - Hai run thử không được tham chiếu: `runs/path_sp42`, `runs/path_minlen42`.
  - `retrieve/exp_qrdde/data/`: subset của EXP-11, tạo lại được bằng `prepare_subsets.py`.
  - Mọi `__pycache__`, `reason/wandb`, `retrieve/exp_qrdde/wandb`, `runs/a1_cwq/bench40*`.
  - `RAPL/run_rapl_sp_eval.py`, `RAPL/eval_rapl.py` (bản lỗi của Phase 16).
- **Docs:** 01, 06, 07, 08 và `prompts/` được chuyển vào `docs/archive/relrag_phase/`.
- **Giữ nguyên:** log, dữ liệu GTSQA / MetaQA (`data_ext/`).

## EXP-28 — Phân tích lỗi (bằng chứng bị bỏ sót) và các sửa đổi không cần train: ADR, CVT có định danh relation (10-05)
Script: `/relrag/failure_analysis.py` (output ở `/relrag/runs/analysis/`), `retrieve/exp_qrdde/adr_probe.py`, `ext_eval.py` (mode `adr*`), scratchpad `multitopic.py`, `connector_stats.py`, `cvt_confusion.py`.

**1. Bằng chứng bị bỏ sót nằm ở đâu** (gold GPT, CWQ transfer, recall@50: AER / SubgraphRAG train trên CWQ, tức trần):
- Câu nhiều topic (53% câu CWQ; chỉ 1.8% câu WebQSP train): gold recall 45–51, so với 66–77 ở trần. Câu một topic: 65.8 so với 78.1.
- Triple "gần mọi anchor" chỉ chiếm 8.2% số triple nhưng chứa 37.2% gold (giàu gấp 5.7 lần).
- Hop thứ hai ra khỏi CVT ("d1 out cvt", 19% gold CWQ, 17% gold WebQSP) là nhóm yếu nhất. Ngay trên WebQSP in-distribution, SubgraphRAG chỉ đạt 62.8, kém REL-RAG (74.1).
  - Mỗi triple loại này bị sót bị đẩy ra bởi khoảng 10 (WebQSP) / 16.6 (CWQ) triple hop-2 của các CVT khác.
  - 86–92% số CVT đó nối với topic qua relation **khác**. Nguyên nhân: mọi CVT dùng chung một embedding.
- 20% triple gold có CVT bị trùng điểm tuyệt đối với triple khác.

**2. ADR (anchor-decomposed retrieval), không cần train.** Chấm riêng từng topic như một câu một topic, rồi gộp bằng max z.
- Trên CWQ transfer @50, theo thứ tự ansR / SP-R / GPT-R:
  - DDE: 70.05 / 49.77 / 53.59.
  - DDE + ADR + AER (không train): 78.58 / 54.04 / 59.20.
  - AER đã train + ADR: 79.02 / 53.75 / 59.62, so với AER 77.95 / 52.01 / 56.86.
- WebQSP gần như không đổi (98% câu một topic).
- Thêm lượt giao mềm (mean/min z theo anchor, kiểu CQD): chỉ +0.1 đến +0.4.
- GTSQA (zero-shot, ansR@50 / goldR@50):
  - QR-DDE: 44.0 / 42.2. QR-DDE + ADR: **51.2 / 48.9**.
  - Câu 3 seed: 18.2 → 47.3. Câu 4 seed: 4.9 → 25.3.
  - AER làm hại trên GTSQA.
- Chiều CWQ → WebQSP (checkpoint train CWQ): neo lại làm hại (−1.5 ansR), ADR không đổi.

**3. Đang chạy:** `sweep_queue.sh`.
- `dde+cvtrel`: CVT được biểu diễn bằng trung bình embedding văn bản của các relation chạm vào nó, cộng vector học chung. Có thêm bản kết hợp AER.
- Seed 0/1 cho DDE và AER.
- `dde_rounds` 1/3, MLP hidden 256, MLP 3 lớp.

*Ghi chú:* thư mục `retrieve/exp_qrdde/logs/` cũ biến mất lúc 18:59 ngày 10-05 (không nằm trong danh sách xóa đã duyệt). Log mới được tạo lại từ 19:24.

**Bổ sung EXP-28 (10-05 tối).**
- QR-DDE + ADR + AER (không train), CWQ transfer @50: ansR 75.84 / SP-R 58.33 / GPT-R 63.92 / GPTp-R 66.78.
  - So với SubgraphRAG train WebQSP: 70.05 / 49.77 / 53.59 / 59.07.
  - Đây là cấu hình tốt nhất về bằng chứng. ansR thấp hơn DDE + ADR + AER (78.58).
  - WebQSP: QR-DDE + ADR đạt 89.39 / 84.71 / 82.39, so với SubgraphRAG 90.38 / 82.26 / 81.03.
- **RT-combined (GTE) và weighted loss (effective-num):** checkpoint và scored triples không còn trên đĩa, chỉ còn config.
  - Bản GTE không có selector được train: điểm relation = 0.5 × cos(q, r) GTE + 0.5 × lexical, mask top-5, weighted DDE (sàn 0.1).
  - Đang train lại bằng `retrieve/train.py` (`queue_v3.sh`), inference trên WebQSP và CWQ (config `configs/retriever/transfer/*_to_cwq.yaml`), sau đó chạy `gate_report.py` và ghép với ADR/AER.

## EXP-29 — RT-combined (GTE) train lại, ghép ADR/AER; các run seed; `cvtrel`; reranker ngữ cảnh (10-05/06)
- **Hạ tầng:** vast.ai RTX 4090 (instance 54322473), chạy song song 7 run.
  - GPU là nút thắt: 7 run song song chỉ cho thông lượng hơn 4 run khoảng 10%.
  - Tổng chi phí $1.24. Khoảng $0.7 trong đó là thời gian máy ngồi không, do watchdog không nhận ra dòng "ALL DONE" và chỉ hủy instance khi chạm mốc 4 giờ.
- **RT-combined** (`configs/retriever/webqsp/webqsp_rt_combined.yaml`, train lại bằng `retrieve/train.py`; checkpoint `retrieve/webqsp_rt_combined_Oct05-13:52:20/`). ADR/AER chạy bằng `retrieve/sg_adr_infer.py`.
  - *Lỗi phát hiện:* `inference.py` gốc bỏ qua `--config_file`, nên kết quả "CWQ" thực chất là WebQSP. Đã dùng `sg_adr_infer.py` thay thế.
  - CWQ transfer (ansR / SP-R / GPT-R / GPTp-R @50):
    - RT base: 72.41 / 51.11 / 54.71 / 61.41.
    - **RT + ADR + AER: 82.37 / 57.11 / 63.47 / 69.76**; @100 ansR 89.20; @200 ansR 93.89.
  - Gate (`/relrag/runs/gate/gate_*_rt_adr_aer.md`):
    - WebQSP: PASS, không kém baseline nào; ansR @100 / @200 hơn SubgraphRAG chính thức +1.04 / +0.92 (có ý nghĩa).
    - CWQ: PASS so với cả hai baseline ở mọi budget, trên **mọi** metric.
      - So với SubgraphRAG train WebQSP @50: ansR +12.32, SP-R +7.34, GPT-R +9.88.
      - So với REL-RAG: ansR +20.46.
  - **Hiệu quả budget CWQ, RT+ADR+AER@50 trừ SubgraphRAG@100:** ansR +2.78 [+1.33, +4.25], GPTp-R +2.23, GPT-R +0.42 (hòa), SP-R −1.53. Đây là cấu hình đầu tiên đạt tiêu chí này.
- **Dao động theo seed** (test WebQSP ansR@50, id-match):
  - DDE: 90.07 / 86.72 / 89.90 / 90.36 (seed 42 / 0 / 1 / 2). Seed 0 dừng sớm ở epoch 31.
  - AER: 89.68 / 88.59 (seed 42 / 0).
  - QR-DDE: 89.20 / 90.17 / 89.79 (seed 42 / 0 / 1).
  - Hệ quả: chênh lệch dưới khoảng 2 điểm giữa các cấu hình chỉ chạy một seed không kết luận được.
- **`cvtrel`** (CVT = trung bình embedding của các relation chạm vào nó): **không giúp**. CWQ base 67.31, so với DDE 70.05; WebQSP 89.77. Giả thuyết CVT ở dạng đặc trưng đơn giản này bị bác bỏ.
- **Reranker ngữ cảnh** (`rerank.py`): lần train đầu **vô hiệu**. Eval mode của `nn.TransformerEncoderLayer` (torch 2.1) trả về NaN, nên thứ tự giữ nguyên như tầng 1 và checkpoint bị kẹt ở epoch 0. Đã thay bằng lớp attention tự viết; đang train lại.
- **GrailQA:** chuyển đổi được 1,855/6,763 câu dev (4,664 câu không có topic trong file subgraph). Lần embedding đầu bị OOM do chạy chồng GPU; đang chạy lại với batch 128.

**Ghi chú 10-06 11:15 (sắp xếp lại tài nguyên theo yêu cầu người dùng: chạy tuần tự, ưu tiên việc cần nhất cho nghiên cứu).**
- **svuet:** quá chậm để dùng cho việc ra quyết định. Kể cả khi chỉ chạy 3 job, mỗi job 1 thread, tốc độ cũng chỉ 1.1–1.7 it/s, so với khoảng 30 it/s ở local, vì CPU 6 core bị gunicorn chiếm.
  - Lần đầu còn tệ hơn: 7 job × 16 thread, load 16, mất khoảng 4 giờ.
  - Hiện chỉ giữ WL seed 0 chạy nền. Các run V1 đã được chuyển về local.
- **RT-combined train trên CWQ (in-distribution):** dừng ở epoch 27, checkpoint tốt nhất khoảng epoch 25, **trước khi hết patience 10**.
  - Đây là lệch protocol, chấp nhận để nhường GPU cho V1.
  - Số liệu CWQ in-distribution của run này phải ghi chú như vậy.
- **Hàng đợi tuần tự duy nhất** (`retrieve/exp_qrdde/queue_main.sh`):
  1. ADR/AER cho RT-combined CWQ (in-distribution và CWQ → WebQSP).
  2. V1 seed 42.
  3. V1 seed 0.
  4. Reranker.
  - V2 chỉ chạy nếu V1 cho thấy anchor-set encoding có lợi.

### EXP-29 (tiếp) — RT-combined train trên CWQ (dừng sớm ở epoch 27), 10-06

Chấm bằng `/relrag/gate_report.py --setting cwq | cwq_to_webqsp` (2 setting mới). Kết quả ở `/relrag/runs/gate/gate_cwq*_rt_*.md`. Trong cả hai setting đều không có REL-RAG train trên CWQ.

| Setting (@50) | Retriever | ansR | SP-R | GPT-R |
|---|---|---|---|---|
| CWQ in-dist | SubgraphRAG official | 86.20 | 66.28 | 75.54 |
| | DDE repro (CWQ) | 86.67 | 66.79 | 72.89 |
| | RT (CWQ) | 86.21 | 64.02 | 72.30 |
| | RT + ADR + AER | 87.04 | 64.42 | 72.72 |
| CWQ → WebQSP | SubgraphRAG repro (train CWQ) | 82.22 | 71.66 | 71.84 |
| | RT (CWQ) | 84.97 | 72.79 | 72.78 |
| | RT + ADR + AER | 83.65 | 70.40 | 71.19 |

- **CWQ in-dist.** Gate ansR PASS: không thua ở budget nào, ADR+AER thắng ở @100. Tuy vậy, SP-R/GPT-R thấp hơn bản official khoảng 2–3 điểm, nên RT không mang lại gì khi train trên CWQ.
- **CWQ → WebQSP.** RT base thắng repro ở cả 3 budget (+2.75 @50). Khi thêm ADR+AER thì kết quả lại kém base: WebQSP gần như toàn câu 1 topic, nên các pass bổ sung chỉ chèn nhiễu vào top-k.
- **svuet.** WL seed 0 chết lúc 11:15 (khoảng epoch 4), bị kill cùng runner. Đã chuyển sang chạy local qua `exp_qrdde/queue_after.sh`, chạy sau queue_main.

## EXP-30 — A2: downstream toàn bộ CWQ test cho WL+ADR+AER (train WebQSP), 10-06
- **Setup:** giống EXP-26: Llama-3.1-8B-Instruct bf16, vLLM, RTX 4090 vast.ai, `sys_icl_dc`, @50, raw. Chạy toàn bộ 3,531 câu; 1,000 câu đầu theo đúng thứ tự tập A1.
  - Instance 54455052, 15:25–17:25.
  - Tốn $0.96 credit (2.00 → 1.03), gồm khoảng $0.1 cho 3 máy hỏng: lỗi quyền `authorized_keys` phía host; kẹt tải image 2 giờ; launcher hủy nhầm.
  - File: `reason/results/KGQA/cwq/SubgraphRAG/vllm-bf16-4090-full/`. Script chấm: scratchpad `a2_score.py`.
- **1,000 câu đầu (cùng tập với EXP-26):**

| Retriever @50 | Hit | Macro-F1 |
|---|---|---|
| REL-RAG P | 45.10 | 34.07 |
| DDE (SubgraphRAG repro) | 48.50 | 35.69 |
| AER | 49.60 | 36.90 |
| **WL+ADR+AER** | **51.80** | **40.94** |
| *SubgraphRAG official @100, train CWQ (trần)* | *58.20* | *47.47* |

- **Hit, WL+ADR+AER − X (bootstrap):**
  - vs REL-RAG P: +6.70 [+3.20, +10.30]; riêng composition +13.49.
  - vs DDE: +3.30 [+0.10, +6.60].
  - vs AER: +2.20 [−0.90, +5.40].
  - vs trần: −6.40.
- **Toàn bộ 3,531 câu:**

| | Hit | Macro-F1 |
|---|---|---|
| WL+ADR+AER @50 (train WebQSP) | 54.18 | 42.84 |
| SubgraphRAG official @100 (train CWQ), chấm bằng script của mình | 56.87 | 47.06 |
| *SubgraphRAG + Llama3.1-8B, số công bố (REL-RAG Table 1)* | *56.9* | *47.2* |
| *REL-RAG paper, WebQSP→CWQ, Llama3.1-8B, 50 triple (Table 12)* | *64.1* | *53.6* |
| *REL-RAG paper, CWQ in-dist, Llama3.1-8B, 50 triple (Table 1/12)* | *67.2* | *56.8* |

  - So với trần: Hit −2.69 [−4.33, −1.16].
- **Nhận xét:**
  - Script chấm của mình khớp số công bố của SubgraphRAG (56.87/47.06 so với 56.9/47.2), nên pipeline đánh giá đã được hiệu chuẩn.
  - Số REL-RAG công bố cho transfer (64.1 Hit) **cao hơn cả SubgraphRAG train in-distribution trên CWQ** (56.9). Cùng pipeline với chúng ta, bản REL-RAG tái hiện chỉ đạt 45.1 Hit (1,000 câu) và ansR@50 = 61.9 (so với 86.2 của SubgraphRAG CWQ).
  - Khác biệt đã biết: REL-RAG dùng prompt riêng (input dạng path, "think step by step", beam 5), còn mình dùng `sys_icl_dc` của SubgraphRAG (xem 02_BASELINE §2, M0). Vì vậy không so trực tiếp số tuyệt đối với bảng của REL-RAG. So sánh hợp lệ là cùng pipeline: WL+ADR+AER hơn REL-RAG tái hiện +6.7 Hit.

## EXP-31 — Kiểm chứng số liệu REL-RAG: checkpoint chính thức của tác giả, rò rỉ nhãn test, giới hạn trên của Hit (10-06)
Mục đích: tìm xem paper làm gì mà code chưa phản ánh, để giải thích con số 64.1 Hit (WebQSP→CWQ) và 67.2 Hit (CWQ in-dist) với Llama-8B @50 (Table 1, 12).

**Setup:**
- Checkpoint tác giả công bố (HF `TianJun1/RAPL`: `webqsp.pt`, `cwq.pt`), map sang `RelRagRetriever` bằng `/relrag/authors_ckpt.py`. Load khớp mọi key.
- Thêm `infer.py --leak_topic sp|sp_minlen`: giống `Trainerv3.decoding(way='normal')`, gộp thêm triple đầu của các shortest path q→đáp án trên test vào tập điểm xuất phát.
- Queue `run_exp31.sh`, log `logs/exp31.log`. Phép chiếu Hit: scratchpad `exp31_project.py`.

**1. Subgraph giống hệt nhau.** `test_id_entity_mapping.pkl` của tác giả có tập entity trùng 100% với subgraph SubgraphRAG/RoG: 3,531/3,531 câu CWQ, 1,639/1,639 câu WebQSP. Tỷ lệ câu có đáp án trong subgraph cũng như nhau: CWQ 80.7%, WebQSP 95.1%. Subgraph ODA (Sun et al. 2024b, `lanjiuqing64/KGdata`) chỉ dùng cho GrailQA.

**2. Retrieval của checkpoint tác giả** (ansR / ansHit tính trên các câu có đáp án trong graph; #triple):

| Checkpoint → test | Decode | ansR | ansHit | #triple |
|---|---|---|---|---|
| webqsp.pt → WebQSP | path_bf@50 | 89.91 | 93.65 | 49.9 |
| | tác giả (K=800, M=1200, không budget) | 91.11 | 95.19 | 42.9 |
| | tác giả + rò rỉ nhãn test | 95.91 | 97.82 | 47.7 |
| webqsp.pt → CWQ (OOD) | path_bf@50 | 66.83 | 68.96 | 49.4 |
| | tác giả | 66.88 | 68.36 | 66.8 |
| | tác giả + rò rỉ | 72.95 | 74.26 | 70.1 |
| cwq.pt → CWQ (ID) | path_bf@50 | 80.75 | 83.81 | 49.3 |
| | tác giả | 72.16 | 74.96 | 70.0 |
| | tác giả + rò rỉ | 76.18 | 78.76 | 72.7 |
| *Tham chiếu: SubgraphRAG official, CWQ* | @50 | *86.20* | – | 50 |

- Bản tái hiện của mình khớp checkpoint tác giả trên WebQSP: 90.60 so với 89.91. Khi transfer thì bản của mình thấp hơn khoảng 5 điểm (61.9 so với 66.8).
- Decode của tác giả không giữ budget 50 triple: trung bình 67–73 triple/câu trên CWQ (P4).
- Rò rỉ nhãn test (P1) làm ansR tăng +4.8 (WebQSP), +6.1 (OOD), +4.0 (ID).

**3. Giới hạn trên của Hit** (Llama-3.1-8B bf16, prompt `sys_icl_dc`, đo từ A1/A2):
- P(Hit | đáp án nằm trong retrieval) ≈ 70.5%; P(Hit | không nằm) ≈ 17%.
- Hiệu chuẩn: phép chiếu cho SubgraphRAG official 58.0 (thực đo 58.2) và REL-RAG repro 48.2 (thực đo 45.1).
- Hit chiếu từ retrieval của chính tác giả:
  - OOD: 46.5–49.1 (lạc quan 49.9–52.2), so với paper 64.1.
  - ID: 49.4–53.2 (lạc quan 52.4–55.8), so với paper 67.2.
- **Retrieval hoàn hảo** (ansHit = 80.7%) cũng chỉ cho khoảng 60 Hit. Để đạt 64.1 cần khoảng 88% câu có đáp án trong retrieval, vượt trần 80.7%.

**4. Đọc lại các bảng của paper:**
- Khoảng cách ID−OOD trong paper nhỏ (67.2 so với 64.1). Điều này *nhất quán* với checkpoint của tác giả khi dùng decode của họ: ansR ID 72.2 so với OOD 66.9, vì decode tự do làm yếu model CWQ (path_bf@50 cho 80.8).
- Cái *không nhất quán* là **mức tuyệt đối**: paper cao hơn mức chiếu khoảng 13–17 Hit.
- Số của SubgraphRAG trong Table 1 lấy nguyên từ paper gốc (chấm lại file chính thức ra 56.87/47.06), dù paper nói dùng "the same reasoning modules" (P8).
- Tức là REL-RAG (prompt riêng, beam 5, cách chấm không công bố) được so với SubgraphRAG (prompt `sys_icl_dc`, cách chấm gốc).

**Kết luận:**
- Không thể đạt 64.1 / 67.2 chỉ bằng retrieval: không đạt với checkpoint của tác giả, kể cả khi có rò rỉ, và không đạt kể cả với retrieval hoàn hảo dưới reasoner và cách chấm của SubgraphRAG.
- Nguyên nhân còn lại nằm ở **reasoner và cách chấm** (P2, P5): prompt riêng, beam 5, và Hit có thể chấm trên toàn bộ text thô. Ở EXP-17 (WebQSP), cùng prompt tác giả, Hit trên text thô là 91.95, trên đáp án parse là 68.03.
- Tác giả không công bố script chấm, nên chưa xác minh được. Đây là bằng chứng *không tái hiện được và không nhất quán*, chưa phải bằng chứng về ý định.
- Bước có thể làm tiếp (tốn tiền, cần duyệt): chạy prompt và beam 5 của tác giả trên CWQ cho checkpoint của họ, rồi chấm theo hai cách (text thô và đáp án parse).

## EXP-32 — GrailQA theo setting của paper REL-RAG: retrieval theo hop / level / domain (10-06)
- **Dữ liệu.** Tập 1,000 câu của ODA (`lanjiuqing64/KGdata` `QAdata/graliqa.json`) với subgraph 2-hop Freebase của ODA (HF `CCaccept/Subgraph`); đây là nguồn paper REL-RAG dùng (§5.1, App. D).
  - Giữ 897 câu (paper giữ 874): bỏ 12 câu không có topic, 42 câu không có đáp án trong graph, 49 câu quá 30k triple.
  - Phân bố: 645 câu 1-hop, 219 câu 2-hop, 33 câu 3-hop; 513 zero-shot / 216 i.i.d. / 168 compositional; 857 câu 1 topic.
  - Code: `ext_data.py convert_grailqa_oda`, `make_sg_ext.py` (format pipeline SubgraphRAG, fp16), `/relrag/ext_prepare.py`.
- **Retriever** (đều train trên WebQSP, không train thêm):
  - SubgraphRAG repro: DDE, `runs/webqsp_dde`.
  - REL-RAG repro: code-faithful, `path_bf`.
  - WL: RT-combined + effective-number loss, `webqsp_wl_effective_num_Oct05-15:23:05`, chạy qua `sg_adr_infer.py` với base / ADR / AER / ADR+AER.
- **Chấm.** `/relrag/grail_report.py`, text-match đáp án với head/tail của top-k, như nhau cho mọi phương pháp. Bảng đầy đủ: `/relrag/runs/grail_report_oda_full.md`.

| ansR @50 / @100 / @200 | n | SubgraphRAG repro | REL-RAG repro | WL | WL+ADR+AER |
|---|---|---|---|---|---|
| Tất cả | 897 | 71.7 / 79.1 / 86.9 | 70.0 / 76.5 / 79.8 | 83.2 / 87.0 / 91.8 | **83.4 / 89.2 / 93.0** |
| 1-hop | 645 | 83.5 / 89.1 / 94.2 | 81.5 / 87.0 / 89.6 | 94.0 / 96.4 / 98.1 | 93.7 / 96.3 / 98.0 |
| 2-hop | 219 | 43.2 / 57.1 / 73.4 | 42.7 / 52.4 / 58.6 | 59.6 / 67.8 / 79.4 | **60.3 / 76.0 / 84.9** |
| 3-hop | 33 | 30.5 / 30.5 / 33.7 | 27.4 / 30.5 / 30.5 | 27.6 / 31.1 / 50.7 | 36.8 / 37.4 / 47.8 |
| i.i.d. | 216 | 72.9 / 78.9 / 85.0 | 74.8 / 80.7 / 84.7 | 82.3 / 86.7 / 91.0 | 82.4 / 88.6 / 92.2 |
| compositional | 168 | 69.9 / 76.4 / 80.2 | 69.8 / 73.8 / 78.3 | 78.6 / 81.4 / 88.5 | 80.0 / 84.8 / 89.7 |
| zero-shot | 513 | 71.8 / 80.2 / 89.9 | 68.0 / 75.6 / 78.3 | 85.0 / 88.9 / 93.2 | 85.0 / 90.8 / 94.4 |
| 2 topic | 40 | 73.3 / 77.5 / 85.0 | 60.0 / 65.0 / 73.8 | 81.2 / 83.8 / 87.5 | 86.2 / 90.0 / 90.0 |

**Nhận xét:**
- **Nguồn chính của lợi thế là backbone WL**, tức điều kiện hóa theo relation-topic (RT-combined) cộng với weighted loss: hơn SubgraphRAG repro +11.5 và REL-RAG repro +13.2 ansR@50.
  - Phần lớn đến từ 1-hop (+10.5) và zero-shot (+13.2).
  - Tức là khớp câu hỏi với relation tổng quát hóa được sang domain mới. Còn DDE và line graph REL-RAG thì không.
- **ADR/AER thêm vào WL:** gần như không đổi ở @50 (+0.2), nhưng +2.2 @100 và +1.2 @200.
  - Lợi nằm ở 2-hop (+8.2 @100, +5.5 @200), 3-hop @50 (+9.2, n = 33) và câu 2 topic (+5.0 @50, n = 40).
  - Đúng với nơi bridge / đa-anchor có tác dụng.
- **REL-RAG repro không hơn SubgraphRAG repro ở cả 3 budget.** Riêng @200, decode theo path cạn đường đi (trung bình chỉ 132 triple). Điều này trái với kết luận ở Table 2 / Figure 4 của paper, nhưng paper đo downstream với GPT-4o-mini và 500 triple, còn đây là đo retrieval.
- **Lưu ý về baseline:** SubgraphRAG ở đây là bản repro DDE (khớp official trên WebQSP, −0.31 R@50). Checkpoint official chưa chạy trên GrailQA.

### EXP-31b — Checkpoint của tác giả so với bản repro: ansR / SP-R / GPT-R ở @50/100/200 (10-07)
`infer.py --mode path_bf --budgets 50,100,500` với `runs/authors_ckpt/webqsp_model`. Mọi câu đều prefix-consistent (@50 = phần đầu @500), nên @200 lấy prefix như baseline. Chấm bằng `gate_report.py`; kết quả ở `/relrag/runs/gate/gate_{webqsp,cwq_transfer}_authors_ckpt.md`.

| Setting | k | Checkpoint tác giả (ansR / SP-R / GPT-R) | Repro code-faithful | Tác giả − repro (bootstrap 95%) |
|---|---|---|---|---|
| WebQSP | 50 | 89.91 / 83.71 / 64.02 | 90.60 / 85.77 / 61.92 | ansR −0.70 ~; SP-R −2.07 ❌; GPT-R +2.10 ✅ |
| WebQSP | 100 | 95.16 / 89.26 / 67.93 | 94.84 / 90.77 / 65.45 | +0.31 ~; −1.51 ❌; +2.48 ✅ |
| WebQSP | 200 | 97.51 / 92.82 / 71.74 | 96.95 / 93.92 / 69.91 | +0.56 ~; −1.10 ❌; +1.83 ✅ |
| WebQSP→CWQ | 50 | 66.82 / 54.40 / 39.39 | 61.90 / 50.82 / 34.35 | **+4.92 ✅**; +3.58 ✅; +5.04 ✅ |
| WebQSP→CWQ | 100 | 74.09 / 60.71 / 44.57 | 72.26 / 58.89 / 40.80 | +1.82 ✅; +1.82 ✅; +3.77 ✅ |
| WebQSP→CWQ | 200 | 80.38 / 65.69 / 49.17 | 79.33 / 65.71 / 46.14 | +1.05 ~; −0.02 ~; +3.03 ✅ |

**Đọc kết quả:**
- **In-distribution (WebQSP): ansR tương đương**, chênh không có ý nghĩa thống kê.
  - Repro cao hơn về SP-R khoảng 1–2 điểm; checkpoint tác giả cao hơn về GPT-R khoảng 2 điểm.
  - Khớp với khác biệt về nhãn: tác giả dùng GPT-4o cho toàn bộ, repro dùng GPT-4o + DeepSeek.
- **Transfer: checkpoint tác giả tốt hơn có ý nghĩa** (+4.9 ansR@50), khoảng cách giảm dần ở budget lớn. Bản repro không "y hệt" ở transfer.
- Dù vậy, checkpoint tác giả vẫn thấp hơn SubgraphRAG repro trên CWQ:
  - ansR@50: 66.8 so với 70.1;
  - GPT-R@50: 39.4 so với 53.6.
- Chưa có run downstream nào dùng checkpoint tác giả. Mọi downstream REL-RAG trước đây (EXP-06/07/17/25/26) đều dùng bản repro.

## EXP-33 — ASE (Anchor-Set Encoding) train trên toàn bộ dữ liệu: WebQSP và WebQSP → CWQ (10-07)
Phương pháp: xem `09_NEW_DIRECTION_ANCHOR_EQUIVARIANCE.md` §8.
- **Train.** `qrdde.py --feat dde+ase --seed 42 --full --skip_test` (patience 10).
  - Chạy trên vast #54539039: A4000, 126 GB RAM, $0.09/giờ, tổng chi khoảng $1.0.
  - `webqsp_ase_s42`: best epoch 40, khoảng 1 giờ.
  - `cwq_ase_full_s42`: best epoch 15, khoảng 5 giờ. Chưa chấm.
- **Chấm.** `adr_probe.py` (mode base, adr_aer) rồi `gate_report.py --setting webqsp | cwq_transfer`, bootstrap 1,000.
  - Trên 1,559 câu WebQSP và 2,848 câu CWQ có đáp án trong graph; cùng quy ước với `retrieve/eval.py` gốc.
  - `*` = khác biệt có ý nghĩa (khoảng tin cậy 95% không chứa 0).

**WebQSP (in-distribution), ASE base:**

| k | ansR: ASE / SubgraphRAG / REL-RAG | SP-R: ASE / SG / REL | GPT-R: ASE / SG / REL |
|---|---|---|---|
| 50 | 90.10 / 90.38 / 90.60 | 82.99 / 82.26 / 85.77 | 80.56 / 81.03 / 61.92 |
| 100 | 94.92 / 94.42 / 94.84 | 89.39 / 87.76 / 90.77 | 86.92 / 87.42 / 65.45 |
| 200 | 97.41 / 96.94 / 96.95 | 93.62 / 91.98 / 93.92 | 91.80 / 93.12 / 69.91 |

- **ansR: không thua baseline ở budget nào**, nên gate in-distribution PASS.
- SP-R hơn SubgraphRAG (+1.6* ở @100 và @200), thua REL-RAG ở @50/@100 (−2.8*, −1.4*).
- GPT-R thua SubgraphRAG nhẹ ở @200 (−1.3*), hơn REL-RAG khoảng +19–22*.

**WebQSP → CWQ (transfer):**

| k | ansR: ASE / ASE+ADR+AER / SubgraphRAG / REL-RAG / WL+ADR+AER | SP-R: ASE+ADR+AER / SG | GPT-R: ASE+ADR+AER / SG |
|---|---|---|---|
| 50 | 71.49 / **80.71** / 70.05 / 61.90 / 83.18 | 56.66 / 49.77 | 61.32 / 53.59 |
| 100 | 80.72 / **87.75** / 79.58 / 72.26 / 89.82 | 66.25 / 58.65 | 70.46 / 63.05 |
| 200 | 88.61 / **94.28** / 87.70 / 79.33 / 94.66 | 76.10 / 67.84 | 79.31 / 72.04 |
| 500 | 95.66 / 97.55 / 94.81 / 85.21 / – | 86.80 / 80.51 | 88.98 / 84.53 |

- ASE+ADR+AER so với SubgraphRAG: ansR +10.7* / +8.2* / +6.6* ở @50/100/200. So với REL-RAG: +18.8* / +15.5* / +15.0*.
  - SP-R, GPT-R và GPTp-R đều hơn có ý nghĩa ở mọi budget.
  - **Gate transfer PASS.**
- **ASE base một mình chỉ hơn SubgraphRAG +1.4 ansR@50**; phần lớn mức tăng đến từ ADR+AER lúc inference.
  - So với DDE+ADR+AER (78.6 @50, EXP-20), ASE+ADR+AER cao hơn khoảng +2.1.
- **ASE+ADR+AER vẫn thấp hơn WL+ADR+AER** ở mọi budget và ở cả hai setting:
  - transfer @50: −2.5 ansR, −4.3 GPT-R;
  - WebQSP @50: −1.8 ansR (90.10 so với 91.92).
  - Điều kiện hóa theo relation-topic (WL) mạnh hơn mã hóa tập anchor (ASE) khi đứng riêng.
- **Bước tiếp theo:**
  - Chấm `cwq_ase_full_s42` (CWQ in-distribution, so với QR-DDE full 89.69 @50).
  - Thử ghép ASE vào backbone WL (ASE + relation-topic): đây là phép thử còn thiếu để biết ASE có cộng thêm vào WL hay không.
- Biểu đồ: `docs/figures/ase_*.png` (sinh bằng `docs/figures/make_ase_figs.py`).

**EXP-33b — `cwq_ase_full_s42` (train CWQ full), chấm 10-07** (`/relrag/runs/gate/{cwq,cwq_to_webqsp}_ase_full_*.md`):
- **CWQ in-distribution:** ASE base ansR **90.34 / 94.18 / 96.96**, SP-R 75.07, GPT-R 79.11 ở @50.
  - So với official: ansR +4.1* / +2.8* / +1.6*; SP-R +8.8* / +6.4* / +4.4*; GPT-R +3.6* ở @50 nhưng −2.5* ở @200.
  - Ngang QR-DDE full (89.69 / 75.71 / 79.60) và cao hơn RT-combined CWQ (86.21).
  - Thêm ADR làm giảm còn 89.30, ADR+AER còn 88.33.
- **CWQ → WebQSP:** ASE base 81.83 / 88.63 / 93.45, ngang SubgraphRAG repro (82.22 / 88.15 / 92.92; không có chênh lệch nào có ý nghĩa). Gate transfer không đạt.
  - RT-combined đạt 84.97 @50, tốt hơn ASE.
  - ADR+AER gây hại (77.86, −4.4*).
- **Kết luận:** ASE mạnh khi train trên dữ liệu nhiều anchor, nhưng không tổng quát hoá tốt hơn WL/RT. Bước tiếp theo là thử WL+ASE. Đã ghi vào doc báo cáo, mục 12.

## EXP-34 — Chọn backbone cho ASE bằng chẩn đoán, không train (10-07)
`/relrag/diag_base_choice.py` và `diag_anchor_aug.py`; kết quả ở `/relrag/runs/diag_*.md`. Chấm trên CWQ test (2,848 câu, 53% có ≥ 2 topic); mọi retriever đều train trên WebQSP.

- **A/B — họ retriever và độ nhiễu giữa các seed (ansR@50, transfer):**
  - DDE: 70.05 / 64.45 / 67.89 / 65.44 (4 seed). **Chênh 5.6 điểm giữa các seed.**
  - QR-DDE: trung bình 62.8 (3 seed). Câu 1 topic sụp còn 47–56.
  - RT (không weighted loss): 72.4.
  - WL: 74.04 / 74.42 (2 seed), chỉ chênh 0.37.
  - REL-RAG P: 61.9; câu 1 topic chỉ 49.0.
  - ⇒ Mức hơn +1.4 của ASE s42 so với DDE s42 nằm trong độ nhiễu seed. WL là backbone tốt nhất và ổn định nhất. Weighted loss đóng góp +1.6 so với RT, chủ yếu ở câu 1 topic (64.7 → 70.1).
- **C — tính bổ sung so với WL** (gộp xếp hạng bằng RRF; đối chứng là WL s42 + WL s0, cho biết mức tăng do ensemble thuần):
  - Đối chứng: +1.59 @50.
  - Mức tăng vượt đối chứng ở @50 / @100: ASE +0.55 (không ý nghĩa) / +0.97 (biên); DDE −0.46 / −0.61; QR-DDE −2.10* / −1.15*; REL-RAG −2.39* / +0.49.
  - ⇒ ASE train trên WebQSP gần như không mang thông tin mà WL thiếu. QR-DDE và REL-RAG làm WL kém đi.
- **D — cơ chế của ASE** (ASE − DDE, @50):
  - Train trên WebQSP: câu 1 topic +3.3*, câu ≥ 2 topic −0.2 (không ý nghĩa).
  - Train trên CWQ: câu 1 topic +1.4 (không ý nghĩa), câu ≥ 2 topic **+5.65***.
  - Tỉ lệ câu train có ≥ 2 topic: WebQSP 1.9% (52 câu), CWQ 49.7%.
  - ⇒ **Nút thắt là tín hiệu train, không phải backbone.** ASE chỉ học được phép giao khi dữ liệu train có câu nhiều anchor.
- **E — augmentation anchor** (anchor thứ hai tổng hợp trên WebQSP so với anchor thứ hai thật của CWQ):

| Anchor | Khoảng cách tới đáp án 1 / 2 / ≥3 | Median selectivity | Median degree |
|---|---|---|---|
| CWQ thật (test / train) | 35 / 57 / 8 % | 0.138 / 0.132 | 195 / 158 |
| Tổng hợp, ngẫu nhiên (`--pseudo_anchor_p` cũ) | 81 / 19 / 0 % | 0.015 | 8 |
| Tổng hợp, khớp phân phối | 32 / 68 / 0 % | 0.125 | 27 |

  - Anchor ngẫu nhiên gần như chỉ thẳng vào đáp án, nên sẽ dạy model một lối tắt.
  - Bản khớp phân phối giống CWQ về khoảng cách và selectivity; độ phủ 98% số câu.
  - Chưa khớp: bậc của anchor (27 so với 158), và câu hỏi không nhắc tới anchor tổng hợp.
- **Quyết định:**
  - Backbone là **WL** (port sang `qrdde.py` dưới tên `dde+rt` với `--pos_weight effective_num`; điểm relation khớp WL gốc, sai lệch ≤ 1.1e-4, mask top-5 trùng 50/50 câu).
  - Thêm **ASE** và **augmentation anchor khớp phân phối** (`make_anchor_aug.py`, `--anchor_aug_p`).
  - Loại QR-DDE (transfer kém nhất, làm WL kém đi) và REL-RAG (kém nhất ở câu 1 topic, line graph lớn gấp 83–107×, làm WL kém đi).

## EXP-35 — 2×2 trên backbone WL: {ASE có / không} × {augmentation có / không}, train WebQSP (bắt đầu 10-07 13:24)
Augmentation: 7,708 anchor khớp phân phối. Bản đầu gán trung bình 29.7 nhãn mỗi anchor (p90 75, cao nhất 1,410, do câu có nhiều đáp án), trong khi một topic CWQ thật chỉ mang median 3 nhãn. Vì vậy chỉ giữ anchor có ≤ 20 nhãn (phân vị 90 của CWQ thật): còn 5,714 anchor trên 2,275 câu (86% số câu 1 topic), mỗi anchor median 4 nhãn, trung bình 5.5. Ngưỡng này được chốt trước khi R2/R3 chạy.
**10-07 15:13 (giờ VN): chuyển sang vast.** Lần chạy local bị dừng theo yêu cầu: R0 ở khoảng epoch 35, R1 ở khoảng epoch 27, cả hai vẫn đang cải thiện. Thư mục được giữ lại dưới tên `*_stopped_local_1007`, không dùng để báo cáo. Cả 4 run chạy lại từ đầu, song song trên vast #54539039 (`exp_qrdde/vast/exp35_remote.sh`), không resume, để giữ đúng protocol. Watchdog `vast/exp35_watchdog.sh` (chạy local) kéo kết quả về, stop máy (không xoá) khi xong, rồi chạy `adr_probe` trên CWQ và WebQSP.
Các run: R0 `webqsp_wlq_s42`, R1 `webqsp_wlq_ase_s42`, R2 `webqsp_wlq_aug_s42`, R3 `webqsp_wlq_ase_aug_s42`. Seed 42, patience 10, `anchor_aug_p` 0.5, chạy local (`queue_exp35.sh`).

**Tiêu chí đặt trước khi có kết quả** (CWQ test, mode base, ansR, bootstrap 95%):
1. **Port hợp lệ:** R0 nằm trong khoảng WL gốc ±1 điểm @50 (74.0–74.4).
2. **Cơ chế:** R3 − R2 ở câu ≥ 2 topic @50 > 0, có ý nghĩa. Tức là ASE học được phép giao khi có augmentation, như phép đo D đã thấy khi train trên CWQ.
3. **Cần cả hai thành phần:** R3 − R1 > 0 và R3 − R2 > 0 ở mức tổng thể @50.
4. **Đáng giá:** R3 − R0 vượt mức tăng của đối chứng ensemble (+1.6 @50). Trên WebQSP test, R3 không thua R0 có ý nghĩa.

Nếu (2) không đạt: augmentation không truyền được phép giao; dừng hướng ASE-transfer và giữ ASE như kết quả in-distribution (EXP-33b). Nếu (2) đạt: chạy seed 0 cho R3, rồi ADR/AER và downstream.

**Trạng thái 10-07 14:55: DỪNG theo yêu cầu**, chưa có kết quả.
- R0 dừng ở khoảng epoch 35 (val triple_recall@100 ≥ 0.899 ở ep 23, vẫn đang tăng). R1 dừng ở khoảng epoch 27 (0.902 ở ep 16, vẫn đang tăng). Tốc độ khoảng 2.5 và 3.2 phút/epoch khi chạy 2 run song song trên máy local.
- `best.pt` của R0/R1 vẫn giữ, nhưng chưa có `result.json`. R2/R3 chưa bắt đầu. Ước tính lúc dừng: cả queue còn khoảng 4–5 giờ.
- Muốn chạy tiếp thì có `--resume runs/<run>/best.pt --start_epoch <n>`. Tuy nhiên state của optimizer bị mất và bộ đếm patience reset, nên chỉ khớp protocol một cách xấp xỉ. Chạy lại từ đầu mới sạch.

## EXP-36 — SubgraphRAG / RAPL (checkpoint tác giả) / REL-RAG repro, chấm bằng `retrieve/eval.py` gốc (10-07)
Script `/relrag/sg_eval_compare.py`; kết quả đầy đủ (recall và precision ở k = 50/100/200/400/500) ở `/relrag/runs/sg_eval_compare.md`.
- Dùng nguyên `eval.py` của SubgraphRAG, không sửa.
- RAPL và REL-RAG không xuất trường nhãn, nên lấy nhãn từ file chính thức của SubgraphRAG cho đúng từng câu.
- Trong mỗi setting, mọi phương pháp chấm trên cùng tập câu.
- RAPL/REL-RAG: path best-first ở budget 500; k nhỏ hơn lấy prefix.

| Setting (n câu) | Retriever | ansR @50/100/200 | SP-R @50/100/200 | GPT-R @50/100/200 |
|---|---|---|---|---|
| WebQSP (1,638) | SubgraphRAG official | 90.4 / 94.4 / 96.9 | 82.7 / 88.3 / 92.7 | 79.8 / 86.5 / 92.5 |
| | RAPL `webqsp.pt` | 89.9 / 95.2 / 97.5 | 58.9 / 63.9 / 68.4 | 63.1 / 67.1 / 71.2 |
| | REL-RAG repro | 90.6 / 94.8 / 97.0 | 59.6 / 64.0 / 68.7 | 61.4 / 65.0 / 69.5 |
| CWQ (3,531) | SubgraphRAG official | 86.2 / 91.4 / 95.3 | 72.2 / 81.1 / 88.0 | 75.3 / 84.0 / 91.7 |
| | RAPL `cwq.pt` | 80.8 / 85.9 / 89.8 | 42.1 / 47.1 / 52.8 | 46.4 / 50.8 / 54.9 |
| WebQSP→CWQ (2,848) | SubgraphRAG repro | 70.0 / 79.6 / 87.7 | 53.6 / 63.4 / 73.2 | 53.6 / 63.1 / 72.0 |
| | RAPL `webqsp.pt` | 66.8 / 74.1 / 80.4 | 37.2 / 42.3 / 47.5 | 39.4 / 44.6 / 49.2 |
| | REL-RAG repro | 61.9 / 72.3 / 79.3 | 34.4 / 40.6 / 46.5 | 34.3 / 40.8 / 46.1 |
| CWQ→WebQSP (1,559) | SubgraphRAG repro | 82.2 / 88.2 / 92.9 | 71.5 / 79.8 / 86.5 | 71.8 / 79.7 / 86.5 |
| | RAPL `cwq.pt` | 90.9 / 95.8 / 97.7 | 56.6 / 61.4 / 65.0 | 62.3 / 65.6 / 68.7 |

- **SP-R/GPT-R thấp ở RAPL/REL-RAG phần lớn do cách biểu diễn triple, không phải thiếu bằng chứng.**
  - `eval.py` so khớp chính xác (h, r, t), trong khi decode theo path thường trả về triple dùng relation nghịch, ví dụ `(Lou Seal, sports.mascot.team, San Francisco Giants)` thay vì triple gốc.
  - Trên WebQSP @50, nếu tính SP-R theo cặp entity (không xét relation và hướng): SubgraphRAG 86.6, RAPL 83.6, REL-RAG 85.3, so với 82.7 / 58.9 / 59.6 khi khớp chính xác.
  - Hệ quả: khoảng cách 24 điểm thu lại còn khoảng 3 điểm. `gate_report.py` dùng nhãn trên graph của REL-RAG, nên không bị hiệu ứng này.
- **ansR** (không phụ thuộc biểu diễn triple):
  - Ngang nhau trên WebQSP.
  - Trên CWQ in-distribution, RAPL thấp hơn SubgraphRAG −5.4 @50.
  - Khi transfer WebQSP→CWQ, RAPL thấp hơn −3.2 @50.
- **Bất thường chưa giải thích:** `cwq.pt` đạt 90.9 ansR@50 trên WebQSP, cao hơn chính nó trên CWQ (80.8) và cao hơn SubgraphRAG train CWQ (82.2).
  - Hai checkpoint được train độc lập: cosine giữa trọng số 0.09; top-50 trên WebQSP trùng 55%.
  - CWQ được xây từ truy vấn của WebQSP, nên có thể có trùng lặp; nhưng điều này cũng đúng với SubgraphRAG train CWQ. Chưa có dữ liệu train của tác giả để kiểm chứng.
- Chưa có bản REL-RAG repro train trên CWQ.

### EXP-35 — kết quả (10-07 20:00)
Bảng đầy đủ ở `relrag/runs/exp35_report.md` (script `relrag/exp35_report.py`) và `relrag/runs/gate/gate_*_exp35_r3_*.md`.

**Cả 4 tiêu chí đặt trước đều PASS:**
1. **Port hợp lệ:** R0 = 73.93 ansR@50 (WL gốc 74.04 / 74.42).
2. **Cơ chế:** R3 − R2 trên câu ≥ 2 topic = **+7.43** [+5.75, +9.10]. ASE học được phép giao khi có augmentation.
3. **Cần cả hai thành phần:** R3 − R1 = +12.70*, R3 − R2 = +3.69*.
   - ASE không có augmentation còn **gây hại**: R1 − R0 = −6.01*. Đúng như phép đo D dự báo.
4. **Đáng giá:**
   - CWQ: R3 − R0 = **+6.69** [+5.05, +8.23], gấp hơn 4 lần mức tăng do ensemble thuần (+1.59).
   - WebQSP: R3 − R0 = +0.85 [−0.01, +1.76], không thua.

**Transfer WebQSP → CWQ, ansR @50 / @100 / @200:**

| Retriever | ansR | SP-R @50 | GPT-R @50 |
|---|---|---|---|
| R3 base | 80.62 / 86.57 / 91.33 | 58.51 | 64.79 |
| **R3 + ADR + AER** | **86.52 / 92.59 / 95.81** | 63.05 | 70.81 |
| WL + ADR + AER (EXP-30) | 83.18 / 89.82 / 94.66 | 56.83 | 65.59 |
| SubgraphRAG repro | 70.05 / 79.58 / 87.70 | 49.77 | 53.59 |
| REL-RAG repro | 61.90 / 72.26 / 79.33 | 50.82 | 34.35 |
| *SubgraphRAG official train CWQ (trần)* | *86.20 / 91.43 / 95.35* | *66.28* | *75.54* |

- **Gate transfer PASS** so với cả hai baseline ở mọi budget: R3+ADR+AER hơn SubgraphRAG +16.5 / +13.0 / +8.1 và hơn REL-RAG +24.6 / +20.3 / +16.5. Tất cả 17 ô metric phụ đều thắng.
- R3+ADR+AER train trên WebQSP đã **ngang mức trần** của SubgraphRAG train thẳng trên CWQ về ansR (86.5 so với 86.2). SP-R và GPT-R vẫn thấp hơn mức trần khoảng 3–5 điểm.
- **WebQSP in-distribution:** R3 base ansR 92.72 / 96.21 / 97.83, hơn SubgraphRAG official có ý nghĩa ở cả 3 budget (+2.3 / +1.8 / +0.9). Riêng GPT-R @200 thua −2.1*. Gate PASS.

**Cần thận trọng trước khi đưa vào paper:**
- **Mới một seed.** DDE chênh 5.6 điểm giữa các seed. WL ổn định (0.37), nhưng độ dao động của ASE và augmentation chưa đo. Cần thêm seed 0 cho R3 và R2.
- **Câu 1 topic bị trả giá:** R3 base − R0 = −2.66* @50 trên CWQ. Khi thêm ADR+AER thì ngang lại (82.49 so với 82.48).
- **Augmentation dùng thống kê của CWQ train** (khoảng cách, selectivity và degree của anchor thứ hai) để chọn anchor. Không dùng nhãn hay câu hỏi của CWQ, cũng không dùng gì của CWQ test. Nhưng đây vẫn là thông tin về domain đích, nên phải nêu rõ.
  - Cần một ablation không dùng CWQ: chọn anchor theo luật cố định (khoảng cách 1–2, selectivity nằm trong khoảng cho trước), để chứng minh mức tăng không đến từ việc khớp phân phối với CWQ.
- WebQSP chỉ có 26 câu ≥ 2 topic, nên các con số ở nhóm này (ví dụ 100.00) không có ý nghĩa thống kê.

## EXP-37 — Đính chính: EXP-32 dùng sai tập GrailQA so với paper REL-RAG (10-07)
- **Phát hiện** (người dùng nêu, đã kiểm chứng):
  - Figure 4 của paper REL-RAG cho top-5 domain là People, Biology, Computer, Religion, Digicams.
  - Bộ đã dùng ở EXP-32 là file `graliqa.json` của ODA: 1,000 câu, **lấy từ tập dev** của GrailQA (có trường level). Top domain của nó là measurement_unit 221, cvg 209, conferences 56, rail 38. Năm domain của paper ở bộ này chỉ có 9–37 câu mỗi domain.
- **Cách paper dựng tập** (App. D): "874 câu **trong tập train** của GrailQA", là các câu có subgraph 2-hop của ODA, ghép theo MID của topic entity. "Sun et al., 2024b" chính là ODA.
- **Dựng lại:**
  - Bộ subgraph của ODA (`CCaccept/Subgraph`, `grailqa_labels_2_hop_all.json`) có 879 MID topic.
  - Lọc GrailQA train (44,337 câu, HF `dki-lab/grail_qa`) theo điều kiện "ít nhất một topic entity có subgraph ODA" được **995 câu**. Top domain: religion 104, computer 83, biology 75, digicams 69, medicine 59, people 52. Khớp với Figure 4, chỉ thêm medicine.
  - Lọc theo điều kiện "mọi topic entity đều có subgraph" thì được 723 câu.
  - Sau khi bỏ các câu không có đáp án trong graph, số câu dự kiến gần 874.
- **Hệ quả:** mọi số GrailQA ở EXP-32 (và bảng GrailQA trong doc báo cáo) **không so được với paper**. Chúng chỉ là một phép thử OOD trên tập dev, theo kiểu khác. Cần dựng lại tập và chấm lại toàn bộ.
- **Đã dựng (10-11)**: `ext_data.py convert grailqa_paper` → `data_ext/grailqa_paper/test.pkl`.
  - 995 câu train có topic trong ODA. Giữ **728 câu** có đáp án (entity) nằm trong subgraph. Bỏ 176 câu đáp án entity không có trong subgraph, 90 câu `count` (đáp án là số), 1 câu so sánh.
  - Không cắt theo kích thước: trung vị 11.6k triple, p90 50.6k, max 184k. Nếu cắt ở 30k như EXP-32 thì chỉ còn 584 câu.
  - Domain: religion 89, computer 68, biology 66, digicams 63, medicine 44. Hop: 342 / 241 / 144 (1/2/3). 716 câu 1 topic.
  - Paper giữ 874 câu. Khoảng chênh gần với 90 câu `count` cộng các câu mà paper không yêu cầu đáp án nằm trong graph. Với ansR thì cần đáp án là entity trong graph, nên dùng 728.
  - Còn thiếu: embedding GTE (652k entity, 4k relation, cần GPU).

## EXP-38 — Chạy lại RAPL bằng chính code của tác giả, WebQSP → CWQ (10-10 → 10-11)
Chi tiết, các điểm khác code gốc (D1–D12) và giải thích về rò rỉ `topic_labels`: `rapl_faithful/REPORT_DEVIATIONS.md`.
- **Retrieval:** checkpoint `webqsp.pt` của tác giả; `model_inference.py` chạy nguyên văn trên svuet; nhãn test kiểu SP.
  - 3,531/3,531 câu, trung bình **123.5 triple/câu** (paper nói khoảng 50).
  - Chi phí dựng dữ liệu: line graph 77 MB/câu, tổng khoảng 175 GB × 2, đã giảm bằng D12; dựng graph 3.5 giờ, post-process 1 giờ, inference 2.6 giờ, đều đơn luồng.
- **LLM:** Llama-3.1-8B qua OpenRouter (DeepInfra fp8, greedy), $0.20. GPU của svuet hỏng tản nhiệt (99–103 °C, xung nhịp 105–210 MHz), không dùng được.
- **Kết quả (3,531 câu):**
  - Hit trên toàn bộ text đã lưu (có cả prompt) **70.6**;
  - Hit chỉ trên phần trả lời **48.1**;
  - Dòng "The answer is:" 29.0, Macro-F1 24.9;
  - Paper: 64.1 / 53.6.
- **Kết luận:** con số của paper không tái hiện được với cách chấm trên câu trả lời. Chỉ khi chấm cả prompt (đúng là file tác giả có lưu kèm prompt) mới vượt được mức của paper.

## EXP-39 — RBF-Ret: retriever chấm điểm phân rã theo bước (10-11)
Kế hoạch và giả thuyết: `docs/11_KE_HOACH_TUAN_1011.md` §3–4, §7.
- **Mô hình** (`retrieve/exp_qrdde/rbf.py`, `qrdde.py --feat rbfonly`):
  - Chạy trên entity graph. Không có MLP chấm triple, không có DDE.
  - Nhân tử một bước dùng chung cho mọi bước: π(r|q) = σ(MLP[z_q; z_r; z_q⊙z_r; lex; cos]), tách hai chiều fwd/rev.
  - Max-product Bellman-Ford từ từng anchor, T = 3. Điểm triple là log của đường đi tốt nhất qua triple đó (lấy max qua các anchor).
  - Loss BCE trên nhãn SP, như SubgraphRAG. 0.85M tham số, train trên CPU khoảng 40 s/epoch.
- **WebQSP test (in-dist, 1,559 câu, `gate_webqsp_rbfonly.md`):** ansR@50/100/200 = 89.36 / 93.85 / 96.97. SubgraphRAG official được 90.38 / 94.42 / 96.94, REL-RAG 90.60 / 94.84 / 96.95. → **Gate in-dist PASS** (không thua có ý nghĩa ở budget nào). ansHit thua SubgraphRAG 2.4 / 1.5 điểm ở @50 / @100 (chỉ số phụ).
- **WebQSP → CWQ (2,848 câu, `gate_cwq_transfer_rbfonly.md`), không dùng mẹo inference:** ansR@50/100/200/500 = 80.28 / 87.50 / 93.55 / 97.90.
  - So với SubgraphRAG train trên WebQSP: +10.2 / +7.9 / +5.9 / +3.1.
  - So với REL-RAG: +18.4 / +15.2 / +14.2 / +12.7.
  - Mọi ô đều có CI > 0, kể cả SP-R và GPT-R. → **Gate generalisation PASS.**
  - Budget efficiency: RBF@k ≈ SubgraphRAG@2k (CI chứa 0). AER chưa từng đạt được điều này.
  - Vẫn thấp hơn WL+ADR+AER (83.2 / 89.8 / 94.7), là bản dùng 3 mẹo inference.
- **Theo nhóm** (ansR@50): tổ hợp mới 83.7 so với DDE 49.2. Câu 1 topic 85.6, gần trần train CWQ (86.1). Điểm yếu: câu ≥ 2 topic (75.3, so với 84.4 của WL+ADR+AER) và câu 1 bước (85.7, so với 94.5 của DDE).
- **Ablation / kết quả âm:**
  - π chỉ dùng cổng 6 tham số (`webqsp_rbfgate_s42`): val ansR@50 dừng ở 48.6.
  - Soft-AND, π đối xứng, chia budget đều theo anchor: đều làm giảm. Chi tiết trong docs/11 §7.
- **Đang chạy:**
  - ψ(v|q) (`webqsp_rbfnode_s42`): val thấp hơn bản không có ψ, 84 so với 88 ở cùng số epoch.
  - Nhãn rational path (`webqsp_rbfrat_s42`, `make_rational.py`).
  - Train CWQ (`cwq_rbfonly_s42`, CPU, ước khoảng 9–20 giờ).
- **Cập nhật 10-11 (sáng): biến thể RBF, tất cả train trên WebQSP, không dùng mẹo inference.**

  | Biến thể | WebQSP test ansR@50/100/200 | CWQ transfer ansR@50/100/200 | Gate in-dist / transfer |
  |---|---|---|---|
  | RBF-only | 89.36 / 93.85 / 96.97 | 80.3 / 87.5 / 93.5 | PASS / PASS |
  | **RBF + non-backtracking (train với nb)** | 89.63 / 93.85 / **97.73** | **81.8 / 89.8 / 94.6** | PASS (@200 hơn SubgraphRAG có ý nghĩa, +0.79) / PASS (+11.8 / +10.2 / +6.9 so với SubgraphRAG) |
  | RBF + nb + nhãn SP ∪ rational | 89.53 / 94.40 / 97.58 | 82.0 / 89.5 / 94.6 | PASS / PASS |
  | RBF nhãn rational, nb chỉ lúc test | 89.81 / 93.97 / 96.97 | 82.1 / 89.0 / 94.8 | — |
  | Tham chiếu: WL + ADR + AER (3 mẹo) | — | 83.2 / 89.8 / 94.7 | |

  - nb giúp rõ ở câu so sánh (+7.6), câu cực trị (+5.5) và câu 1 bước (+3.9 / +5.2), nhưng giảm ở câu ≥ 3 bước (66.7 → 59.9).
  - Nhãn rational giúp câu 1 bước và câu conjunction, nhưng giảm ở câu ≥ 3 bước.
- **Hybrid (train trên vast A4000, `vast/rbf_hybrid_remote.sh`):**
  - `webqsp_rbfmlp_nb_s42`: MLP entity, không DDE, cộng 4 feature đường đi RBF. Val ansR@50 **91.52**.
  - `webqsp_wlrbf_nb_s42`: WL cộng RBF. Val ansR@50 **91.62**.
  - So với WL 91.01 và DDE 89.14. Test và CWQ được chấm trên CPU local (`queue_hybrid_probe.sh`).
- **Đang chạy:**
  - Seed 0 của `rbfnb` và `rbfmlp_nb` trên vast.
  - Embedding GTSQA trên vast.
  - Train CWQ cho `rbfonly` trên CPU local (epoch 16, val ansR@50 90.3).
  - GrailQA tập paper (`queue_grailqa_paper.sh`).
- **GrailQA, tập của paper (728 câu, train trên WebQSP, sơ bộ 10-11):** ansR@50/100/200.
  - RBF-only: **49.9 / 57.8 / 62.3**.
  - SubgraphRAG repro (DDE): 40.3 / 48.3 / 54.4.
  - DDE + ADR + AER: 42.4 / 51.4 / 59.3.
  - Theo số bước: 2 bước 47.2 so với 29.0 (DDE). 3 bước 12.3 so với 9.3 (DDE) và 14.5 (AER).
  - Theo domain: biology 50.8 so với 29.4. Digicams thấp ở cả hai (22.3 so với 21.5).
  - Đang chạy: các biến thể nb, WL (8 phần), REL-RAG.
- **GrailQA, tập của paper (728 câu, train trên WebQSP, 10-11), ansR@50/100/200:**

  | Phương pháp | @50 / @100 / @200 |
  |---|---|
  | SubgraphRAG repro | 40.3 / 48.3 / 54.4 |
  | WL | 51.9 / 57.9 / 62.6 |
  | WL + ADR + AER | 54.7 / 62.9 / 67.7 |
  | RBF-only | 49.9 / 57.8 / 62.3 |
  | RBF + nb (chỉ lúc test) | 58.1 / 65.4 / 71.3 |
  | **RBF + nb (train với nb)** | **58.8 / 67.1 / 73.2** |
  | RBF + nb + nhãn SP ∪ rational | 58.8 / 69.1 / 75.0 |
  | RBF nhãn rational + nb | 61.5 / 69.2 / 76.9 |

  - Đi không quay lui giúp nhiều hơn trên GrailQA (+8.2 @50) so với CWQ (+0.8), vì các domain Freebase có nhiều hub và nút type.
  - Đang chạy REL-RAG, sau đó bootstrap có ghép cặp.
- **GrailQA tập paper, đủ baseline** (`relrag/runs/grail_report_paper.md`):
  - REL-RAG repro (path_bf): 42.7 / 49.5 / 54.7.
  - Bootstrap ghép cặp, RBF+nb (train với nb) so với từng baseline:
    - SubgraphRAG: **+18.5 / +18.8 / +18.7**.
    - REL-RAG: **+16.1 / +17.7 / +18.4**.
    - WL + ADR + AER: +4.1 / +4.3 / +5.4.
    - Mọi CI 95% đều > 0.
  - → **Gate WebQSP → GrailQA PASS.** Script: `relrag/ext_boot.py`.
- **Seed 0 của RBF+nb** (`webqsp_rbfnb_s0`, train trên vast):
  - WebQSP 88.78 / 94.11 / 97.28. **Gate in-dist FAIL ở @50**: −1.59 so với SubgraphRAG official, CI [−2.89, −0.36]; −1.82 so với REL-RAG. @100 và @200 hoà.
  - CWQ transfer 82.0 / 88.5 / 94.7, gate PASS. Nhóm tổ hợp mới 80.8 (seed 42: 84.0).
  - → RBF+nb thuần ổn định khi transfer, nhưng in-dist @50 thấp hơn khoảng 1 điểm (trung bình hai seed 89.2 so với 90.4). RBF-PoE nhắm đúng vào khoảng hụt này.
- **LỖI ĐÃ SỬA (10-11 trưa): `torch_scatter.scatter_max` điền 0 vào ô không nhận được message, chứ không phải −∞.**
  - Nhánh không-nb của RBF tách bước xuôi và bước ngược thành hai lần scatter. Vì vậy nút nào không có cạnh vào, hoặc không có cạnh ra (mọi nút lá), đều nhận điểm 0, tức được đối xử như một anchor phụ. Soft-AND (backward) cũng dính lỗi này.
  - Nhánh **nb không bị**: mọi bước gom vào một lần scatter, nên ô rỗng chỉ có thể là nút cô lập, không ảnh hưởng điểm triple. Đây là lý do nhánh nb khớp brute force.
  - Đã sửa bằng `rbf.smax`: ô rỗng gán NEG. Cả ba phép tính (nb, không-nb, completion) giờ khớp brute force tuyệt đối, kiểm trên 300 đồ thị ngẫu nhiên mỗi loại.
  - **Ảnh hưởng tới kết quả:**
    - Các run train **không có nb** đều được train với lỗi: `webqsp_rbfonly_s42`, `webqsp_rbfrat_s42`, `rbfgate`, `rbfnode`, `cwq_rbfonly_s42`. Run cuối đã dừng và thay bằng `cwq_rbfnb_s42`.
    - Các kết luận về soft-AND và π đối xứng được rút ra trên checkpoint dính lỗi.
    - Phần lợi ích tôi gán cho "nb" (+0.85 WebQSP, +8.2 GrailQA khi chỉ bật lúc test) **có thể một phần là do sửa lỗi**. Cần train lại bản không-nb đã sửa để tách hai phần.
    - Các run **có nb** (`rbfnb_s42/s0`, `rbfnb_union`, `rbfmlp_nb`, `wlrbf_nb`, `rbfpoe_nb`) **không bị ảnh hưởng**.
- **Điểm max-marginal của đường đi trọn vẹn (`--rbf_mm`, 10-11 chiều).** Ràng buộc kết thúc dựa trên cấu trúc, không có tham số: walk chỉ được dừng ở entity có text, vì CVT không bao giờ là đáp án. Điểm của mỗi triple là điểm của walk trọn vẹn tốt nhất đi qua nó (forward × completion). Kiểm chính xác bằng brute force. Mới bật lúc test trên `webqsp_rbfnb_s42`, chưa train lại:
  - **WebQSP:** 89.63 / 93.85 / 97.73 → **91.37 / 95.18 / 98.05**. Gate PASS; @200 hơn SubgraphRAG có ý nghĩa (+1.11), GPT-R@50 +2.09 có ý nghĩa, SP-R@100 −1.17.
  - **WebQSP → CWQ:** 81.8 / 89.8 / 94.6 → **87.1 / 92.1 / 96.0**, vượt cả trần SubgraphRAG official train trên CWQ (86.2 / 91.4 / 95.4).
    - So với SubgraphRAG: ansR +17.0 / +12.6 / +8.3 / +4.0, SP-R +8.7, GPT-R +13.2 @50. So với REL-RAG: +25.2 @50. Mọi ô có ý nghĩa.
    - **Budget efficiency PASS**: RBF@50 hơn SubgraphRAG@100 +7.5.
  - **Theo nhóm:** tổ hợp mới 85.7; tổ hợp đã thấy 85.9 (trước đó 76.5); ≥ 2 topic 85.6 (WL+ADR+AER 84.4); ≥ 3 bước 73.1 (trước đó 59.9).
  - **Cơ chế:** điểm theo tiền tố chỉ giảm dần dọc đường đi, nên bước 1 sang CVT luôn đứng trên bước tiếp theo, và "anchor → member → [80 CVT]" lấp đầy budget. Max-marginal trả cho bước vào CVT đúng điểm của phần hoàn chỉnh tốt nhất của nó.
- **Hybrid seed 0** (`webqsp_rbfmlp_nb_s0`), nhóm tổ hợp mới trên CWQ: head MLP 62.7, head đường đi 81.8, hợp nhất 70.9 (seed 42: 51.2 / 80.8 / 66.0). → Cơ chế "MLP học thuộc tổ hợp" lặp lại qua seed.
- **Đang chạy:** `webqsp_rbfnbmm_s42` (train với nb + mm), để biết train với mm có hơn chỉ bật mm lúc test không.
- **mm trên GrailQA và GTSQA:**
  - GrailQA: mm gần như không đổi gì (+0.1, CI chứa 0), có lẽ vì subgraph ODA ít nút trung gian. RBF+nb+mm 58.9 / 67.1 / 73.1. So với SubgraphRAG +18.6, so với REL-RAG +16.3 @50.
  - GTSQA (Wikidata, không có CVT): mm không có tác dụng.
- **WebQSP → GTSQA (1,622 câu), RBF+nb:**
  - ansR@50/100/200 = 57.7 / 64.9 / 71.9. Gold-subgraph recall@50 53.7, gold full@50 32.7.
  - Bootstrap ghép cặp: so với SubgraphRAG (DDE, 35.4 / 41.2 / 47.9) **+22.3 / +23.7 / +24.0**; so với DDE+ADR+AER +19.0 / +16.5 / +12.9. Mọi CI > 0.
  - Theo loại: in-distribution 67.3, unseen relation type 69.2, unseen graph type 44.5. Theo số seed: 1 seed 71.2, 4 seed 27.6 (câu nhiều anchor vẫn yếu, như mọi retriever trong paper GTSQA).
  - Đang chờ REL-RAG và WL.
- **Chuyển train sang vast (10-11 chiều, theo yêu cầu người dùng: chỉ train trên vast).**
  - Đã dừng 3 run local: `cwq_rbfnb_s42` (CPU, khoảng 30–45 phút/epoch), `webqsp_rbfnbmm_s42`, `webqsp_rbffix_s42`. Train lại từ đầu trên box #55286305.
  - Dữ liệu CWQ: `make_global_emb.py` trích một vector cho mỗi tên duy nhất, đúng giá trị trong shard, fp32.
    - Train: 701,550 entity, 6,468 relation, 2.9 GB. Val: 1.1 GB. Shard gốc là 59 GB.
    - Lệch lớn nhất giữa các bản lặp của cùng một tên là 6.1e-05. Kiểm 50 câu ngẫu nhiên so với shard: shape bằng nhau, lệch lớn nhất 6.1e-05.
  - Khác với run CWQ cũ: với bảng gộp, `epoch_order` xáo trộn phẳng thay vì xáo theo nhóm shard. Chỉ thứ tự mẫu khác, dữ liệu và loss giữ nguyên.
- **Gate WebQSP → GTSQA PASS** (1,622 câu, bootstrap ghép cặp). RBF+nb 57.7 / 64.9 / 71.9:
  - so với SubgraphRAG (35.4 / 41.2 / 47.9): **+22.3 / +23.7 / +24.0**;
  - so với REL-RAG repro (path_bf, 38.8 / 46.0 / 50.8): **+18.9 / +18.9 / +21.1**;
  - so với DDE + ADR + AER: +19.0 / +16.5 / +12.9.
  - Mọi CI > 0. WL trên GTSQA đang chạy lại; `make_sg_ext` đã sửa cho bản ghi không có `n_text`.
- **Tách đóng góp: sửa lỗi scatter hay non-backtracking?** `webqsp_rbffix_s42` (không nb, đã sửa lỗi, train trên vast). Một seed:

  | | WebQSP ansR @50/100/200 | CWQ ansR @50/100/200 | CWQ tổ hợp mới |
  |---|---|---|---|
  | RBF dính lỗi, không nb | 89.36 / 93.85 / 96.97 | 80.3 / 87.5 / 93.5 | 84.5 |
  | RBF đã sửa, không nb | 89.42 / 94.26 / 97.41 | 82.1 / 88.7 / 94.0 | 81.5 |
  | RBF + nb | 89.63 / 93.85 / 97.73 | 81.8 / 89.8 / 94.6 | 84.0 |

  → **Phần lớn lợi ích trước đây gán cho nb thật ra đến từ việc sửa lỗi.** Bản thân nb chỉ lệch từ −0.4 đến +1.1, nằm trong dao động của một seed (seed 0 / 42 lệch nhau tới 1.5).
  - **ĐÍNH CHÍNH các ghi chú trước:** "nb +0.85 WebQSP / +0.8 CWQ / +8.2 GrailQA" là nb cộng với sửa lỗi, đo trên checkpoint dính lỗi.
  - Đóng góp chính còn lại là (1) điểm phân rã theo bước và (2) max-marginal của đường đi trọn vẹn (mm). nb giữ lại như một ràng buộc cấu trúc có lý (đã kiểm chính xác) nhưng không được tính là nguồn lợi ích. Đang kiểm trên GrailQA.
- **nb trên GrailQA tập paper** (bootstrap ghép cặp, cùng seed 42, đều đã sửa lỗi): RBF+nb hơn RBF không nb **+2.9 / +3.9 / +4.3**, mọi CI > 0. Riêng bản không nb đã hơn SubgraphRAG khoảng +15.6 @50.
  - → **Đóng góp của nb phụ thuộc domain:** trung tính trên WebQSP và CWQ, có ý nghĩa trên các domain mới của GrailQA, nơi có nhiều hub và nút type, đúng với chẩn đoán về đường "nảy". Phần đính chính phía trên được hiểu lại theo kết quả này.
- **Train với mm (`webqsp_rbfnbmm_s42`, vast) so với chỉ bật mm lúc test:**
  - WebQSP: 91.46 / 95.44 / 97.86, so với 91.37 / 95.18 / 98.05.
  - CWQ: 85.2 / 91.2 / 95.5, so với 87.1 / 92.1 / 96.0. Nhóm tổ hợp mới 85.2 so với 85.7.
  - → Train với mm không đem thêm lợi ích. mm là ràng buộc lúc inference. **Cấu hình chính: train với nb, test với nb + mm.**
- **WL trên GTSQA** (16 phần): WL 40.9 / 46.0 / 52.8; WL + ADR + AER 48.6 / 56.9 / 64.9.
  - RBF+nb hơn WL +16.8 / +18.9 / +19.1, hơn WL + ADR + AER +9.1 / +8.1 / +7.0. Mọi CI > 0.
- **Đính chính ví dụ minh hoạ mm.** Câu "George Clinton → member → [80 CVT] lấp đầy budget" là sai: thực tế có 23 bước member trong top-50, và mm chỉ đưa đáp án từ hạng 57 lên 51. Ví dụ đã đo đúng là "who is the governor of arizona 2009":
  - Điểm theo tiền tố: cả 50/50 triple đầu là "Arizona → governing_officials → CVT" (−2.64), triple đáp án ở hạng 92 (−3.63).
  - Với mm: hai bước cùng điểm −3.63, xếp xen kẽ 27 / 23, đáp án lên hạng 5.
  - Trên toàn WebQSP test, mm sửa 31 câu trượt @50 và làm hỏng 3 câu.
- **Seed 0 của cấu hình chính** (`webqsp_rbfnb_s0` + mm lúc test):
  - WebQSP 90.61 / 95.27 / 97.44, gate PASS. Không mm thì seed này FAIL ở @50; mm lấp được.
  - CWQ 84.4 / 90.4 / 95.5, gate PASS: so với SubgraphRAG +14.3 / +10.8 / +7.8 / +4.2, so với REL-RAG +22.5 @50. Nhóm tổ hợp mới 82.1.
  - **Trung bình 2 seed:** WebQSP @50 91.0, CWQ @50 85.8.
- **WebQSP → MetaQA-3hop** (14,118 câu; REL-RAG chỉ có 4,710 câu trong bộ của nó). RBF+nb+mm 42.4 / 57.6 / 70.0:
  - so với SubgraphRAG (21.5 / 33.7 / 51.8): **+20.9 / +23.8 / +18.2**;
  - so với WL: +18.7 / +24.5 / +21.6; so với WL + ADR + AER: +21.1 / +25.9 / +22.0;
  - so với REL-RAG: +27.6 / +42.3 / +55.0. Mọi CI > 0.
  - **Lưu ý về REL-RAG:** path_bf tự dừng, trung bình chỉ trả **9.6 triple/câu** (trung vị 8), nên recall đứng yên ở 14.4 cho cả ba budget. So ở cùng số triple của từng câu, REL-RAG nhỉnh hơn: 14.4 so với RBF 12.8 (−1.6, CI [−2.1, −1.1]). RBF@10 được 15.3. → Ở budget rất nhỏ, path của REL-RAG chính xác hơn. Ở budget của protocol thì nó không lấp đầy được.
- **Baseline train trên CWQ cho hàng CWQ → \* của ma trận** (ansR@50/100/200). SubgraphRAG = repro `cwq_dde`; REL-RAG = checkpoint `cwq.pt` của tác giả trong bản repro, path_bf.

  | Tập test | SubgraphRAG (CWQ) | REL-RAG (ckpt CWQ) |
  |---|---|---|
  | GrailQA | 41.3 / 50.1 / 58.0 | 37.3 / 44.3 / 52.0 |
  | GTSQA | 26.1 / 32.1 / 39.8 | 37.2 / 45.1 / 49.8 |
  | MetaQA | 33.1 / 44.9 / 61.4 | 14.4 / 14.4 / 14.4 (4,710 câu, path tự dừng) |

  - `cwq_rbfnb_s42` (vast) có best mới ở epoch khoảng 30, vẫn đang train.
- **Hàng CWQ → \* của ma trận** (`cwq_rbfnb_s42`, train trên vast, nb khi train + mm khi test; best ở epoch khoảng 30). Bootstrap ghép cặp; ansR@50/100/200:

  | CWQ → | RBF (CWQ) | so với SubgraphRAG (CWQ) | so với REL-RAG (ckpt CWQ của tác giả) | Gate |
  |---|---|---|---|---|
  | CWQ (in-dist) | 90.54 / 94.09 / 96.82 | +4.35 / +2.65 / +1.47 (official) | +9.79 / +8.22 / +7.04 | **PASS**, hơn có ý nghĩa |
  | WebQSP | 87.23 / 90.90 / 95.01 | +5.02 / +2.75 / +2.09 (repro) | **−3.62 / −4.86 / −2.74** | **FAIL** so với REL-RAG |
  | GrailQA (paper) | 48.4 / 54.8 / 60.8 | +7.1 / +4.7 / +2.8 (CI @200 chứa 0) | +11.2 / +10.5 / +8.8 | Chưa đạt chặt (@200 hoà) |
  | GTSQA | 52.4 / 58.4 / 64.8 | +26.2 / +26.3 / +25.0 | +15.1 / +13.3 / +15.0 | **PASS** |
  | MetaQA | 43.3 / 57.5 / 69.9 | +10.2 / +12.6 / +8.5 | +28.4 / +42.3 / +54.9 (REL-RAG chỉ trả 9.6 triple) | **PASS** |

  → **Điểm yếu thật: CWQ → WebQSP.** Checkpoint CWQ của REL-RAG transfer sang WebQSP rất tốt (90.85, ngang mức train trên WebQSP), trong khi RBF train trên CWQ chỉ được 87.2 (train trên WebQSP: 91.4). Cần phân tích: có thể π học trên CWQ nghiêng về relation ràng buộc/nhiều bước, còn câu WebQSP chủ yếu 1–2 bước, 1 topic.

## EXP-40 — REL-RAG train lại bằng nhãn SPARQL v2, hai setup paper-faithful / code-faithful (10-18)
- **Nhãn:** `relrag/sparql_to_pathlabels.py webqsp` đi lại chain gold (WebQSP InferentialChain) trên triple của `relrag/data/{train,val}.pkl`, chỉ giữ các trail **kết thúc ở đáp án gold** (giống v2), xuất ra `relrag/data/labels_sparqlv2/{train,val}{,_minlen}.jsonl`. Mỗi path là dãy line-graph node có thứ tự.
  - Train: 2,511 / 2,798 câu có nhãn, trung bình 5.2 trail/câu. Còn 287 câu không tới được đáp án trong subgraph, `train.py` dùng SP thay thế (206 câu trong 2,710 câu train có nhãn). Val: 215 / 246.
  - `_minlen`: giữ 1 trail ngắn nhất (code-faithful).
- **Setup:** giữ nguyên cờ của EXP-16 (`path_paper`) và EXP-17 (`path_codefaithful`), chỉ đổi file nhãn. Train trên 2× T4 (cloudflared), 15 epoch, khoảng 1.5 giờ.
  - paper-faithful (`runs/path_paper_sparqlv2`): `--labels paths:labels_sparqlv2/train.jsonl --val_labels sp --no_self_loops --batch_size 10 --seed 42` (combine mean, gnn dropout, topic loss all). Chọn checkpoint theo **val loss** như EXP-16, ra **epoch 6** (`epoch_6.pt`). Tham khảo: `best.pt` = epoch 15 chọn theo path_f1.
  - code-faithful (`runs/path_codefaithful_sparqlv2`): `--labels paths:labels_sparqlv2/train_minlen.jsonl --val_labels paths:labels_sparqlv2/val_minlen.jsonl --no_self_loops --combine sum --gnn_dropout 0 --topic_loss sample --batch_size 4 --seed 1`. Chọn theo path_f1, ra epoch 14.
- **Kết quả WebQSP test (`infer.py`, metric theo SP; ansR / ansHit):**

| Run | path_bf @50 | @100 | @500 | free (số triple) |
|---|---|---|---|---|
| paper, nhãn hybrid (EXP-16, ep 7) | 80.4 / 85.1 | 89.5 / 92.6 | 98.0 / 99.0 | 91.7 / 95.4 (192.7) |
| **paper, nhãn SPARQL v2 (ep 6, val loss)** | **84.3 / 88.5** | **90.9 / 93.8** | 98.3 / 99.0 | **92.2 / 96.2** (148.2) |
| *paper, nhãn SPARQL v2 (ep 15, path_f1)* | *88.1 / 91.8* | *93.0 / 95.3* | *98.4 / 99.0* | *90.4 / 94.2 (66.1)* |
| code-faithful, nhãn hybrid (EXP-17, ep 15) | 90.6 / 94.2 | 94.8 / 96.5 | 98.8 / 99.6 | 90.9 / 94.7 (44.6) |
| **code-faithful, nhãn SPARQL v2 (ep 14)** | 90.4 / 93.4 | **95.1 / 96.9** | **99.0 / 99.6** | 90.7 / 94.5 (42.3) |

- **Metric theo gold query** (`query_recall.py webqsp`, 1,496 câu, 257 câu có triple ràng buộc; @50 / @100):

| Run | Chain-R | Cons-R | Q-F1 | PathHit | PathAll |
|---|---|---|---|---|---|
| SubgraphRAG (official) | 84.8 / 91.0 | 11.5 / 15.4 | 15.6 / 11.4 | 84.1 / 90.1 | 76.6 / 84.0 |
| paper, hybrid | 76.1 / 84.4 | 24.8 / 38.4 | 16.5 / 12.2 | 79.9 / 87.4 | 68.5 / 77.1 |
| **paper, SPARQL v2 (ep 6)** | 80.1 / 86.0 | 31.1 / 45.5 | 17.4 / 12.5 | 84.0 / 88.6 | 73.7 / 80.1 |
| *paper, SPARQL v2 (ep 15)* | *83.7 / 88.7* | *35.5 / 45.9* | *18.5 / 13.2* | *87.2 / 90.8* | *76.9 / 82.4* |
| code-faithful, hybrid | 85.7 / 89.6 | 32.3 / 43.4 | 18.7 / 13.2 | 88.1 / 91.6 | 77.9 / 83.4 |
| **code-faithful, SPARQL v2** | **86.7 / 91.1** | 34.5 / 44.2 | **19.2 / 13.6** | **89.6 / 93.6** | **79.4 / 85.0** |

- **Nhận xét:**
  - **paper-faithful hưởng lợi rõ từ nhãn mới:** cùng tiêu chí chọn checkpoint (val loss), ansR@50 +3.9 (80.4 → 84.3), PathHit@50 +4.1, Cons-R@50 +6.3. Nhãn hybrid dài hơn (d..d+2) và lẫn trail không chứa đáp án; nhãn v2 chỉ giữ trail tới đáp án.
  - Tiêu chí chọn checkpoint vẫn là yếu tố lớn nhất của paper-faithful: chọn theo path_f1 (ep 15) thêm +3.8 ansR@50 nữa so với val loss (ep 6), như đã thấy ở EXP-16.
  - **code-faithful gần như không đổi về ansR** (@50 −0.2, @100 +0.3), nhưng tăng nhẹ ở mọi metric theo gold query: PathHit@50 +1.5 / @100 +2.0, PathAll@50 +1.5, Chain-R@50 +1.0, Q-F1 +0.5. Nhãn minlen của hybrid vốn đã gần với chain gold, nên nhãn mới chỉ giúp ít.
  - Với nhãn v2, code-faithful đạt PathHit@50 89.6, ngang RBF + BGE + WL nhãn SP ∪ rational (90.0, docs/12 §16) và cao hơn SubgraphRAG (84.1).
  - **Transfer WebQSP → CWQ (2,848 câu, cùng tập với BENCHMARK §6):** nhãn mới làm REL-RAG kém đi. paper-faithful ansR@50 / @100 61.2 / 69.2 → 56.2 / 65.6; code-faithful 61.9 / 72.3 → 57.8 / 67.9; PathHit@50 gần như không đổi (38.2 → 37.7, 38.4 → 37.3). Nhãn SPARQL của WebQSP gần như toàn chain 1 hop, decoder học đi ngắn hơn. Cùng nhãn này lại giúp RBF + BGE + WL (87.9 → 90.3).
  - Bảng thống nhất của cả phase: BENCHMARK.md §6 (`relrag/sparql_phase_table.py`). QA chưa chạy (cần server bf16).
- **Artifact:** `relrag/runs/path_{paper,codefaithful}_sparqlv2/` (`args.json`, `log.json`, `best.pt`, `epoch_6.pt` cho paper, `retrieval_test_path_{bf,free}{,_ep6}/`, `retrieval_cwq_test_path_bf{,_ep6}/`). Checkpoint từng epoch còn lại chỉ có trên server.

## EXP-41 — BGE-M3 LoRA tuning lại bằng nhãn SPARQL v2; RT_combined + WL dùng BGE đó (10-20)
- **BGE:** recipe Phase 11 `v1_dense_entity_disjoint` giữ nguyên (LoRA r 8, 3 epoch, lr 1e-4, chỉ WebQSP train, split entity-disjoint). Chỉ đổi tập relation dương: câu có nhãn v2 lấy relation của chain + constraint (`retrieve/exp_qrdde/bge_sparql_data.py`), câu không có nhãn v2 giữ SP. 2,605 / 2,848 câu dùng nhãn v2; trung bình 1.8 relation dương/câu (SP: 5.0). Train trên Kaggle T4 (`vast/bge_sparql_remote.sh`, transformers 4.57, gradient checkpointing non-reentrant). Dev: recall@10 0.837, MRR 0.78. Adapter + điểm: `data/bge_p11/{adapter_sparql2,scores_ft_sparql2}`.
- **Độ phủ relation của gold query trên test** (đủ cả chain trong top-5 / top-10; recall@10 chain + constraint):

  | | base | tuned SP (cũ) | tuned SPARQL v2 |
  |---|---|---|---|
  | WebQSP (1,496 câu) | 55.8 / 68.6; 72.7 | 74.1 / 84.2; 86.1 | **76.4 / 89.2; 89.4** |
  | CWQ (2,510 câu) | 24.3 / 40.4; 55.8 | 33.8 / 50.4; 66.1 | **39.0 / 56.1; 68.8** |

- **RT_combined + WL** (qrdde `dde+rt` + `effective_num`, nhãn SPARQL v2, seed 42, patience 10; cờ mới `--rt_src bge --rel_prior bge2`: nửa ngữ nghĩa của điểm relation lấy từ BGE mới, embedding vẫn GTE, nửa lexical là lexical của WL). Đối chứng cùng nhãn với điểm relation GTE. Bảng thống nhất BENCHMARK §6 (`relrag/sparql_phase_table.py`), ansR@50/100, PathHit@50, Cons-R@50:

  | Retriever · nhãn | WebQSP (ID) | WebQSP → CWQ |
  |---|---|---|
  | RT_combined + WL (GTE) · SP (`webqsp_wlq_s42`) | 91.9 / 95.1; 87.4; 11.7 | 73.9 / 83.7; 46.8; 31.1 |
  | RT_combined + WL (GTE) · SPARQL v2 | **94.8 / 97.3**; 94.3; **68.9** | 79.1 / 86.9; 51.8; 19.4 |
  | RT_combined + WL (BGE SPARQL v2) · SPARQL v2 | 94.1 / 96.8; **95.4**; 67.6 | 79.1 / 86.7; 52.5; 20.5 |
  | RBF_nb_mm · SPARQL v2 | 92.8 / 96.4; 92.4; 59.3 | 87.5 / 92.7; 66.1; 39.8 |
  | RBF + BGE (cũ) + WL · SPARQL v2 | 93.5 / 97.0; 93.6; 65.8 | **90.3 / 94.6**; 64.0; 51.3 |

- **Kết luận:**
  - Nhãn SPARQL là thứ làm WL tăng (+2.9 ansR@50 ID, +5.2 transfer); thay GTE bằng BGE tuned SPARQL trong kênh RT **không thêm gì** (−0.7 / 0.0 ansR@50, +1.1 PathHit@50 ID). MLP trên entity + DDE quyết định thứ hạng, điểm relation chỉ là một feature phụ.
  - WL + SPARQL v2 là retriever ID mạnh nhất hiện có (94.8), nhưng transfer kém RBF 8–11 điểm ansR@50: phần entity-aware không tổng quát hoá sang tổ hợp mới. RBF ngược lại. Đây là căn cứ cho hướng thêm nhân tử entity / constraint ở bước cuối của RBF (chưa chạy).
- **Artifact:** `retrieve/exp_qrdde/runs/webqsp_wl_{gte,bge2}_sparql2_s42/` (`best.pt`, `result.json`, `adr_{webqsp,cwq}_base_mm1_full.pth`).
- **Cập nhật 10-20 (trưa): AER / ADR trên hai checkpoint WL SPARQL v2** (`adr_probe.py`, B = 3, không train lại; chạy trên GPU 1080 Ti local, ~7 phút / tập CWQ). Cùng tập câu với BENCHMARK §6; ansR@50 / @100, PathHit@50, Cons-R@50:

  | Checkpoint · mode | WebQSP (ID) | WebQSP → CWQ (OOD) |
  |---|---|---|
  | WL (GTE) · base | 94.8 / 97.3; 94.3; 68.9 | 79.1 / 86.9; 51.8; 19.4 |
  | WL (GTE) · AER | 94.6 / 97.0; 93.9; 69.1 | 86.3 / 92.7; 53.7; 22.6 |
  | WL (GTE) · ADR | 94.8 / 97.3; 94.1; 69.4 | 80.6 / 87.9; 52.2; 28.7 |
  | WL (GTE) · ADR + AER | 94.6 / 97.0; 93.6; 70.6 | **87.4 / 93.5**; 53.7; 34.4 |
  | WL (BGE SPARQL v2) · base | 94.1 / 96.8; 95.4; 67.6 | 79.1 / 86.7; 52.5; 20.5 |
  | WL (BGE SPARQL v2) · ADR + AER | 93.8 / 96.6; 95.1; 70.4 | 87.1 / 92.4; 52.9; 34.4 |
  | RBF_nb_mm · SPARQL v2 | 92.8 / 96.4; 92.4; 59.3 | 87.5 / 92.7; **66.1**; 39.8 |
  | RBF + BGE + WL · SPARQL v2 | 93.5 / 97.0; 93.6; 65.8 | **90.3 / 94.6**; 64.0; **51.3** |

  - AER gánh gần hết phần tăng OOD (+7.2 ansR@50), ADR riêng +1.5, cộng lại +8.3 (79.1 → 87.4): ngang RBF_nb_mm v2, vẫn kém RBF + BGE + WL 2.9. ID gần như không đổi (−0.2 / −0.3).
  - **PathHit@50 OOD không nhích** (~53 ở mọi mode, RBF 64–66): bridge của AER đưa đáp án vào top-50 nhưng không đưa được đường đi gold từ topic. Cons-R tăng (19 → 34) nhưng còn dưới RBF + BGE + WL (51).
  - Chi phí: ~6.5 lượt mô hình / câu nhiều topic (RBF: 1 lượt). Profile GPU: một lượt WL ~55 ms, chủ yếu là `nonzero` / `index` trong DDE (đồng bộ host), không phải MLP. `adr_probe.py` giờ dùng chung cache lượt giữa các mode của cùng một câu (kết quả y hệt, nhanh gấp đôi).
