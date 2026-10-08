# 02. Baseline: REL-RAG

*Cập nhật: 2026-09-30.* Paper: [`papers/REL_RAG_Relation_Aware_Retrieval.pdf`](papers/REL_RAG_Relation_Aware_Retrieval.pdf). Code gốc của tác giả (phiên bản RAPL) ở `RAPL/`. **Bản tái hiện của mình ở `/relrag/`.**

## 1. Phương pháp (theo paper)

### 1.1 Line graph
Mỗi triple (h, r, t) của subgraph trở thành một node. Có cạnh t₁ → t₂ khi tail của t₁ là head của t₂ (Definition 4). Feature của node lấy theo code gốc: `[z_h ‖ z_r ‖ z_t]` (GTE-large, 3072 chiều).

### 1.2 Encoder
Hai GCN 2 lớp: một chạy trên line graph, một chạy trên line graph đảo chiều. Biểu diễn cuối là **z = MEAN(z_fwd, z_bwd)** (Eq. 86). Hidden 512, dropout 0.2, không batch norm.

### 1.3 Hai regime train, cùng kiến trúc nhưng **loss khác nhau nên là hai model riêng**

| | Path-based (đóng góp chính) | Triple-based |
|---|---|---|
| Nhãn | Rational path: shortest path đã được LLM (GPT-4o) lọc | Mọi triple nằm trên rational path |
| Loss | **L_q** (Eq. 10–11): BCE chọn triple khởi đầu trong V_cand (head là topic entity); V_pos = triple khởi đầu nhãn cộng các triple có relation tương tự (> τ). **L_path** (Eq. 8): tại mỗi bước, softmax trên các triple kế tiếp trong line graph cộng node STOP | **L_triple** (Eq. 12): BCE trên ⟨z_q, z_v⟩, lấy mẫu positive:negative = 1:5 |
| Inference | Lấy mẫu triple khởi đầu, mở rộng từng bước tới STOP hoặc độ sâu tối đa, lặp lại **tới khi đủ budget triple** (STOP quyết định độ dài từng path, không quyết định tổng budget) | Chấm điểm mọi triple, lấy top-k |
| Budget trong paper | 50 triple | 500 triple |

### 1.4 Setup (App. F.1)
Adam lr 1e-3, batch 10, 15 epoch, chọn model theo val loss. Lọc rational path bằng GPT-4o với prompt ở Figure 12. Reasoner: Llama3.1-8B, GPT-4o-mini, GPT-4o.

### 1.5 Kết quả paper cần đối chiếu (WebQSP)

| Setting | Macro-F1 | Hit |
|---|---|---|
| Ours(P) + Llama3.1-8B, 50 triple (Table 1, 3) | 77.3 | 88.8 |
| Ours(P) + Llama3.1-8B, 500 triple | 74.9 | 92.1 |
| Ours(P) + GPT-4o-mini, 50 triple | 80.4 | 92.5 |
| SubgraphRAG + Llama3.1-8B, 100 triple (trích từ paper gốc) | 70.5 | 86.6 |

- Table 2 (transfer, triple-based, 500 triple, GPT-4o-mini): line graph so với raw graph trên CWQ là 57.0 / 67.5 so với 51.8 / 62.8 (F1 / Hit).
- Table 7: runtime end-to-end 1.74 s (50 triple) so với SubgraphRAG 2.63 s (100 triple).

## 2. Bản tái hiện (`/relrag/`)

| Bước | Script | Ghi chú |
|---|---|---|
| Dữ liệu line graph | `prepare_data.py [--dataset cwq]` | Node là cạnh của `nx.DiGraph` (như code gốc); forward shortest path q→a; ứng viên relation-path |
| Lọc rational path | `gpt_label.py`, `merge_labels.py` | Theo đúng code gốc (`FindRationalPath.py`, ứng viên d..d+2 top 10, chỉ tập train). GPT-4o cho 901 câu, DeepSeek-V4.1-Flash cho phần còn lại (EXP-15). Bản gemma3 trước đây (EXP-01) đã bỏ, các run gemma đã xóa ngày 10-01 |
| Embedding | `make_memmap.py` | GTE-large, lưu fp16 memmap (cosine với fp32 = 1.000000) |
| Train | `train.py --regime path/triple/joint`; cấu hình paper: `run_paper.sh` | Khớp F.1; GCN tự viết bằng `torch.sparse.mm` (khớp PyG `GCNConv` tới sai số 3.6e-7). Từ 10-01: seed 42, không self-loop, nhãn train `paths:` (path cụ thể d..d+2 của tác giả mà LLM chọn), val dùng nhãn SP |
| Inference | `infer.py --mode path_bf/triple` | `path_bf` = best-first theo xác suất path cho tới khi đủ budget |
| Eval retrieval | `eval_retrieval_text.py` | Text-match, cùng tập đáp án với SubgraphRAG |
| Reasoning | `run_reason.sh` | Module `reason/` của SubgraphRAG, llama3.1 qua ollama |

### 2.1 Sai khác đã biết so với paper

| Điểm | Paper | Bản tái hiện | Ảnh hưởng đo được |
|---|---|---|---|
| Lọc path | GPT-4o | Từ 10-01: GPT-4o cho 901/2,710 câu train (hết credit), DeepSeek-V4.1-Flash cho phần còn lại; regex parse bỏ ràng buộc cuối dòng vì GPT-4o hiện tại viết giải thích ngay trên dòng chọn path | DeepSeek lọc lỏng hơn GPT-4o: 2.47 so với 1.72 relation path mỗi câu; khoảng 1/5 câu nhiều path có thêm path vòng vèo (EXP-15b). Bản gemma trước đây (EXP-01) đã bỏ |
| Reasoner | Llama3.1-8B đầy đủ | llama3.1 Q4 (ollama) | ~−1 Hit, −0.9 F1 (EXP-07) |
| Prompt reasoning | Của tác giả (input dạng path, "think step by step", beam 5) | Prompt SubgraphRAG `sys_icl_dc` | Chưa đo (M0) |
| Self-loop trong GCN | "Standard GCN" (không nói rõ) | Theo code gốc: tắt self-loop (chốt 10-01) | Ở các run gemma cũ, không self-loop tốt hơn ~1.7 R@50 |
| Graph | Không nêu | DiGraph gộp cạnh song song (như code gốc): mất 9% triple, 6.4% bằng chứng GPT | Chưa đo downstream (M0) |
| τ (Eq. 10), độ sâu tối đa | Không nêu | τ = 0.95, độ sâu 4 | – |
| **P1. Nhãn test lúc inference** | Không nêu | `Trainerv3.decoding(way='normal')` gộp `topic_labels` (triple đầu của rational path, cần đáp án) vào điểm xuất phát; bản tái hiện bỏ | +4.0 đến +6.1 ansR trên CWQ (EXP-31) |
| **P2. Cách chấm Hit** | "answer appears among the predictions", không có script | Không có script; Hit trên text thô cao hơn nhiều | WebQSP: text thô 91.95 so với parse 68.03 (EXP-17) |
| **P3. Câu bị bỏ** | Không nêu | Decode lỗi / OOM thì `continue`, không ghi file | Chưa đo được |
| **P4. Budget triple** | 50 | Decode của tác giả không có budget | CWQ: 67–73 triple/câu (EXP-31) |
| **P6. Checkpoint tác giả** | HF `TianJun1/RAPL` | Đã chạy (`authors_ckpt.py`) | WebQSP 89.91 R@50; WebQSP→CWQ 66.8; CWQ ID 80.8 (path_bf@50) |
| **P8. Baseline SubgraphRAG** | "same reasoning modules" | Số lấy nguyên từ paper SubgraphRAG (56.9/47.2), tức prompt và cách chấm của SubgraphRAG | So hai pipeline reasoning khác nhau |

Kết luận EXP-31: với checkpoint của tác giả (kể cả khi có rò rỉ), Hit chiếu ra là 46–55, trong khi paper báo 64.1 / 67.2. Retrieval hoàn hảo cũng chỉ cho khoảng 60. Phần chênh còn lại chỉ có thể nằm ở reasoner và cách chấm không công bố (P2, P5).

## 3. Kết quả tái hiện

### 3.1 Retrieval, WebQSP test (ans recall, text-match, 1,559 câu)

| Method | R@20 | R@50 | R@100 | R@500 |
|---|---|---|---|---|
| SubgraphRAG (checkpoint chính thức) | 82.58 | 90.38 | 94.42 | 98.93 |
| REL-RAG path, best-first, nhãn gemma3, không self-loop | 78.91 | 88.77 | 93.80 | 98.78 |
| REL-RAG path, best-first, nhãn gemma3, có self-loop | 75.90 | 87.06 | 92.38 | 98.29 |
| REL-RAG path, best-first, nhãn SP | 76.13 | 86.50 | 91.51 | 98.61 |
| REL-RAG triple, nhãn gemma3 | 68.65 | 80.43 | 87.80 | 97.56 |
| REL-RAG triple + residual (cải tiến của mình) | 81.54 | 89.92 | 94.54 | 99.10 |

### 3.2 Reasoning @50 (1,608 câu, eval gốc = protocol của paper)

| Run | Hit | Macro-F1 |
|---|---|---|
| SubgraphRAG@100 + Llama3.1-8B đầy đủ (dự đoán chính thức) | 86.63 | – |
| SubgraphRAG@100 + llama3.1 Q4 | 85.57 | 70.88 |
| SubgraphRAG@50 + llama3.1 Q4 | 84.27 | 70.82 |
| **REL-RAG(P)@50 + llama3.1 Q4** | **83.77** | **72.33** |
| *Paper: REL-RAG(P)@50 + Llama3.1-8B* | *88.8* | *77.3* |

- Paired bootstrap giữa REL-RAG@50 và SubgraphRAG@50 (eval corrected): ΔF1 +1.65 [−0.07, +3.33], chưa có ý nghĩa thống kê.
- **Kết luận hiện tại:** hướng kết quả giống paper, nhưng độ lớn nhỏ hơn nhiều. Khoảng cách tuyệt đối (−5 Hit, −5 F1) chưa được giải thích; nghi vấn chính là prompt và cách trích đáp án của tác giả (M0).

### 3.3 Chi phí
Line graph có số cạnh gấp 83× (WebQSP) / 107× (CWQ) graph gốc. Retriever 69.5 ms/query, so với 10.1 ms của SubgraphRAG (1080 Ti). Chi tiết ở EXP-08 … EXP-10.

### 3.4 Transfer WebQSP → CWQ (retrieval)
REL-RAG đúng paper: R@50 53.6 (path) / 51.8 (triple), R@500 81.0 / 90.1, trong khi SubgraphRAG train in-distribution trên CWQ đạt 86.2 / 98.4. Triple + residual: 74.6 / 97.2. Chi tiết ở EXP-12 trong [`04_EXPERIMENT_LOG.md`](04_EXPERIMENT_LOG.md).
