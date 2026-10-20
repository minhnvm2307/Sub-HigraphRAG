# BENCHMARK — KGQA (retrieval + QA)

*Cập nhật 2026-10-18 (thêm mục 6: phase nhãn SPARQL).* Mọi số retrieval sinh bởi `relrag/benchmark_all.py`, ra `relrag/runs/bench/bench_all.json`. Mọi số QA sinh bởi `reason/metrics/bench_qa.py`, ra `reason/results/bench_qa.json`. Riêng dataset ngoài (mục 4) và ablation RBF (mục 5) lấy từ nhật ký thí nghiệm, có ghi rõ EXP.

**Tóm tắt vị trí hiện tại:**

| | WebQSP (ID) | CWQ (ID) | WebQSP → CWQ (OOD) |
|---|---|---|---|
| Retrieval tốt nhất của mình, ansR@50 | 92.8 (WL + ASE) · RBF+nb+mm 91.4 | **90.5** (RBF, train CWQ) | **88.7** (RBF+nb+mm + BGE) · WL+ASE+aug+AER 87.4 (*) |
| SubgraphRAG, ansR@50 / @200 | 90.4 / 96.9 | 86.2 / 95.3 | 70.0 / 87.7 (repro train WebQSP) |
| Gate "@50 > SubgraphRAG @200" | chưa đạt | chưa đạt | **đạt** (88.7 > 87.7) |
| QA của mình, Hit / Macro-F1 (full) | RBF @100: 84.3 / 69.4 (fp8). AER @50 chỉ có 1,559 câu: 86.4 / 69.8, cùng tập SubgraphRAG 88.6 / 72.1 | **58.0 / 48.2** (RBF @100) | 54.2 / 42.8 (WL+ADR+AER @50) · RBF @50 52.5 / 44.6 |
| SubgraphRAG QA (full) | 86.6 / 70.6 (@100) | 56.9 / 47.1 (@100) | — (repro @50, 1,000 câu: 48.5 / 35.7) |
| Paper REL-RAG QA (@50) | 88.8 / 77.3 | 67.2 / 56.8 | 64.1 / 53.6 |

(*) aug dùng thống kê của CWQ train, nên không còn là OOD thuần.

---

## 0. Cách đọc

### 0.1 Setting: train ở đâu, test ở đâu

| Ký hiệu | Train retriever trên | Test trên | Loại | Số câu chấm retrieval | Số câu chấm QA (full / subset) |
|---|---|---|---|---|---|
| **WebQSP** | WebQSP train | WebQSP test | **ID** (in-distribution) | 1,559 | 1,628 / 1,559 |
| **CWQ** | CWQ train | CWQ test | **ID** | 2,848 | 3,531 / 2,848 |
| **WebQSP → CWQ** | WebQSP train | CWQ test | **OOD** (compositional: câu CWQ ghép nhiều ràng buộc, nhiều bước) | 2,848 | 3,531 / 2,848 |
| **CWQ → WebQSP** | CWQ train | WebQSP test | **OOD** | 1,559 | 1,628 / 1,559 |
| WebQSP → GrailQA / GTSQA / MetaQA | WebQSP train | tập ngoài | **OOD** (domain hoặc KG khác) | xem mục 4 | chưa chạy QA |

- **"SubgraphRAG" trong các bảng.** Nếu là ID, đó là **checkpoint chính thức** của tác giả (file `reason/scored_triples/*_unidir_test.pth`). Nếu là OOD, đó là **bản repro DDE** train trên tập nguồn, vì tác giả không công bố checkpoint train trên tập kia. Trên WebQSP, repro lệch chính thức −0.3 ansR@50.
- **"REL-RAG" có ba loại khác nhau:**
  - (a) **số paper**: chỉ có QA;
  - (b) **checkpoint tác giả** (`webqsp.pt` / `cwq.pt` trên HF `TianJun1/RAPL`), chạy bằng decoder path_bf của mình;
  - (c) **repro** của mình (`relrag/runs/path_codefaithful`), chỉ có bản train WebQSP.

### 0.2 Full set và subset: vì sao có hai con số
- **Full** là toàn bộ câu test mà cả hai paper báo cáo, đúng bằng tập câu trong file dự đoán chính thức của SubgraphRAG:
  - **WebQSP 1,628 câu.** Split test có 1,639 câu, nhưng 11 câu không có đáp án gold (`a_entity` rỗng, ví dụ WebQTest-238), nên không chấm được và bị bỏ.
  - **CWQ 3,531 câu.**
  - Số trong paper (SubgraphRAG 86.6 / 70.5, REL-RAG 88.8 / 77.3, …) đều tính trên full. Mình đã kiểm lại: chấm lại file chính thức ra đúng 86.61 / 70.57 và 56.87 / 47.06.
- **Subset** là các câu full có **ít nhất một đáp án nằm trong subgraph KG** của câu (`a_entity_in_graph` khác rỗng): **WebQSP 1,559, CWQ 2,848**.
  - Với các câu còn lại, retriever không thể lấy được đáp án, nên **mọi metric retrieval chỉ tính trên subset**. LLM vẫn có thể trả lời đúng nhờ kiến thức sẵn có, nên QA vẫn chấm trên full.
  - QA trên subset cao hơn full khoảng 2 điểm (WebQSP) và 8–9 điểm (CWQ). Hai số này không được trộn lẫn khi so sánh.
- **Run chỉ có một phần câu** (A1: 1,000 câu đầu của CWQ; AER WebQSP: 1,559 câu; hiệu chuẩn T4: 300 câu) được chấm trên đúng các câu nó có. Cột "SubgraphRAG @100, cùng tập" chấm file chính thức trên **đúng tập id đó**, để vẫn so được.

### 0.3 Metric
- **Retrieval** (`benchmark_all.py`, cùng cách chấm với `gate_report.py`). Mọi dòng trong một setting được chấm trên **cùng tập câu**.
  - **ansR@k**: tỉ lệ đáp án gold xuất hiện làm head hoặc tail của k triple đầu, so khớp text. Đây là chỉ số số 1.
  - **SP-R / SP-P / SP-F1@k**: recall, precision, F1 của k triple so với triple trên đường ngắn nhất (nhãn SP của REL-RAG). P = số khớp / số triple đã lấy; F1 tính theo từng câu rồi lấy trung bình.
    - SP-P thấp là do cấu tạo: mỗi câu chỉ có vài triple SP. Ví dụ câu có 5 triple SP thì P@50 tối đa là 10%. Vì vậy chỉ nên so F1 giữa các phương pháp **ở cùng budget**.
  - **GPT-R@k**: recall của các triple bằng chứng do GPT-4o chọn, khớp chính xác (h, r, t). Chỉ số này đánh giá thấp RAPL/REL-RAG, vì hai phương pháp này hay trả triple dùng relation nghịch (EXP-36).
- **Budget:**
  - **top-k**: lấy k triple điểm cao nhất (@50, @100, @200, @500).
  - **dynamic τ** (chỉ RBF): giữ các triple có log-xác suất đường đi ≥ (điểm tốt nhất của câu − τ), tối đa 100 triple. **τ chọn trên WebQSP val** (τ = 6 là τ nhỏ nhất có val ansR vượt top-50: 92.9 so với 92.6). Bảng ghi số triple trung bình và trung vị.
- **QA** (`reason/metrics/paper_score.py`):
  - **Hit**: eval gốc của SubgraphRAG, đúng nếu có đáp án gold xuất hiện (khớp chuỗi con) trong response thô.
  - **Macro-F1**: eval corrected, trích các dòng `ans:` nhưng dùng **match gốc**.
  - ⚠️ Các file `*-eval_result_corrected*.txt` mà `reason/main.py` tự in ra dùng `match` bị nới thành "trùng một cụm 3 ký tự", nên **cao giả**. Ví dụ REL-RAG repro hiện 87.9 / 74.9 trong file `.txt`, còn số đúng là 84.95 / 70.15. **Không dùng các file `.txt` đó.**
  - Hit chỉ so được với paper khi lưu response thô (`--pred_format raw`). Các run đánh dấu "ans" chỉ lưu dòng đáp án, nên Hit là cận dưới. Các run "raw*" có response thô dựng lại từ log bằng `metrics/raw_from_log.py`, kiểm khớp 100%.
- **Reasoner:** luôn là Llama-3.1-8B-Instruct, prompt `sys_icl_dc` của SubgraphRAG, temperature 0. Khác nhau ở độ chính xác số và nơi chạy, có ghi trong từng dòng:
  - bf16: file chính thức của tác giả, và vLLM trên RTX 4090;
  - fp16: vLLM trên T4;
  - fp8: DeepInfra `Meta-Llama-3.1-8B-Instruct-Turbo`, gọi trực tiếp hoặc qua OpenRouter;
  - Q4: ollama.
  - So hai dòng khác reasoner thì lệch khoảng ±1 điểm là bình thường. Hiệu chuẩn: cùng retrieval SubgraphRAG @100, file bf16 cho 88.33 Hit, pipeline fp16 T4 cho 89.33 trên 300 câu.
- **Một seed (42)** cho mọi run trừ khi có ghi. Dao động giữa các seed đã đo, theo ansR@50:
  - DDE: tới 3.6 (WebQSP), 5.6 (CWQ transfer);
  - RBF+nb+mm: 0.8 (WebQSP), 2.7 (CWQ transfer).
  - Chênh lệch dưới khoảng 2 điểm giữa hai cấu hình chỉ chạy một seed thì chưa kết luận được.

---

## 1. Thẻ cấu hình từng phương pháp

| Phương pháp | Kiến trúc | Chọn relation / topic | Train trên, nhãn, loss | Lúc inference | Checkpoint / export |
|---|---|---|---|---|---|
| **SubgraphRAG (paper / official)** | MLP chấm từng triple + DDE (khoảng cách tới topic) | không có | WebQSP hoặc CWQ, nhãn SP, BCE | top-k (paper: @100) | `reason/scored_triples/*_unidir_test.pth` |
| **SubgraphRAG repro = DDE** | như trên, code của mình | không có | như trên, seed 42 | top-k | `retrieve/exp_qrdde/runs/{webqsp,cwq}_dde` |
| **REL-RAG (paper)** | GNN trên line graph, giải mã theo path (P) hoặc triple (T) | relation tương tự với triple khởi đầu (> τ) | nhãn rational path do GPT-4o chọn | P: lấy mẫu path tới đủ 50 triple | chỉ có số paper |
| **RAPL / REL-RAG (checkpoint tác giả)** | như paper | như paper | checkpoint `webqsp.pt` / `cwq.pt` của tác giả | decoder path_bf của mình (không dùng nhãn test) | `relrag/runs/authors_ckpt/` |
| **REL-RAG repro (P / T)** | code-faithful theo code tác giả (EXP-17) | như paper | WebQSP, nhãn GPT-4o + DeepSeek | P path_bf (đủ budget), P path_free (tự dừng, khoảng 43 triple), T top-k | `relrag/runs/{path,triple}_codefaithful` |
| **QR-DDE** | DDE lan truyền có điều kiện theo câu hỏi | không có | như DDE | top-k | `runs/{webqsp_qrdde,cwq_qrdde_resume}` |
| **AER** | neo lại ở cầu nối: B = 3 triple cầu, 2 lượt, gộp bằng max z-score | không có | "AER-trained" = DDE train với `reanchor_p` 0.5; "+ AER" trên backbone khác = chỉ bật lúc test | thêm lượt neo lại | `runs/webqsp_dde_aer`, `adr_*_aer.pth` |
| **ADR** | một lượt riêng cho từng topic entity, gộp max z-score | không có | không train (chỉ inference) | thêm lượt | `adr_*_adr.pth` |
| **ASE** | mã hoá tập anchor (anchor-set encoding) trên DDE hoặc WL | không có | WebQSP hoặc CWQ, BCE | top-k | `runs/{webqsp_ase_s42,cwq_ase_full_s42}` |
| **RT** (relation-topic) | DDE có mask và trọng số theo relation-topic | **GTE hybrid**: 0.5 × lexical (trùng token câu hỏi với tên relation) + 0.5 × cosine GTE-large-en-v1.5; giữ top-5 relation; `weighted_dde_floor` 0.1 | WebQSP (CWQ: dừng ở epoch 27, chưa hết patience), BCE | top-k | `retrieve/webqsp_rt_combined_Oct05-13:52:20` |
| **WL** | = RT + **weighted loss** | GTE hybrid như RT | WebQSP, BCE có trọng số effective-number (β = 0.999) | top-k | `retrieve/webqsp_wl_effective_num_Oct05-15:23:05` |
| **aug** | augmentation tập anchor | — | ⚠️ **dùng thống kê của CWQ train** để khớp phân phối, nên WebQSP → CWQ không còn OOD thuần | — | `runs/webqsp_wlq_*aug*` |
| **RBF+nb+mm** (phương pháp tổng quát) | Bellman-Ford max-product phân rã theo bước trên entity graph (T = 3); π(r\|q) = MLP[z_q; z_r; z_q⊙z_r; **lex; cos**]; **nb** = walk không quay lui (train và test); **mm** = max-marginal của đường đi trọn vẹn, chỉ dừng ở entity có tên (**chỉ bật lúc test**) | **lexical + cosine GTE làm đặc trưng, π học được**; **không dùng BGE** | WebQSP hoặc CWQ, nhãn SP, BCE, seed 42, chọn theo val triple_recall@100, patience 10 | top-k; hoặc dynamic τ | `retrieve/exp_qrdde/runs/{webqsp,cwq}_rbfnb_s42` |
| **RBF+nb+mm + BGE prior** | như trên, cộng β · z(cos BGE) vào logit của π **lúc test** (β = 0.5, chọn trên WebQSP val) | **BGE-M3 + LoRA** (công thức Phase 11, train lại trên vast chỉ với WebQSP train) | không train lại | top-k | `adr_*_base_mm_bge0.5.pth` (`bge_bonus_export.py`) |
| RBF+nb+mm + answer loss | + softmax mức đáp án (`--rbf_ans_loss`) | như RBF | WebQSP | top-k | **đã loại**: overfit WebQSP, CWQ −7.8 |

---

## 2. QA end-to-end (Llama-3.1-8B, prompt `sys_icl_dc`)

Cột "SubgraphRAG @100, cùng tập" là file chính thức (bf16) chấm trên đúng các câu của dòng đó: Hit / F1 trên full của dòng, rồi trên subset.

### 2.1 WebQSP test

| Retriever | Train → Test (loại) | Budget | Reasoner | Format | n full / subset | **Full: Hit / F1** | Subset: Hit / F1 | SubgraphRAG @100, cùng tập |
|---|---|---|---|---|---|---|---|---|
| *Paper SubgraphRAG* | WebQSP → WebQSP (ID) | @100 | bf16 | — | 1,628 | *86.6 / 70.5* | — | — |
| *Paper REL-RAG (P)* | WebQSP → WebQSP (ID) | @50 | bf16 | — | 1,628 | *88.8 / 77.3* | — | — |
| *Paper REL-RAG (P)* | WebQSP → WebQSP (ID) | @500 | bf16 | — | 1,628 | *92.1 / 74.9* | — | — |
| SubgraphRAG official | ID | @100 | bf16 (file tác giả) | raw | 1,628 / 1,559 | 86.61 / 70.57 | 88.58 / 72.10 | (chính nó) |
| SubgraphRAG official | ID | @500 | bf16 (file tác giả) | raw | 1,628 / 1,559 | 85.44 / 68.72 | 87.17 / 70.14 | 86.61 / 70.57 |
| SubgraphRAG official (hiệu chuẩn, EXP-25) | ID | @100 | fp16 T4 | raw | 300 / 286 | 89.33 / 70.60 | 91.61 / 72.37 | 88.33 / 70.76 |
| REL-RAG repro (P, path_bf) (EXP-25) | ID | @50 | fp16 T4 | raw | 1,628 / 1,559 | 84.95 / 70.15 | 86.59 / 71.41 | 86.61 / 70.57 |
| AER (EXP-25) | ID | @50 | fp16 T4 | raw | 1,559 / 1,559 | 86.40 / 69.83 | 86.40 / 69.83 | 88.58 / 72.10 |
| **RBF+nb+mm** | ID | **@50** | fp8 (OpenRouter → DeepInfra) | raw* | 1,628 / 1,559 | 82.74 / 68.65 | 84.48 / 70.20 | 86.61 / 70.57 |
| **RBF+nb+mm** | ID | **@100** | fp8 (OpenRouter → DeepInfra) | raw* | 1,628 / 1,559 | 84.28 / 69.41 | 86.08 / 70.89 | 86.61 / 70.57 |
| SubgraphRAG official (EXP-17) | ID | @100 | Q4 ollama | ans | 1,628 / 1,559 | 85.63 / 69.29 | 87.68 / 70.78 | 86.61 / 70.57 |
| SubgraphRAG official (EXP-17) | ID | @50 | Q4 ollama | ans | 1,608 / 1,539 | 84.27 / 69.10 | 86.22 / 70.55 | 86.63 / 70.55 |
| REL-RAG repro cũ (P, nhãn gemma, path_bf) (EXP-17) | ID | @50 | Q4 ollama | ans | 1,628 / 1,559 | 83.72 / 69.88 | 85.70 / 71.40 | 86.61 / 70.57 |
| REL-RAG repro (P, path_free) (EXP-17) | ID | khoảng 43 triple | Q4 ollama | ans | 1,628 / 1,559 | 83.54 / 68.88 | 85.18 / 70.13 | 86.61 / 70.57 |

- Chưa có run QA cho: RBF dynamic τ, RBF + BGE, WL, ASE, QR-DDE và các tổ hợp trên WebQSP. Lý do: retrieval in-dist không hơn baseline rõ ràng, nên gate không yêu cầu.
- Chênh lệch giữa RBF (fp8) và SubgraphRAG (bf16) có lẫn phần do reasoner. **Chưa có** baseline SubgraphRAG chạy qua đúng reasoner fp8 (khoảng $0.11).

### 2.2 CWQ test

| Retriever | Train → Test (loại) | Budget | Reasoner | Format | n full / subset | **Full: Hit / F1** | Subset: Hit / F1 | SubgraphRAG @100, cùng tập |
|---|---|---|---|---|---|---|---|---|
| *Paper SubgraphRAG* | CWQ → CWQ (ID) | @100 | bf16 | — | 3,531 | *56.9 / 47.2* | — | — |
| *Paper REL-RAG (P)* | CWQ → CWQ (ID) | @50 | bf16 | — | 3,531 | *67.2 / 56.8* | — | — |
| *Paper REL-RAG (P)* | WebQSP → CWQ (OOD) | @50 | bf16 | — | 3,531 | *64.1 / 53.6* | — | — |
| SubgraphRAG official | CWQ → CWQ (ID) | @100 | bf16 (file tác giả) | raw | 3,531 / 2,848 | 56.87 / 47.06 | 65.66 / 54.62 | (chính nó) |
| SubgraphRAG official | CWQ → CWQ (ID) | @500 | bf16 (file tác giả) | raw | 3,531 / 2,848 | 56.70 / 45.98 | 65.13 / 53.06 | 56.87 / 47.06 |
| **RBF+nb+mm (train CWQ)** | CWQ → CWQ (ID) | **@50** | fp8 (OpenRouter → DeepInfra) | raw* | 3,531 / 2,848 | 55.82 / 46.66 | 64.68 / 54.50 | 56.87 / 47.06 |
| **RBF+nb+mm (train CWQ)** | CWQ → CWQ (ID) | **@100** | fp8 (DeepInfra) | raw | 3,531 / 2,848 | **57.97 / 48.24** | **67.13 / 56.00** | 56.87 / 47.06 |
| RT Path+Memory hybrid (bản SubgraphRAG cũ, 09-06) | CWQ → CWQ (ID) | @100 | Q4 ollama | ans | 3,531 / 2,848 | 56.50 / 47.68 | 65.73 / 55.60 | 56.87 / 47.06 |
| WL + ADR + AER (EXP-30, A2) | WebQSP → CWQ (OOD) | @50 | bf16 vLLM 4090 | raw | 3,531 / 2,848 | 54.18 / 42.84 | 61.87 / 49.15 | 56.87 / 47.06 (ID) |
| **RBF+nb+mm** | WebQSP → CWQ (OOD) | **@50** | fp8 (DeepInfra) | raw | 3,531 / 2,848 | 52.53 / 44.59 | 60.67 / 51.63 | 56.87 / 47.06 (ID) |
| REL-RAG repro (P, path_bf) (EXP-26, A1) | WebQSP → CWQ (OOD) | @50 | bf16 vLLM 4090 | raw | 1,000 / 833 | 45.10 / 34.07 | 49.34 / 37.35 | 58.20 / 47.47 (ID) |
| DDE = SubgraphRAG repro (A1) | WebQSP → CWQ (OOD) | @50 | bf16 vLLM 4090 | raw | 1,000 / 833 | 48.50 / 35.69 | 53.54 / 38.91 | 58.20 / 47.47 (ID) |
| DDE + neo lại lúc test, không train (A1) | WebQSP → CWQ (OOD) | @50 | bf16 vLLM 4090 | raw | 1,000 / 833 | 48.90 / 37.27 | 53.78 / 41.19 | 58.20 / 47.47 (ID) |
| AER-trained (A1) | WebQSP → CWQ (OOD) | @50 | bf16 vLLM 4090 | raw | 1,000 / 833 | 49.60 / 36.90 | 54.74 / 40.98 | 58.20 / 47.47 (ID) |
| RAPL, code + prompt của tác giả (EXP-38) | WebQSP → CWQ (OOD) | khoảng 123 triple | fp8 (OpenRouter) | prompt tác giả | 3,531 | Hit 48.1 (phần trả lời) / F1 24.9 | — | — |

- Ở các dòng OOD, cột "SubgraphRAG @100, cùng tập" là trần **train trên CWQ (ID)**, không phải baseline OOD. Baseline OOD là DDE repro (A1).
- Trên 1,000 câu A1, bootstrap Hit của WL + ADR + AER cho thấy: hơn REL-RAG repro +6.70 [+3.20, +10.30], hơn DDE +3.30 [+0.10, +6.60], hơn AER +2.20 [−0.90, +5.40].
- RAPL (EXP-38) dùng prompt "The answer is" của tác giả. Nếu chấm Hit trên toàn bộ text lưu ra (có cả prompt) thì được 70.6, nhưng con số này vô nghĩa. Không so trực tiếp được với các dòng khác.

---

## 3. Retrieval: bảng đầy đủ theo setting

Mọi dòng trong cùng một bảng được chấm trên cùng tập câu. "+ AER / + ADR" trên một backbone nghĩa là bật thêm lúc test. Dòng "dynamic" là RBF với τ = 6, chọn trên **WebQSP val**, tối đa 100 triple; với các dòng này các cột @k không áp dụng, giá trị nằm ở cột @50.

**Tóm tắt các phương pháp chính** (ansR@50 / @100 / @200 · SP-F1@50 · GPT-R@50):

| Phương pháp | WebQSP (ID) | WebQSP → CWQ (OOD) | CWQ (ID) | CWQ → WebQSP (OOD) |
|---|---|---|---|---|
| SubgraphRAG (ID: official; OOD: repro) | 90.4 / 94.4 / 96.9 · 15.2 · 81.0 | 70.0 / 79.6 / 87.7 · 6.9 · 53.6 | 86.2 / 91.4 / 95.3 · 10.4 · 75.5 | 82.2 / 88.2 / 92.9 · 12.6 · 71.8 |
| REL-RAG, checkpoint tác giả (path_bf) | 89.9 / 95.2 / 97.5 · 16.2 · 64.0 | 66.8 / 74.1 / 80.4 · 7.4 · 39.4 | 80.8 / 85.9 / 89.8 · 9.0 · 45.1 | **90.9 / 95.8 / 97.7** · 16.3 · 62.3 |
| REL-RAG repro (P, path_bf) | 90.6 / 94.8 / 97.0 · 16.6 · 61.9 | 61.9 / 72.3 / 79.3 · 7.0 · 34.3 | — | — |
| QR-DDE | 89.2 / 94.0 / 97.2 · 16.3 · 82.2 | 61.0 / 71.3 / 81.3 · 7.0 · 53.1 | 89.7 / 93.2 / 96.6 · **12.5** · **79.6** | 81.0 / 87.2 / 91.9 · 14.1 · 74.5 |
| AER-trained + AER | 89.7 / 94.2 / 97.2 · 15.2 · 80.9 | 77.9 / 85.2 / 91.7 · 7.5 · 56.9 | — | — |
| DDE + AER + ADR | 89.5 / 93.9 / 96.9 · 14.5 · 80.6 | 78.6 / 85.8 / 91.8 · 7.8 · 59.2 | 87.1 / 91.8 / 95.3 · 10.3 · 72.9 | 80.6 / 87.4 / 92.7 · 12.0 · 69.9 |
| ASE (trên DDE) | 90.1 / 94.9 / 97.4 · 15.4 · 80.6 | 71.5 / 80.7 / 88.6 · 7.3 · 50.9 | 90.3 / 94.2 / 97.0 · 12.1 · 79.1 | 81.8 / 88.6 / 93.4 · 13.0 · 70.5 |
| WL | 92.6 / 96.1 / 98.2 · 15.7 · 84.5 | 74.0 / 83.6 / 89.9 · 7.2 · 57.6 | — (chỉ có RT, dừng sớm: 86.2) | — (RT: 85.0) |
| WL + ADR + AER | 91.9 / 96.0 / 98.1 · 15.3 · 84.6 | 83.2 / 89.8 / 94.7 · 8.5 · 65.6 | — | — |
| WL + ASE | **92.8** / 96.1 / 98.2 · 16.3 · 84.0 | 67.9 / 78.3 / 86.5 · 7.5 · 54.9 | — | — |
| WL + ASE + aug + AER (*) | 92.3 / 96.0 / 97.6 · 15.4 · 84.7 | 87.4 / 92.4 / 95.2 · **9.3** · **71.7** | — | — |
| **RBF+nb+mm** | 91.4 / 95.2 / 98.0 · 15.4 · 83.1 | 87.1 / 92.1 / 96.0 · 8.7 · 66.8 | **90.5 / 94.1 / 96.8** · 10.0 · 73.4 | 87.2 / 90.9 / 95.0 · 14.4 · **80.5** |
| **RBF+nb+mm, dynamic τ = 6** | 94.5 với 59 triple · **22.4** · 84.7 | 90.2 với 81 triple · 8.8 · 73.0 | — (τ chưa chọn trên CWQ val) | — |
| **RBF+nb+mm + BGE (β 0.5)** | 92.0 / 95.7 / 97.9 · 15.3 · 84.5 | **88.7 / 93.5 / 96.4** · 8.8 · 70.1 | — | — |

(*) aug dùng thống kê CWQ train.

- **Gate "ansR@50 của mình > ansR@200 của SubgraphRAG":**
  - **Đạt** ở WebQSP → CWQ với RBF+nb+mm + BGE (88.7 > 87.7).
  - RBF+nb+mm (87.1) và WL+ASE+aug+AER (87.4) sát mốc nhưng chưa đạt.
  - WebQSP ID và CWQ ID chưa đạt (cần lần lượt 96.9 và 95.3).
- **Budget động trên WebQSP** đạt ansR 94.5 với trung bình 59 triple, SP-F1 22.4, so với top-50 là 91.4 và F1 15.4.
- **Budget động trên CWQ OOD** dùng cùng τ nhưng phải giữ trung bình 81 triple mới đạt 90.2, trong khi top-100 đạt 92.1. Ở đây cắt động không hiệu quả hơn top-k.

### 3.1 WebQSP: train WebQSP → test WebQSP (ID) (1559 câu)

| Method | Budget | ansR@50 | ansR@100 | ansR@200 | ansR@500 | SP-R@50 | SP-P@50 | SP-F1@50 | SP-R@100 | SP-F1@100 | GPT-R@50 | GPT-R@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SubgraphRAG (official ckpt) | top-k | 90.4 | 94.4 | 96.9 | 98.9 | 82.3 | 11.2 | 15.2 | 87.8 | 10.8 | 81.0 | 87.4 |
| RAPL / REL-RAG (author ckpt webqsp.pt, path_bf) | top-k | 89.9 | 95.2 | 97.5 | 99.0 | 83.7 | 12.6 | 16.2 | 89.3 | 11.3 | 64.0 | 67.9 |
| REL-RAG repro (P, path_bf) | top-k | 90.6 | 94.8 | 97.0 | 98.8 | 85.8 | 13.0 | 16.6 | 90.8 | 11.7 | 61.9 | 65.5 |
| REL-RAG repro (P, path_free, ~43 triples) | top-k | 84.9 | 88.8 | 90.2 | 90.9 | 80.5 | 40.4 | 45.4 | 84.0 | 45.9 | 54.0 | 55.7 |
| REL-RAG repro (T, triple) | top-k | 75.6 | 84.8 | 91.5 | 96.3 | 73.9 | 10.5 | 13.6 | 83.5 | 10.2 | 49.7 | 56.4 |
| RBF+nb+mm | top-k | 91.4 | 95.2 | 98.0 | 99.1 | 81.6 | 11.4 | 15.4 | 86.6 | 10.7 | 83.1 | 87.8 |
| RBF+nb+mm, dynamic tau=6 cap 100 | dyn, mean 59 / median 59 triple | 94.5 (dyn) | | | | 85.0 | 15.5 | 22.4 | | | 84.7 | |
| RBF+nb+mm + BGE prior (beta 0.5) | top-k | 92.0 | 95.7 | 97.9 | 99.2 | 82.2 | 11.3 | 15.3 | 87.4 | 10.7 | 84.5 | 88.8 |
| RBF+nb+mm, seed 0 | top-k | 90.6 | 95.3 | 97.4 | 99.1 | 81.1 | 10.8 | 14.7 | 87.0 | 10.6 | 81.5 | 87.3 |
| RBF+nb+mm + answer loss (rejected) | top-k | 93.0 | 96.8 | 98.6 | 99.4 | 80.1 | 10.2 | 14.2 | 85.9 | 10.1 | 83.4 | 89.2 |
| DDE (SubgraphRAG repro) | top-k | 90.1 | 94.4 | 97.2 | 98.9 | 81.9 | 10.9 | 14.8 | 87.1 | 10.5 | 81.0 | 86.9 |
| DDE (SubgraphRAG repro) + AER | top-k | 89.7 | 94.1 | 97.0 | 98.8 | 81.2 | 10.5 | 14.5 | 86.8 | 10.4 | 80.7 | 87.0 |
| DDE (SubgraphRAG repro) + ADR | top-k | 89.8 | 94.3 | 97.0 | 98.8 | 81.9 | 10.8 | 14.8 | 87.1 | 10.5 | 81.0 | 87.0 |
| DDE (SubgraphRAG repro) + AER + ADR | top-k | 89.5 | 93.9 | 96.9 | 98.8 | 81.2 | 10.5 | 14.5 | 86.8 | 10.4 | 80.6 | 87.0 |
| QR-DDE | top-k | 89.2 | 94.0 | 97.2 | 99.1 | 84.6 | 12.2 | 16.3 | 90.0 | 11.6 | 82.2 | 87.2 |
| QR-DDE + AER | top-k | 88.7 | 93.9 | 97.1 | 99.0 | 83.8 | 12.1 | 16.2 | 89.4 | 11.5 | 81.9 | 87.2 |
| QR-DDE + ADR | top-k | 89.4 | 94.0 | 97.2 | 99.0 | 84.7 | 12.2 | 16.4 | 90.0 | 11.6 | 82.4 | 87.5 |
| QR-DDE + AER + ADR | top-k | 89.0 | 93.9 | 97.1 | 98.9 | 83.9 | 12.2 | 16.2 | 89.4 | 11.5 | 82.2 | 87.5 |
| DDE, AER-trained (reanchor_p 0.5) | top-k | 90.1 | 94.2 | 97.2 | 98.9 | 81.4 | 11.4 | 15.2 | 87.0 | 10.8 | 79.8 | 85.9 |
| DDE, AER-trained (reanchor_p 0.5) + AER | top-k | 89.7 | 94.2 | 97.2 | 98.9 | 81.2 | 11.4 | 15.2 | 86.8 | 10.8 | 80.9 | 86.8 |
| DDE, AER-trained (reanchor_p 0.5) + ADR | top-k | 89.8 | 94.1 | 97.2 | 98.9 | 81.4 | 11.4 | 15.2 | 87.0 | 10.8 | 79.6 | 86.0 |
| DDE, AER-trained (reanchor_p 0.5) + AER + ADR | top-k | 89.4 | 93.8 | 97.0 | 98.8 | 81.2 | 11.3 | 15.1 | 86.8 | 10.8 | 80.8 | 86.7 |
| RT (relation-topic, no weighted loss) | top-k | 91.8 | 95.7 | 97.9 | 99.1 | 83.3 | 12.1 | 15.9 | 88.4 | 11.1 | 82.6 | 87.9 |
| RT (relation-topic, no weighted loss) + AER | top-k | 91.5 | 95.7 | 97.9 | 99.1 | 82.4 | 11.8 | 15.5 | 88.1 | 11.0 | 82.3 | 88.3 |
| RT (relation-topic, no weighted loss) + ADR | top-k | 91.6 | 95.5 | 97.7 | 99.2 | 83.3 | 12.1 | 15.9 | 88.4 | 11.1 | 82.6 | 88.0 |
| RT (relation-topic, no weighted loss) + AER + ADR | top-k | 91.2 | 95.5 | 97.9 | 99.1 | 82.4 | 11.8 | 15.5 | 88.0 | 11.0 | 82.4 | 88.3 |
| WL (RT + weighted loss) | top-k | 92.6 | 96.1 | 98.2 | 99.3 | 84.7 | 11.7 | 15.7 | 90.0 | 11.2 | 84.5 | 89.7 |
| WL (RT + weighted loss) + AER | top-k | 92.3 | 96.3 | 98.2 | 99.3 | 84.0 | 11.4 | 15.4 | 89.7 | 11.0 | 84.5 | 89.7 |
| WL (RT + weighted loss) + ADR | top-k | 92.3 | 95.9 | 98.1 | 99.2 | 84.5 | 11.7 | 15.7 | 89.8 | 11.1 | 84.4 | 89.6 |
| WL (RT + weighted loss) + AER + ADR | top-k | 91.9 | 96.0 | 98.1 | 99.2 | 83.8 | 11.4 | 15.3 | 89.5 | 10.9 | 84.6 | 89.6 |
| ASE (on DDE) | top-k | 90.1 | 94.9 | 97.4 | 99.2 | 83.0 | 11.4 | 15.4 | 89.4 | 11.1 | 80.6 | 86.9 |
| ASE (on DDE) + AER | top-k | 89.2 | 94.3 | 97.2 | 99.1 | 82.3 | 11.2 | 15.1 | 88.5 | 10.9 | 81.5 | 87.3 |
| ASE (on DDE) + ADR | top-k | 89.7 | 94.7 | 97.5 | 99.2 | 82.7 | 11.3 | 15.3 | 89.1 | 11.1 | 80.3 | 86.8 |
| ASE (on DDE) + AER + ADR | top-k | 88.9 | 94.0 | 97.2 | 99.1 | 82.0 | 11.1 | 15.0 | 88.2 | 10.8 | 81.3 | 87.2 |
| WL + ASE | top-k | 92.8 | 96.1 | 98.2 | 99.3 | 85.6 | 12.4 | 16.3 | 90.9 | 11.5 | 84.0 | 88.9 |
| WL + ASE + AER | top-k | 92.5 | 96.1 | 98.0 | 99.2 | 85.2 | 12.2 | 16.0 | 90.6 | 11.4 | 84.3 | 89.5 |
| WL + ASE + ADR | top-k | 92.4 | 96.1 | 98.1 | 99.2 | 85.3 | 12.4 | 16.3 | 90.7 | 11.5 | 83.5 | 88.7 |
| WL + ASE + AER + ADR | top-k | 91.9 | 95.5 | 97.9 | 99.2 | 84.7 | 12.1 | 16.0 | 90.2 | 11.3 | 83.8 | 89.2 |
| WL + aug | top-k | 91.9 | 96.0 | 98.0 | 99.3 | 84.9 | 12.1 | 16.0 | 90.3 | 11.3 | 84.8 | 89.2 |
| WL + aug + AER | top-k | 90.6 | 95.3 | 98.0 | 99.2 | 83.6 | 11.6 | 15.4 | 89.6 | 11.1 | 84.7 | 89.3 |
| WL + aug + ADR | top-k | 91.7 | 95.8 | 98.0 | 99.3 | 84.8 | 12.1 | 15.9 | 90.2 | 11.3 | 84.6 | 89.2 |
| WL + aug + AER + ADR | top-k | 90.5 | 95.0 | 98.0 | 99.2 | 83.6 | 11.6 | 15.4 | 89.5 | 11.1 | 84.6 | 89.3 |
| WL + ASE + aug | top-k | 92.7 | 96.2 | 97.8 | 99.2 | 84.6 | 11.6 | 15.6 | 89.5 | 11.1 | 82.7 | 87.4 |
| WL + ASE + aug + AER | top-k | 92.3 | 96.0 | 97.6 | 99.2 | 84.5 | 11.4 | 15.4 | 89.4 | 11.0 | 84.7 | 89.4 |
| WL + ASE + aug + ADR | top-k | 92.3 | 96.1 | 97.8 | 99.2 | 84.5 | 11.6 | 15.6 | 89.3 | 11.1 | 82.5 | 87.4 |
| WL + ASE + aug + AER + ADR | top-k | 92.2 | 95.9 | 97.6 | 99.2 | 84.4 | 11.4 | 15.4 | 89.3 | 11.0 | 84.7 | 89.5 |

### 3.2 WebQSP → CWQ: train WebQSP → test CWQ (OOD) (2848 câu)

| Method | Budget | ansR@50 | ansR@100 | ansR@200 | ansR@500 | SP-R@50 | SP-P@50 | SP-F1@50 | SP-R@100 | SP-F1@100 | GPT-R@50 | GPT-R@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SubgraphRAG (DDE repro, train WebQSP) | top-k | 70.0 | 79.6 | 87.7 | 94.8 | 49.8 | 4.3 | 6.9 | 58.6 | 5.0 | 53.6 | 63.1 |
| RAPL / REL-RAG (author ckpt webqsp.pt, path_bf) | top-k | 66.8 | 74.1 | 80.4 | 85.1 | 54.4 | 4.6 | 7.4 | 60.7 | 4.9 | 39.4 | 44.6 |
| REL-RAG repro (P, path_bf) | top-k | 61.9 | 72.3 | 79.3 | 85.2 | 50.8 | 4.3 | 7.0 | 58.9 | 4.8 | 34.3 | 40.8 |
| REL-RAG repro (P, path_free, ~43 triples) | top-k | 51.2 | 60.2 | 64.9 | 66.3 | 42.6 | 10.4 | 12.6 | 49.0 | 12.4 | 26.5 | 30.5 |
| REL-RAG repro (T, triple) | top-k | 47.8 | 60.4 | 74.1 | 90.4 | 32.1 | 3.0 | 4.6 | 40.6 | 3.5 | 21.4 | 27.5 |
| RBF+nb (no mm) | top-k | 81.8 | 89.8 | 94.6 | 98.4 | 57.1 | 5.2 | 8.5 | 65.7 | 5.8 | 65.2 | 74.2 |
| RBF+nb+mm | top-k | 87.1 | 92.1 | 96.0 | 98.8 | 58.4 | 5.3 | 8.7 | 66.9 | 5.8 | 66.8 | 75.7 |
| RBF+nb+mm, dynamic tau=6 cap 100 | dyn, mean 81 / median 100 triple | 90.2 (dyn) | | | | 64.2 | 5.3 | 8.8 | | | 73.0 | |
| RBF+nb+mm + BGE prior (beta 0.5) | top-k | 88.7 | 93.5 | 96.4 | 98.9 | 59.8 | 5.4 | 8.8 | 67.7 | 5.8 | 70.1 | 78.1 |
| RBF+nb+mm, seed 0 | top-k | 84.4 | 90.4 | 95.5 | 99.0 | 56.4 | 5.0 | 8.3 | 64.4 | 5.6 | 65.8 | 74.5 |
| RBF+nb+mm + answer loss (rejected) | top-k | 79.3 | 89.0 | 94.5 | 98.5 | 54.2 | 4.6 | 7.6 | 61.7 | 5.0 | 64.4 | 73.0 |
| DDE (SubgraphRAG repro) | top-k | 70.0 | 79.6 | 87.7 | 94.8 | 49.8 | 4.3 | 6.9 | 58.6 | 5.0 | 53.6 | 63.1 |
| DDE (SubgraphRAG repro) + AER | top-k | 75.8 | 84.8 | 91.1 | 96.0 | 51.7 | 4.5 | 7.2 | 61.2 | 5.2 | 55.7 | 65.6 |
| DDE (SubgraphRAG repro) + ADR | top-k | 72.5 | 80.5 | 88.3 | 94.8 | 51.6 | 4.5 | 7.2 | 60.4 | 5.1 | 56.9 | 66.0 |
| DDE (SubgraphRAG repro) + AER + ADR | top-k | 78.6 | 85.8 | 91.8 | 96.0 | 54.0 | 4.9 | 7.8 | 63.9 | 5.6 | 59.2 | 68.7 |
| QR-DDE | top-k | 61.0 | 71.3 | 81.3 | 91.7 | 49.5 | 4.4 | 7.0 | 57.5 | 4.9 | 53.1 | 62.1 |
| QR-DDE + AER | top-k | 70.5 | 81.7 | 89.7 | 96.1 | 55.0 | 4.8 | 7.9 | 64.1 | 5.5 | 59.1 | 69.4 |
| QR-DDE + ADR | top-k | 64.9 | 74.7 | 82.7 | 91.9 | 51.3 | 4.6 | 7.3 | 59.1 | 5.1 | 56.1 | 64.9 |
| QR-DDE + AER + ADR | top-k | 75.8 | 85.7 | 91.7 | 96.4 | 58.3 | 5.3 | 8.6 | 67.0 | 5.9 | 63.9 | 73.2 |
| DDE, AER-trained (reanchor_p 0.5) | top-k | 74.8 | 82.4 | 89.9 | 95.8 | 50.9 | 4.5 | 7.3 | 60.8 | 5.2 | 54.4 | 64.5 |
| DDE, AER-trained (reanchor_p 0.5) + AER | top-k | 77.9 | 85.2 | 91.7 | 96.8 | 52.0 | 4.7 | 7.5 | 61.7 | 5.4 | 56.9 | 66.3 |
| DDE, AER-trained (reanchor_p 0.5) + ADR | top-k | 76.7 | 84.0 | 90.4 | 95.5 | 52.2 | 4.7 | 7.5 | 61.4 | 5.3 | 57.5 | 66.8 |
| DDE, AER-trained (reanchor_p 0.5) + AER + ADR | top-k | 79.0 | 86.4 | 91.8 | 96.4 | 53.8 | 5.0 | 8.0 | 63.0 | 5.6 | 59.6 | 69.1 |
| RT (relation-topic, no weighted loss) | top-k | 72.4 | 80.3 | 88.0 | 95.5 | 51.1 | 4.5 | 7.3 | 60.3 | 5.2 | 54.7 | 63.6 |
| RT (relation-topic, no weighted loss) + AER | top-k | 80.4 | 87.1 | 93.0 | 97.5 | 54.0 | 4.8 | 7.8 | 63.9 | 5.6 | 59.6 | 68.8 |
| RT (relation-topic, no weighted loss) + ADR | top-k | 74.1 | 82.5 | 88.9 | 95.6 | 53.1 | 4.8 | 7.6 | 62.2 | 5.4 | 58.5 | 67.2 |
| RT (relation-topic, no weighted loss) + AER + ADR | top-k | 82.4 | 89.2 | 93.9 | 97.7 | 57.1 | 5.4 | 8.6 | 66.6 | 6.1 | 63.5 | 72.5 |
| WL (RT + weighted loss) | top-k | 74.0 | 83.6 | 89.9 | 96.3 | 50.9 | 4.5 | 7.2 | 60.6 | 5.3 | 57.6 | 67.6 |
| WL (RT + weighted loss) + AER | top-k | 80.3 | 88.5 | 93.9 | 97.8 | 53.4 | 4.8 | 7.7 | 63.1 | 5.5 | 61.7 | 71.9 |
| WL (RT + weighted loss) + ADR | top-k | 77.4 | 85.4 | 90.8 | 96.0 | 53.1 | 4.7 | 7.6 | 62.7 | 5.4 | 61.1 | 70.6 |
| WL (RT + weighted loss) + AER + ADR | top-k | 83.2 | 89.8 | 94.7 | 98.0 | 56.8 | 5.3 | 8.5 | 66.6 | 6.1 | 65.6 | 74.9 |
| ASE (on DDE) | top-k | 71.5 | 80.7 | 88.6 | 95.7 | 51.0 | 4.5 | 7.3 | 60.3 | 5.2 | 50.9 | 60.7 |
| ASE (on DDE) + AER | top-k | 74.9 | 84.6 | 91.6 | 96.9 | 53.5 | 4.8 | 7.7 | 62.8 | 5.4 | 54.0 | 64.0 |
| ASE (on DDE) + ADR | top-k | 76.3 | 83.7 | 91.5 | 96.7 | 53.7 | 4.9 | 7.8 | 63.2 | 5.6 | 58.1 | 67.1 |
| ASE (on DDE) + AER + ADR | top-k | 80.7 | 87.7 | 94.3 | 97.5 | 56.7 | 5.4 | 8.5 | 66.3 | 6.0 | 61.3 | 70.5 |
| WL + ASE | top-k | 67.9 | 78.3 | 86.5 | 95.3 | 51.7 | 4.8 | 7.5 | 60.6 | 5.4 | 54.9 | 63.2 |
| WL + ASE + AER | top-k | 77.2 | 85.9 | 92.5 | 97.9 | 57.3 | 5.2 | 8.3 | 67.1 | 5.9 | 62.0 | 70.9 |
| WL + ASE + ADR | top-k | 74.5 | 82.7 | 89.3 | 95.5 | 54.4 | 4.9 | 7.8 | 63.9 | 5.5 | 60.1 | 68.3 |
| WL + ASE + AER + ADR | top-k | 82.5 | 89.7 | 94.7 | 98.1 | 60.1 | 5.4 | 8.8 | 69.6 | 6.1 | 66.9 | 75.4 |
| WL + aug | top-k | 76.9 | 85.2 | 91.7 | 97.1 | 51.9 | 4.5 | 7.3 | 62.4 | 5.4 | 57.9 | 68.3 |
| WL + aug + AER | top-k | 79.7 | 88.7 | 93.8 | 97.9 | 54.7 | 4.7 | 7.7 | 64.8 | 5.6 | 61.4 | 71.9 |
| WL + aug + ADR | top-k | 77.7 | 85.7 | 91.7 | 96.6 | 52.6 | 4.5 | 7.4 | 63.3 | 5.4 | 59.4 | 69.9 |
| WL + aug + AER + ADR | top-k | 80.5 | 89.2 | 93.9 | 97.4 | 56.3 | 4.9 | 8.1 | 66.6 | 5.8 | 64.5 | 74.3 |
| WL + ASE + aug | top-k | 80.6 | 86.6 | 91.3 | 96.3 | 58.5 | 5.3 | 8.4 | 66.6 | 5.9 | 64.8 | 72.7 |
| WL + ASE + aug + AER | top-k | 87.4 | 92.4 | 95.2 | 98.4 | 64.2 | 5.8 | 9.3 | 72.5 | 6.4 | 71.7 | 79.6 |
| WL + ASE + aug + ADR | top-k | 79.5 | 86.7 | 91.9 | 96.4 | 54.6 | 4.7 | 7.6 | 64.2 | 5.5 | 62.0 | 71.4 |
| WL + ASE + aug + AER + ADR | top-k | 86.5 | 92.6 | 95.8 | 98.6 | 63.0 | 5.6 | 9.1 | 71.8 | 6.3 | 70.8 | 79.0 |

### 3.3 CWQ: train CWQ → test CWQ (ID) (2848 câu)

| Method | Budget | ansR@50 | ansR@100 | ansR@200 | ansR@500 | SP-R@50 | SP-P@50 | SP-F1@50 | SP-R@100 | SP-F1@100 | GPT-R@50 | GPT-R@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SubgraphRAG (official ckpt) | top-k | 86.2 | 91.4 | 95.3 | 98.4 | 66.3 | 6.8 | 10.4 | 74.5 | 7.2 | 75.5 | 84.4 |
| RAPL / REL-RAG (author ckpt cwq.pt, path_bf) | top-k | 80.8 | 85.9 | 89.8 | 92.6 | 63.5 | 5.5 | 9.0 | 69.5 | 5.8 | 45.1 | 49.3 |
| RBF+nb+mm (train CWQ) | top-k | 90.5 | 94.1 | 96.8 | 98.4 | 67.4 | 6.2 | 10.0 | 73.8 | 6.5 | 73.4 | 81.6 |
| DDE (SubgraphRAG repro) | top-k | 86.7 | 91.3 | 95.1 | 98.2 | 66.8 | 6.7 | 10.5 | 74.0 | 7.1 | 72.9 | 80.9 |
| DDE (SubgraphRAG repro) + AER | top-k | 86.4 | 90.8 | 94.5 | 98.2 | 65.1 | 6.6 | 10.2 | 73.1 | 7.0 | 71.0 | 80.2 |
| DDE (SubgraphRAG repro) + ADR | top-k | 87.6 | 91.9 | 95.3 | 98.3 | 66.4 | 6.5 | 10.2 | 74.0 | 6.9 | 73.2 | 81.0 |
| DDE (SubgraphRAG repro) + AER + ADR | top-k | 87.1 | 91.8 | 95.3 | 98.2 | 66.3 | 6.5 | 10.3 | 74.0 | 7.0 | 72.9 | 81.5 |
| QR-DDE | top-k | 89.7 | 93.2 | 96.6 | 98.5 | 75.7 | 8.1 | 12.5 | 80.9 | 8.1 | 79.6 | 85.2 |
| QR-DDE + AER | top-k | 89.7 | 93.6 | 96.5 | 98.6 | 75.0 | 8.0 | 12.3 | 80.7 | 8.0 | 79.5 | 85.5 |
| QR-DDE + ADR | top-k | 90.9 | 94.3 | 97.1 | 98.7 | 74.5 | 7.5 | 11.9 | 80.4 | 7.8 | 79.4 | 85.4 |
| QR-DDE + AER + ADR | top-k | 90.6 | 94.4 | 97.1 | 98.8 | 74.4 | 7.6 | 11.9 | 80.6 | 7.9 | 79.1 | 85.6 |
| RT (relation-topic, no weighted loss, stopped ep 27) | top-k | 86.2 | 92.0 | 95.6 | 98.1 | 64.0 | 6.4 | 9.9 | 72.9 | 6.9 | 72.3 | 81.3 |
| RT (relation-topic, no weighted loss, stopped ep 27) + AER | top-k | 85.6 | 91.9 | 95.2 | 98.0 | 62.6 | 6.3 | 9.7 | 71.9 | 6.8 | 70.8 | 80.0 |
| RT (relation-topic, no weighted loss, stopped ep 27) + ADR | top-k | 87.2 | 92.4 | 95.4 | 98.2 | 63.6 | 5.9 | 9.5 | 72.3 | 6.5 | 72.7 | 81.2 |
| RT (relation-topic, no weighted loss, stopped ep 27) + AER + ADR | top-k | 87.0 | 92.5 | 95.4 | 98.3 | 64.4 | 6.3 | 10.0 | 73.5 | 6.9 | 72.7 | 81.4 |
| ASE (on DDE) | top-k | 90.3 | 94.2 | 97.0 | 98.7 | 75.1 | 7.8 | 12.1 | 80.9 | 8.0 | 79.1 | 84.9 |
| ASE (on DDE) + AER | top-k | 89.4 | 93.7 | 96.6 | 98.4 | 74.8 | 8.0 | 12.2 | 81.0 | 8.1 | 77.8 | 84.2 |
| ASE (on DDE) + ADR | top-k | 89.3 | 93.6 | 96.5 | 98.7 | 71.4 | 6.7 | 10.9 | 78.3 | 7.2 | 77.5 | 84.1 |
| ASE (on DDE) + AER + ADR | top-k | 88.3 | 92.9 | 96.2 | 98.5 | 71.1 | 6.9 | 11.0 | 78.8 | 7.4 | 76.4 | 83.5 |

### 3.4 CWQ → WebQSP: train CWQ → test WebQSP (OOD) (1559 câu)

| Method | Budget | ansR@50 | ansR@100 | ansR@200 | ansR@500 | SP-R@50 | SP-P@50 | SP-F1@50 | SP-R@100 | SP-F1@100 | GPT-R@50 | GPT-R@100 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| SubgraphRAG (DDE repro, train CWQ) | top-k | 82.2 | 88.2 | 92.9 | 97.1 | 71.7 | 9.2 | 12.6 | 79.8 | 9.2 | 71.8 | 79.7 |
| RAPL / REL-RAG (author ckpt cwq.pt, path_bf) | top-k | 90.9 | 95.8 | 97.7 | 99.1 | 83.7 | 12.6 | 16.3 | 89.8 | 11.5 | 62.3 | 65.6 |
| RBF+nb+mm (train CWQ) | top-k | 87.2 | 90.9 | 95.0 | 97.1 | 79.8 | 10.5 | 14.4 | 84.7 | 10.0 | 80.5 | 84.8 |
| DDE (SubgraphRAG repro) | top-k | 82.2 | 88.2 | 92.9 | 97.1 | 71.7 | 9.2 | 12.6 | 79.8 | 9.2 | 71.8 | 79.7 |
| DDE (SubgraphRAG repro) + AER | top-k | 80.7 | 87.2 | 92.5 | 97.0 | 69.0 | 8.7 | 11.9 | 78.2 | 8.9 | 69.5 | 78.7 |
| DDE (SubgraphRAG repro) + ADR | top-k | 82.1 | 88.2 | 93.0 | 97.2 | 71.9 | 9.3 | 12.7 | 80.1 | 9.3 | 72.1 | 80.0 |
| DDE (SubgraphRAG repro) + AER + ADR | top-k | 80.6 | 87.4 | 92.7 | 97.2 | 69.3 | 8.8 | 12.0 | 78.6 | 9.0 | 69.9 | 79.1 |
| QR-DDE | top-k | 81.0 | 87.2 | 91.9 | 96.3 | 76.3 | 10.3 | 14.1 | 83.0 | 10.1 | 74.5 | 82.0 |
| QR-DDE + AER | top-k | 79.9 | 86.3 | 91.3 | 96.1 | 74.2 | 9.9 | 13.6 | 81.9 | 9.9 | 73.0 | 81.3 |
| QR-DDE + ADR | top-k | 81.4 | 87.3 | 91.9 | 96.3 | 76.6 | 10.3 | 14.2 | 83.3 | 10.1 | 74.7 | 82.2 |
| QR-DDE + AER + ADR | top-k | 80.2 | 86.7 | 91.3 | 96.1 | 74.6 | 10.0 | 13.7 | 82.2 | 10.0 | 73.4 | 81.5 |
| RT (relation-topic, no weighted loss, stopped ep 27) | top-k | 85.0 | 90.8 | 95.0 | 97.6 | 72.8 | 9.4 | 12.9 | 80.4 | 9.6 | 72.8 | 80.5 |
| RT (relation-topic, no weighted loss, stopped ep 27) + AER | top-k | 83.6 | 90.2 | 94.8 | 97.5 | 70.0 | 8.9 | 12.3 | 79.6 | 9.3 | 70.8 | 79.6 |
| RT (relation-topic, no weighted loss, stopped ep 27) + ADR | top-k | 85.1 | 90.8 | 95.2 | 97.8 | 73.1 | 9.4 | 13.0 | 80.7 | 9.6 | 73.2 | 80.8 |
| RT (relation-topic, no weighted loss, stopped ep 27) + AER + ADR | top-k | 83.6 | 90.3 | 95.0 | 97.6 | 70.4 | 9.0 | 12.4 | 79.9 | 9.3 | 71.2 | 79.9 |
| ASE (on DDE) | top-k | 81.8 | 88.6 | 93.4 | 97.6 | 71.8 | 9.7 | 13.0 | 80.3 | 9.7 | 70.5 | 78.6 |
| ASE (on DDE) + AER | top-k | 77.9 | 87.1 | 92.0 | 97.2 | 69.0 | 9.4 | 12.7 | 79.3 | 9.6 | 67.5 | 77.6 |
| ASE (on DDE) + ADR | top-k | 81.7 | 88.5 | 93.4 | 97.5 | 71.7 | 9.7 | 13.0 | 80.2 | 9.7 | 70.4 | 78.5 |
| ASE (on DDE) + AER + ADR | top-k | 77.9 | 86.9 | 92.0 | 97.1 | 68.9 | 9.4 | 12.7 | 79.1 | 9.6 | 67.3 | 77.6 |

---

## 4. Dataset ngoài (OOD, zero-shot), ansR@50 / @100 / @200

### 4.1 GrailQA, tập theo setting paper REL-RAG (728 câu; EXP-39)

| Retriever | Train WebQSP | Train CWQ |
|---|---|---|
| SubgraphRAG (repro DDE) | 40.3 / 48.3 / 54.4 | 41.3 / 50.1 / 58.0 |
| REL-RAG (WebQSP: repro path_bf; CWQ: checkpoint tác giả) | 42.7 / 49.5 / 54.7 | 37.3 / 44.3 / 52.0 |
| WL | 51.9 / 57.9 / 62.6 | — |
| WL + ADR + AER | 54.7 / 62.9 / 67.7 | — |
| RBF+nb (không mm) | 58.8 / 67.1 / 73.2 | — |
| **RBF+nb+mm** | **58.9 / 67.1 / 73.1** | 48.4 / 54.8 / 60.8 |

- Bootstrap, RBF+nb so với SubgraphRAG: +18.5 / +18.8 / +18.7; so với REL-RAG: +16.1 / +17.7 / +18.4; so với WL+ADR+AER: +4.1 / +4.3 / +5.4. Mọi CI > 0.
- Trên GrailQA, nb có ích thật: hơn RBF không nb +2.9 / +3.9 / +4.3, CI > 0.

### 4.2 GTSQA (Wikidata, 1,622 câu; EXP-24, EXP-39)

| Retriever | Train WebQSP | Train CWQ |
|---|---|---|
| SubgraphRAG (repro DDE) | 35.4 / 41.2 / 47.9 | 26.1 / 32.1 / 39.8 |
| REL-RAG (WebQSP: repro path_bf; CWQ: checkpoint tác giả) | 38.8 / 46.0 / 50.8 | 37.2 / 45.1 / 49.8 |
| DDE + ADR + AER (suy từ chênh lệch bootstrap so với RBF) | 38.7 / 48.4 / 59.0 | — |
| WL | 40.9 / 46.0 / 52.8 | — |
| WL + ADR + AER | 48.6 / 56.9 / 64.9 | — |
| **RBF+nb** (mm không đổi gì: KG không có CVT) | **57.7 / 64.9 / 71.9** | 52.4 / 58.4 / 64.8 |

### 4.3 MetaQA-3hop (14,118 câu; EXP-39)

| Retriever | Train WebQSP | Train CWQ |
|---|---|---|
| SubgraphRAG (repro DDE) | 21.5 / 33.7 / 51.8 | 33.1 / 44.9 / 61.4 |
| REL-RAG (path_bf tự dừng, trung bình 9.6 triple/câu; chỉ 4,710 câu) | 14.4 / 14.4 / 14.4 | 14.4 / 14.4 / 14.4 |
| WL (suy từ chênh lệch bootstrap) | 23.7 / 33.1 / 48.4 | — |
| WL + ADR + AER (suy từ chênh lệch bootstrap) | 21.3 / 31.7 / 48.0 | — |
| **RBF+nb+mm** | **42.4 / 57.6 / 70.0** | 43.3 / 57.5 / 69.9 |

### 4.4 Số cũ (họ DDE, EXP-24; GrailQA-ODA dev, EXP-32)

> **Cảnh báo (EXP-37):** tập GrailQA đã chấm (897 câu từ `graliqa.json` của ODA, tập **dev**) **không phải** tập 874 câu của paper (GrailQA **train**, top-5 domain People / Biology / Computer / Religion / Digicams). Số dưới chỉ là phép thử OOD tạm, không so với paper. Tập đúng đã dựng lại được (995 câu trước lọc) nhưng chưa embed.

| WebQSP → GrailQA-ODA-dev (897 câu), ansR@50 / @100 / @200 | Tất cả | 1-hop (645) | 2-hop (219) | 3-hop (33) | zero-shot (513) |
|---|---|---|---|---|---|
| SubgraphRAG repro | 71.7 / 79.1 / 86.9 | 83.5 / 89.1 / 94.2 | 43.2 / 57.1 / 73.4 | 30.5 / 30.5 / 33.7 | 71.8 / 80.2 / 89.9 |
| REL-RAG (path_bf) repro | 70.0 / 76.5 / 79.8 | 81.5 / 87.0 / 89.6 | 42.7 / 52.4 / 58.6 | 27.4 / 30.5 / 30.5 | 68.0 / 75.6 / 78.3 |
| DDE + AER + ADR | 72.1 / 79.4 / 87.3 | 82.6 / 88.6 / 94.2 | 46.3 / 59.0 / 73.7 | 36.6 / 36.7 / 42.8 | 71.9 / 79.8 / 90.2 |
| RT + weighted loss (WL) | 83.2 / 87.0 / 91.8 | 94.0 / 96.4 / 98.1 | 59.6 / 67.8 / 79.4 | 27.6 / 31.1 / 50.7 | 85.0 / 88.9 / 93.2 |
| WL + AER + ADR | 83.4 / 89.2 / 93.0 | 93.7 / 96.3 / 98.0 | 60.3 / 76.0 / 84.9 | 36.8 / 37.4 / 47.8 | 85.0 / 90.8 / 94.4 |

**GTSQA (Wikidata, 1,622 câu) và MetaQA-3hop (4,710 câu), train WebQSP (EXP-24, chỉ họ DDE):**

| ansR@50 | GTSQA 1 / 2 / 3 / 4 hop | GTSQA unseen graph type (767) | MetaQA-3hop ansR@20 / @50 / @100 |
|---|---|---|---|
| DDE | 44.1 / 30.3 / 33.9 / 20.0 | 27.6 | 12.2 / 21.0 / 33.5 |
| DDE 2 lượt | 38.4 / 34.6 / 33.4 / 18.7 | 27.3 | 12.4 / 21.4 / 34.3 |
| AER (2 lượt) | 37.0 / 32.0 / 33.8 / 18.2 | 26.5 | 9.3 / 18.9 / 32.1 |
| QR-DDE | 50.6 / 41.4 / 42.2 / 21.3 | 34.7 | 14.9 / 21.7 / 30.8 |
| SubgraphRAG / RAPL / REL-RAG / WL / ASE | — | — | — (chưa chạy; WL cần thêm nhánh relation-topic vào `ext_eval.py`) |

---

## 5. Ablation RBF (EXP-39; train WebQSP, ansR@50 / @100 / @200, cùng tập 1,559 / 2,848 câu)

| Biến thể | WebQSP (ID) | WebQSP → CWQ (OOD) | Ghi chú |
|---|---|---|---|
| RBF không nb, **dính lỗi scatter** | 89.36 / 93.85 / 96.97 | 80.3 / 87.5 / 93.5 | lỗi `scatter_max` điền 0 vào ô rỗng |
| RBF không nb, đã sửa lỗi | 89.42 / 94.26 / 97.41 | 82.1 / 88.7 / 94.0 | |
| RBF + nb | 89.63 / 93.85 / 97.73 | 81.8 / 89.8 / 94.6 | nb trung tính trên WebQSP/CWQ |
| **RBF + nb + mm lúc test** (phương pháp chính) | **91.37 / 95.18 / 98.05** | **87.1 / 92.1 / 96.0** | mm sửa 31 câu trượt, làm hỏng 3 câu (WebQSP) |
| RBF + nb + mm, train với mm | 91.46 / 95.44 / 97.86 | 85.2 / 91.2 / 95.5 | không hơn chỉ bật lúc test |
| RBF + nb + mm, seed 0 | 90.61 / 95.27 / 97.44 | 84.4 / 90.4 / 95.5 | trung bình 2 seed: 91.0 / 85.8 @50 |
| RBF + nb, nhãn SP ∪ rational | 89.53 / 94.40 / 97.58 | 82.0 / 89.5 / 94.6 | không mm |
| RBF nhãn rational, nb chỉ lúc test | 89.81 / 93.97 / 96.97 | 82.1 / 89.0 / 94.8 | không mm |
| + prior Path+Memory (in-dist) | 91.3 / 95.1 / 97.9 | — | âm |
| + prior memory | 91.0 / 95.2 / 97.9 | 86.9 / 91.4 / 95.8 | âm |
| + thưởng memory lúc test (β = 2) | 91.6 / 95.6 / — | 87.8 / 93.0 / — | +0.8 trên CWQ |
| **+ BGE prior lúc test (β = 0.5)** | 92.0 / 95.7 / 97.9 | **88.7 / 93.5 / 96.4** | relation selection top-5 Hit: WebQSP 0.93, CWQ 0.89 (π: 0.85 / 0.78) |
| + Path+Memory + loss đáp án | 92.5 / 96.1 / 98.5 | — | checkpoint epoch 2 |
| + loss đáp án | 93.0 / 96.8 / 98.6 | 79.3 / 89.0 / 94.5 | **loại**: overfit WebQSP |
| soft-AND / max-marginal có ràng buộc (cmm) | ±0.2 | 84.7–85.3 @50 | âm |
| phá hoà điểm bằng cos(entity, câu hỏi) | 90.9 @50 | 87.1 @50 | âm |
| π chỉ là cổng 6 tham số | val 48.6 @50 | — | âm |
| Hybrid MLP + RBF (`rbfmlp_nb`, `wlrbf_nb`) | val 91.5 / 91.6 @50 | nhóm tổ hợp mới: head MLP 51.2 so với head đường đi 80.8 | MLP học thuộc tổ hợp |
| Trần: π + thưởng relation oracle (SP) | 95.3 @50 | 95.5 @50 | chọn relation là nút thắt trên CWQ |

---

## 6. Phase nhãn SPARQL (gold query, 10-15 → 10-18)

Nhãn lấy từ truy vấn gold (WebQSP: InferentialChain + Constraints; CWQ: SPARQL gold), gắn vào subgraph:
- **SPARQL v1:** mọi trail đi theo chain gold từ topic, cộng triple ràng buộc. Trail kết thúc ở entity không phải đáp án vẫn là nhãn dương.
- **SPARQL v2:** chỉ trail **kết thúc ở đáp án gold**, cộng triple ràng buộc gắn vào các trail đó. Câu không tới được đáp án trong subgraph giữ nhãn SP.
- REL-RAG cần path có thứ tự, nên dùng `relrag/sparql_to_pathlabels.py` (cùng quy tắc v2; code-faithful giữ 1 trail ngắn nhất). Mọi cờ khác giữ nguyên như run nhãn cũ (EXP-16, EXP-17, EXP-40). REL-RAG paper-faithful chọn checkpoint theo val loss như paper.

### 6.1 Retrieval

Sinh bởi `relrag/sparql_phase_table.py` → `relrag/runs/bench/sparql_phase.{json,md}`. Trong một bảng, mọi dòng chấm trên cùng tập câu. PathHit / Cons-R / Q-F1 tính theo truy vấn gold (như `query_recall.py`, mục 3 không có các cột này). Mỗi kiến trúc có dòng nhãn cũ ngay trên dòng nhãn SPARQL.

#### WebQSP (ID) (1559 câu; metric theo truy vấn gold trên 1496 câu)

| Phương pháp | Nhãn | ansR@50 | ansR@100 | ansR@200 | PathHit@50 | PathHit@100 | Cons-R@50 | Q-F1@50 | SP-F1@50 | GPT-R@50 |
|---|---|---|---|---|---|---|---|---|---|---|
| SubgraphRAG | SP | 90.4 | 94.4 | 96.9 | 84.1 | 90.1 | 11.5 | 15.6 | 15.2 | 81.0 |
| SubgraphRAG (DDE repro) | SPARQL v1 | 92.1 | 96.5 | 98.4 | 92.1 | 96.5 | 50.6 | 20.5 | 15.8 | 69.2 |
| REL-RAG paper-faithful | hybrid GPT-4o+DeepSeek | 80.4 | 89.5 | 94.6 | 79.9 | 87.4 | 24.8 | 16.5 | 14.6 | 55.1 |
| REL-RAG paper-faithful | SPARQL v2 | 84.3 | 90.9 | 95.0 | 84.0 | 88.6 | 31.1 | 17.4 | 14.4 | 58.1 |
| REL-RAG code-faithful | hybrid GPT-4o+DeepSeek (minlen) | 90.6 | 94.8 | 97.0 | 88.1 | 91.6 | 32.3 | 18.7 | 16.6 | 61.9 |
| REL-RAG code-faithful | SPARQL v2 (minlen) | 90.4 | 95.1 | 97.5 | 89.6 | 93.6 | 34.5 | 19.2 | 16.0 | 61.6 |
| RBF+nb+mm | SP | 91.4 | 95.2 | 98.0 | 85.6 | 90.6 | 16.3 | 16.6 | 15.4 | 83.1 |
| RBF+nb+mm | SPARQL v1 | 94.1 | 97.1 | 98.7 | 94.4 | 97.1 | 58.1 | 21.2 | 15.5 | 80.8 |
| RBF+nb+mm + BGE + WL | SP | 91.9 | 95.6 | 97.4 | 89.0 | 93.0 | 22.2 | 17.8 | 15.4 | 84.8 |
| RBF+nb+mm + BGE + WL | SP ∪ rational | 92.7 | 95.6 | 97.3 | 90.0 | 93.8 | 23.5 | 18.2 | 15.7 | 85.0 |
| RBF+nb+mm + BGE + WL | SPARQL v2 | 93.5 | 97.0 | 98.3 | 93.6 | 96.9 | 65.8 | 20.7 | 15.6 | 82.1 |

#### WebQSP → CWQ (OOD) (2848 câu; metric theo truy vấn gold trên 2510 câu)

| Phương pháp | Nhãn | ansR@50 | ansR@100 | ansR@200 | PathHit@50 | PathHit@100 | Cons-R@50 | Q-F1@50 | SP-F1@50 | GPT-R@50 |
|---|---|---|---|---|---|---|---|---|---|---|
| SubgraphRAG | SP | 70.0 | 79.6 | 87.7 | 44.8 | 53.9 | 27.1 | 12.9 | 6.9 | 53.6 |
| SubgraphRAG (DDE repro) | SPARQL v1 | 70.7 | 81.3 | 89.6 | 42.0 | 49.8 | 12.2 | 19.0 | 7.4 | 41.2 |
| REL-RAG paper-faithful | hybrid GPT-4o+DeepSeek | 61.2 | 69.2 | 77.7 | 38.2 | 42.6 | 12.2 | 17.4 | 6.8 | 34.4 |
| REL-RAG paper-faithful | SPARQL v2 | 56.2 | 65.6 | 75.2 | 37.7 | 41.8 | 10.5 | 17.7 | 6.3 | 31.0 |
| REL-RAG code-faithful | hybrid GPT-4o+DeepSeek (minlen) | 61.9 | 72.3 | 79.3 | 38.4 | 43.9 | 12.0 | 18.0 | 7.0 | 34.3 |
| REL-RAG code-faithful | SPARQL v2 (minlen) | 57.8 | 67.9 | 76.7 | 37.3 | 42.5 | 9.9 | 17.2 | 6.6 | 32.7 |
| RBF+nb+mm | SP | 87.1 | 92.1 | 96.0 | 62.7 | 72.8 | 37.8 | 20.3 | 8.7 | 66.8 |
| RBF+nb+mm | SPARQL v1 | 86.4 | 92.2 | 96.2 | 65.6 | 74.5 | 37.6 | 24.0 | 8.9 | 62.2 |
| RBF+nb+mm + BGE + WL | SP | 87.9 | 92.3 | 95.2 | 62.0 | 71.8 | 46.9 | 18.9 | 8.5 | 68.4 |
| RBF+nb+mm + BGE + WL | SP ∪ rational | 87.2 | 92.0 | 95.0 | 60.4 | 69.4 | 44.3 | 18.3 | 8.5 | 65.8 |
| RBF+nb+mm + BGE + WL | SPARQL v2 | 90.3 | 94.6 | 96.6 | 64.0 | 73.7 | 51.3 | 23.7 | 8.9 | 65.8 |

#### CWQ (ID) (2848 câu; metric theo truy vấn gold trên 2510 câu)

| Phương pháp | Nhãn | ansR@50 | ansR@100 | ansR@200 | PathHit@50 | PathHit@100 | Cons-R@50 | Q-F1@50 | SP-F1@50 | GPT-R@50 |
|---|---|---|---|---|---|---|---|---|---|---|
| SubgraphRAG | SP | 86.2 | 91.4 | 95.3 | 67.4 | 78.0 | 39.9 | 18.2 | 10.4 | 75.5 |
| REL-RAG (author ckpt cwq.pt) | GPT-4o (authors) | 80.8 | 85.9 | 89.8 | 45.5 | 49.3 | 12.3 | 19.7 | 9.0 | 45.1 |
| RBF+nb+mm | SP | 90.5 | 94.1 | 96.8 | 71.2 | 80.2 | 47.7 | 20.9 | 10.0 | 73.4 |
| RBF+nb+mm | SPARQL v1 | 92.5 | 95.5 | 97.5 | 82.0 | 87.5 | 60.5 | 26.9 | 8.7 | 70.1 |
| RBF+nb+mm | SPARQL v2 | 93.1 | 95.7 | 98.1 | 80.6 | 86.4 | 59.8 | 26.7 | 9.1 | 70.5 |
| RBF+nb+mm + BGE + WL | SP | 94.5 | 96.9 | 98.4 | 77.3 | 84.9 | 58.0 | 21.4 | 10.6 | 78.7 |
| RBF+nb+mm + BGE + WL | SPARQL v2 | 95.1 | 97.3 | 98.7 | 85.7 | 90.8 | 65.2 | 24.2 | 9.6 | 75.4 |

#### CWQ → WebQSP (OOD) (1559 câu; metric theo truy vấn gold trên 1496 câu)

| Phương pháp | Nhãn | ansR@50 | ansR@100 | ansR@200 | PathHit@50 | PathHit@100 | Cons-R@50 | Q-F1@50 | SP-F1@50 | GPT-R@50 |
|---|---|---|---|---|---|---|---|---|---|---|
| SubgraphRAG | SP | 82.2 | 88.2 | 92.9 | 77.3 | 85.2 | 16.5 | 12.3 | 12.6 | 71.8 |
| REL-RAG (author ckpt cwq.pt) | GPT-4o (authors) | 90.9 | 95.8 | 97.7 | 89.7 | 93.2 | 33.5 | 18.8 | 16.3 | 62.3 |
| RBF+nb+mm | SP | 87.2 | 90.9 | 95.0 | 83.6 | 88.5 | 24.3 | 15.1 | 14.4 | 80.5 |
| RBF+nb+mm | SPARQL v1 | 89.9 | 94.0 | 96.9 | 88.2 | 92.3 | 29.3 | 18.8 | 14.7 | 79.2 |
| RBF+nb+mm | SPARQL v2 | 90.4 | 93.8 | 96.0 | 90.0 | 91.9 | 35.6 | 18.3 | 14.8 | 82.4 |
| RBF+nb+mm + BGE + WL | SP | 89.7 | 94.1 | 96.2 | 86.2 | 91.4 | 24.8 | 16.2 | 14.6 | 81.7 |
| RBF+nb+mm + BGE + WL | SPARQL v2 | 91.6 | 94.9 | 97.3 | 90.4 | 94.1 | 24.4 | 18.6 | 14.8 | 81.8 |

**Nhận xét:**
- **ID:** nhãn SPARQL tăng PathHit@50 cho mọi kiến trúc train bằng code của mình: DDE 84.1 → 92.1 (so với SubgraphRAG official), RBF+nb+mm 85.6 → 94.4 (WebQSP) và 71.2 → 80.6 (CWQ), RBF + BGE + WL 89.0 → 93.6 (WebQSP) và 77.3 → 85.7 (CWQ). Cons-R tăng mạnh nhất (WebQSP: khoảng 11–23 lên 50–66), vì nhãn SP coi triple ràng buộc là nhãn âm.
- **REL-RAG hưởng lợi ít nhất.** Paper-faithful +3.9 ansR@50 / +4.1 PathHit@50 (nhãn cũ dài hơn và lẫn path không tới đáp án). Code-faithful gần như không đổi về ansR, PathHit@50 +1.5.
- **OOD WebQSP → CWQ:** nhãn v2 giúp RBF + BGE + WL (ansR@50 87.9 → 90.3, PathHit@50 62.0 → 64.0), nhưng **làm REL-RAG kém đi** (ansR@50 −5.0 paper, −4.1 code). Nhãn SPARQL của WebQSP gần như toàn chain 1 hop, nên decoder theo path của REL-RAG học đi ngắn hơn, trong khi câu CWQ cần nhiều bước. DDE nhãn v1 cũng không cải thiện (70.0 → 70.7, PathHit giảm 2.8).
- **OOD CWQ → WebQSP:** nhãn v2 giúp RBF (PathHit@50 83.6 → 90.0, ngang REL-RAG checkpoint tác giả 89.7).
- **Tốt nhất phase này:** RBF + BGE + WL nhãn v2 dẫn ansR@50 ở CWQ (95.1), WebQSP → CWQ (90.3) và CWQ → WebQSP (91.6). Trên WebQSP, RBF+nb+mm nhãn v1 cao hơn (94.1 so với 93.5).
- **Gate "ansR@50 > ansR@200 của SubgraphRAG":** đạt ở WebQSP → CWQ (90.3 > 87.7). Chưa đạt ở WebQSP (94.1 < 96.9), CWQ (95.1 < 95.3, sát mốc) và CWQ → WebQSP (91.6 < 92.9).

### 6.2 QA (Llama-3.1-8B bf16, vLLM 0.8.4 trên H100, toàn tập test, k = 100)

`reason/prompt_lab.py --test --api --n 0`. **ans-Hit** chỉ chấm trên các dòng `ans:` (server này lặp tới 4,000 token ở khoảng 11% câu CWQ, làm Hit trên text thô bị thổi lên; docs/12 §27). ans-Hit dùng để **so cặp**. Để so với số paper, dùng Hit theo cách chấm paper đã hiệu chuẩn bằng ô SubgraphRAG × T0 so với file tác giả: WebQSP dùng Hit trên text thô (86.7 so với 86.61 của tác giả); CWQ dùng bản loop-safe (56.6 so với 56.87). Theo cách này, RBF v2 + P2: WebQSP **88.4**, CWQ **62.8** (Hit trong log là 68.4, chưa hiệu chuẩn). Prompt T0 = `sys_icl_dc` gốc (triple); P2 = path từ topic + danh sách ứng viên + chỉ dẫn (docs/12 §25–26). Mỗi ô ghi ans-Hit / Macro-F1. Không so trực tiếp với số paper.

| Retrieval | Nhãn | WebQSP T0 | WebQSP P2 | CWQ T0 | CWQ P2 |
|---|---|---|---|---|---|
| SubgraphRAG (official) | SP | 83.3 / 68.6 | 85.1 / 69.4 | 54.2 / 45.6 | 56.2 / 48.7 |
| RBF+nb+mm + BGE + WL | SPARQL v2 | 83.2 / 69.9 | **86.4 / 72.2** | 57.6 / 49.3 | **60.8 / 53.4** |
| REL-RAG paper-faithful (WebQSP) | hybrid | 80.2 / 66.9 | 80.7 / 65.9 | | |
| REL-RAG paper-faithful (WebQSP) | SPARQL v2 | 82.0 / 69.3 | 80.8 / 68.2 | | |
| REL-RAG code-faithful (WebQSP) | hybrid | 82.9 / 70.1 | 85.3 / 70.8 | | |
| REL-RAG code-faithful (WebQSP) | SPARQL v2 | 82.6 / 70.3 | 85.6 / 70.9 | | |

**WebQSP → CWQ** (2,848 câu chung của mọi export; SubgraphRAG ở đây là bản train CWQ, tức một mốc ID):

| Retrieval (train) | Nhãn | T0 | P2 |
|---|---|---|---|
| SubgraphRAG (train CWQ, ID) | SP | 62.0 / 52.2 | 63.0 / 55.0 |
| RBF+nb+mm + BGE + WL (train WebQSP) | SPARQL v2 | **64.5 / 55.0** | 61.8 / 54.0 |
| REL-RAG paper-faithful (train WebQSP) | hybrid → SPARQL v2 | 49.5 / 40.7 → 45.9 / 37.1 | 48.6 / 40.6 → 43.5 / 35.9 |
| REL-RAG code-faithful (train WebQSP) | hybrid → SPARQL v2 | 47.6 / 38.7 → 47.0 / 38.7 | 46.4 / 38.7 → 46.2 / 38.8 |

- REL-RAG chạy trên server bf16 thứ hai (cùng trọng số, template và config; hiệu chuẩn lại ô SubgraphRAG × T0 WebQSP: 83.2 / 68.6 so với 83.3 / 68.6, McNemar p = 1.0).
- **Nhãn mới không giúp QA của REL-RAG trên WebQSP** (code-faithful: −0.3 / +0.3 ans-Hit, p ≥ 0.76; paper-faithful T0 +1.8, p 0.06) và **làm transfer kém đi** ở paper-faithful (−3.7 / −5.1, p ≤ 2e-4), khớp với retrieval ở 6.1.
- Trên WebQSP, REL-RAG code-faithful ngang SubgraphRAG và RBF (P2: 85.6 so với 85.1 và 86.4, p ≥ 0.43). Ở WebQSP → CWQ, REL-RAG thấp hơn RBF 15–18 ans-Hit (p < 1e-45).

- RBF v2 + P2 so với SubgraphRAG + T0: WebQSP +3.1 ans-Hit (McNemar p 0.001), F1 +3.6; CWQ +6.5 (p 6e-15), F1 +7.8.
- Transfer (subset, ans-Hit / F1): WebQSP → CWQ, RBF v2 + T0 64.5 / 55.0 so với SubgraphRAG train CWQ 62.0 / 52.2 (P2 kém T0 2.7 ở setting này). CWQ → WebQSP, RBF v2 + P2 86.5 / 72.9 so với SubgraphRAG ID 85.2 / 70.0.

---

## 7. Tái tạo số

```bash
# retrieval (local CPU, khoảng 10 phút): mọi bảng ở mục 3
python relrag/benchmark_all.py                      # -> relrag/runs/bench/bench_all.json, bench_all_tables.md
# QA (local, khoảng 5 phút, không gọi LLM): mọi bảng ở mục 2
cd reason && python -m metrics.bench_qa              # -> reason/results/bench_qa.json
python -m metrics.paper_score <predictions.jsonl>    # chấm một file
# phase SPARQL (mục 6): python relrag/sparql_phase_table.py; QA: reason/prompt_lab.py <T0|P2> <ds> --test --api --n 0 --k 100 --src <export>
# export RBF: train qrdde.py (vast) -> adr_probe.py --rbf_mm 1 -> score_extra.py (đủ tập test) -> make_dynamic_export.py
```

---

## Phụ lục A. Theo loại câu CWQ (ansR@50 / ansR@100; bảng cũ, 10-07)

### WebQSP → CWQ (train WebQSP, OOD / compositional)

| Method | composition (940) | conjunction (1526) | comparative (199) | superlative (183) | 0–1 hop (810) | 2 hop (1867) | ≥ 3 hop (171) |
|---|---|---|---|---|---|---|---|
| SubgraphRAG (repro, train WebQSP) | 54.6 / 66.0 | 78.1 / 87.2 | 75.5 / 82.0 | 76.0 / 83.1 | 94.5 / 98.4 | 60.5 / 72.2 | 59.0 / 70.8 |
| RAPL (author ckpt, webqsp.pt) | 30.4 / 38.6 | 84.4 / 91.0 | 85.7 / 93.1 | 86.9 / 94.5 | 89.1 / 93.4 | 57.1 / 65.1 | 66.7 / 80.6 |
| REL-RAG (path_bf) (reprod) | 26.0 / 37.4 | 77.6 / 87.9 | 84.7 / 92.8 | 90.2 / 98.4 | 88.5 / 94.4 | 49.6 / 61.6 | 70.2 / 83.9 |
| DDE (SubgraphRAG repro) + AER | 74.1 / 82.5 | 76.8 / 86.6 | 76.0 / 83.0 | 75.4 / 82.5 | 92.4 / 97.5 | 70.2 / 80.4 | 58.1 / 71.3 |
| DDE (SubgraphRAG repro) + AER + ADR | 74.9 / 83.4 | 81.6 / 88.3 | 75.9 / 81.2 | 75.4 / 82.0 | 94.9 / 97.8 | 73.5 / 81.7 | 57.2 / 73.5 |
| RT + weighted loss (WL) | 58.0 / 69.9 | 80.7 / 89.5 | 87.7 / 93.0 | 85.8 / 94.0 | 95.2 / 98.3 | 65.7 / 77.5 | 65.4 / 79.5 |
| RT + weighted loss (WL) + AER + ADR | 77.1 / 84.2 | 86.0 / 92.3 | 87.2 / 93.5 | 86.9 / 94.0 | 94.1 / 98.0 | 79.7 / 86.9 | 69.6 / 83.1 |
| ASE (on DDE) | 51.5 / 65.6 | 80.2 / 86.9 | 84.7 / 92.9 | 86.9 / 93.4 | 95.0 / 98.0 | 64.1 / 75.0 | 41.3 / 61.3 |
| ASE (on DDE) + AER + ADR | 69.0 / 79.9 | 86.5 / 91.4 | 87.0 / 91.5 | 85.8 / 93.4 | 96.0 / 98.7 | 76.0 / 84.7 | 59.4 / 69.3 |
| WL + ASE + aug | 54.9 / 66.1 | 94.0 / 96.9 | 90.7 / 95.8 | 90.7 / 95.1 | 98.4 / 99.5 | 74.4 / 81.8 | 64.7 / 77.2 |
| WL + ASE + aug + AER + ADR | 77.9 / 85.2 | 91.3 / 96.4 | 87.7 / 95.4 | 89.6 / 96.2 | 97.3 / 99.4 | 83.3 / 90.4 | 70.3 / 84.3 |

### CWQ (train CWQ, in-distribution)

| Method | composition (940) | conjunction (1526) | comparative (199) | superlative (183) | 0–1 hop (810) | 2 hop (1867) | ≥ 3 hop (171) |
|---|---|---|---|---|---|---|---|
| SubgraphRAG (official ckpt) | 83.5 / 89.9 | 89.8 / 93.7 | 75.9 / 80.4 | 81.4 / 92.9 | 91.2 / 94.7 | 86.6 / 91.6 | 58.1 / 74.1 |
| RAPL (author ckpt, cwq.pt) | 60.6 / 66.8 | 90.7 / 95.3 | 88.8 / 93.8 | 92.9 / 96.7 | 93.4 / 96.9 | 76.0 / 81.4 | 73.4 / 81.9 |
| DDE (SubgraphRAG repro) + AER | 83.7 / 88.3 | 89.8 / 93.6 | 75.1 / 81.9 | 83.6 / 89.6 | 91.2 / 94.1 | 86.8 / 91.3 | 59.5 / 70.2 |
| DDE (SubgraphRAG repro) + AER + ADR | 84.5 / 89.1 | 90.9 / 94.9 | 74.8 / 83.2 | 82.0 / 89.6 | 91.6 / 95.4 | 87.5 / 92.1 | 60.9 / 71.7 |
| ASE (on DDE) | 86.6 / 92.5 | 94.5 / 96.7 | 80.7 / 85.7 | 85.8 / 91.3 | 94.5 / 97.4 | 91.1 / 95.1 | 61.8 / 68.7 |
| ASE (on DDE) + AER + ADR | 86.2 / 91.5 | 92.4 / 95.7 | 74.8 / 82.7 | 79.8 / 87.4 | 92.4 / 95.5 | 89.0 / 93.8 | 62.1 / 71.1 |

## Phụ lục B. Siêu tham số AER (EXP-23, WebQSP → CWQ, ansR@50)

| reanchor_p | ALL | composition | conjunction | comparative | superlative | WebQSP |
|---|---|---|---|---|---|---|
| 0.25 | 75.38 | 68.1 | 79.7 | 77.0 | 75.4 | 89.17 |
| **0.5** | **77.95** | **70.0** | 81.3 | 85.4 | 83.1 | 89.68 |
| 1.0 | 77.13 | 63.4 | 83.3 | 84.7 | 88.5 | 89.62 |

B (số triple lấy cầu nối) chỉ chạy B = 3; K (số lượt) chỉ chạy 2. Chưa có ablation B / K có nhãn đầy đủ.
