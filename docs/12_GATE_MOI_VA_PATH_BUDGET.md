# Giai đoạn 10-12: gate mới, chọn relation tốt hơn, budget theo path

## 1. Gate mới (người dùng đặt, 10-12)
- **Không dùng REL-RAG repro làm thước đo chính**, vì số của nó lệch với benchmark thật. Thước đo là các phương pháp tốt nhất **của mình**: WL, WL+ADR+AER, AER, ADR, ASE. Các phương pháp này đã có Hit và Macro-F1 tốt hơn SubgraphRAG.
- **Thắng = recall@50 của phương pháp mới > recall@200/300 của phương pháp cũ tốt nhất.** Chỉ khi đạt mức này mới chạy reasoning.
- Recall vẫn là chỉ số số 1, nhưng luôn báo kèm **precision và F1 của triple** (so với nhãn SP), vì ít triple chất lượng cao thì Llama/SLM suy luận tốt hơn.
- Đánh giá thêm **retrieval theo path với budget động**, không chỉ top-k cố định.
- Công cụ: `relrag/budget_table.py` (ansR, hit, SP-P/R/F1 ở @50/100/200/300, trần ansR@k), `relrag/dynamic_cut.py`, `retrieve/exp_qrdde/dyn_anchor_probe.py`.

## 2. Vị trí hiện tại (ansR; train trên WebQSP)

| Tập | RBF+nb+mm @50 | Cũ tốt nhất @200 / @300 | Trần @50 |
|---|---|---|---|
| WebQSP | 91.4 | WL 98.2 / 98.9 | 99.3 |
| CWQ | 87.1 | WL+ADR+AER 94.7 / 96.4 | 100 |

→ **Chưa đạt gate mới ở ô nào.** Với SubgraphRAG gốc thì CWQ gần chạm (87.1 so với 87.7 @200).

**Chẩn đoán WebQSP @50** (8.6 điểm mất):
- Mất nhiều nhất ở câu 1 đáp án bị trượt hoàn toàn: 58 câu, 3.7 điểm. Đây là lỗi xếp hạng relation của π.
- Câu có hơn 20 đáp án mất 1.5 điểm, do budget.
- Path+Memory chọn trúng relation đúng ở 61/91 câu RBF trượt hoàn toàn; trung vị hạng của relation đúng tốt nhất là 1.

**Budget động** (giữ triple có log-prob đường đi ≥ tốt nhất − τ):
- WebQSP, τ = 2: ansR 91.6 với trung bình 41 triple (trung vị 10), SP-F1 **55.8**. Top-50 cho 91.4 với F1 21.0.
- Nhưng WL cũng được lợi y như vậy (τ = 4: 96.6 với 74 triple, F1 39.8). → Lợi ích chủ yếu đến từ việc cắt động, không riêng gì RBF.
- CWQ transfer: cắt theo cả câu thì **kém hơn** top-k (τ = 3: 84.7 với trung bình 93 triple, top-100 được 92.1). Ngưỡng của anchor mạnh cắt mất đường đi của anchor còn lại → cần cắt riêng theo từng anchor.

## 3. Giả thuyết đang kiểm

| | Giả thuyết | Thí nghiệm | Bác bỏ nếu |
|---|---|---|---|
| H6 | Prior relation tốt hơn (Path+Memory: P 0.58 / R 0.76 / Hit 0.93 trên WebQSP) sửa được lỗi xếp hạng của π | `webqsp_rbfnb_pmh_s42` (vast), chỉ in-dist vì memory dùng nhãn train cùng tập | WebQSP @50 tăng < 2 điểm |
| H7 | Loss listwise ở mức đáp án (softmax trên điểm đường đi tốt nhất) xếp hạng top-k tốt hơn BCE | `webqsp_rbfnb_ans_s42` (vast) | WebQSP @50 và CWQ @50 không tăng |
| H8 | Budget động cắt riêng theo từng anchor giữ được recall trên CWQ với ít triple hơn | `dyn_anchor_probe.py` (CPU) | Ở cùng số triple trung bình, recall không hơn top-k |
| H9 | Bản Path+Memory dựng lại bằng GTE, với memory lấy từ tập nguồn, giúp được cả các ô transfer | Dựng lại selector (checkpoint BGE fine-tune đã mất) | Không hơn prior lex + cos trên CWQ / GrailQA |

## 4. Dọn dẹp lưu trữ (10-12)
- **Đã xoá:**
  - Các phần tạm của WL (`data_files/*_p*`, export `adr_*_pN_*`), dựng lại được trong vài phút.
  - `data_ext/grailqa_raw` (tải lại được từ HF).
  - `grailqa_oda` và bản `cwq/gemb_*` local đã không còn từ trước.
- **Đã nén** (zstd, kiểm trước khi xoá bản gốc): export của các run đã kết luận hoặc dính lỗi scatter.
- **Đề xuất, chờ duyệt:**
  - Shard CWQ train/val (59 GB). Đã có `gemb` gọn, 4 GB, trên vast; có thể dựng lại shard từ đó.
  - `RAPL/data` trên svuet (48 GB).

## 5. Kết quả 10-12
- **H8 (budget động cắt riêng theo anchor) bị bác ở ô transfer.** Trên CWQ: τ = 3 cho ansR 87.1 với trung bình 112 triple, trong khi top-50 cho 87.1 với 50 triple. In-dist trên WebQSP thì budget động rất tốt (ansR 91.7 với trung vị 10 triple, F1 55.7), nhưng WL cũng được lợi tương tự.
- **Trần khi π biết relation đúng** (π học được + thưởng cho relation SP; ansR@50):

  | Tập | π học được | + relation oracle | Cũ tốt nhất @200 |
  |---|---|---|---|
  | CWQ | 87.2 | **95.5** | 94.7 |
  | WebQSP | 91.5 | 95.3 | 98.2 |

  → Ở **CWQ (transfer), chọn relation là nút thắt**: nếu chọn gần đúng như oracle thì vượt gate mới. Ở **WebQSP**, relation hoàn hảo cũng chỉ lên 95.3. Phần còn lại mất *bên trong* một relation (câu nhiều đáp án, entity hoà điểm, ràng buộc thời gian/kiểu), nên cần bằng chứng ở mức entity.
- **Prior memory (H9)**, vote từ câu train của tập nguồn:
  - Chọn relation trên CWQ đạt top-5 P 0.43 / R 0.48 / Hit 0.85, gấp khoảng đôi so với GTE cos (0.23 / 0.26 / 0.63).
  - Trên GrailQA phủ được 80% câu. GTSQA và MetaQA thì không phủ được câu nào (0%), vì tên relation khác KG; muốn dùng phải khớp mềm qua embedding.
- **Max-marginal có ràng buộc (`rbf.cmm_scores`, kết quả âm).**
  - Cách làm: walk phải kết thúc ở entity có tên. Nút cuối được thưởng mềm nếu cũng tới được từ các entity khác của câu (topic khác, entity được nhắc tên trong câu). Với ε = 1 thì trùng mm (kiểm: 0/40 lệch).
  - **CWQ @50:** mm 87.1, soft-AND giữa các topic 84.7–85.3, entity nhắc tên 83.7–83.9. **WebQSP:** ±0.2.
  - → Đã kiểm lại soft-AND với code đúng (lần đo trước dính lỗi scatter): vẫn âm. Phần thưởng kéo vào các triple quanh entity chung chung được nhắc trong câu ("country", nút type) hơn là quanh đáp án.
  - Chẩn đoán vẫn đúng: 186/352 câu CWQ bị trượt có entity nhắc tên nằm cạnh đáp án. Nhưng một ràng buộc đối xứng, không học, thì chưa khai thác được tín hiệu này.
- **Phá hoà điểm bằng độ giống giữa entity và câu hỏi** (cộng 1e-4 · cos(z_q, z_entity), chỉ đổi thứ tự trong nhóm hoà): WebQSP @50 91.4 → 90.9, CWQ không đổi. → **Kết quả âm.**
- **Batch 3 trên vast (train trên WebQSP, nb, seed 42; test bật mm), ansR@50 / @100 / @200:**

  | Biến thể | WebQSP | CWQ (transfer) | Ghi chú |
  |---|---|---|---|
  | RBF+nb (gốc) | 91.4 / 95.2 / 98.0 | 87.1 / 92.1 / 96.0 | |
  | + prior Path+Memory (H6) | 91.3 / 95.1 / 97.9 | — (chỉ dùng in-dist) | **Âm**: π đã học được độ liên quan tương đương |
  | + prior memory (H9) | 91.0 / 95.2 / 97.9 | 86.9 / 91.4 / 95.8 | **Âm** tổng thể. Tổ hợp mới 85.7 → 90.6 nhưng tổ hợp đã thấy 85.9 → 83.8 |
  | + Path+Memory + loss đáp án | **92.5 / 96.1 / 98.5** | — | Checkpoint epoch 2 (tiêu chí chọn mô hình lệch với loss đáp án) |
  | + loss đáp án (H7) | **93.0 / 96.8 / 98.6** | **79.3** / 89.0 / 94.5 | In-dist tốt nhất (hơn WL 92.6 / 96.1 / 98.2), nhưng transfer sụp −7.8 @50 → **bác** |

- **Thưởng memory lúc test** (logit π + β · vote; β = 2 chọn trên WebQSP val): CWQ 87.8 / 93.0 (+0.8 / +0.9), WebQSP 91.6 / 95.6. Nhỏ, còn xa mức +8.3 của relation oracle. Lý do: memory chọn relation còn nhiễu (P top-5 0.43).

## 6. Tổng kết 10-12
- **Không ứng viên nào đạt gate mới.**
  - Các cách đưa thông tin vào ở mức relation (Path+Memory, memory, thưởng memory, ràng buộc mềm, entity được nhắc tên) cho tối đa +0.8 trên CWQ, trong khi relation oracle cho +8.3.
  - Loss mức đáp án tăng in-dist (+1.6) nhưng phá transfer (−7.8).
- **Hai kết luận định hướng:**
  1. Trên CWQ (transfer), *chất lượng chọn chuỗi relation* là đòn bẩy đã đo được: oracle đưa @50 lên 95.5, vượt gate. Các selector rẻ đều quá nhiễu (P top-5 khoảng 0.43). Cần một bộ chấm *chuỗi relation* mạnh hơn, chỉ train trên tập nguồn, ví dụ cross-encoder fine-tune trên WebQSP. Đây đúng là vai trò mà BGE-M3 fine-tune của Path+Memory đã đóng (Hit 0.93 in-dist).
  2. Trên WebQSP, gate có vẻ ngoài tầm với nếu chỉ chỉnh relation: relation oracle chỉ cho 95.3, gate cần 98.2. Phần còn lại là ràng buộc ở mức entity và câu nhiều đáp án.

## 7. BGE-M3 fine-tune (công thức Phase 11, train lại trên vast 10-12)

**Chất lượng chọn relation** (so với relation trên SP triple, tập test):

| Bộ chọn | WebQSP top-5 P / R / Hit | CWQ (transfer) top-5 P / R / Hit |
|---|---|---|
| BGE fine-tune | 0.395 / 0.709 / **0.930** | 0.397 / 0.464 / **0.891** |
| BGE base | 0.293 / 0.520 / 0.766 | 0.302 / 0.338 / 0.768 |
| memory | 0.423 / 0.722 / 0.883 | 0.428 / 0.476 / 0.847 |
| π (RBF) | 0.367 / 0.627 / 0.853 | 0.337 / 0.361 / 0.775 |

- Adapter chỉ train trên WebQSP train, nên dùng cho CWQ không rò rỉ nhãn.
- Kết quả chọn relation trên WebQSP khớp với số cũ của Phase 11 (Hit 0.93).

**Cộng prior lúc test, không train lại.** Logit của π được cộng thêm β · z(cos BGE), trong đó z chuẩn hoá theo từng câu. Phương pháp là RBF+nb+mm (`webqsp_rbfnb_s42`); β chọn trên WebQSP val. Script: scratchpad `bgebonus.py`.

| β | WebQSP val @50 / @100 | WebQSP test @50 / @100 | CWQ test @50 / @100 |
|---|---|---|---|
| 0 | 92.0 / 94.8 | 91.5 / 95.3 | 86.9 / 92.2 |
| **0.5** (chọn trên val) | **92.2 / 95.0** | **92.1 / 95.8** | **88.6 / 93.5** |
| 1 | 91.8 / 94.8 | 91.8 / 95.7 | 89.2 / 93.7 |
| 2 | 90.6 / 94.1 | 91.4 / 95.4 | 90.4 / 94.2 |

- Với β = 0.5: CWQ @50 = 88.6, **vượt SubgraphRAG @200 (87.7)** nhưng chưa vượt @300 (91.2). WebQSP chưa đạt gate (96.9 @200).
- Trên CWQ, β tối ưu (2) lớn hơn β chọn trên val, đúng hướng với trần relation oracle (95.5).

## 8. Reasoning WebQSP (10-13): phương pháp tổng quát RBF+nb+mm

**Cấu hình chạy:**
- Retrieval: `webqsp_rbfnb_s42`, mm bật lúc test. File `reason/custom_retrieval/webqsp_rbfnb_mm_top100.pth`, phủ đủ 1639 câu test; chỉ 1 câu có context rỗng.
- Reasoner: Llama-3.1-8B qua OpenRouter (DeepInfra fp8, temperature 0), `sys_icl_dc`.
- Chi phí: top-50 khoảng $0.08, top-100 khoảng $0.11.

**Kết quả (đã sửa 10-13).** Bảng cũ dùng `match` của `evaluate_results_corrected.py`, hàm này chỉ cần trùng một cụm 3 ký tự là tính khớp, nên số bị thổi phồng. Số dưới đây chấm lại bằng `reason/metrics/paper_score.py`: Hit theo eval gốc trên response thô, Macro-F1 theo eval corrected với match gốc. Bộ chấm này cho lại đúng số EXP-25. Response thô được dựng lại từ log bằng `metrics/raw_from_log.py`; kiểm 0 dòng lệch.

| Retrieval | Reasoner | Toàn tập: Hit / F1 (số câu) | 1559 câu chung: Hit / F1 |
|---|---|---|---|
| RBF+nb+mm top-50 | Llama-3.1-8B fp8 (OpenRouter) | 82.18 / 68.19 (1639) | 84.48 / 70.20 |
| RBF+nb+mm top-100 | fp8 (OpenRouter) | 83.71 / 68.94 (1639) | 86.08 / 70.89 |
| REL-RAG repro P @50 (EXP-25) | fp16 (vLLM, T4) | 84.95 / 70.15 (1628) | 86.59 / 71.41 |
| SubgraphRAG @100, file chính thức (EXP-25) | bf16 (của tác giả) | 86.61 / 70.57 (1628) | 88.58 / 72.10 |
| Paper REL-RAG @50 | bf16 | 88.8 / 77.3 | — |

- Cùng cách chấm, RBF thấp hơn cả SubgraphRAG lẫn REL-RAG repro, khoảng 1–4 Hit và 0.6–2 F1.
- Nhưng reasoner khác nhau (fp8 OpenRouter so với fp16/bf16), nên chưa tách được phần nào do retrieval, phần nào do reasoner.

**Ghi chú:**
- Đã sửa `reason/metrics/evaluate_results_corrected.py:355`. HAL score tra retrieval của RoG, file này thiếu `WebQTest-238` nên evaluator báo lỗi; nay dùng `.get(id, [])`. Hit và F1 không bị ảnh hưởng.
- Chưa có baseline SubgraphRAG chạy qua cùng reasoner (fp8 OpenRouter). Số trong cột "paper" là từ vLLM bf16.

- Run top-100 có 2 câu bị RateLimitError, đã chạy lại ở chế độ raw.

## 9. Reasoning CWQ in-dist (10-13): RBF+nb+mm train trên CWQ, top-50

**Cách chạy:**
- Retrieval: `cwq_rbfnb_s42` + mm, file `reason/custom_retrieval/cwq_rbfnbcwq_mm_top100.pth`, đủ 3531 câu.
- Reasoner: Llama-3.1-8B fp8 qua OpenRouter.
- Chấm bằng `metrics/paper_score.py`. Response thô dựng lại từ log; chỉ 2/3531 câu phải giữ dạng ans.

**Sự cố trong lúc chạy:** run chạm trần ngân sách $0.17 ở câu khoảng thứ 3216. Hàm `safe_inf` ghi "ans: not available" cho 173 câu, mà tiến trình vẫn trả exit 0.
- Đã chạy lại các câu đó cùng 20 câu pilot ở chế độ raw.
- Đã sửa `main.py`: khi gặp lỗi ngân sách thì dừng hẳn, không ghi đáp án rỗng.
- Tổng chi phí khoảng $0.18.

| Retrieval | Reasoner | Toàn tập (3531): Hit / F1 | Subset (2848): Hit / F1 |
|---|---|---|---|
| **RBF+nb+mm (CWQ) top-50** | fp8 (OpenRouter) | 55.82 / 46.66 | 64.68 / 54.50 |
| SubgraphRAG @100 chính thức | bf16 (tác giả) | 56.87 / 47.06 | 65.66 / 54.62 |
| WL+ADR+AER @50, train WebQSP (A2, transfer) | bf16 (vLLM) | 54.18 / 42.84 | 61.87 / 49.15 |
| Paper REL-RAG CWQ in-dist @50 | bf16 | 67.2 / 56.8 | — |

- RBF với 50 triple xấp xỉ SubgraphRAG với 100 triple (−1.0 Hit, −0.4 F1), dù reasoner khác nhau.
- Còn cách xa số paper của REL-RAG (−11 Hit).

**Bổ sung 10-13 qua DeepInfra trực tiếp** (`-m deepinfra`, model `Meta-Llama-3.1-8B-Instruct-Turbo`):
- Đây đúng là bản fp8 mà backend OpenRouter đã được ghim vào. Pilot 20 câu: 18/20 cho cùng đáp án.
- Lưu response thô ngay khi chạy (raw). Tổng chi phí khoảng $0.44.

| Retrieval | Toàn tập (3531): Hit / F1 | Subset (2848): Hit / F1 |
|---|---|---|
| RBF+nb+mm (CWQ) **top-100** | **57.97 / 48.24** | 67.13 / 56.00 |
| RBF+nb+mm (**WebQSP→CWQ**) top-50 | 52.53 / 44.59 | 60.67 / 51.63 |
| *Đối chiếu:* SubgraphRAG @100 chính thức (train CWQ, bf16) | 56.87 / 47.06 | 65.66 / 54.62 |
| *Đối chiếu:* WL+ADR+AER @50, WebQSP→CWQ (A2, bf16) | 54.18 / 42.84 | 61.87 / 49.15 |
| *Đối chiếu:* paper REL-RAG, WebQSP→CWQ @50 / CWQ in-dist @50 | 64.1 / 53.6 · 67.2 / 56.8 | — |

- CWQ in-dist @100: hơn SubgraphRAG khoảng +1.1 Hit và +1.2 F1, dù reasoner khác nhau.
- Transfer @50: F1 cao hơn WL+ADR+AER (+1.8) nhưng Hit thấp hơn (−1.7). Cả hai đều cách xa số paper của REL-RAG.

**Đã sửa thêm trong evaluator:** dataset giờ được nhận theo thư mục `/webqsp/` hoặc `/cwq/` trong đường dẫn. Trước đây evaluator tìm chuỗi con trong tên file, nên file `cwq_rbfnbwebqsp_*` bị chấm như WebQSP.

## 10. Đính chính 10-13: tập full của WebQSP là 1,628 câu

- Các số RBF trên WebQSP ở mục 8 được chấm trên 1,639 câu. Trong đó có 11 câu không có đáp án gold, nên không chấm được và bị tính là sai.
- Trên đúng tập full 1,628 câu (tập mà paper và file chính thức dùng), kết quả là:
  - top-50: **82.74 / 68.65**;
  - top-100: **84.28 / 69.41**.
- Mọi số QA và retrieval chính thức giờ nằm ở `BENCHMARK.md`, sinh bởi `relrag/benchmark_all.py` và `reason/metrics/bench_qa.py`.

## 11. RBF + ADR / AER (10-13, task 1): kết quả âm

**Cách chạy:** `adr_probe.py --run webqsp_rbfnb_s42 --modes aer,adr,adr_aer --rbf_mm 1`. Không train lại. Chấm bằng `relrag/eval_rows.py` (cùng metric với BENCHMARK, có thêm ansHit).

Kết quả theo dạng ansR@50 / @100 / @200 · ansHit@50 · SP-F1@50 · GPT-R@50:

| Biến thể | WebQSP (ID, 1,559 câu) | WebQSP → CWQ (OOD, 2,848 câu) |
|---|---|---|
| RBF+nb+mm | 91.4 / 95.2 / 98.0 · 94.2 · 15.4 · 83.1 | **87.1 / 92.1 / 96.0** · 87.6 · 8.7 · 66.8 |
| + AER | 91.2 / 95.0 / 97.8 · 94.4 · 15.5 · 83.1 | 86.0 / 91.9 / 96.0 · 86.6 · 8.5 · 66.1 |
| + ADR | 91.5 / 95.4 / 98.0 · 94.4 · 15.4 · 83.0 | 82.9 / 89.4 / 94.9 · 83.5 · 8.0 · 61.6 |
| + AER + ADR | 91.2 / 95.2 / 97.8 · 94.4 · 15.5 · 82.9 | 81.9 / 89.1 / 94.5 · 82.7 · 7.9 · 60.9 |
| *đối chiếu:* RBF+nb+mm + BGE | 92.0 / 95.7 / 97.9 · — · 15.3 · 84.5 | 88.7 / 93.5 / 96.4 · 89.3 · 8.8 · 70.1 |
| *đối chiếu:* WL + AER + ADR | 91.9 / 96.0 / 98.1 · — · 15.3 · 84.6 | 83.2 / 89.8 / 94.7 · 84.9 · 8.5 · 65.6 |

**Nhận xét:**
- ADR và AER không giúp RBF. Trên CWQ chúng còn làm giảm: AER −1.1, ADR −4.2, cả hai −5.2 ansR@50.
- Lý do: RBF đã tính đường đi riêng cho từng anchor và có thêm hạng tử giao giữa các anchor (`inter`).
  - ADR thêm các lượt chỉ có một anchor rồi gộp bằng max z-score. Cách gộp này đẩy lên các triple quanh từng anchor riêng lẻ, nên mất tín hiệu giao.
  - AER coi cầu nối là anchor mới, nhưng các cầu nối này vốn đã nằm trong đường đi T = 3 của RBF.
  - Hai mẹo này chỉ bù cho backbone chấm từng triple (DDE, WL), không bù cho backbone chấm theo đường đi.

## 12. RBF + BGE + WL (+ nhãn rational), train trên vast (10-13, task 2)

**Cấu hình:**
- `qrdde.py --feat rbfonly --rbf_nb 1 --rel_prior bge --pos_weight effective_num --label_src {sp,union} --lean_emb 1`, seed 42, patience 10. Test bật mm.
- Box #55761477 (RTX 4060 Ti, 32 core). Mỗi epoch WebQSP khoảng 66 s.

**Bộ embedding "lean"** (`make_lean_emb.py`):
- Chỉ giữ embedding câu hỏi, bảng relation dùng chung và số entity có text. `lex` tính sẵn.
- Dung lượng: WebQSP train 40 MB (trước 5.2 GB); CWQ train 164 MB (trước: shard 59 GB, hoặc bản gemb 4 GB).
- `to_device` nhanh hơn 17 lần. Logit lệch tối đa 5e-4 do khử trùng lặp vector relation.
- Mỗi epoch CWQ khoảng 7 phút, trước là khoảng 12 phút.

**Kết quả train trên WebQSP** (ansR@50 / @100 / @200 · ansHit@50 / @100 · SP-F1@50 · GPT-R@50):

| Run | WebQSP (ID, 1,559 câu) | WebQSP → CWQ (OOD, 2,848 câu) |
|---|---|---|
| SubgraphRAG (OOD: repro) | 90.4 / 94.4 / 96.9 · 94.6 / 97.0 · 15.2 · 81.0 | 70.0 / 79.6 / 87.7 · 73.4 / 81.9 · 6.9 · 53.6 |
| RBF+nb+mm (bản chính) | 91.4 / 95.2 / 98.0 · 94.2 / 96.5 · 15.4 · 83.1 | 87.1 / 92.1 / 96.0 · 87.6 / 92.4 · 8.7 · 66.8 |
| + BGE chỉ lúc test (β 0.5) | 92.0 / 95.7 / 97.9 · 94.8 / 97.1 · 15.3 · 84.5 | **88.7 / 93.5 / 96.4 · 89.3 / 93.8 · 8.8 · 70.1** |
| train BGE + WL, nhãn SP | 91.9 / 95.6 / 97.4 · 94.8 / **97.4** · 15.4 · 84.8 | 87.9 / 92.3 / 95.2 · 88.7 / 92.8 · 8.5 · 68.4 |
| **train BGE + WL, nhãn SP ∪ rational** | **92.7** / 95.6 / 97.3 · **95.4** / 97.2 · **15.7** · **85.0** | 87.2 / 92.0 / 95.0 · 87.9 / 92.5 · 8.5 · 65.8 |

**Nhận xét** (một seed; dao động giữa các seed của RBF khoảng 0.8 điểm trên WebQSP, nên các chênh lệch dưới 1 điểm ở đây chưa chắc chắn):
- **In-dist:** cấu hình đầy đủ (BGE + WL + nhãn rational) tốt nhất trên WebQSP @50: ansR 92.7, ansHit 95.4 (vượt SubgraphRAG 94.6), GPT-R 85.0.
  - Nhãn rational đóng góp khoảng +0.8 ansR@50 và +0.6 ansHit@50 so với nhãn SP.
- **OOD:** không có cấu hình train nào hơn bản cộng BGE lúc test. Nhãn rational còn làm giảm GPT-R (−2.6) trên CWQ.
- **Đuôi phân bố:** WL làm @200 / @500 kém đi ở cả hai tập (−0.6 đến −1.0).

**Kết quả train trên CWQ, `cwq_rbfnb_bge_wl_sp_s42`** (BGE + WL, nhãn SP; CWQ không có nhãn rational):
- Box #55761477 biến mất khỏi tài khoản trước khi run ghi xong `result.json`.
- `best.pt` (val tốt nhất ở epoch 4) đã được đồng bộ về local trước đó. Lịch sử wandb cho thấy val không cải thiện từ epoch 5 đến 12, và với patience 10 run sẽ dừng sau epoch 14. Vì vậy coi checkpoint này là kết quả cuối. `result.json` được dựng lại và có ghi chú.

Kết quả (ansR@50 / @100 / @200 · ansHit@50 / @100 · SP-F1@50 · GPT-R@50):

| Run | CWQ (ID, 2,848 câu) | CWQ → WebQSP (OOD, 1,559 câu) |
|---|---|---|
| SubgraphRAG (ID: official; OOD: repro train CWQ) | 86.2 / 91.4 / 95.3 · 87.8 / 92.6 · 10.4 · 75.5 | 82.2 / 88.2 / 92.9 · 90.8 / 94.0 · 12.6 · 71.8 |
| REL-RAG, checkpoint tác giả `cwq.pt` | 80.8 / 85.9 / 89.8 · — · 9.0 · 45.1 | **90.9 / 95.8 / 97.7 · 94.7 / 97.8** · 16.3 · 62.3 |
| RBF+nb+mm (CWQ) | 90.5 / 94.1 / 96.8 · 90.9 / 94.5 · 10.0 · 73.4 | 87.2 / 90.9 / 95.0 · 91.0 / 93.4 · 14.4 · 80.5 |
| **RBF+nb+mm, train BGE + WL (CWQ)** | **94.5 / 96.9 / 98.4 · 94.8 / 97.1 · 10.6 · 78.7** | 89.7 / 94.1 / 96.2 · 93.3 / 96.4 · 14.6 · **81.7** |

- **CWQ (ID):** tăng +4.0 ansR@50 so với RBF, và +8.3 so với SubgraphRAG. Con số này ngang trần "relation oracle" đã đo trước đây (95.5), đúng với chẩn đoán rằng chọn relation là nút thắt trên CWQ.
  - Gate "@50 > SubgraphRAG @200" **sát nhưng chưa đạt** (94.5 so với 95.3). Ở @100 đã vượt SubgraphRAG @300 (96.9 so với 96.8).
- **CWQ → WebQSP:** tăng +2.5 @50, nhưng vẫn thua checkpoint của REL-RAG (−1.2 @50, −1.7 @100).
- **Không rò nhãn test:** adapter BGE chỉ train trên WebQSP train, và CWQ train/val chỉ dùng điểm tính từ adapter đó.
- **Kiểm tra lợi thế không công bằng qua adapter BGE:** 2,328/3,531 câu CWQ test được dựng từ câu WebQSP *train*, mà adapter BGE train trên WebQSP train. Tách ansR@50 theo nguồn gốc câu (RBF CWQ → BGE + WL):
  - nhóm dựng từ WebQSP test (adapter chưa từng thấy câu gốc): 90.7 → **94.1** (+3.4);
  - nhóm dựng từ WebQSP train: 90.5 → 94.7 (+4.2).
  - → Mức tăng chủ yếu là thật. Phần có thể do adapter đã thấy câu gốc chỉ khoảng 0.8 điểm.
- **BGE lúc test hay lúc train** (model `cwq_rbfnb_s42`, CWQ ID, ansR@50 / @100 / @200):
  - không BGE: 90.5 / 94.1 / 96.8;
  - cộng BGE lúc test: 91.8 / 94.7 / 96.8;
  - train BGE + WL: **94.5 / 96.9 / 98.4**.
  - → Khi tập train lớn (CWQ: 22k câu), π học cách dùng đặc trưng BGE tốt hơn nhiều so với cộng prior lúc test. Trên WebQSP (2.7k câu train) thì hai cách ngang nhau.

## 13. QA CWQ với retriever RBF + BGE + WL (train CWQ), 10-14

**Cách chạy:**
- File retrieval `reason/custom_retrieval/cwq_rbfbgewl_mm_top100.pth`: 3,531 câu, mm bật. 683 câu không có đáp án trong graph được chấm bằng `score_extra.py`; điểm BGE của chúng tính trên CPU local, khớp điểm tính trên box (lệch tối đa 8e-4).
- Reasoner: Llama-3.1-8B fp8 (DeepInfra), `sys_icl_dc`, response thô.
- Chấm theo cách của paper: Hit của eval gốc, Macro-F1 của eval corrected với match gốc. Bộ chấm cho lại đúng 56.87 / 47.06 với SubgraphRAG.

| Run | Full 3,531: Hit / Macro-F1 | Subset 2,848: Hit / Macro-F1 |
|---|---|---|
| SubgraphRAG official @100 (train CWQ, bf16) | 56.87 / 47.06 | 65.66 / 54.62 |
| RBF+nb+mm (CWQ) @50 | 55.82 / 46.66 | 64.68 / 54.50 |
| RBF+nb+mm (CWQ) @100 | 57.97 / 48.24 | 67.13 / 56.00 |
| **RBF + BGE + WL (CWQ) @50** | **57.89 / 49.98** | **67.28 / 58.37** |
| RBF + BGE + WL (CWQ) @100 | 58.06 / 49.13 | 67.45 / 57.26 |
| *Paper REL-RAG CWQ in-dist @50* | *67.2 / 56.8* | — |

**Nhận xét:**
- Retrieval tăng +4.0 ansR@50 kéo theo QA tăng (cùng reasoner, @50): Hit +2.1, F1 +3.3.
- **So với SubgraphRAG @100:** chỉ dùng một nửa số triple mà Hit +1.0 và Macro-F1 **+2.9**. Reasoner khác nhau (fp8 so với bf16), nên chưa phải phép so ngang hàng hoàn toàn.
- @100 không hơn @50: Hit +0.2, F1 −0.9.
- Còn cách số paper REL-RAG khoảng −9.3 Hit và −6.8 F1.

## 14. Nhãn từ truy vấn gold (SPARQL / InferentialChain): độ phủ trên subgraph, 10-15

**Cách đo:**
- Code: `retrieve/exp_qrdde/sparql_labels.py` (bộ trích nhãn người dùng cung cấp, giữ nguyên) và `sparql_coverage.py` (đo trên các record của mình). Chạy CPU, không gọi LLM.
- WebQSP đọc thẳng `InferentialChain` và `Constraints` từ annotation: ràng buộc thời gian của WebQSP nằm trong `FILTER(NOT EXISTS … EXISTS {…})`, nên `parse_sparql` xếp chúng vào filters chứ không phải pattern.
- CWQ dùng `parse_sparql` + `extract_label`. MID được đổi sang tên bằng annotation WebQSP và tên topic có trong record.
- Local chỉ có CWQ **test** gốc; bản train cần tải về.

| | WebQSP train (2,674 câu có nhãn) | CWQ test (2,691/2,848 câu có nhãn) |
|---|---|---|
| chain_complete (trần của nhãn mới) | **97.7%** | **94.5%** |
| Ứng viên cuối chain so với đáp án gold trong graph | trung vị 2, trung bình 7.5 so với 5.0; ứng viên > gold ở 23% số câu | trung vị 4, trung bình 9.5 so với 1.6; **ứng viên > gold ở 66% số câu** |
| Triple chain: tỉ lệ đang dương dưới nhãn SP / tỉ lệ nhãn SP nằm trên chain | 54% / 49% | **16% / 35%** |
| Triple ràng buộc mỗi câu (số câu có ít nhất 1) | 2.8 (20%) | 9.3 (55%) |
| Tỉ lệ triple ràng buộc đang là **nhãn âm** dưới SP | **99.4%** | **92.6%** |
| Hop (topo, ngữ nghĩa) | (1, 1) 1,629 · (2, 1) 983 | (2, 1) 1,016 · (1, 1) 677 · (2, 2) 641 · (3, 2) 136 · (4, 2) 72 |
| Trong tầm walk T = 3 của RBF | 100% | 100% (subgraph là 2-hop nên mọi triple đều với tới được) |
| Ràng buộc gold có mặt trong subgraph, theo loại | entity 97.7% (556) · **thời gian / literal 0% (332)** · **ORDER BY 0% (70)** | type 97.2% (216) · entity 92.8% (774) · khác 47.5% (1,496) · **ORDER BY 0% (208)** |

**Kết luận:**
- **Trần nhãn mới rất cao** (95–98%), nên dùng thẳng nhãn mới, không cần bản lai với SP.
- **Nhãn SP rất nhiễu:** trên CWQ chỉ 16% triple của chain gold đang dương dưới SP, và 65% nhãn SP nằm ngoài chain. Gần như mọi triple ràng buộc đang bị dạy là nhãn âm.
- **Giá trị lớn nhất nằm ở CWQ:** 66% số câu có nhiều ứng viên hơn đáp án, và bằng chứng để chọn đúng ứng viên chính là các triple ràng buộc đó.
- **Giới hạn cứng:** subgraph của SubgraphRAG/RoG không có triple literal (ngày tháng, giá trị số). Ràng buộc thời gian và khoá ORDER BY vắng mặt hoàn toàn, nên mọi retriever (kể cả các baseline) đều không lấy được. Câu "khi nào / lần cuối" phải dựa vào kiến thức sẵn có của LLM.

## 15. So sánh nguồn nhãn: SP, hybrid (GPT-4o + DeepSeek), SPARQL (10-15)

**Cách đo:**
- Code: `retrieve/exp_qrdde/label_compare.py`. Đã tải CWQ train/dev gốc vào `relrag/data/cwq_orig/` (mọi câu đều có SPARQL).
- Mỗi câu tính trên chính subgraph của nó; chỉ lấy các câu có đáp án gold trong graph.
- Nếu một nguồn nhãn không có nhãn cho một câu, câu đó được tính là không phủ gì.
- Nhãn SPARQL được lưu ở `data/<ds>/sparql_<split>.pkl` (chỉ số triple chain và triple ràng buộc).

| Tập | Nhãn | #triple TB / trung vị | phủ ≥1 đáp án | phủ đủ đáp án | topic → đủ đáp án | hop topo 1 / 2 / 3 / 4+ (%) | % triple chain SPARQL mà nhãn này cũng đánh dấu | % triple của nhãn nằm trên truy vấn |
|---|---|---|---|---|---|---|---|---|
| WebQSP train (2,710) | SP | 24.7 / 4 | 100 | 100 | 100 | 70 / 30 / 0 / 0 | 77.3 | 47.8 |
| | hybrid | 6.3 / 4 | 98.8 | 84.5 | 84.5 | 65 / 34 / 0 / 0 | 70.7 | 63.6 |
| | SPARQL chain | 11.0 / 3 | 96.1 | 93.5 | 93.5 | 62 / 38 / 0 / 0 | — | — |
| | SPARQL chain + ràng buộc | 13.7 / 3 | 96.1 | 93.5 | 93.5 | 62 / 38 / 0 / 0 | — | — |
| CWQ train (21,943) | SP | 13.7 / 5 | 100 | 100 | 100 | 56 / 43 / 0 / 0 | 42.6 | 39.2 |
| | SPARQL chain | 11.8 / 4 | 88.8 | 88.6 | 88.6 | 26 / 65 / 6 / 2 | — | — |
| | SPARQL chain + ràng buộc | 19.8 / 7 | 89.2 | 89.0 | 89.0 | 50 / 44 / 5 / 0 | — | — |
| CWQ test (2,848) | SP | 11.7 / 5 | 99.4 | 99.4 | 100 | 54 / 44 / 0 / 0 | 39.9 | 40.4 |
| | SPARQL chain | 12.3 / 5 | 88.1 | 88.1 | 88.2 | 27 / 64 / 5 / 2 | — | — |

- Nhãn SPARQL gắn được vào subgraph cho 96% câu WebQSP và 90% câu CWQ train (19,732/21,943). Trong số các câu gắn được, khoảng 98% phủ đủ đáp án. Khoảng 10% câu CWQ không có nhãn SPARQL: 945 câu không parse được hoặc không đổi được MID sang tên, và 1,266 câu có chain không đi được trong subgraph.
- **SP là nhãn đi đường tắt.** Trên CWQ, 56% đường SP chỉ dài 1 hop, trong khi chain gold chỉ 26% là 1 hop (65% là 2 hop, 8% từ 3 hop trở lên). SP và truy vấn gold chỉ trùng khoảng 40% theo cả hai chiều.
- **Hybrid** gọn (6 triple) và nằm trên truy vấn nhiều hơn SP (64% so với 48%), nhưng chỉ phủ đủ đáp án ở 84.5% số câu. Hybrid cũng chỉ có cho WebQSP.
- **Đề xuất nhãn train:** SPARQL chain + ràng buộc. Với khoảng 4–10% câu không có nhãn SPARQL, dùng SP thay thế.

## 16. Retrieval đo theo truy vấn gold (10-15, `relrag/query_recall.py`)

**Câu được chấm:** câu test có chain gold đi được trong subgraph và tới được đáp án gold. WebQSP: 1,496 câu (257 câu có triple ràng buộc). CWQ: 2,510 câu (1,376 câu có triple ràng buộc).

**Metric** (khớp triple theo tên, nên RAPL / REL-RAG bị tính thiệt khi trả relation theo chiều ngược):
- **Chain-R:** recall của triple trên chain gold.
- **Cons-R:** recall của triple ràng buộc gold. Object phải khớp đúng tên.
- **Q-F1:** F1 so với toàn bộ truy vấn (chain + ràng buộc).
- **PathHit:** top-k chứa trọn đường từ topic tới ít nhất 1 đáp án.
- **PathAll:** top-k chứa trọn đường tới mọi đáp án.

Mỗi ô ghi @50 / @100:

| Setting | Phương pháp | Chain-R | Cons-R | Q-F1 | PathHit | PathAll |
|---|---|---|---|---|---|---|
| **WebQSP (ID)** | SubgraphRAG (official) | 84.8 / 91.0 | 11.5 / 15.4 | 15.6 / 11.4 | 84.1 / 90.1 | 76.6 / 84.0 |
| | REL-RAG repro (path_bf) | 85.7 / 89.6 | **32.3 / 43.4** | **18.7 / 13.2** | 88.1 / 91.6 | 77.9 / 83.4 |
| | WL | 88.2 / 93.1 | 13.0 / 21.0 | 16.3 / 11.7 | **89.0 / 93.9** | 81.0 / 88.0 |
| | WL + ADR + AER | 87.1 / 92.5 | 17.4 / 24.7 | 15.9 / 11.4 | 88.5 / 93.2 | 79.5 / 87.5 |
| | RBF+nb+mm | 86.5 / 91.6 | 16.3 / 27.4 | 16.6 / 11.9 | 85.6 / 90.6 | **83.0 / 89.7** |
| | *(thêm) RBF + BGE + WL, nhãn SP ∪ rational* | *89.1 / 93.8* | *23.5 / 34.2* | *18.2 / 13.0* | *90.0 / 93.8* | *86.2 / 91.8* |
| **WebQSP → CWQ (OOD)** | SubgraphRAG (repro) | 52.2 / 62.4 | 27.1 / 34.0 | 12.9 / 10.2 | 44.8 / 53.9 | 40.6 / 50.4 |
| | REL-RAG repro (path_bf) | 43.5 / 50.1 | 12.0 / 15.5 | 18.0 / 12.8 | 38.4 / 43.9 | 37.6 / 43.0 |
| | WL | 58.0 / 69.0 | 32.7 / 40.2 | 16.3 / 12.7 | 46.4 / 57.8 | 41.6 / 53.4 |
| | WL + ADR + AER | 62.6 / 73.3 | 37.5 / 46.6 | 16.8 / 13.0 | 56.8 / 68.4 | 52.5 / 64.8 |
| | RBF+nb+mm | **67.8 / 76.5** | **37.8 / 49.9** | **20.3 / 15.4** | **62.7 / 72.8** | **61.8 / 72.7** |
| | *(thêm) RBF+nb+mm + BGE lúc test* | *69.3 / 77.8* | *41.6 / 53.1* | *20.7 / 15.7* | *65.1 / 75.1* | *64.5 / 74.8* |
| **CWQ (ID)** | SubgraphRAG (official) | 66.1 / 76.6 | 39.9 / 49.4 | 18.2 / 13.8 | 67.4 / 78.0 | 63.9 / 75.0 |
| | REL-RAG (checkpoint tác giả `cwq.pt`) | 54.3 / 59.7 | 12.3 / 16.4 | 19.7 / 13.6 | 45.5 / 49.3 | 44.9 / 48.4 |
| | RBF+nb+mm (CWQ) | 72.0 / 80.5 | 47.7 / 59.4 | 20.9 / 15.6 | 71.2 / 80.2 | 71.0 / 80.0 |
| | *(thêm) RBF + BGE + WL (CWQ)* | *72.9 / 81.6* | *58.0 / 65.9* | *21.4 / 15.7* | *77.3 / 84.9* | *77.0 / 84.7* |
| **CWQ → WebQSP (OOD)** | SubgraphRAG (repro) | 72.5 / 81.4 | 16.5 / 21.8 | 12.3 / 9.2 | 77.3 / 85.2 | 60.9 / 71.0 |
| | REL-RAG (checkpoint tác giả `cwq.pt`) | 85.3 / 90.1 | **33.5 / 46.0** | **18.8 / 13.2** | **89.7 / 93.2** | 79.1 / 84.8 |
| | RBF+nb+mm (CWQ) | 83.5 / 88.2 | 24.3 / 36.5 | 15.1 / 10.4 | 83.6 / 88.5 | 80.9 / 85.7 |
| | *(thêm) RBF + BGE + WL (CWQ)* | *85.7 / 90.9* | *24.8 / 32.3* | *16.2 / 11.6* | *86.2 / 91.4* | *82.6 / 89.1* |

**Nhận xét:**
- **Triple ràng buộc bị bỏ sót nặng ở mọi phương pháp.** Trên WebQSP @50, các model train bằng nhãn SP chỉ lấy được 11–17% triple ràng buộc. REL-RAG (nhãn GPT) cao nhất với 32%. Điều này khớp với việc nhãn SP đang dạy ràng buộc là nhãn âm.
- **WebQSP (ID):** RBF lấy trọn đường cho **mọi** đáp án tốt nhất (PathAll@50 83.0). Nhưng tỉ lệ có trọn đường cho **ít nhất một** đáp án (PathHit@50 85.6) lại kém WL (89.0) và REL-RAG (88.1). Điều này khớp với ansHit thấp hơn và QA Hit thấp hơn đã thấy ở mục 8–10.
- **OOD WebQSP → CWQ:** RBF dẫn ở mọi metric (PathHit@50 62.7, so với 56.8 của WL+ADR+AER và 44.8 của SubgraphRAG).
- **CWQ (ID):** RBF + BGE + WL dẫn rõ (PathHit@50 77.3 so với 67.4 của SubgraphRAG; Cons-R@50 58.0).
- **CWQ → WebQSP:** REL-RAG dẫn ở PathHit và Cons-R.

## 17. Train lại bằng nhãn SPARQL, WebQSP (Kaggle 2× T4, 10-16)

**Cấu hình:**
- `qrdde.py --label_src sparql`: nhãn = chain gold + triple ràng buộc, áp cho cả train và val. 97 câu train và 7 câu val không gắn được chain nên giữ nhãn SP.
- Seed 42, patience 10. Hai run:
  - `webqsp_rbfnb_sparql_s42` (RBF, nb, mm bật lúc test): best ở epoch 27;
  - `webqsp_dde_sparql_s42` (SubgraphRAG repro): best ở epoch 34.
- Môi trường Kaggle: torch 2.4 cu124 (local là torch 2.1).

**Metric độc lập với nhãn** (ansR@50 / @100 / @200 · ansHit@50 / @100 · GPT-R@50):

| Run | WebQSP (ID, 1,559 câu) | WebQSP → CWQ (OOD, 2,848 câu) |
|---|---|---|
| SubgraphRAG official | 90.4 / 94.4 / 96.9 · 94.6 / 97.0 · 81.0 | — |
| DDE, nhãn SP | 90.1 / 94.4 / 97.2 · 94.9 / 97.3 · 81.0 | 70.0 / 79.6 / 87.7 · 73.4 / 81.9 · 53.6 |
| **DDE, nhãn SPARQL** | 92.1 / 96.5 / 98.4 · 95.1 / 97.9 · 69.2 | 70.7 / 81.3 / 89.6 · 73.0 / 83.3 · 41.2 |
| RBF+nb+mm, nhãn SP | 91.4 / 95.2 / 98.0 · 94.2 / 96.5 · 83.1 | **87.1** / 92.1 / 96.0 · **87.6** / 92.4 · **66.8** |
| **RBF+nb+mm, nhãn SPARQL** | **94.1 / 97.1 / 98.7 · 96.7 / 98.1** · 80.8 | 86.4 / **92.2 / 96.2** · 87.1 / **92.6** · 62.2 |

**Metric theo truy vấn gold** (PathHit@50 / @100 · Cons-R@50 · Q-F1@50):

| Run | WebQSP (1,496 câu) | WebQSP → CWQ (2,510 câu) |
|---|---|---|
| SubgraphRAG official | 84.1 / 90.1 · 11.5 · 15.6 | — |
| DDE, nhãn SP | 83.4 / 88.8 · 12.9 · 14.6 | 44.8 / 53.9 · 27.1 · 12.9 |
| **DDE, nhãn SPARQL** | 92.1 / 96.5 · 50.6 · 20.5 | 42.0 / 49.8 · 12.2 · 19.0 |
| RBF+nb+mm, nhãn SP | 85.6 / 90.6 · 16.3 · 16.6 | 62.7 / 72.8 · 37.8 · 20.3 |
| **RBF+nb+mm, nhãn SPARQL** | **94.4 / 97.1 · 58.1 · 21.2** | **65.6 / 74.5** · 37.6 · **24.0** |
| WL / REL-RAG repro (đối chiếu) | 89.0 / 93.9 · 13.0 · 16.3 / 88.1 / 91.6 · 32.3 · 18.7 | WL+ADR+AER: 56.8 / 68.4 · 37.5 · 16.8 |

**Nhận xét:**
- **In-dist tăng mạnh ở cả hai backbone**, kể cả trên metric không phụ thuộc nhãn train:
  - RBF: ansR@50 +2.7, ansHit@50 +2.5. RBF @100 (97.1) đã vượt SubgraphRAG @200 (96.9); @50 (94.1) vẫn chưa vượt @200.
  - DDE: ansR@50 +2.0.
  - Trên metric theo truy vấn: PathHit@50 +8.8 (RBF) và +8.7 (DDE); Cons-R@50 từ 13–16 lên 51–58.
- **Một tín hiệu ngược chiều:** GPT-R@50 giảm (RBF −2.3, DDE −11.8). Nhãn GPT và truy vấn gold không đồng nhất.
- **OOD (WebQSP → CWQ) hầu như không đổi:**
  - RBF: ansR@50 −0.7, nhưng PathHit@50 +2.9 và Q-F1@50 +3.7.
  - DDE: Cons-R và PathHit giảm.
  - Nhãn SPARQL của WebQSP gần như toàn chain 1 hop ngữ nghĩa, nên không dạy được các tổ hợp của CWQ.
- **Lưu ý:** metric theo truy vấn đo cùng loại nhãn với nhãn train, nên mức tăng trên các metric này một phần là do mục tiêu train khớp với metric. Bằng chứng độc lập là ansR / ansHit tăng.

## 18. Nhãn SPARQL: QA WebQSP và retrieval CWQ (10-16)

**QA WebQSP** (RBF+nb+mm nhãn SPARQL, Llama-3.1-8B fp8 qua DeepInfra, chấm theo cách của paper; chi phí khoảng $0.18). Mỗi ô ghi Hit / Macro-F1:

| Run | Full 1,628 câu | Subset 1,559 câu |
|---|---|---|
| SubgraphRAG official @100 (bf16) | 86.61 / 70.57 | 88.58 / 72.10 |
| REL-RAG repro @50 (fp16 T4) | 84.95 / 70.15 | 86.59 / 71.41 |
| RBF+nb+mm, nhãn SP @50 / @100 | 82.74 / 68.65 · 84.28 / 69.41 | 84.48 / 70.20 · 86.08 / 70.89 |
| **RBF+nb+mm, nhãn SPARQL @50 / @100** | **83.78 / 69.12** · 84.15 / 69.15 | 85.76 / 70.61 · 86.02 / 70.53 |

**Phân tích theo câu** (subset 1,559 câu):

| Run | Đáp án có trong context (ansHit) | QA Hit | P(đúng \| đáp án có trong context) |
|---|---|---|---|
| SubgraphRAG @100 (bf16) | 97.0 | 88.6 | **90.9** |
| RBF SPARQL @100 (fp8) | **98.1** | 86.0 | 87.3 |

- Retrieval của RBF SPARQL giờ đã **hơn** SubgraphRAG: có 30 câu đáp án chỉ nằm trong context của RBF, ngược lại chỉ có 12 câu.
- Toàn bộ khoảng thua QA nằm ở **bước LLM đọc context**. Trên 1,500 câu mà cả hai context đều có đáp án, LLM đúng 91.1% với context SubgraphRAG so với 87.6% với context RBF.
- Hai nguyên nhân chưa tách được:
  1. reasoner fp8 so với bf16 (riêng Q4 so với bf16 đã mất 1.2 điểm);
  2. thành phần context: RBF ít relation khác nhau hơn (19 so với 27) và ít triple CVT hơn (28 so với 50).
- Phép thử tách hai nguyên nhân: chạy SubgraphRAG official @100 qua đúng reasoner fp8, khoảng $0.08.

**Retrieval CWQ** (`cwq_rbfnb_sparql_s42`: RBF+nb+mm nhãn SPARQL, không BGE/WL; train trên Kaggle, best ở epoch 23). Mỗi ô ghi ansR@50 / @100 · ansHit@50 · PathHit@50 · Cons-R@50 · Q-F1@50:

| Run | CWQ (ID) | CWQ → WebQSP (OOD) |
|---|---|---|
| SubgraphRAG (ID: official; OOD: repro) | 86.2 / 91.4 · 87.8 · 67.4 · 39.9 · 18.2 | 82.2 / 88.2 · 90.8 · 77.3 · 16.5 · 12.3 |
| REL-RAG (checkpoint tác giả) | — | **90.9 / 95.8 · 94.7 · 89.7** · **33.5** · **18.8** |
| RBF+nb+mm, nhãn SP | 90.5 / 94.1 · 90.9 · 71.2 · 47.7 · 20.9 | 87.2 / 90.9 · 91.0 · 83.6 · 24.3 · 15.1 |
| RBF + BGE + WL, nhãn SP | **94.5 / 96.9 · 94.8** · 77.3 · 58.0 · 21.4 | 89.7 / 94.1 · 93.3 · 86.2 · 24.8 · 16.2 |
| **RBF+nb+mm, nhãn SPARQL** | 92.5 / 95.5 · 92.9 · **82.0 · 60.5 · 26.9** | 89.9 / 94.0 · 92.7 · 88.2 · 29.3 · **18.8** |

- **CWQ (ID):** nhãn SPARQL cho mức tăng lớn nhất trên metric theo truy vấn (PathHit@50 +10.8 so với nhãn SP). ansR@50 +2.0, nhưng vẫn thấp hơn bản dùng BGE + WL.
- **CWQ → WebQSP:** khoảng cách với REL-RAG thu hẹp còn −1.0 ansR@50 và −1.5 PathHit@50.
- **Bước tiếp:** ghép nhãn SPARQL với BGE + WL; hai thành phần này bổ sung cho nhau.

## 19. Context của RBF làm LLM chọn sai: chẩn đoán và cách trình bày context (10-16)

**Phép thử cùng reasoner** trên 167 câu WebQSP bất đồng. Đây là các câu mà cả hai context đều chứa đáp án, nhưng SubgraphRAG (bf16, file tác giả) và RBF nhãn SPARQL (fp8) cho kết quả đúng/sai khác nhau. Tất cả các dòng dưới đây dùng Llama-3.1-8B fp8 qua DeepInfra, @100 (`reason/context_variants.py`):

| Context | Đúng trên 110 câu (SubgraphRAG bf16 đúng) | Đúng trên 57 câu (RBF đúng) | Tổng /167 |
|---|---|---|---|
| SubgraphRAG | 81 | 33 | **114** |
| RBF, thứ tự gốc, chính lần chạy dùng để chọn ra 167 câu | 0 | 57 | 57 (thiên lệch do cách chọn) |
| **RBF, thứ tự gốc, chạy lại (đối chứng)** | 36 | 54 | **90** |
| RBF pathorder (mỗi triple đi sau đường từ topic tới nó) | 50 | 45 | 95 |
| **RBF diverse** (bỏ cặp entity trùng theo hai chiều, tối đa 8 triple mỗi relation, bù từ top-500) | 65 | 41 | **106** |
| RBF both (diverse + pathorder) | 62 | 42 | 104 |

**Nhận xét:**
- **Reasoner fp8 rất nhiễu dù temperature 0:** chỉ chạy lại cùng context đã đổi đúng/sai ở 36/110 câu. Mọi so sánh QA phải có đối chứng chạy lại, và các chênh lệch toàn tập dưới khoảng 1 Hit chưa kết luận được.
- **pathorder gần như không giúp** (+5 so với đối chứng, nằm trong mức nhiễu).
- **diverse giúp rõ** (+16 so với đối chứng), thu hẹp còn khoảng một nửa khoảng cách với context SubgraphRAG (114).
- Điều này khớp với file phân tích `reason/results/analysis/webqsp_sgr_right_rbf_wrong.xlsx` (110 câu):
  - context RBF chứa đủ chain gold nhiều hơn (88% so với 65%), nhưng ít đa dạng relation hơn (18.6 so với 25.6), có nhiều ứng viên có tên hơn (51 so với 37) và ít triple chạm topic hơn (47% so với 64%);
  - LLM bị nhiễu bởi nhiều chain hoàn chỉnh dẫn tới các ứng viên sai (ví dụ "who is niall ferguson's wife": cả hai người vợ cũ và hiện tại đều có chain đầy đủ).
- **Bước tiếp:**
  - chạy diverse trên toàn bộ WebQSP;
  - sau đó đưa tính đa dạng vào chính retriever: phạt các triple cùng relation hoặc cùng cặp entity khi xếp hạng, thay vì xử lý sau khi đã xếp hạng.

## 20. Vì sao "retrieval tốt hơn mà QA kém hơn": câu trả lời khi so cùng reasoner (10-17)

**Cách làm:** mọi run dưới đây chạy trên toàn bộ WebQSP test (1,628 câu), Llama-3.1-8B fp8 qua DeepInfra, @100. RBF nhãn SPARQL được chạy **hai lần** để đo mức nhiễu của reasoner. Chi phí khoảng $0.34.

| Run | Full: Hit / Macro-F1 | Subset 1,559 câu: Hit / Macro-F1 |
|---|---|---|
| SubgraphRAG, bf16 (file tác giả, *khác reasoner*) | 86.61 / 70.57 | 88.58 / 72.10 |
| **SubgraphRAG, fp8** | **85.87 / 68.12** | 87.68 / 69.47 |
| RBF nhãn SPARQL, fp8, lần 1 | 84.15 / 69.15 | 86.02 / 70.53 |
| RBF nhãn SPARQL, fp8, lần 2 | 85.07 / 70.26 | 86.79 / 71.57 |
| **RBF nhãn SPARQL + diverse, fp8** | **85.57** / 68.82 | 87.43 / 70.15 |

**Kiểm định McNemar trên Hit** (1,628 câu):

| Cặp so sánh | Chỉ A đúng | Chỉ B đúng | % câu bất đồng | p |
|---|---|---|---|---|
| RBF lần 1 vs lần 2 (mức nhiễu) | 29 | 44 | 4.5% | 0.10 |
| SubgraphRAG bf16 vs fp8 | 82 | 69 | 9.3% | 0.33 |
| SubgraphRAG fp8 vs RBF lần 1 | 123 | 92 | 13.2% | 0.04 |
| SubgraphRAG fp8 vs RBF lần 2 | 111 | 95 | 12.7% | 0.30 |
| SubgraphRAG fp8 vs RBF + diverse | 109 | 104 | 13.1% | **0.78** |
| RBF + diverse vs RBF lần 1 / lần 2 | 100 / 90 | 74 / 79 | 10.7% / 10.4% | 0.06 / 0.44 |

**Kết luận:**
- **Khoảng thua "vô lý" (86.6 so với 83.8–84.2 Hit) phần lớn không phải lỗi của RBF.**
  - Khoảng 0.7–0.8 Hit (và 2.5 F1) là do **khác reasoner**: file SubgraphRAG là bf16 của tác giả.
  - Khoảng 0.9 Hit là **nhiễu giữa các lần chạy**: hai lần chạy y hệt của RBF lệch nhau 0.9 Hit và 1.1 F1.
  - Phần còn lại, khoảng 1 Hit, là do **thành phần context**: các chain song song, đồng điểm, dẫn tới ứng viên sai (mục 19).
- **Cùng reasoner,** RBF ngang SubgraphRAG về Hit: 84.6 (trung bình hai lần) và 85.6 với diverse, so với 85.9 (p = 0.78). RBF **hơn** về Macro-F1: 69.7 so với 68.1.
- **Ràng buộc đa dạng (diverse)** xoá khoảng thua Hit còn lại, nhưng mức tăng so với RBF gốc chưa có ý nghĩa thống kê.
- **Hệ quả về phương pháp:**
  - Trên WebQSP in-dist, retrieval chỉ hơn khoảng 1 điểm ansHit, nhỏ hơn mức nhiễu của QA (khoảng ±1 Hit), nên một run đơn lẻ **không thể** cho thấy RBF thắng ở QA.
  - Muốn chứng minh ở QA thì phải: (a) so cùng reasoner, (b) chạy lặp lại hoặc dùng McNemar, (c) tập trung vào các setting có khoảng cách retrieval lớn, như CWQ in-dist (+8 ansR@50) và OOD.

## 21. Nhãn SPARQL v2, cắt theo ngưỡng và gộp chain (10-17)

**Nhãn SPARQL v2** (`make_sparql_v2.py`): chỉ giữ các trail của chain gold kết thúc ở đáp án gold, cùng các ràng buộc gắn vào chúng.
- Số triple nhãn mỗi câu giảm: WebQSP từ 12.6 xuống 7.9, CWQ từ 18.3 xuống 5.7.
- Số câu có chain dẫn cả tới ứng viên sai: 23% (WebQSP), 61–66% (CWQ).

**Train `webqsp_rbfnb_bge_wl_sparql2_s42`** (Kaggle 2× T4; BGE + WL + v2):
- WebQSP test: ansR@50 93.6, ansHit@100 98.3, PathHit@50 90.9.
- Số ứng viên sai cuối chain gold @100 vẫn là **1.85** (v1: 1.98; SubgraphRAG: 0.78).
- → **Nhãn không sửa được nhiễu chain song song.** RBF chấm điểm chỉ theo chuỗi relation, nên các trail có cùng chuỗi relation luôn cùng điểm, bất kể nhãn.

**Cắt theo ngưỡng τ** (giữ triple có log-prob ≥ tốt nhất − τ):
- Ở mọi τ, tỉ lệ ứng viên sai trong số node cuối chain vẫn khoảng 16%, vì nhóm triple đồng điểm bị giữ hoặc bỏ cùng nhau.
- Ngưỡng chỉ làm context gọn lại: τ = 2 còn 36 triple (trung vị 16), ansHit 96.4.

**QA với τ = 2 + gộp chain** (`context_variants.py taumerge`): gộp cặp bước qua CVT thành `x, r1 > r2, y`, gộp các nhánh song song thành `[y1, y2, …]`, bỏ CVT không nối được về topic.
- Trung bình 9.3 dòng mỗi câu; ansHit giảm không đáng kể (96.3 → 96.0).
- Toàn WebQSP, fp8: **84.46 / 67.88**.
- McNemar: so với SubgraphRAG fp8 p = 0.15, so với RBF lần 2 p = 0.68, so với RBF + diverse p = 0.26.

**Tổng hợp tất cả cấu hình WebQSP cùng reasoner fp8:** Hit nằm trong khoảng 84.2–85.9, Macro-F1 trong khoảng 67.9–70.3. Không cặp nào vượt mức nhiễu (hai lần chạy y hệt lệch nhau 0.9 Hit).
- → Trên WebQSP in-dist, **QA không nhạy với khác biệt retrieval trong khoảng này**. Thứ quyết định là reasoner 8B và các thông tin vắng mặt trong subgraph (thời gian), không phải cách chọn hay trình bày context.
- Lỗi chủ yếu giống nhau ở mọi hệ thống: chọn một entity khác trong context (khoảng 40%) và trả lời ngoài context (khoảng 30%).
- **Hướng tiếp:** đo QA ở các setting có khoảng cách retrieval lớn (CWQ in-dist, OOD), có baseline SubgraphRAG chạy cùng reasoner fp8.

## 22. CWQ: RBF + BGE + WL + nhãn SPARQL v2 (`cwq_rbfnb_bge_wl_sparql2_s42`, Kaggle, 10-17)

Run này dừng ở epoch 28; best ở epoch 17, val ansR@50 96.5.

**CWQ (ID, 2,848 câu):**

| Retriever | ansR@50 / @100 / @200 | ansHit@50 / @100 | PathHit@50 / @100 | Cons-R@50 | Q-F1@50 |
|---|---|---|---|---|---|
| SubgraphRAG official | 86.2 / 91.4 / 95.3 | 87.8 / 92.6 | 67.4 / 78.0 | 39.9 | 18.2 |
| RBF+nb+mm, nhãn SP | 90.5 / 94.1 / 96.8 | 90.9 / 94.5 | 71.2 / 80.2 | 47.7 | 20.9 |
| RBF + BGE + WL, nhãn SP | 94.5 / 96.9 / 98.4 | 94.8 / 97.1 | 77.3 / 84.9 | 58.0 | 21.4 |
| RBF+nb+mm, nhãn SPARQL v1 | 92.5 / 95.5 / 97.5 | 92.9 / 95.8 | 82.0 / 87.5 | 60.5 | **26.9** |
| **RBF + BGE + WL, nhãn SPARQL v2** | **95.1 / 97.3 / 98.7** | **95.5 / 97.6** | **85.7 / 90.8** | **65.2** | 24.2 |

**CWQ → WebQSP (OOD, 1,559 câu):**

| Retriever | ansR@50 / @100 / @200 | ansHit@50 / @100 | PathHit@50 / @100 |
|---|---|---|---|
| SubgraphRAG repro (train CWQ) | 82.2 / 88.2 / 92.9 | 90.8 / 94.0 | — |
| REL-RAG, checkpoint tác giả `cwq.pt` | 90.9 / **95.8 / 97.7** | **94.7 / 97.8** | 89.7 / 93.2 |
| RBF + BGE + WL, nhãn SP | 89.7 / 94.1 / 96.2 | 93.3 / 96.4 | 86.2 / 91.4 |
| **RBF + BGE + WL, nhãn SPARQL v2** | **91.6** / 94.9 / 97.3 | 94.4 / 96.3 | **90.4 / 94.1** |

**Nhận xét:**
- **CWQ (ID):**
  - Gate "@50 > SubgraphRAG @200" **gần như đạt**: 95.1 so với 95.3 (−0.2). Ở @100 đã vượt SubgraphRAG @300 (97.3 so với 96.8).
  - PathHit@50 hơn SubgraphRAG **+18.3**; Cons-R@50 cao nhất trong mọi cấu hình.
  - Nhãn v2 và BGE + WL bổ sung cho nhau: hơn từng thành phần riêng lẻ.
- **CWQ → WebQSP:**
  - Lần đầu RBF vượt checkpoint của REL-RAG ở ansR@50 (+0.7) và PathHit@50 (+0.7).
  - Ở @100 vẫn thua (−0.9 ansR); ansHit gần ngang.
- Run nhãn v2 thuần (`cwq_rbfnb_sparql2_s42`) vẫn đang chạy trên Kaggle.
- **Bước tiếp:** QA CWQ cùng reasoner fp8 (SubgraphRAG fp8 so với cấu hình này).

## 23. QA CWQ với RBF + BGE + WL + nhãn v2, và trần do reasoner (10-17)

**QA CWQ** (Llama-3.1-8B fp8, @50, 3,531 câu; chi phí khoảng $0.23), Hit / Macro-F1:

| Run | Full | Subset 2,848 câu |
|---|---|---|
| SubgraphRAG @100 (bf16, file tác giả) | 56.87 / 47.06 | 65.66 / 54.62 |
| RBF + BGE + WL, nhãn SP, @50 | 57.89 / **49.98** | 67.28 / **58.37** |
| **RBF + BGE + WL, nhãn SPARQL v2, @50** | 57.83 / 49.50 | **67.59** / 57.89 |

→ Nhãn v2 tăng retrieval (PathHit@50 +8.4, ansR@50 +0.6) nhưng QA giữ nguyên.

**Vì sao retrieval tốt hơn nhiều mà QA chỉ nhích:** tỉ lệ LLM trả lời đúng khi đã có đủ bằng chứng.

| Context | Số câu có **trọn đường suy luận** gold | Hit khi có trọn đường | Hit khi có đáp án nhưng thiếu đường | Hit khi không có đáp án |
|---|---|---|---|---|
| SubgraphRAG @100 (bf16) | 73.1% | 71.1 | 60.3 | 4.8 |
| RBF + BGE + WL v2 @50 (fp8) | **80.1%** | 70.3 | 59.7 | 4.0 |

- **Ngay cả khi context chứa trọn đường suy luận gold, Llama-3.1-8B chỉ đúng khoảng 70% câu CWQ.** Đây là trần do reasoner.
- Giữa "có trọn đường" và "có đáp án nhưng thiếu đường", tỉ lệ đúng chỉ chênh khoảng 11 điểm. Vì vậy tăng tỉ lệ câu có trọn đường thêm 7 điểm (73 → 80) chỉ đem lại khoảng 7% × 11 ≈ **0.8 Hit**, khớp với mức tăng quan sát được (khoảng 1 Hit so với SubgraphRAG).
- **Kết luận:**
  - Với reasoner 8B, lợi ích QA từ retrieval bị chặn bởi (thay đổi về độ phủ) × (khoảng cách tỉ lệ đúng có điều kiện), mà khoảng cách sau rất nhỏ.
  - Muốn QA tăng mạnh thì phải cải thiện **bước reasoning**: cách đưa đường suy luận vào prompt, hoặc reasoner mạnh hơn. Retrieval không còn là nút thắt.
  - Con số của paper REL-RAG (67.2 Hit trên CWQ) cao hơn cả trần khoảng 71% đo được ở đây khi đã có trọn đường suy luận. Điều này củng cố kết luận EXP-31: phần chênh nằm ở reasoner và cách chấm của paper.

## 24. Tách tác dụng của nhãn v2 (`cwq_rbfnb_sparql2_s42`: RBF+nb+mm, nhãn v2, không BGE/WL; best ở epoch 14)

Ở bảng này, PathHit được tính trên **mọi** câu có nhãn SPARQL, kể cả câu mà chain không tới được đáp án (tính là trượt). Vì vậy số thấp hơn ở mục 16–22, nhưng các dòng trong bảng vẫn so được với nhau.

| Train CWQ → | Retriever | ansR@50 / @100 | ansHit@50 | PathHit@50 / @100 |
|---|---|---|---|---|
| CWQ | RBF+nb+mm, nhãn v1 | 92.5 / 95.5 | 92.9 | 76.7 / 81.8 |
| CWQ | RBF+nb+mm, nhãn v2 | 93.1 / 95.7 | 93.5 | 75.4 / 80.8 |
| CWQ | RBF + BGE + WL, nhãn v2 | **95.1 / 97.3** | **95.5** | **80.1 / 84.7** |
| WebQSP | RBF+nb+mm, nhãn v1 | 89.9 / 94.0 | 92.7 | 85.6 / 89.6 |
| WebQSP | RBF+nb+mm, nhãn v2 | 90.4 / 93.8 | 93.1 | 87.2 / 89.2 |
| WebQSP | RBF + BGE + WL, nhãn v2 | **91.6 / 94.9** | **94.4** | **88.1 / 91.2** |

- Nhãn v2 so với v1, cùng kiến trúc: chênh lệch nhỏ (ansR@50 +0.5 đến +0.6; PathHit lúc tăng lúc giảm khoảng 1.5 điểm).
- Phần lớn mức tăng của cấu hình tốt nhất đến từ **BGE + WL** (ansR@50 thêm +2.0 / +1.2).

## 25. Prompt lab: cách đưa bằng chứng vào reasoner (Llama-3.1-8B Q4 qua ollama, val, 10-17)

**Mục đích:** retrieval đã đủ recall (mục 23), nên nút thắt nằm ở cách đưa bằng chứng vào prompt. Để tối ưu prompt mà không tốn tiền API, mình dùng Llama-3.1-8B Q4_K_M chạy local qua ollama (temperature 0, seed 0).

**Thiết lập:**
- **Dữ liệu dev:** tập **validation**, retrieval RBF + BGE + WL + nhãn v2, top-500.
- **Số câu:** WebQSP 234 câu (mọi câu val có đáp án), CWQ 300 câu (lấy mẫu, seed 0).
- **Cách chấm:** như paper (Hit theo eval gốc; Macro-F1 theo eval corrected với match chuỗi con gốc).
- **Code:** `reason/prompt_lab.py`. Kết quả ở `reason/results/prompt_lab/<variant>_<ds>_k<k>.jsonl`.
- **Khung prompt:** mọi biến thể giữ khung `sys_icl_dc` (system + một ví dụ ICL + câu hỏi). Ví dụ ICL được chuyển đổi cùng cách với context thật.

| Biến thể | Mô tả | WebQSP Hit / F1 | CWQ Hit / F1 | McNemar gộp (Hit) so với T0 |
|---|---|---|---|---|
| T0 (k50) | triple `(h,r,t)` theo thứ tự điểm (pipeline hiện tại) | 83.8 / 68.4 | 72.3 / 61.0 | – |
| T1 | triple, tên relation rút gọn, CVT thành nhãn đọc được | 81.6 / 64.8 | 74.0 / 62.4 | – |
| P1 | đường suy luận từ topic, dạng `A -> rel -> B`, gấp CVT | 85.5 / 67.8 | 73.7 / 63.4 | p 0.40 |
| P2 | P1 + danh sách ứng viên (điểm cuối path) + chỉ dẫn chọn chain khớp câu hỏi | 88.0 / 70.0 | 76.3 / 63.9 | p 0.018, ΔF1 +2.3 |
| P2a | P1 + chỉ dẫn, không có danh sách ứng viên | 87.6 / 69.4 | 74.7 / 64.9 | p 0.085 |
| P2b | P1 + danh sách ứng viên, không có chỉ dẫn | 86.8 / 67.9 | 77.0 / 66.1 | p 0.014 |
| P3 | P2 + ví dụ ICL minh hoạ quy trình (gọi tên chain → liệt kê path khớp → trả lời) | 88.5 / 71.6 | 78.7 / 58.2 | p 0.001, ΔF1 −0.1 |
| **P2 (k100)** | P2 với top-100 | **89.3 / 71.5** | 75.7 / **64.2** | p 0.013, **ΔF1 +3.2** |
| P4 (k100) | P3 sửa định dạng: dòng `ans:` chỉ có tên entity, chỉ dẫn "mỗi entity một lần" | 90.6 / 71.3 | 77.3 / 56.9 | p 0.001, ΔF1 −1.0 |

**Chi tiết theo precision / recall (corrected eval):**

| Biến thể | WebQSP P / R | CWQ P / R | Đáp án/câu (CWQ) | Câu không có dòng `ans:` (WebQSP / CWQ) |
|---|---|---|---|---|
| T0 k50 | 75.9 / 68.9 | 64.8 / 61.7 | 1.8 | 10 / 22 |
| P2 k100 | 78.9 / 73.5 | 67.3 / 64.3 | 1.6 | 4 / 23 |
| P3 k50 | 74.9 / 77.3 | 57.8 / 63.9 | 1.8 | 7 / 39 |
| P4 k100 | 73.2 / 79.0 | 55.8 / 65.6 | 2.2 | 9 / 32 |

**Kết luận vòng 1–3:**
- **Định dạng path giúp, nhưng chỉ khi có thêm chỉ dẫn hoặc danh sách ứng viên.**
  - P1 (chỉ đổi triple thành path) không khác T0 có ý nghĩa.
  - P2 / P2b hơn T0 có ý nghĩa thống kê (p < 0.02 trên 534 câu).
  - Danh sách ứng viên là thành phần mạnh nhất trên CWQ (P2b: F1 +5.1).
- **Top-100 không làm nhiễu khi đã trình bày dạng path:** P2 k100 ≥ P2 k50 ở cả hai tập, ΔF1 so với T0 là +3.2.
- **Ví dụ ICL minh hoạ quy trình (P3, P4) làm tăng Hit nhưng giảm F1 trên CWQ, do precision tụt khoảng 10 điểm.** Ba kiểu lỗi:
  1. Model chép câu "mọi entity tới qua chain này đều là đáp án" của ví dụ, nên trả lời mọi điểm cuối path. Ví dụ, câu hỏi "I'm Not There" được trả lời cả Bob Dylan lẫn Dylan Thomas.
  2. Model liệt kê path rồi lặp đến hết 512 token mà không viết được dòng `ans:` (CWQ: 32–39 câu, so với 23 ở P2).
  3. Model kết luận "không path nào khớp" rồi bỏ cuộc, thay vì dùng kiến thức riêng như P2.
  Hit tăng một phần là do trả lời nhiều hơn, không phải chọn đúng hơn. Với reasoner 8B Q4, ví dụ ICL dài kiểu chain-of-thought có hại cho câu nhiều ràng buộc.
- **Ứng viên được chọn:** P2 với k=100. Kiểm định trên test (mẫu 600 câu mỗi tập, seed 1, so với T0 k100 cùng bằng chứng) ở mục 26.

## 26. Kiểm định prompt P2 trên test (Llama-3.1-8B Q4, ollama, 10-17)

**Thiết lập:**
- Mỗi tập lấy mẫu 600 câu test, seed 1.
- Retrieval là export test của `webqsp_rbfnb_bge_wl_sparql2_s42` / `cwq_rbfnb_bge_wl_sparql2_s42`. Topic và đáp án lấy từ file `scored_triples` của SubgraphRAG.
- Hai prompt dùng **cùng top-100 bằng chứng**, nên chênh lệch chỉ do cách trình bày.
- Lệnh chạy: `python prompt_lab.py <P2|T0> <ds> --test --n 600 --k 100`.

| Tập (600 câu) | T0 (triple, prompt gốc) Hit / F1 | **P2** (path + ứng viên + chỉ dẫn) Hit / F1 | Chỉ P2 đúng / chỉ T0 đúng | McNemar p | ΔF1 |
|---|---|---|---|---|---|
| WebQSP | 86.8 / 65.6 | **89.0 / 67.4** | 38 / 25 | 0.13 | +1.8 |
| CWQ | 60.8 / 48.6 | **65.5 / 51.9** | 73 / 45 | 0.013 | +3.2 |
| Gộp (1,200) | | | 111 / 70 | **0.003** | **+2.5** |

- **Kết quả val được giữ trên test:** P2 hơn T0 ở cả hai tập. Trên CWQ và khi gộp, chênh lệch có ý nghĩa thống kê. Prompt P2 cũng ngắn hơn khoảng 12% số ký tự.
- **Số câu thiếu dòng `ans:`:** WebQSP: P2 27, T0 45. CWQ: P2 65, T0 58. Phần lớn do lab giới hạn 512 token đầu ra. Khi chạy thật sẽ dùng `max_tokens` 4000 như SubgraphRAG.
- **Bước tiếp theo:** chạy toàn bộ tập test (WebQSP 1,628, CWQ 3,531) trên Llama-8B bf16 bằng `prompt_lab.py --test --api --n 0`, gồm bốn cấu hình: {RBF, SubgraphRAG} × {T0, P2}, k=100. Bốn cấu hình này tách được phần đóng góp của retrieval và của prompt.

## 27. Chạy toàn tập test trên Llama-3.1-8B bf16 (vLLM 0.8.4, H100 FPT, 10-17)

**Thiết lập:**
- Model `unsloth/Meta-Llama-3.1-8B-Instruct`: cùng trọng số với bản gốc, không gated.
- Sinh: temperature 0, seed 0, `max_tokens` 4000, 128 request đồng thời.
- Lệnh: `prompt_lab.py --test --api --n 0`. Hàng đợi ở scratchpad `bf16_queue.sh`, kết quả ở `reason/results/prompt_lab/bf16/`.
- Khoảng 60k lượt gọi trong khoảng 2.3 giờ.

**Cảnh báo về server:** khoảng 11% câu CWQ bị lặp tới hết 4000 token (403 câu trong ô SubgraphRAG × T0 CWQ). File bf16 tác giả công bố chỉ có 11 câu như vậy.
- **Cách kiểm tra:** gửi lại 30 câu bị lặp trên server rảnh với đúng prompt của tác giả.
  - 16/30 câu vẫn lặp; đặt `top_p=1, top_k=-1` vẫn còn 12/30 câu lặp.
  - Hai lần chạy cùng một câu chỉ cho output trùng nhau 1/30 lần, tức là output không tất định khi chạy batch.
  - Output khởi đầu giống hệt của tác giả rồi mới rẽ sang lặp. Đây là khác biệt số học của kernel bf16 giữa hai phiên bản vLLM, không phải do prompt.
  - Đặt `frequency_penalty=0.16` thì hết lặp (0/30), nhưng làm vậy là lệch protocol.
- **Hệ quả:**
  - Phần text lặp nhắc tên rất nhiều entity, nên **Hit trên text thô bị thổi lên**. Ví dụ ô SubgraphRAG × T0 CWQ đạt 63.2 Hit trên text thô, so với 56.87 của tác giả.
  - F1 thì bị kéo xuống.
  - Vì vậy dùng thêm **ans-Hit**: Hit chỉ tính trên các dòng `ans:`, không bị ảnh hưởng bởi text lặp.
  - Các so sánh dưới đây đều là so sánh cặp trên cùng server. **Không so trực tiếp với số paper.**

**Bảng 2×2 (toàn tập test, k=100; ô ghi ans-Hit / Macro-F1):**

| Retrieval \ Prompt | WebQSP T0 | WebQSP P2 | CWQ T0 | CWQ P2 |
|---|---|---|---|---|
| SubgraphRAG | 83.3 / 68.6 | 85.1 / 69.4 | 54.2 / 45.6 | 56.2 / 48.7 |
| RBF + BGE + WL v2 | 83.2 / 69.9 | **86.4 / 72.2** | 57.6 / 49.3 | **60.8 / 53.4** |

**So sánh cặp (McNemar trên ans-Hit):**
- **Phương pháp (RBF + P2) so với pipeline gốc (SubgraphRAG + T0):**
  - WebQSP: +3.1 ans-Hit (p 0.001), F1 +3.6.
  - CWQ: **+6.5 ans-Hit (p 6e-15), F1 +7.8**.
- **Riêng tác dụng của retrieval:**
  - Khi dùng T0: WebQSP −0.1 (p 0.95); CWQ +3.3 (p 5e-5).
  - Khi dùng P2: WebQSP +1.3 (p 0.16); CWQ +4.6 (p 1e-8).
- **Riêng tác dụng của prompt P2:**
  - Trên bằng chứng của SubgraphRAG: +1.8 / +2.0.
  - Trên bằng chứng của RBF: **+3.2 / +3.2** (p ≤ 5e-5).
  - Prompt và retrieval cộng hưởng nhau: dạng path phát huy tốt hơn khi bằng chứng là chain đúng.
- **Ngân sách k (P2):** k50 / k100 / k200 không khác nhau có ý nghĩa ở cả hai tập, vì P2 giới hạn ở 40 path. SubgraphRAG với k200 cũng không tốt hơn k100.
- **Ablation trên bf16 (so với P2):**
  - P2a (bỏ danh sách ứng viên) ngang P2: WebQSP −1.0, CWQ +1.0, đều không có ý nghĩa.
  - P2b (bỏ chỉ dẫn) kém hơn trên CWQ (raw-Hit −1.8, p 0.003).
  - P1 (chỉ dạng path) kém hơn trên WebQSP (raw-Hit −1.5, p 0.04).
  - Vậy **chỉ dẫn** là thành phần quan trọng trên bf16. Kết quả này khác với Q4, nơi danh sách ứng viên mạnh nhất.
- **Transfer (đánh giá trên subset):**
  - CWQ→WebQSP, RBF + P2: 86.5 / 72.9, so với SubgraphRAG ID 85.2 / 70.0 (+1.3, p 0.21).
  - WebQSP→CWQ: **RBF + T0 đạt 64.5 / 55.0**, cao hơn SubgraphRAG train trên CWQ (62.0 / 52.2). Còn P2 lại kém T0 2.7 điểm (p 0.003). Như vậy prompt P2 không giúp khi retriever chạy ngoài miền huấn luyện.
