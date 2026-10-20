# RBF-Ret: lập luận cho paper, lợi và hại so với SubgraphRAG / REL-RAG, và việc còn thiếu (10-19)

Tổng hợp từ docs/11 (thiết kế, giả thuyết), EXP-39 (RBF), docs/12 (gate, nhãn SPARQL, QA), EXP-40 và BENCHMARK.md §3–6. Số retrieval là ansR@50 trên cùng tập câu của từng setting (1,559 / 2,848); số QA là ans-Hit / Macro-F1, Llama-3.1-8B bf16 (vLLM), toàn tập test, k = 100.

## 0. Trả lời thẳng: RBF đã hơn REL-RAG paper chưa?

**Hơn REL-RAG khi cùng điều kiện: có, rõ ở mọi ô transfer. Hơn con số trong paper REL-RAG: chưa, và cũng không nên so trực tiếp.**

**Số chốt (10-19): vLLM 0.10.1, `meta-llama/Llama-3.1-8B-Instruct` bf16, k = 100, `prompt_lab.py --loop_fix`.**
- `--loop_fix`: câu bị cắt ở 4,000 token (greedy rơi vào vòng lặp; 2–11% câu) được cắt ở câu lặp đầu tiên, rồi hỏi thêm một lượt chỉ lấy dòng `ans:`. Áp như nhau cho mọi ô.
- Server được kiểm chứng bằng ô SubgraphRAG × T0 so với file bf16 tác giả: WebQSP 86.67 / 69.87 so với 86.61 / 70.57; CWQ 59.5 / 47.9 so với 56.87 / 47.06 (server này cao hơn tác giả khoảng 2.6 Hit trên CWQ).
- Kết quả ở `reason/results/prompt_lab/bf16_v010/`.

| QA, Hit / Macro-F1 (chấm như paper) | REL-RAG paper (@50) | RBF | Chênh |
|---|---|---|---|
| WebQSP | 88.8 / 77.3 | 87.7 / 72.7 (RBF + BGE + WL v2, P2) | −1.1 / −4.6 |
| CWQ | 67.2 / 56.8 | 65.4 / 55.4 (v2, P2) | −1.8 / −1.4 |
| WebQSP → CWQ, **toàn tập 3,531** | 64.1 / 53.6 | 59.5 / 48.7 (RBF+nb+mm lõi, T0); v2 ước tính 60.6 / 49.7 | −3.5 đến −4.6 / −3.9 đến −4.9 |
| *(WebQSP → CWQ, tập con 2,848)* | — | *69.2 / 57.0 (v2, T0)* | *không so được: paper chấm toàn tập* |

- Hiệu chuẩn công bằng thì **RBF chưa vượt số trong paper REL-RAG ở ô nào**. Số 69.2 / 57.0 ở WebQSP → CWQ chỉ tính trên tập con; 683 câu còn lại (đáp án không có trong subgraph) chỉ đạt 24.7 Hit.
- **So với số trong paper SubgraphRAG** (86.61 / 70.57; 56.87 / 47.06): RBF hơn ở WebQSP (+1.1 / +2.1) và CWQ (+8.5 / +8.3; +5.9 / +7.5 nếu trừ phần server cao hơn).
- Nhưng số của paper REL-RAG **không tái hiện được** (EXP-31):
  - decode dùng nhãn test (rò rỉ, +4–6 ansR);
  - không công bố parser đáp án;
  - Hit của paper cao hơn trần chiếu từ chính retrieval của tác giả 13–17 điểm.
  - Vì vậy paper chỉ nên so **cùng protocol**: REL-RAG chạy lại bằng code tác giả và checkpoint tác giả, cùng reasoner, cùng cách chấm. Bảng gốc của REL-RAG đưa vào kèm chú thích.
- **Cùng protocol thì RBF dẫn ở mọi ô transfer.** Ở ID, RBF ngang hoặc nhỉnh hơn: QA WebQSP 86.4 so với 85.6, p ≈ 0.4, chưa có ý nghĩa thống kê.

## 1. Phương pháp và quá trình đi tới nó

### 1.1 Chuỗi bằng chứng dẫn tới RBF (câu chuyện của paper)

1. **Quan sát.** Retriever kiểu SubgraphRAG (MLP trên [z_q, z_h, z_r, z_t, DDE]) sụt mạnh khi train WebQSP rồi test CWQ: ansR@50 70.0, so với 86.2 khi train trên CWQ. REL-RAG còn tệ hơn: 61.9 (repro), 66.8 (checkpoint tác giả).
2. **Chẩn đoán (D1, docs/11 §1).** Mức sụt **không phải do đường dài hơn**. Câu ≥ 3 bước chỉ chiếm 171 / 2,831 câu; phần sụt nằm ở câu 2 bước. Chia câu 2 bước theo việc *tổ hợp relation* đã xuất hiện trong nhãn train hay chưa:
   - tổ hợp đã thấy: 66.0;
   - **tổ hợp mới nhưng relation đã thấy: 49.2**;
   - mốc train CWQ: 88.5.
   → Lỗi là **tổng quát hoá tổ hợp** (systematicity): mô hình nhớ cặp (r1, r2), không học được từng bước.
3. **Manh mối.** Mẹo inference AER (neo lại ở entity cầu nối, chạy lại 1 bước) tăng nhóm tổ hợp mới 49.2 → 74.8. AER chỉ làm một việc: **phân rã** tổ hợp 2 bước thành hai bước 1 bước đã quen. Nhưng là mẹo, nó làm hỏng các ô khác (ID giảm 0.2–1.4, CWQ → WebQSP thua).
4. **Giả thuyết H1.** Nếu điểm của một đường đi *bằng cấu trúc* là tích các nhân tử một bước, và nhân tử chỉ phụ thuộc (q, r), thì tổ hợp mới của relation đã thấy được chấm đúng mà không cần đã thấy tổ hợp đó. Phân rã được đưa vào trong mô hình thay vì làm lúc inference.
5. **Hiện thực trên entity graph.** REL-RAG lập luận rằng MPNN trên entity graph bị "relation mixing" nên phải chuyển sang line graph. Nhưng lập luận đó chỉ đúng cho thông điệp **cộng dồn**. Max-product (Bellman-Ford tổng quát, như NBFNet nhưng dùng max) giữ mỗi đường đi là một tích riêng, không trộn, với chi phí O(|E|·T). Không cần line graph: REL-RAG tốn 77 MB/câu, khoảng 175 GB × 2 và 3.5 giờ dựng graph cho CWQ test (EXP-38).
6. **Kiểm chứng cơ chế, đây là bằng chứng mạnh nhất cho paper.** Thêm lại một MLP nhìn thấy đặc trưng bước sau cùng relation (`rbfmlp_nb`, *không* có DDE) thì nhóm tổ hợp mới **tụt từ 84 về 51.2**, ngang WL (53.1). Mọi bộ chấm kết hợp được đặc trưng nhiều bước đều học thuộc tổ hợp. Chỉ khi điểm thuần phân rã theo bước thì mới tổng quát hoá. Cách kết hợp PoE và max-z giữa hai head cũng thất bại theo đúng cách đó (docs/11 §7).
7. **Hai sửa lỗi có nguyên tắc** (không thêm tham số):
   - **nb**, walk không quay lui: hub (múi giờ, type) sinh đường "nảy" r⁻¹ rồi r, làm ngập budget. Max-product chính xác trên walk không quay lui chỉ cần 2 trạng thái mỗi nút (đã kiểm brute force trên 300 đồ thị). Đây chính là tính chất mà line graph có được nhờ trạng thái là cạnh. Tác dụng: trung tính trên WebQSP/CWQ, **+2.9 / +3.9 / +4.3 trên GrailQA**.
   - **mm**, max-marginal của đường đi trọn vẹn: điểm tiền tố không tăng dọc đường đi, nên hop anchor → CVT luôn xếp trên CVT → đáp án. Kết quả là CVT ("member" × 80) ngập budget. mm chấm mỗi triple bằng **walk trọn vẹn tốt nhất đi qua nó**, kết thúc ở entity có tên (ràng buộc cấu trúc, không tham số). Tác dụng: WebQSP 89.6 → 91.4, WebQSP → CWQ 81.8 → 87.1, chỉ bật lúc test.

### 1.2 Mô hình (phần cốt lõi)

- π(r | q) = σ(MLP[z_q; z_r; z_q ⊙ z_r; lex(q, r); cos(q, r)]), hai đầu ra cho chiều xuôi và ngược. **Đây là tham số học duy nhất.**
- Với mỗi anchor a: log p_aᵗ⁺¹(v) = max(log p_aᵗ(v), max₍u,r,v₎ log p_aᵗ(u) + log π(r)), T = 3, walk không quay lui.
- Điểm triple = max-marginal của walk trọn vẹn tốt nhất qua triple đó (mm), hợp qua các anchor bằng max. Lấy top-k triple, nên giao diện với LLM giống hệt SubgraphRAG.
- Train: BCE trên nhãn triple (SP hoặc SPARQL), điểm triple là logit của xác suất đường đi.

## 2. Bằng chứng tổng hợp

**Retrieval, ansR@50** (BENCHMARK §3, §4, §6; REL-RAG ID dùng repro code-faithful, OOD dùng checkpoint tác giả nếu có):

| Setting | SubgraphRAG | REL-RAG | **RBF+nb+mm** (nhãn SP) | RBF + BGE + WL, nhãn SPARQL v2 |
|---|---|---|---|---|
| WebQSP (ID) | 90.4 | 90.6 | 91.4 (2 seed: 91.0) | 93.5 |
| WebQSP → CWQ | 70.0 | 61.9 (tác giả 66.8) | **87.1** (seed 0: 84.4) | **90.3** |
| CWQ (ID) | 86.2 | 80.8 (tác giả) | 90.5 | 95.1 |
| CWQ → WebQSP | 82.2 | **90.9** (tác giả) | 87.2 | 91.6 |
| WebQSP → GrailQA (728) | 40.3 | 42.7 | **58.9** | — |
| WebQSP → GTSQA (1,622) | 35.4 | 38.8 | **57.7** | — |
| WebQSP → MetaQA-3hop | 21.5 | 14.4 (9.6 triple/câu) | **42.4** | — |
| Nhóm "tổ hợp mới" (587 câu CWQ 2 bước) | 49.2 (DDE) | — | **85.7** | — |

**QA** (bf16, ans-Hit / F1; BENCHMARK §6.2, docs/12 §27):

| | SubgraphRAG + T0 | REL-RAG code-faithful (tốt nhất) | RBF + BGE + WL v2 (tốt nhất) |
|---|---|---|---|
| WebQSP | 83.3 / 68.6 | 85.6 / 70.9 (P2) | **86.4 / 72.2** (P2) |
| CWQ | 54.2 / 45.6 | — | **60.8 / 53.4** (P2) |
| WebQSP → CWQ (2,848) | 62.0 / 52.2 (*train CWQ*) | 47.6 / 38.7 (T0) | **64.5 / 55.0** (T0) |

## 3. Lợi và hại

### 3.1 So với SubgraphRAG

| Lợi | Bằng chứng |
|---|---|
| Tổng quát hoá tổ hợp, đúng bản chất lỗi của SubgraphRAG | Nhóm tổ hợp mới 49.2 → 85.7; WebQSP → CWQ +17.1 @50 (RBF@k ≈ SubgraphRAG@2k) |
| Transfer sang KG / domain khác | GrailQA +18.6, GTSQA +22.3, MetaQA +20.9 @50 (bootstrap CI > 0) |
| Ít tham số, ít thứ để học thuộc | Chỉ học π(r \| q); không DDE, không MLP trên entity |
| Lấy được ràng buộc và trọn đường suy luận khi có nhãn SPARQL | Cons-R@50 11.5 → 65.8, PathHit@50 84.1 → 93.6 (WebQSP) |
| Giao diện y hệt (top-k triple), chi phí cùng bậc | Lan truyền O(\|E\|·T) trên entity graph, không cần lưu gì thêm |
| QA tăng khi trình bày dạng path | CWQ +6.5 ans-Hit (p 6e-15), WebQSP +3.1 (p 0.001) |

| Hại | Bằng chứng / hệ quả |
|---|---|
| **π chỉ phụ thuộc relation, không nhìn entity.** Các chain song song cùng mẫu relation có điểm bằng nhau → nhiễu (đáp án sai cùng loại) | Mất bằng chứng kiểu / entity ("Hollywood studio"); 87 / 91 câu 1 bước thua DDE là do π sai hoặc thiếu bằng chứng entity (docs/11 §7). Bản RBF đầu chỉ được 85.7 ở câu 1 bước so với 94.5 của DDE (mm thu hẹp lại) |
| **Câu nhiều anchor** không được mô hình hoá: hợp bằng max, giao mềm bị bác (H3) | Peek ban đầu: câu ≥ 2 topic 75.3 so với 75.7 (DDE); soft-AND làm giảm. mm đưa lên 85.6, nhưng vẫn là hợp, không phải giao |
| Lợi ích retrieval **không tự chuyển thành QA** với prompt gốc | WebQSP T0: 83.2 so với 83.3. Phải đổi cách trình bày (P2) mới ăn được |
| Điểm là log-prob đường đi tốt nhất (max, không sum); mọi điểm âm, khó hiệu chuẩn thành "độ liên quan" | Không cộng dồn được bằng chứng từ nhiều đường; cắt theo ngưỡng chỉ có lợi trên WebQSP |
| Phương sai seed ở transfer lớn | WebQSP → CWQ: seed 42 87.1, seed 0 84.4 |
| Gate ID chưa đạt | WebQSP 91.4 < 96.9 (SubgraphRAG @200); CWQ 95.1 < 95.3 (bản v2, sát mốc) |

### 3.2 So với REL-RAG

| Lợi | Bằng chứng |
|---|---|
| Phủ định có điều kiện luận điểm "relation mixing": đạt tính chất của line graph trên entity graph | Max-product giữ từng đường đi; nb = 2 trạng thái/nút thay vì O(Σ deg²) chuyển trạng thái của line graph |
| Rẻ hơn nhiều | Không dựng line graph (REL-RAG: 77 MB/câu, 3.5 giờ dựng + 1 giờ post-process cho CWQ test) |
| Không cần nhãn từ LLM | REL-RAG cần rational path từ GPT-4o; RBF dùng SP hoặc truy vấn gold |
| Transfer tốt hơn rất nhiều | WebQSP → CWQ +25.2 @50 (repro), +20.3 (checkpoint tác giả); GrailQA +16.2; MetaQA: REL-RAG tự dừng ở 9.6 triple |
| Budget kiểm soát được | top-k; REL-RAG decode tự do trả số triple thay đổi (42–193 triple/câu tuỳ checkpoint) |
| Bền với nguồn nhãn | Nhãn SPARQL v2 giúp RBF ở mọi ô, nhưng làm REL-RAG kém đi khi transfer (−4 đến −5 ansR@50, −3.7 đến −5.1 QA) |

| Hại | Bằng chứng |
|---|---|
| **CWQ → WebQSP: checkpoint tác giả REL-RAG vẫn tốt hơn RBF nhãn SP** | 90.9 so với 87.2 @50. Chỉ bản RBF + BGE + WL v2 (91.6) mới vượt |
| REL-RAG trả về **đường đi tường minh**, precision cao khi decode tự do | path_free: SP-F1 45.4 so với khoảng 15 của mọi bản top-k |
| REL-RAG có định lý biểu diễn (line graph), RBF chưa có phát biểu hình thức tương ứng | Cần mệnh đề và chứng minh (§4) |
| Max-product giữ một đường tốt nhất mỗi nút (Viterbi); line graph với GNN tổng hợp được nhiều đường | Khi đáp án được hỗ trợ bởi nhiều đường yếu, RBF không cộng dồn |

## 4. Lập luận cho paper

### 4.1 Đóng góp nên khẳng định (theo thứ tự sức nặng của bằng chứng)

1. **Chẩn đoán:** lỗi transfer của retriever KGQA là lỗi tổng quát hoá *tổ hợp*, không phải độ dài (D1). Đây là phát hiện độc lập với phương pháp, có thể kiểm lại trên mọi retriever.
2. **Nguyên lý:** chấm điểm phân rã theo bước (path = tích nhân tử một bước chỉ phụ thuộc (q, r)) tổng quát hoá tổ hợp *bằng cấu trúc*. Ablation cơ chế (§1.1, mục 6) cho thấy chỉ cần thêm lại một bộ chấm nhìn được nhiều bước là mất ngay tính chất này.
3. **Hiện thực hiệu quả:** max-product Bellman-Ford trên entity graph, cộng nb chính xác và mm. Hai thành phần này cho tính chất của line graph mà không phải trả chi phí line graph, và bác bỏ có điều kiện lập luận "relation mixing" của REL-RAG.
4. **Kết quả:** đạt hoặc hơn SubgraphRAG / REL-RAG ở ID; thắng lớn ở 6 ô transfer (2 benchmark chéo, 3 dataset ngoài, nhóm tổ hợp mới); QA tăng có ý nghĩa thống kê.

### 4.2 Reviewer sẽ hỏi gì, và trả lời bằng gì

| Câu hỏi | Trả lời hiện có | Còn thiếu |
|---|---|---|
| "Đây chỉ là NBFNet / A*Net áp vào retrieval?" | Khác ở ba điểm: dùng max-product (Viterbi) thay vì sum; nhân tử là xác suất dùng chung giữa các bước, chỉ phụ thuộc (q, r); đầu ra là budget triple cho LLM, không phải điểm link prediction. Thêm nb chính xác và mm, cả hai không có trong NBFNet. TransferNet dùng attention riêng cho từng bước | Thí nghiệm "NBFNet-style (sum, step-specific)" cùng khung để chứng minh hai lựa chọn này là cần thiết |
| "Tổng quát hoá tổ hợp có chứng minh không?" | Bằng chứng thực nghiệm mạnh (49 → 86; ablation MLP 84 → 51) | **Mệnh đề hình thức + bài toán tổng hợp có kiểm soát (D3 / H2 chưa chạy)** |
| "Gain QA có phải do prompt?" | 2×2 trên bf16: retrieval và prompt bổ sung nhau. Retrieval riêng: CWQ +3.3 (p 5e-5); prompt riêng: +2.0 | Cùng 2×2 cho **cấu hình chính của paper** (xem 4.3); reasoner lớn hơn |
| "Seed?" | 2 seed trên WebQSP; transfer dao động 2.7 điểm | ≥ 3 seed cho mọi ô chính, báo mean ± std |
| "So với số của REL-RAG paper?" | EXP-31: số paper không tái hiện được (rò rỉ, parser) | Viết thành phụ lục tái hiện; chạy checkpoint tác giả qua QA cùng protocol |
| "Chi phí?" | Lý thuyết O(\|E\|·T); line graph có số đo | Đo wall-clock, bộ nhớ của SubgraphRAG / REL-RAG / RBF trên cùng máy |

### 4.3 Phải chốt "phương pháp" là cấu hình nào

Hiện có hai ứng viên, và QA chỉ mới có số bf16 cho ứng viên thứ hai:

- **(A) RBF+nb+mm, nhãn SP:** sạch nhất về đóng góp mô hình. Chưa có QA bf16, chỉ có QA fp8 cũ: WebQSP @100 84.3 / 69.4.
- **(B) RBF + BGE + WL + nhãn SPARQL v2:** mạnh nhất. Nhưng gộp ba yếu tố ngoài mô hình: BGE prior, WL (phương pháp trước của bạn) và nhãn gold query. Reviewer sẽ hỏi phần nào làm nên kết quả.

Đề xuất: **phương pháp = (A)**. BGE / WL / nhãn SPARQL là "tín hiệu train tương thích", trình bày như ablation cộng dần, mỗi bước có cột riêng. Cần chạy **QA bf16 cho (A)** cùng protocol (khoảng 20 phút trên server H100).

## 5. Việc tiếp theo, theo mức ưu tiên

| # | Việc | Vì sao | Chi phí ước tính |
|---|---|---|---|
| 1 | QA bf16 cho cấu hình (A): T0 và P2, WebQSP / CWQ / hai chiều transfer | Bảng QA chính phải là của phương pháp được claim | khoảng 30 phút H100 (khoảng $2) |
| 2 | ≥ 3 seed cho (A) ở WebQSP và CWQ, chấm toàn ma trận transfer | Phương sai transfer 2.7 điểm; claim chính nằm ở transfer | khoảng 6–8 giờ GPU (T4 / Kaggle miễn phí) |
| 3 | Bài toán tổng hợp có kiểm soát (D3): KG nhân tạo, tổ hợp relation chưa thấy, so DDE / line graph (REL-RAG) / RBF / NBFNet-sum | Biến chẩn đoán thành thí nghiệm nhân quả; trả lời câu hỏi "chỉ là NBFNet" | CPU, khoảng 1–2 ngày công |
| 4 | Mệnh đề hình thức: (i) điểm phân rã bất biến với tổ hợp chưa thấy khi các relation đã thấy; (ii) nb đúng (max-product trên walk không quay lui); (iii) mm là max-marginal chính xác. Kèm độ phức tạp so với line graph | Đóng góp lý thuyết đối trọng với định lý của REL-RAG | Viết tay |
| 5 | Đo chi phí thực trên cùng máy | Lợi thế chi phí là một trong ba claim | khoảng 2 giờ |
| 6 | Điểm yếu thật: π không nhìn entity, câu nhiều anchor, câu 1 bước. Đây là phần Limitation, hoặc một bước tiếp: nhân tử nút ψ(v \| q) đã thử và thất bại, còn có thể thử ψ chỉ ở bước cuối (kiểu / ràng buộc), không đưa vào lan truyền | Reviewer sẽ chỉ ra; có hướng sửa thì tốt hơn | khoảng 1 ngày |
| 7 | Reasoner thứ hai (Llama-70B hoặc GPT-4o-mini) cho bảng QA chính | Trần reasoner 8B khoảng 70% (docs/12 §23) che mất phần retrieval | cần ước tính và duyệt riêng |
| 8 | Phụ lục tái hiện REL-RAG (EXP-16, 17, 31, 40) | Giải thích vì sao không so với số paper | Viết |

**Rủi ro lớn nhất cho mức A*:** novelty bị quy về "NBFNet cho retrieval". Mục 3 và 4 của bảng trên là phần trả lời. Chẩn đoán tổ hợp (đóng góp 1) cùng ablation cơ chế là phần khó bị phản bác nhất. Nên đặt chúng làm trục chính của paper, phương pháp là hệ quả của chẩn đoán.

## 6. Vì sao retrieval gần 100% mà QA thấp (10-19, `reason/qa_anatomy.py`)

**Thiết lập:** vLLM 0.10.1, `--loop_fix`, k = 100. Mỗi câu sai được xếp vào đúng một nhóm theo thứ tự trong docstring của script. Cách xếp dùng truy vấn gold (chain và ràng buộc), context top-100 và đáp án đã parse.

**CWQ, RBF + BGE + WL v2 + P2 (Hit 65.4):**

| Nguyên nhân | % toàn bộ câu | Ghi chú |
|---|---|---|
| Đáp án **không có trong subgraph** | 13.3 | 683 / 3,531 câu (19.3%) không có đáp án trong graph; LLM vẫn đoán đúng 31% số này. 606 / 683 là câu composition: subgraph dựng quanh topic của câu WebQSP gốc nên thiếu bước nhảy thêm (truy vấn gold dài 3.39 so với 2.89 triple pattern) |
| Retriever bỏ sót | 1.8 | SubgraphRAG: 5.5 |
| Có trong context, **cần ràng buộc literal** (ngày, số, thứ tự) | khoảng 7.8 | KG không có literal. Câu có ràng buộc literal: Hit 50.5, so với 68.6 cho câu không có. Superlative chỉ đạt 44.7 Hit |
| Có trong context, **nhãn gold hoặc câu hỏi có vấn đề** | khoảng 4.4 | Gold thiếu đáp án (Willow cũng là phim UK), câu yes/no, "who played" mà gold là tên nhân vật |
| Có trong context, **LLM suy luận sai dù đủ bằng chứng** | khoảng 5.4 | Ví dụ Dorothy Gale thay vì Judy Garland; Christianity thay vì các deity |
| Có trong context, thiếu triple ràng buộc | khoảng 1.9 | |

- Bốn dòng "có trong context" (tổng 19.5% câu) được ước lượng từ mẫu 40 câu phân loại tay (literal 16, suy luận 11, nhãn 9, bằng chứng 4), nên sai số khoảng ±5 điểm tỉ lệ cho mỗi nhóm.
- Ở các câu sai, context có trọn đường gold 88–92%, gần bằng câu đúng (93.8%). Triple ràng buộc đủ 57–61%, so với 67% ở câu đúng.
- **Kết luận cho CWQ:** khoảng 21 / 35 điểm mất là do dữ liệu (subgraph thiếu đáp án và thiếu literal), khoảng 4 do nhãn, khoảng 5 do reasoner 8B, khoảng 4 do retrieval. "Bao phủ gần 100%" chỉ đúng trên tập 2,848 câu có đáp án trong graph, và chỉ là bao phủ *đáp án*, không phải bao phủ *bằng chứng để chọn đúng đáp án*.

**WebQSP, RBF v2 + P2 (Hit 87.7 / F1 72.7):**
- 96.4% câu có đáp án trong context, Hit khi có đáp án là 91.0. Sai: không có trong graph 2.6, retrieval 1.0, LLM chọn entity khác 4.7, còn lại 4.0.
- **Khoảng cách F1 với paper (−4.6) là do liệt kê thiếu đáp án**: gold trung bình 8.8 đáp án/câu, LLM chỉ đưa 3.6; 25.9% câu có danh sách ngắn và thiếu. Precision / recall là 79.9 / 74.3. Câu có ≥ 4 đáp án: Hit 91.7 nhưng F1 chỉ 63.0.

**So với SubgraphRAG, cùng phép phân rã:** context của RBF không làm LLM kém đi. CWQ: có đáp án trong context 84.8% so với 79.8%, Hit khi có đáp án 77.1 so với 74.5. WebQSP: 96.4% so với 95.4%, 91.0 so với 90.9.

**Hướng xử lý, theo chi phí:**
1. **WebQSP F1:** chỉ dẫn trong prompt "liệt kê *mọi* entity ở cuối các path khớp". Thử trên val bằng Q4 local, miễn phí.
2. **Ràng buộc literal (khoảng 8 điểm CWQ):** đưa literal (ngày, số) của các ứng viên vào context. Cần giá trị literal từ Freebase, mà subgraph hiện tại không có. Đây là đóng góp phương pháp thật, nhưng tốn công lấy dữ liệu.
3. **Đáp án ngoài subgraph (13 điểm CWQ):** chung cho mọi phương pháp dùng subgraph SubgraphRAG, kể cả REL-RAG. Paper nên báo thêm số trên tập trả lời được (2,848 câu), ghi rõ cạnh số toàn tập.
4. **Nhãn gold (khoảng 4 điểm):** không sửa được. Ghi vào phần phân tích lỗi.

## 7. Mức trần của nhãn SPARQL (10-19, `reason/oracle_label_context.py`)

**Thiết lập:** context là **chính các triple trong nhãn** (chain, rồi ràng buộc); prompt P2, vLLM 0.10.1, `--loop_fix`. Câu không có nhãn (không có đáp án trong subgraph, hoặc chain không đi được) giữ dự đoán của RBF v2 khi tính số toàn tập.

| | Câu có nhãn: mức trần | Câu có nhãn: RBF v2 cùng câu | **Toàn tập: mức trần** | REL-RAG paper |
|---|---|---|---|---|
| WebQSP, nhãn v1 | 92.7 / 79.6 (1,502) | 90.3 / 75.9 | 90.0 / 76.2 | 88.8 / 77.3 |
| WebQSP, nhãn v2 | 94.7 / 83.1 (1,496) | 90.4 / 75.9 | **91.6 / 79.3** | 88.8 / 77.3 |
| CWQ, nhãn v1 | 77.3 / 67.2 (2,542) | 74.0 / 63.4 | 67.8 / 58.2 | 67.2 / 56.8 |
| CWQ, nhãn v2 | 89.8 / 78.7 (2,510) | 74.1 / 63.6 | **76.5 / 66.1** | 67.2 / 56.8 |

**Cách đọc. Hai mức trần khác nhau về bản chất:**
- **v2 rò đáp án.** Nhãn v2 chỉ giữ trail kết thúc **ở đáp án gold**, nên mọi điểm cuối trong context đều là đáp án. Đây là trần của một retriever *đã biết đáp án*; không retriever thật nào đạt được.
- **v1 là trần thực tế của "lấy đúng chain gold".** Context gồm mọi trail đi theo đúng chain gold, kể cả trail tới các điểm cuối không phải đáp án, cộng các triple ràng buộc. Retriever hoàn hảo về relation, nhưng không có literal để lọc ứng viên, chỉ đạt tới mức này.
  - CWQ: 67.8 / 58.2, **chỉ ngang paper** (+0.6 / +1.4).
  - WebQSP: Hit hơn paper (90.0 so với 88.8), F1 kém (76.2 so với 77.3).
- **RBF v2 đã sát trần v1:** trên cùng tập câu, thấp hơn 3.3 Hit ở CWQ và 2.4 Hit ở WebQSP.
- **Chênh v1 → v2 trên CWQ (−12.5 Hit) là tác hại của các điểm cuối anh em trên cùng chain.** Đây đúng là nhóm "chọn điểm cuối khác của đúng chain" trong phân rã lỗi ở mục 6. Muốn loại chúng cần ràng buộc literal (thời gian, thứ tự, số) mà KG không có.

**Kết luận theo tiêu chí đặt trước:** với subgraph này, nhãn này và reasoner 8B này, **một retriever lấy chain gold hoàn hảo cũng chỉ ngang số CWQ của paper REL-RAG.** Cải thiện retrieval hoặc nhãn (kể cả v3) không thể vượt rõ. Muốn vượt phải thêm **thông tin mới** vào context: literal để lọc các điểm cuối anh em, và các bước nhảy ngoài subgraph cho câu composition. Hoặc phải đổi reasoner.

**Nhãn v3** (`make_sparql_v3.py`, `qrdde.py --label_src sparql3`): v2 cộng với các đoạn nối ngắn nhất (≤ 3 bước) từ điểm xa nhất chain đi tới được tới đáp án. Nhãn được thêm: CWQ train +807 câu (+3.7%), val +88, test +99; WebQSP train +25. 14 / 14 nhãn mới trên WebQSP test đều nối được topic với đáp án. Chưa train lại.
