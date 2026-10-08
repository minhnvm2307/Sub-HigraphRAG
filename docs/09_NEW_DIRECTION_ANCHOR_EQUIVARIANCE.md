# 09. Hướng nghiên cứu chính (đề xuất): tổng quát hóa thành phần bằng retrieval bất biến theo anchor

*Tạo: 2026-10-03. Trạng thái: đã có bằng chứng probe (không train) với hiệu ứng lớn. Hướng này đánh trực tiếp vào claim chính của REL-RAG (Def. 2, compositional generalization, WebQSP→CWQ). Hướng witness ([`08_…`](archive/relrag_phase/08_NEW_DIRECTION_EVIDENCE_SUFFICIENCY.md)) trở thành thành phần thứ hai (ràng buộc / conjunction), chờ S0 v3.*

Script (scratchpad `research/`): `composition.py`, `reanchor.py`, `reanchor_eval.py`, `merge_eval.py`, `reanchor_wq.py`. Output probe: `retrieve/exp_qrdde/runs/webqsp_{dde,qrdde}/{retrieval_*_reanchor3.pth, twopass_{webqsp,cwq}_B3.pth}`. Transfer cho checkpoint full: `runs/webqsp_*/retrieval_cwq_test.pth` (`transfer_eval.py`).

---

## 1. Phát hiện

### 1.1 Transfer WebQSP → CWQ: line graph transfer **kém nhất**

Tập đo: CWQ test, 2,848 câu, text-match. Mọi retriever đều train trên WebQSP.

| Retriever | ansR@20 | @50 | @100 | @500 |
|---|---|---|---|---|
| SubgraphRAG **train trên CWQ** (trần in-distribution) | 77.10 | 86.20 | 91.43 | 98.36 |
| **DDE** (MLP kiểu SubgraphRAG, lan truyền không có tham số) | 55.72 | **70.05** | 79.58 | **94.81** |
| RT-DDE | 52.95 | 68.20 | 77.44 | 93.80 |
| QR-DDE bỏ điều kiện câu hỏi | 53.21 | 65.21 | 74.27 | 90.57 |
| REL-RAG P code-faithful | 46.00 | 61.90 | 72.26 | 85.21 |
| QR-DDE | 44.58 | 60.95 | 71.29 | 91.70 |
| REL-RAG T paper | 36.04 | 52.09 | 64.63 | 92.48 |

**Bootstrap ghép cặp:**
- DDE − REL-RAG P = **+8.14 [+6.22, +10.06]** @50; +9.60 @500.
- QR-DDE − DDE = −9.09 @50.

**Thứ tự transfer tỷ lệ nghịch với lượng cấu trúc được học.** Ngược lại, QR-DDE lại hơn DDE +3.0 khi train và test cùng trên CWQ (EXP-14c).

### 1.2 Chỗ sụp: câu composition, ở bước đi tiếp từ entity cầu nối

ansR@50 trên câu **composition** (940 câu):

| Retriever | ansR@50 |
|---|---|
| SubgraphRAG (train trên CWQ) | 83.5 |
| DDE | 54.6 |
| QR-DDE | 34.4 |
| **REL-RAG P** | **26.0** |

- Chỉ tính câu có 1 topic (764 câu): REL-RAG P 17.8, QR-DDE 26.2, DDE 52.1, so với 88.5 in-distribution.
- Ví dụ: "In what location was the composer of *Stop Standing There* born?"
  - Top-1 của DDE là `Stop Standing There → composer → Avril Lavigne`: tìm đúng entity cầu nối.
  - Nhưng nó không lấy `Avril Lavigne → place_of_birth → Belleville`.
- Câu conjunction, comparative và superlative ít bị ảnh hưởng hơn. REL-RAG P mạnh ở comparative/superlative (85–90).

### 1.3 Probe không train: neo lại tại entity cầu nối sửa được phần lớn

Cách làm, với checkpoint WebQSP giữ nguyên:
- **Lượt 1:** chấm từ các topic.
- **Cầu nối:** các entity trong top-3 triple của lượt 1, trừ topic.
- **Lượt 2:** anchor = topic ∪ cầu nối.
- **Trộn:** max của điểm đã chuẩn hóa z ở hai lượt.

| ansR@50 | REL-RAG P | DDE | **DDE 2 lượt (maxz)** | QR-DDE | QR-DDE 2 lượt (top3 + lượt 2) |
|---|---|---|---|---|---|
| **WebQSP (in-distribution)** | 90.60 | 90.07 | **89.86** | 89.20 | 88.02 |
| CWQ transfer, tất cả | 61.90 | 70.05 | **75.94** | 60.95 | 72.84 |
| composition | 26.0 | 54.6 | **74.4** | 34.4 | 72.7 |
| conjunction | 77.6 | 78.1 | 77.0 | 74.8 | 73.7 |
| comparative | 84.7 | 75.5 | 75.9 | 71.9 | 71.2 |
| superlative | 90.2 | 76.0 | 75.4 | 70.0 | 68.3 |

- Neo lại *vô điều kiện* (thứ tự lượt 2) làm hại WebQSP: −1.86 [−2.66, −1.08]. Trộn maxz đưa mức hại về −0.21.
- So sánh các cách trộn (`merge_eval.py`) cho DDE:

| Cách trộn | WebQSP | CWQ |
|---|---|---|
| top3 + lượt 2 | 88.25 | 75.86 |
| RRF (w = 0.5) | 90.32 | 73.76 |
| **maxz** | **89.86** | **75.94** |

### 1.4 Phân rã theo Mệnh đề 2: cầu nối gold đưa retriever về mức in-distribution

Tập đo: 604 câu CWQ composition có 1 topic, đã tìm được cầu nối gold từ SPARQL (`?c pred1 topic . ?c pred2 ?x`). Checkpoint DDE train trên WebQSP (`gold_bridge.py`).

| Neo tại | ansR@50 |
|---|---|
| Chỉ topic (lượt 1) | 53.3 |
| Topic + cầu nối học được (top-3 lượt 1) | 79.0 |
| **Topic + cầu nối gold** | **86.8** (so với 88.5 của SubgraphRAG train trên CWQ, cùng nhóm câu) |

- Recall của bước chọn cầu nối (cầu nối gold nằm trong top-3 entity của lượt 1) ρ_bridge = **90.4%**.
- Khi chọn trúng: 84.3. Khi trượt: 29.5.
- ρ_bridge × ρ_prim ≈ 0.904 × 86.8 ≈ 78.5, khớp với 79.0 đo được.

Kết luận:
- **Khi neo đúng chỗ, hop nguyên thủy học từ WebQSP đủ để đạt mức in-distribution của CWQ.** Tổng quát hóa thành phần quy về bài toán chọn cầu nối.
- Phần còn hở (79.0 so với 86.8) là ρ_bridge. Đây là đích của A2 (học chọn cầu nối / anchor mềm).

### 1.5 Bằng chứng nhân quả và lý do cho từng thiết kế (EXP-21)

**Cơ chế (E1).** Trên WebQSP, giữ nguyên câu hỏi và chỉ dời anchor đi 1 hop, ansR@50 tụt:
- DDE: −20.7 [−22.6, −18.7];
- QR-DDE: −43.3 [−45.8, −40.9].

Neo lại 2 lượt hồi phục +7.6 và +36.8. Đây là can thiệp có kiểm soát, nên loại được giải thích rằng câu CWQ khó hơn ở mặt ngôn ngữ.

**Lý do cho từng thiết kế (E2, CWQ transfer):**

| Lựa chọn | Bằng chứng |
|---|---|
| Lặp neo lại | composition 54.6 → 74.6 (DDE); WebQSP gần như không đổi |
| Trộn **max đơn điệu** thay vì thay hẳn | Thay hẳn trôi khi K tăng (K=3: 68.0 so với 75.7); Mệnh đề 3 đúng ở 100% số câu |
| Cầu nối **theo điểm của mô hình** | Hơn cầu nối ngẫu nhiên +2.62 [+1.85, +3.39], composition +8.2 |
| B = 3, K = 2 | B=1 / 3 / 10: 77.4 / 78.2 / 77.1; K=3 không thêm gì vì CWQ ≤ 2 hop |
| Train neo lại (`reanchor_p`) | 1 lượt 70.0 → 74.8; ít xê dịch lượt 1 hơn (6.5 so với 11.7) |

**Dùng được cho nhiều mô hình (E4, EXP-22):** REL-RAG P + neo lại, không train lại: CWQ 61.90 → 70.45 (+8.54 [+7.33, +9.80]), composition 26.0 → 54.7; WebQSP −0.55. Line graph cũng bị gắn với vị trí.

**Kiểm tra lý thuyết (E3):** bước chọn cầu nối đúng khoảng 90%; trúng thì 79–82, trượt thì 43–45. Phân rã ρ_bridge × ρ_prim khớp với số đo.

---

## 2. Giả thuyết cơ chế: liên quan bị gắn với vị trí (position–relation entanglement)

Retriever học **ở vị trí nào so với anchor** (khoảng cách hop) thì triple liên quan xuất hiện, thay vì học **relation nào** liên quan.
- WebQSP gần như chỉ có bằng chứng kề topic: nhãn minlen trung bình dài 1.37 bước, và mọi đáp án trong subgraph cách topic ≤ 2 hop.
- Câu composition đặt relation được hỏi ở vị trí *sau một cầu nối*, là vị trí không có trong dữ liệu train.

Mức độ gắn tăng theo lượng cấu trúc vị trí mà mô hình học được:

| Mô hình | Cách vị trí đi vào mô hình | Composition @50 |
|---|---|---|
| DDE | One-hot khoảng cách cố định | 54.6 |
| QR-DDE | Lan truyền có học | 34.4 |
| REL-RAG P | Context tự hồi quy cộng dồn + STOP học theo phân bố độ dài | 26.0 |

Neo lại biến "relation ở vị trí 2" thành "relation ở vị trí 1 tính từ anchor mới", tức đúng điều kiện đã gặp lúc train. Vì vậy chỉ riêng việc neo lại đã thu hồi được +20 điểm.

---

## 3. Đóng góp đề xuất

1. **Chẩn đoán (thực nghiệm + phân tích).**
   - Claim compositional generalization của REL-RAG không đứng vững ở tầng retrieval: line graph transfer kém hơn một MLP trên entity graph (+8.1 cho MLP).
   - Nguyên nhân là liên quan bị gắn với vị trí, không phải thiếu mô hình hóa relation. Lát cắt theo loại câu CWQ và theo khoảng cách cho thấy rõ điều này.
2. **Lý thuyết** (cần viết chặt).
   - **Định nghĩa:** retriever *bất biến theo anchor* (anchor-equivariant). Điểm của triple chỉ phụ thuộc (q, ngữ cảnh quan hệ cục bộ, quan hệ với tập anchor *hiện tại*), không phụ thuộc khoảng cách tuyệt đối tới anchor gốc.
   - **Mệnh đề 1 (không xác định được ngoài vùng vị trí đã thấy).** Với lớp giả thuyết phụ thuộc vị trí, nghiệm tối thiểu hóa rủi ro train không bị ràng buộc ở những vị trí có P_train(d) = 0. Có thể dựng phản ví dụ cho recall → 0. Lớp line graph tự hồi quy (REL-RAG Eq. 8 với STOP) thuộc loại này, nên Theorem 3.1 không phủ được dịch chuyển về vị trí/độ dài.
   - **Mệnh đề 2 (tổng quát hóa thành phần từ hop nguyên thủy).**
     - Giả thiết (A1): mỗi relation r_i của chuỗi π đã xuất hiện ở vị trí kề anchor trong dữ liệu train.
     - Giả thiết (A2): bước chọn cầu nối có recall ρ trong top-B.
     - Khi đó retrieval lặp bất biến theo anchor đạt recall ≥ ρ^{L−1}·r_prim trên chuỗi dài L. Tổng quát hóa thành phần quy về tổng quát hóa của từng hop nguyên thủy.
   - **Lập luận độ phức tạp.** Không cần tham số theo relation và không cần line graph, giữ được lập luận √R (`archive/relrag_phase/06_LITERATURE_REVIEW.md` §2.1). Chi phí là K lượt nhân với chi phí tuyến tính.
3. **Phương pháp (bản học được, tên tạm AER: Anchor-Equivariant Retrieval).**
   - **(a) Bộ chấm hop nguyên thủy bất biến theo anchor.** DDE hoặc MLP trên entity graph với **anchor mềm** (kênh topic nhận giá trị liên tục), không có tham số theo relation.
   - **(b) Huấn luyện có tăng cường anchor (anchor augmentation).** Khi train trên WebQSP, ngẫu nhiên thêm entity lân cận làm anchor phụ mà giữ nguyên nhãn. Mục tiêu là làm bộ chấm *ổn định* khi tập anchor phình ra. Cách này sửa cả −1.1 trên conjunction lẫn −0.2 trên WebQSP của bản không train.
   - **(c) Chọn cầu nối và dừng.** Cầu nối là anchor mềm lấy từ điểm của lượt trước. Dừng khi lượt mới không còn nâng điểm cao nhất (điểm cố định), không cần dữ liệu thành phần.
   - **(d) Trộn đã hiệu chỉnh:** max của điểm chuẩn hóa giữa các lượt.
   - Tùy chọn: kết hợp witness (docs/08) cho câu conjunction / ràng buộc.
4. **Đánh giá.**
   - WebQSP in-distribution: không được kém DDE quá 0.5.
   - CWQ transfer và in-distribution.
   - Downstream (Hit/F1, cùng budget, llama3.1 Q4; nếu ngân sách cho phép thêm GPT-4o-mini).
   - GTSQA: tới 5 hop, có kiểu cấu trúc chưa thấy, là nơi K > 2 lượt mới phát huy.
   - Latency.

---

## 4. Tính mới và rủi ro

| Liên quan | Khác biệt |
|---|---|
| Retrieval lặp cho KGQA: PullNet (EMNLP'19), SR (ACL'22), UniKGQA, ToG / KG-Agent (dùng LLM) | Các phương pháp này train / tinh chỉnh trên dữ liệu nhiều hop cùng phân phối, hoặc cần LLM ở mỗi bước. Ở đây: train **chỉ trên câu đơn giản**, tổng quát hóa sang câu thành phần **không cần dữ liệu thành phần, không cần LLM**, có chẩn đoán và mệnh đề lý thuyết |
| Tổng quát hóa thành phần / độ dài (SCAN, tăng cường kiểu GECA, căn chỉnh thuật toán của NBFNet) | Đưa vào retriever graph-RAG; chỉ ra lớp line graph (REL-RAG) bị gắn vị trí |
| REL-RAG (ICML'26), Theorem 3.1 | Cận của họ nói về độ lớn lớp giả thuyết **trong** phân phối pattern; claim compositional của họ không đứng ở tầng retrieval (§1.1) |
| KGFR, GFM-RAG | Transfer chéo KG; không phân tích dịch chuyển vị trí; không so với REL-RAG |

**Rủi ro:**
- "Chỉ là retrieval lặp": cần lý thuyết (§3.2), anchor augmentation (học được), và chứng minh nó tổng quát hóa khi baseline lặp được train cùng điều kiện.
- CWQ composition chỉ dài 2 hop trong subgraph, nên K = 2 là đủ. Cần GTSQA để chứng minh với K > 2.
- Downstream chưa đo. Ưu tiên tiếp theo là reasoning trên CWQ transfer @50 cho REL-RAG, DDE và DDE 2 lượt.

---

## 5. Kế hoạch

| Bước | Nội dung | Tiêu chí go | Chi phí |
|---|---|---|---|
| A0 | Probe không train (đã xong, §1.3) | CWQ +5 và WebQSP không giảm quá 0.5 | ✅ +5.9 / −0.21 |
| A1 | Downstream CWQ transfer @50: REL-RAG P, DDE, DDE 2 lượt; llama3.1 Q4, prompt SubgraphRAG, match gốc; 3 × 3,531 câu (~3 giờ mỗi run) | ΔHit/F1 so với REL-RAG có ý nghĩa (bootstrap) | GPU, ~9 giờ |
| A2 ✅ (EXP-20) | **Kết quả: WebQSP 89.68 (khác REL-RAG −0.92, không có ý nghĩa); CWQ transfer 77.95 (+16.05 [+14.2, +18.0] so với REL-RAG, +7.9 so với DDE).** Code AER: anchor mềm + anchor augmentation trong `qrdde.py` (`--anchor_aug p`), suy luận lặp có điểm dừng; train WebQSP full (seed 42, patience 10) | WebQSP ≥ DDE − 0.5; CWQ ≥ bản không train | Code + ~1–2 giờ train (người dùng chạy) |
| A3 | Lý thuyết: viết chặt Mệnh đề 1–2 và phản ví dụ | – | – |
| A4 | GTSQA (tải, chuyển sang format, embedding GTE) | Thắng trên câu ≥ 3 hop / kiểu chưa thấy | Hạ tầng lớn |
| A5 | Kết hợp witness (docs/08) cho conjunction | Phụ thuộc S0 v3 | – |

---

## 6. Lý thuyết (bản nháp A3, cần viết chặt khi vào paper)

**Ký hiệu.**
- Đồ thị G = (V, E), relation có văn bản z_r, câu hỏi q, tập anchor A ⊆ V.
- Retriever là bộ chấm s(q, x; G, A) trên các triple x = (h, r, t).
- d_A(x) là khoảng cách từ A tới h.
- Câu hỏi có chuỗi đáp án a₀ →r₁→ v₁ → … →r_L→ v_L, với a₀ ∈ A. *Hop được hỏi* thứ i là x_i = (v_{i−1}, r_i, v_i).

**Định nghĩa 1 (phụ thuộc vị trí và bất biến theo anchor).**
- s **phụ thuộc vị trí** nếu nó là hàm của d_A(x), hoặc của các đặc trưng xác định d_A(x): one-hot khoảng cách của DDE, chỉ số bước / context cộng dồn của decoder tự hồi quy (REL-RAG Eq. 8), số vòng lan truyền.
- s **bất biến theo anchor** nếu s(q, x; G, A) = g(q, x, N(x), 1_A|_{N(x)}) với g dùng chung: điểm chỉ phụ thuộc vào việc lân cận của x có chạm anchor hay không. Nó **đơn điệu theo anchor** nếu thêm anchor không làm giảm điểm của một triple liên quan.

**Tiền đề đo được cho Mệnh đề 1 (10-04).**
- Chuỗi suy luận gold của WebQSP **train** (SPARQL `InferentialChain`): 1 hop 56.2%, 2 hop qua CVT 33.1%, **2 hop qua entity văn bản 0.0%**, không có chuỗi 9.7%.
- Như vậy relation chính của câu hỏi *không bao giờ* đứng sau một cầu nối văn bản trong dữ liệu train. Câu CWQ composition thì luôn như vậy theo cách dựng dataset.
- Lưu ý: 17.1% triple trong nhãn train của SubgraphRAG có đầu gần topic hơn là một entity văn bản cách topic 1 hop. Phần này đến từ các path SP giả (đi qua hub), không phải chuỗi suy luận. Vì vậy tiền đề phải phát biểu theo **cặp (relation chính, vị trí)**, không phải theo vị trí đơn thuần: P_train(relation chính ở vị trí sau cầu nối văn bản) = 0.

**Mệnh đề 1 (không xác định được ở vị trí chưa thấy).**
- Phát biểu: gọi H là lớp bộ chấm phụ thuộc vị trí, và giả sử phân phối train có P_tr(d_A(x_i) = δ) = 0 với hop được hỏi. Khi đó, với mọi h* tối thiểu hóa rủi ro train, tồn tại h' ∈ H có cùng rủi ro train nhưng recall bằng 0 trên mọi câu test có hop được hỏi ở vị trí δ.
- Phác thảo chứng minh: sửa h* trên tập đầu vào có d = δ, là tập có độ đo 0 dưới P_tr.
- Hệ quả:
  - Cận tổng quát hóa của REL-RAG (Theorem 3.1 / A.1, giả thiết i.i.d.) không suy ra được tổng quát hóa khi có dịch chuyển vị trí, tức câu composition.
  - Việc REL-RAG được 26.0 so với 83.5 in-distribution trên composition là một trường hợp của mệnh đề này.
  - Thứ tự đo được (DDE > QR-DDE > REL-RAG) phản ánh mức thiên kiến quy nạp còn lại sau khi đã khớp dữ liệu train.

**Mệnh đề 2 (tổng quát hóa thành phần từ hop nguyên thủy).**
- Giả thiết:
  - (A1) Recall nguyên thủy: với mọi A chứa v_{i−1}, P[x_i ∈ topB(s(·; A))] ≥ ρ_i. Điều này đạt được khi relation r_i đã được thấy *kề anchor* lúc train.
  - (A2) Đơn điệu theo anchor: thêm anchor không làm ρ_i giảm. Đây là mục tiêu của anchor augmentation (`--reanchor_p`).
- Phát biểu: retrieval lặp A_t = A_{t−1} ∪ ents(topB(s(·; A_{t−1}))) chứa toàn bộ chuỗi sau L lượt với xác suất ≥ 1 − Σ_i (1 − ρ_i) (cận hợp).
- Phác thảo chứng minh: quy nạp theo i. Sau lượt i, v_i ∈ A_i với xác suất ≥ ρ_i, nên (A1) áp dụng được cho hop i + 1.
- Hệ quả: tổng quát hóa thành phần quy về hai điều kiện, cả hai đều kiểm chứng được trên dữ liệu train đơn giản: tổng quát hóa của hop nguyên thủy, và tính đơn điệu theo anchor.

**Mệnh đề 3 (an toàn in-distribution của trộn max).**
- Phát biểu: gọi z₁, z₂ là điểm chuẩn hóa của lượt 1 và lượt 2, s̄ = max(z₁, z₂), T₁ là top-k của lượt 1, τ_k là ngưỡng z₁ thứ k. Khi đó |T₁ \ topk(s̄)| ≤ |{y ∉ T₁ : z₂(y) > τ_k}|.
- Phác thảo chứng minh: mọi x ∈ T₁ có s̄(x) ≥ τ_k, nên x chỉ bị đẩy ra khỏi top-k bởi những y có z₂(y) > τ_k.
- Ý nghĩa: mức tụt in-distribution bị chặn bởi số triple "bùng lên sai" ở lượt 2, là đại lượng đo được. WebQSP: −0.21 ansR@50 với maxz, so với −1.86 khi thay hẳn bằng lượt 2.

**Việc còn lại:**
- Phát biểu (A1) theo xác suất trên phân phối câu hỏi (cần độ tương tự văn bản giữa phần câu hỏi nói về hop i và cách diễn đạt trong train).
- Đo ρ_i thực nghiệm: recall hop nguyên thủy khi neo tại cầu nối gold, trên CWQ composition.
- Dựng phản ví dụ cụ thể cho line graph tự hồi quy: STOP học trên độ dài ≤ 2.

## 7. Tính mới (kiểm tra 10-03)

| Công trình | Loại | Khác biệt |
|---|---|---|
| [GraphWalker](https://arxiv.org/html/2603.28533) (arXiv 2026) | Agent LLM fine-tune, train trên CWQ, tổ hợp lại relation bằng random walk | Cần LLM; train trên CWQ; không so với REL-RAG / SubgraphRAG |
| [CCoR](https://arxiv.org/html/2608.22762) (arXiv 2026) | Prompting LLM, không train | Không phải retriever; không phân tích transfer WebQSP → CWQ |
| [PullNet](https://arxiv.org/pdf/1904.09537) (EMNLP'19) | Mở rộng subgraph lặp bằng GCN | Train end-to-end trên dữ liệu cùng phân phối, không nhắm tới tổng quát hóa thành phần, không có chẩn đoán vị trí |
| KGFR, GFM-RAG | Retriever nền tảng | Transfer chéo KG; không xét dịch chuyển vị trí |

Chưa thấy công trình nào: (i) đo transfer WebQSP → CWQ của retriever graph-RAG ở tầng retrieval và chỉ ra line graph kém nhất; (ii) chẩn đoán liên quan bị gắn với vị trí; (iii) sửa bằng retrieval bất biến theo anchor, không cần dữ liệu thành phần, không cần LLM.

## 8. Anchor-Set Encoding (ASE): đưa "nhiều anchor" vào kiến trúc (10-07)

### 8.1 Ý tưởng, nói đơn giản
SubgraphRAG đánh dấu mọi topic entity của câu hỏi vào **một** lần DDE (Directional Distance Encoding). Với câu có hai topic, ví dụ "phim do X đạo diễn **và** Y đóng", DDE chỉ cho biết một entity "gần một topic nào đó". Nó không cho biết entity đó gần X, gần Y, hay gần **cả hai**. Mà đáp án của câu giao nhau chính là entity gần cả hai.

ASE sửa điểm này ngay trong kiến trúc, không phải bằng trick lúc inference như ADR/AER:
1. **Chạy DDE riêng cho từng anchor.** Mỗi entity v nhận một vector khoảng cách d_a(v) ứng với từng topic a.
2. **Nhúng qua một mạng dùng chung** φ (MLP 2 lớp): h_a(v) = φ([one-hot_a(v), d_a(v)]).
3. **Gộp theo tập anchor** bằng mean, max và min trên a. Thứ tự anchor không ảnh hưởng kết quả.
   - max: gần **ít nhất một** anchor (giống phép hợp);
   - min: gần **mọi** anchor (giống phép giao);
   - mean: mức độ gần trung bình.
4. Ghép [mean, max, min] vào đặc trưng entity của SubgraphRAG; MLP chấm triple giữ nguyên.

Code: `retrieve/exp_qrdde/qrdde.py`, `--feat dde+ase` (lớp `Scorer`, nhánh `ase_dde` / `ase_phi`). Câu chỉ có một topic thì mean = max = min, nên ASE trở về đúng DDE thường.

### 8.2 Kết quả (chi tiết ở EXP-33)

**Transfer WebQSP → CWQ.** Đây là setting quan trọng nhất, vì 48% câu CWQ có từ 2 topic trở lên, trong khi 98% câu train WebQSP chỉ có một topic.

![ansR theo budget, transfer](figures/ase_transfer_ansR.png)

- Đường xanh đậm (ASE+ADR+AER) nằm cao hơn hẳn SubgraphRAG (xám) và REL-RAG (xanh lá) ở mọi budget: +10.7 điểm ở @50 so với SubgraphRAG.
- Ở @200, ASE+ADR+AER đã vượt mức trần là SubgraphRAG **train trực tiếp trên CWQ** ở @50 (đường gạch).
- Đường cam (WL+ADR+AER, backbone relation-topic) vẫn nhỉnh hơn: khoảng 2.5 điểm ở @50, gần bằng nhau ở @200.

![Các nhóm nhãn @50, transfer](figures/ase_transfer_at50.png)

- Lợi thế không chỉ nằm ở việc lấy được đáp án. Bằng chứng cũng tăng: SP-R +6.9 và GPT-R +7.7 so với SubgraphRAG. Như vậy LLM nhận được cả đường suy luận, không chỉ entity đáp án.
- ASE base (xanh nhạt) chỉ ngang SubgraphRAG. Lợi thế chỉ xuất hiện khi kết hợp với ADR+AER.

**WebQSP (in-distribution).**

![ansR WebQSP](figures/ase_webqsp_ansR.png)

- ASE ngang SubgraphRAG và REL-RAG (chênh ≤ 0.5 điểm, không có ý nghĩa), nên không làm hỏng trường hợp dễ. Đây là điều kiện bắt buộc của gate.

### 8.3 Đánh giá thẳng
- ✅ **Gate đạt ở cả hai setting:** in-distribution không thua, transfer thắng cả SubgraphRAG lẫn REL-RAG ở @50/100/200.
- ⚠️ **Đóng góp riêng của ASE còn nhỏ.**
  - Trên tập con CWQ 4k (in-distribution), ASE hơn DDE +6 ansR@50.
  - Nhưng khi train WebQSP rồi transfer, ASE base chỉ hơn DDE khoảng 1.4, và ASE+ADR+AER chỉ hơn DDE+ADR+AER khoảng 2.
  - Lý do có thể là dữ liệu WebQSP gần như toàn câu 1 topic, nên nhánh min/mean hầu như không được học.
- ❌ **Chưa hơn WL+ADR+AER.** Muốn ASE trở thành đóng góp chính, cần chứng minh ASE cộng thêm được vào WL.
- **Việc tiếp theo:**
  1. Chấm ASE trên CWQ full (đã train xong).
  2. Train WL + ASE.
  3. Seed 0 cho ASE.
