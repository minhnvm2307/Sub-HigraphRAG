# 08. Hướng nghiên cứu (phụ, sau S0): Answer recall ≠ answerability — retrieval bằng chứng *đủ để xác định* đáp án

*Tạo: 2026-10-03. Cập nhật 10-03 chiều: S0 v1 không hợp lệ (§2.7), S0 v2 đang chạy; S1 đạt (§2.8); QR-DDE CWQ cho thấy phần anchor-giao đã giải được, còn witness tại CVT thì chưa (§2.9). Trạng thái: đề xuất, đã có bằng chứng dữ liệu. Thay thế track R ở [`01_RESEARCH_PLAN.md`](01_RESEARCH_PLAN.md); QR-DDE trở thành backbone/ablation.*

Script phân tích (scratchpad, chạy trên output đã có): `shape.py`, `suff.py`, `labels.py`, `contrast.py`, `gold_constraints.py`, `cwq_types.py`, `sg_labels.py`, `suppress.py`, `oracle_build.py`. Annotation gốc tải về: `/relrag/data/webqsp_orig/` (WebQSP parses: SPARQL + `Constraints`), `/relrag/data/cwq_orig/` (CWQ test: SPARQL + `compositionality_type`).


> **Kết luận S0 (10-03 15:23): không đạt theo tiêu chí đặt trước. Hướng này hạ xuống thành phân tích phụ; hướng chính là [`09_…`](09_NEW_DIRECTION_ANCHOR_EQUIVARIANCE.md).**
>
> S0 v3 đối xứng: 151 câu; thêm triple cho cả nhánh đáp án và nhánh nhiễu đang có trong ngữ cảnh; xáo thứ tự.
>
> | | Hit (eval gốc) | Hit@1 | Macro-F1 |
> |---|---|---|---|
> | base | 79.47 | 72.85 | 66.37 |
> | sym-oracle (predicate ràng buộc) | 85.43 | 80.79 | 73.42 |
> | sym-control (predicate khác, cùng nút) | 81.46 | 77.48 | 70.00 |
>
> Bootstrap ghép cặp (Hit theo substring):
> - oracle − control = +2.65 [−2.65, +7.95]; F1 +2.26 [−2.64, +7.29].
> - oracle − base: F1 +6.17 [+0.78, +11.91].
>
> Nội dung ràng buộc có tác dụng dương nhưng nhỏ và không có ý nghĩa thống kê với llama3.1 Q4 trên 151 câu.
>
> Phần **retrieval** vẫn đứng: nhãn SP chứa ràng buộc 0/267; retriever lấy được ràng buộc @50 chỉ 7–31%. Phần này dùng được làm phân tích trong paper.
>
> Các bản trước:
> - v1 (197 câu): ràng buộc không neo, nên chèn nhiễu.
> - v2 (185 câu): chỉ thêm vào nhánh đáp án, nên lộ đáp án qua salience; control +6.5 Hit so với base.
>
> v2 cho thấy một điều: LLM sai chủ yếu ở bước *chọn* giữa các ứng viên. Bất kỳ triple nào làm nổi bật nhánh đúng cũng giúp.

---

## 1. Tóm tắt một đoạn

Mọi retriever graph-RAG hiện nay (SubgraphRAG ICLR'25, REL-RAG ICML'26, GNN-RAG, RoG, PathISE) được supervise bằng **path từ topic tới đáp án**, và được đánh giá bằng **answer recall**. Trên WebQSP/CWQ, answer recall đã bão hòa (~90% @50).

Ta chỉ ra một thực tế khác:
- Với các câu có **ràng buộc** ("who played Obi-Wan *in Episode 2*"; 27% câu WebQSP, 85–92% câu CWQ conjunction/composition), triple ràng buộc **không bao giờ nằm trong nhãn**. Nhãn của REL-RAG chứa 0/271, nhãn của SubgraphRAG chứa 22/287.
- Do đó retriever lấy được đáp án *cùng với các ứng viên nhiễu*, nhưng **không lấy bằng chứng để phân biệt chúng**: chỉ lấy được ràng buộc 12–37% ở budget 50.
- LLM trả lời đúng thấp hơn 9–14 Hit khi thiếu bằng chứng đó, dù đáp án đã có trong ngữ cảnh.

Đề xuất:
1. **Định nghĩa lại mục tiêu retrieval là *answer-identifiability***: bằng chứng phải đủ để tách đáp án khỏi ứng viên nhiễu.
2. **Kết quả lý thuyết**: (a) supervision theo path đáp án mù với ràng buộc; (b) chặn trên độ chính xác của mọi reasoner dựa trên bằng chứng, khi bằng chứng không phân biệt được.
3. **Retriever mới** học chọn "witness" phân biệt **chỉ từ đáp án**, không cần nhãn LLM như REL-RAG, không cần logical form. Cơ chế là một reader ẩn danh hóa ứng viên cộng ràng buộc thưa (information bottleneck).

Hướng này đánh trúng REL-RAG ở chính claim của nó: REL-RAG transfer sang CWQ kém vì CWQ toàn là câu có ràng buộc (§2.5), không phải vì thiếu line graph.

---

## 2. Bằng chứng đã đo (2026-10-03)

### 2.1 Ràng buộc gold của WebQSP (từ SPARQL parse gốc)

Tập đo: WebQSP test, 1,559 câu có đáp án trong graph.

| Loại ràng buộc (parse đầu tiên) | Số câu | % |
|---|---|---|
| Không ràng buộc | 1,140 | 73.1 |
| Có ràng buộc entity | 322 | 20.6 |
| Có ràng buộc thời gian | 89 | 5.7 |
| Có ràng buộc thứ tự (order) | 46 | 2.9 |

- Triple ràng buộc entity **có trong subgraph**: 271/322 câu (84.2%).
- **Nhãn SP (REL-RAG, mọi biến thể nhãn rational) chứa triple ràng buộc: 0/271 (0.0%).** Theo thiết kế, ứng viên rational path của REL-RAG là path tới đáp án, nên LLM lọc thế nào cũng không thêm được ràng buộc.
- **Nhãn train của SubgraphRAG** (`triple_scores > 0`) chứa ràng buộc: **22/287 (7.7%)**.

**Retriever lấy được *toàn bộ* triple ràng buộc của câu** (trên 271 câu):

| Retriever | @20 | @50 | @100 | @(= kích thước REL-RAG auto-stop, ~43) |
|---|---|---|---|---|
| SubgraphRAG | 7.0 | 11.8 | 17.0 | 8.1 |
| DDE (repro) | 5.9 | 13.7 | 20.3 | 8.1 |
| QR-DDE | 8.9 | 15.1 | 22.9 | 11.8 |
| REL-RAG P best-first | 18.5 | **36.5** | **48.0** | 32.5 |
| REL-RAG P auto-stop | – | – | – | 22.5 |
| REL-RAG T | 9.2 | 15.9 | 25.8 | 14.8 |

REL-RAG P cao nhất vì best-first mở rộng các nhánh anh em tại CVT. Cấu trúc path có ích, nhưng mục tiêu train không khuyến khích điều đó.

**Thứ hạng trong danh sách đầy đủ của SubgraphRAG** (296 ràng buộc; phân vị, 0 = đầu danh sách):

| Loại triple | Phân vị trung vị |
|---|---|
| Triple ràng buộc gold | 15.4% |
| Triple anh em cùng nút nguồn (trung vị 19 anh em) | 14.6% |
| Triple ngẫu nhiên | 51.2% |

53.4% ràng buộc gold xếp dưới anh em trung vị. Kết luận: retriever **tìm đúng vùng lân cận nhưng không phân biệt được witness**, như tung đồng xu giữa các anh em. Phân vị 15% tương ứng hạng khoảng 600, nằm ngoài budget.

### 2.2 Có ràng buộc trong ngữ cảnh thì LLM trả lời đúng hơn (tương quan)

Câu có ràng buộc entity và **đã có đáp án trong ngữ cảnh**; SubgraphRAG + llama3.1 Q4; Hit theo match gốc.

| Ngữ cảnh | Thiếu ràng buộc | Có ràng buộc |
|---|---|---|
| top-100 | 78.9 (n=199) | 84.8 (n=46) |
| top-50 | 76.3 (n=194) | **90.3** (n=31) |

Theo bằng chứng GPT (mọi dạng câu, top-100, đáp án đã có trong ngữ cảnh):
- Bằng chứng đầy đủ: Hit 90.1 (n=1,061). Bằng chứng thiếu: 78.9 (n=265).
- Riêng dạng `nary` (CVT có nhánh ràng buộc): 82.3 so với 68.8.

Đây mới là tương quan; thí nghiệm nhân quả S0 ở §6.

### 2.3 Dạng bằng chứng (từ triple bằng chứng GPT), độ phủ của nhãn và của retrieval

| Dataset | Dạng | % câu | Nhãn SP phủ bằng chứng | SubgraphRAG phủ **đủ** bằng chứng @50 | REL-RAG P @50 |
|---|---|---|---|---|---|
| WebQSP | single / chain | 21 / 42 | 93–94% | 94 / 83 | 92 / 85 |
| WebQSP | **nary** (CVT có nhánh ràng buộc) | 13.6 | **39%** | **37** | 54 |
| WebQSP | **other** (entity ràng buộc ngoài CVT) | 22.1 | **40%** | **47** | 40 |
| CWQ | **multi** (≥2 anchor) | 36.9 | 75% | 57 (in-dist) | 22 (transfer) |
| CWQ | **nary** | 16.0 | **25%** | 40 | 41 |
| CWQ | other | 32.6 | 69% | 69 | 13 |

Câu có **ứng viên nhiễu** (cùng relation pattern với đáp án nhưng không phải đáp án): WebQSP `nary` 67%; CWQ chain 96%, CWQ multi 97%.

### 2.4 Nhãn tự động từ đồ thị + đáp án (bằng chứng phân biệt, DISC) chỉ là tập ứng viên

DISC = các triple anh em trên nhánh đáp án có (relation, đầu kia) không xuất hiện ở nhánh nhiễu. Ghép SP với DISC nâng độ phủ bằng chứng:

| Dạng | Độ phủ: SP → SP ∪ DISC | Số triple (trung vị): SP → SP ∪ DISC |
|---|---|---|
| WebQSP `nary` | 39 → 53.5% | 4 → 15 |
| WebQSP `other` | 40 → 57% | 1 → 56 |
| CWQ `nary` | 25 → 34% | – |

DISC chứa cả những khác biệt không liên quan tới câu hỏi, nên không dùng trực tiếp làm nhãn được. Việc chọn ràng buộc nào khớp với câu hỏi phải do mô hình học, có điều kiện theo câu hỏi (§5).

Neo thêm các entity được nhắc trong câu hỏi (mention anchors) chỉ nâng 39 → 47%: ràng buộc thường không được nhắc nguyên văn ("episode 2" so với "Star Wars Episode II…").

### 2.5 CWQ theo loại câu (SPARQL gốc), giải thích transfer của REL-RAG

| Loại | n | % câu có predicate SPARQL không có trong nhãn SP | Nhãn SP phủ % predicate SPARQL |
|---|---|---|---|
| composition | 940 | 92.3 | 40.8 |
| conjunction | 1,526 | 84.9 | 44.6 |
| superlative | 183 | 46.4 | 45.0 |
| comparative | 199 | 48.2 | 44.4 |

**Mọi predicate ràng buộc đều được lấy về @50** (mức relation, có một triple mang predicate đó):

| Loại | SubgraphRAG (train trên CWQ) | DDE (CWQ) | REL-RAG P (WebQSP→CWQ) | REL-RAG T (WebQSP→CWQ) |
|---|---|---|---|---|
| composition | 87.2 | 86.8 | 16.5 | 10.6 |
| conjunction | 63.3 | 64.2 | **7.3** | 8.4 |
| superlative | 50.6 | 44.7 | 17.6 | 15.3 |
| comparative | 54.2 | 51.0 | 16.7 | 18.8 |

Lưu ý: số ở mức relation và có thể gồm predicate ngược chiều của Freebase. Cần chuẩn hóa cặp relation ngược trước khi đưa vào paper.

### 2.6 Số liệu nền đã có

Lấy từ [`REPORT.md`](REPORT.md) và EXP-17:
- ansR@50: WebQSP 90.4–90.9 cho mọi phương pháp; CWQ transfer của REL-RAG 61.9.
- Downstream: llama3.1 Q4 + SubgraphRAG@100 đạt Hit 85.05 / Macro-F1 68.83 (match gốc).
- Các tài liệu bên ngoài cùng chỉ ra benchmark có vấn đề:
  - KGQAGen: WebQSP chỉ đúng 52%, CWQ 49%.
  - GTSQA: retriever sụp khi có ≥3 seed, nhiều hop, hoặc kiểu cấu trúc chưa thấy.
  - BRINK: hiệu năng tụt khi KG thiếu bằng chứng.

### 2.7 S0 v1 không hợp lệ (lỗi dựng oracle), số liệu §2.1 sau khi neo lại

- S0 v1 lấy "triple ràng buộc" theo cặp (predicate, entity) ở bất kỳ đâu trong subgraph. Kết quả là chèn cả `(Alicia Keys, gender, Female)` hay `(Anguilla, notable_types, Country)`: nhiễu không liên quan, làm LLM sai thêm.
  - Kết quả v1 (197 câu): base 77.66, oracle 79.70, control 82.74 Hit. **Không dùng được.**
- **Bản neo (v2):** triple ràng buộc phải có nguồn là nút ở vị trí `SourceNodeIndex + 1` của một instantiation của `InferentialChain` từ topic và kết thúc ở đáp án (CVT, hoặc chính đáp án nếu chuỗi 1 hop).
  - Số liệu gần như không đổi: có ràng buộc trong subgraph 267 câu; nhãn SP chứa **0/267**; nhãn SubgraphRAG chứa **22/267**.
  - Recall ràng buộc của retriever:

| Retriever | @50 | @100 | @500 |
|---|---|---|---|
| SubgraphRAG | 11.2 | 15.4 | 42.7 |
| DDE | 12.4 | 18.0 | – |
| QR-DDE | 13.9 | 21.3 | – |
| REL-RAG P best-first | **31.5** | 44.2 | 73.4 |
| REL-RAG T | 14.2 | 22.5 | 50.9 |

- S0 v2 (185 câu; oracle = ràng buộc đã neo; control = triple anh em ngẫu nhiên của cùng nút nguồn): đang chạy.

### 2.8 S1 — tập ứng viên witness tự động trên WebQSP **train** (447 câu có ràng buộc đã neo)

- 88.1% câu có ứng viên nhiễu theo chuỗi gold; trung vị 4 nhánh trên mỗi nhánh đáp án. Nhãn SP chứa ràng buộc: **0.0%**.
- **DISC mềm**: điểm phân biệt = (tỷ lệ nhánh đáp án mang (r, o, chiều)) − (tỷ lệ nhánh nhiễu mang nó).

| Nguồn pattern | Điểm > 0: chứa gold (trung vị kích thước) | Điểm > 0.5 | Gold ở top-1 theo điểm | top-20 |
|---|---|---|---|---|
| Chuỗi gold | 81.0% (19) | 58.6% (10) | 53.9% | 85.5% |
| **Pattern SP (không cần parse)** | **82.3% (28)** | 40.7% (10) | 36.0% | 72.0% |

- **Kết luận S1: đạt** (mốc ≥ 70% cho túi ứng viên). Riêng điểm phân biệt không đủ để chọn đúng witness: gold chỉ ở top-1 36%. Phải học chọn có điều kiện theo câu hỏi trong túi, đúng là vai trò của reader ẩn danh hóa và κ_x ở §5.
- Định nghĩa DISC chặt ("không xuất hiện ở nhánh nhiễu nào") chỉ được 55–64%, vì một ràng buộc thật thường chỉ loại *một phần* ứng viên nhiễu. Loss phải là MIL / mềm, không phải nhãn cứng.

### 2.9 QR-DDE (CWQ, EXP-14c) đã giải phần giao nhiều anchor, chưa giải witness tại CVT

Mức tăng QR-DDE trừ DDE @50, tính bằng điểm (+); "phủ đủ" là phủ đủ bằng chứng GPT:

| Lát cắt | ansR | Phủ đủ |
|---|---|---|
| multi (≥2 anchor) | +3.2 | **+18.6** (52.6 → 71.2) |
| conjunction | +1.6 | **+13.1** |
| composition | +6.0 | +5.7 |
| **nary** (ràng buộc trên CVT) | +5.1 | **+2.2** (vẫn 37%) |

Hệ quả cho thiết kế:
- Stage A của WITNESS dùng QR-DDE làm backbone; phần giao nhiều anchor đã có bằng chứng.
- Đóng góp mới tập trung vào stage B/C, tức witness tại nút trung gian / CVT, nơi mọi phương pháp còn ≤ 40%.

---

## 3. Vì sao đây là điểm yếu của REL-RAG (và cả họ phương pháp)

1. **Nhãn.**
   - Pipeline rational path của REL-RAG (`FindRationalPath.py`) chỉ cho LLM chọn trong các path topic → đáp án (d..d+2). Ràng buộc nằm ở nhánh anh em, nên không bao giờ là ứng viên: phủ 0/271.
   - Chi phí gán nhãn GPT-4o của REL-RAG không mua được gì cho nhóm câu khó nhất.
2. **Mục tiêu train.**
   - Eq. 8 (softmax trên các triple kế tiếp + STOP): tại CVT, triple ràng buộc cạnh tranh trực tiếp với triple dẫn tới đáp án, nên bị coi là negative.
   - Eq. 12 (triple-based): triple ràng buộc là negative.
3. **Đồ thị.** `nx.DiGraph` gộp cạnh song song làm mất ~8% bằng chứng GPT (đo trong REPORT: chỉ còn 92.2% bằng chứng GPT trong DiGraph). Line graph cũng không có cạnh giữa các triple song song (EXP-13).
4. **Transfer.**
   - CWQ = WebQSP + thêm ràng buộc (conjunction/composition). Khi train trên WebQSP, REL-RAG chỉ lấy được 7–17% predicate ràng buộc của CWQ.
   - Claim "line graph tổng quát hóa tốt hơn" bị lẫn với việc *không bao giờ học lấy ràng buộc*.

---

## 4. Định nghĩa bài toán và lý thuyết (cần chứng minh đầy đủ)

**Ký hiệu.**
- Câu hỏi q có tập đáp án A.
- Relation pattern π (chuỗi relation từ anchor tới đáp án).
- C_π(q) = các entity tới được theo π (ứng viên); D = C \ A là ứng viên nhiễu.
- Bằng chứng S ⊆ G.
- *View* của ứng viên c trong S: V_S(c) = đa tập (relation, văn bản đầu kia) của các triple trong S chạm vào nhánh của c, **không tính danh tính của c**.

**Định nghĩa (answer-identifying evidence).** S *xác định được* A trong C nếu V_S(a) ≠ V_S(d) với mọi a ∈ A, d ∈ D, và sự khác biệt đó khớp với câu hỏi.

Witness tối thiểu tương ứng với **minimum hitting set** trên các cặp (a, d): NP-khó trong trường hợp tổng quát, nên ta học một xấp xỉ.

**Các kết quả cần chứng minh:**
- **T1 (Supervision mù với ràng buộc).** Với họ loss {BCE trên triple thuộc path đáp án; log-likelihood path với STOP (Eq. 8)}, mọi triple witness ngoài path có nhãn 0 hoặc là đối thủ trong softmax. Bộ chấm điểm tối ưu theo rủi ro không ưu tiên witness khớp câu hỏi hơn các triple anh em khác.
  - Bằng chứng thực nghiệm: §2.1, witness xếp như tung đồng xu giữa ~19 anh em.
- **T2 (Chặn trên khả năng xác định).** Nếu V_S(a) = V_S(d) cho k ứng viên nhiễu, thì mọi reasoner *dựa trên bằng chứng* (bất biến với việc đổi tên ứng viên) có P(top-1 đúng) ≤ 1/(k+1). Answer recall = 1 không kéo theo answerability.
  - Phần LLM vẫn trả lời đúng khi thiếu ràng buộc (76–79 Hit) là do tri thức tham số, và đó chính là phần không đáng tin cậy.
- **T3 (Tách biệt về budget).** Để chứa mọi đáp án và xác định được chúng:
  - Retriever chỉ lấy path cần Ω(|C|·|π|) triple, precision ≤ |A|/|C|.
  - Retriever lấy witness cần O(|A|·(|π| + c)) triple, với c là số ràng buộc.
  - Điều này giải thích vì sao ở cùng kích thước, precision theo nhãn GPT của REL-RAG không vượt SubgraphRAG (REPORT).
- **T4 (Nhất quán của mục tiêu).** Với reader ẩn danh hóa và phạt độ thưa (IB), nghiệm tối ưu của mục tiêu ở §5 chọn một hitting set tối thiểu, theo kiểu lập luận rationale/IB (Lei et al. 2016; GSAT, ICML'22).
  - Kết quả này bổ sung (không thay thế) Theorem 3.1 của REL-RAG: một retriever tổng quát hóa hoàn hảo về path vẫn bị chặn bởi T2.

---

## 5. Phương pháp đề xuất (tên tạm: WITNESS)

**Nguyên tắc:**
- Làm trên **entity graph gốc dạng MultiDiGraph**, không dùng line graph: chi phí tuyến tính, giữ cạnh song song.
- **Không có tham số nào đánh chỉ số theo relation ID.** Relation đi vào qua embedding văn bản z_r, nên giữ được lập luận √R ở `06_LITERATURE_REVIEW.md` §2.1.
- **Chỉ dùng đáp án làm supervision.**

### A. Lan truyền theo pattern (tìm nhánh ứng viên)
- Message passing có điều kiện theo câu hỏi từ các anchor (topic, cộng mention anchor mềm), kiểu QR-DDE/NBFNet.
- Trọng số w(r, q) = MLP([z_r; z_q]).
- Đầu ra: trạng thái nút h_v và điểm ứng viên s_c.
- Loss: loss path hiện có (nhãn SP hoặc rational), để giữ answer recall.

### B. Chấm điểm witness theo tương phản trong tập ứng viên (đóng góp mô hình)
- Với mỗi triple x = (v, r, o) chạm vào một nút v thuộc nhánh ứng viên (CVT hoặc ứng viên):

  w_x = MLP([z_q ⊙ z_r ; z_q ⊙ z_o ; h_v ; κ_x])

- κ_x là **đặc trưng tương phản**: set-attention trên *nhóm cạnh tranh* G(v), tức các nút tới được bằng cùng tiền tố pattern với v (các nhánh anh em). Nó cho biết (r, o) phân biệt v với các đối thủ đến mức nào:
  - Triple chung cho mọi nhánh (ví dụ: mọi CVT diễn viên đều có `film = Coronation Street`) không mang thông tin.
  - Triple riêng của một nhánh và khớp câu hỏi là witness.
- So với các mô hình hiện có:
  - SubgraphRAG chấm từng triple độc lập.
  - REL-RAG chấm dọc theo path.
  - Chưa phương pháp nào chấm **so với đối thủ**.

### C. Huấn luyện theo khả năng xác định (không cần nhãn witness)
- **Reader ẩn danh hóa** R: với mỗi ứng viên c trong top-M,

  e_c = Σ_{x ∈ nhánh(c)} σ(w_x) · ψ(q, x)

  ψ **không dùng embedding tên của c**. Reader cho logit ℓ_c = u^T e_c.
- Loss: L_id = −log Σ_{a∈A} softmax(ℓ)_a + β · Σ_x σ(w_x)  (thưa / IB).
- Vì reader không thấy danh tính ứng viên, gradient chỉ chảy vào các witness giúp tách A khỏi D. Đây là T2/T4 vận hành ngược lại thành tín hiệu học.
- Ablation: thêm MIL trên tập DISC (§2.4) làm regularizer.

### D. Retrieval và tự dừng
- Đầu ra gồm path tới các nhánh ứng viên tốt nhất cộng các witness của chúng.
- Dừng khi margin của reader giữa ứng viên tốt nhất và đối thủ vượt ngưỡng: budget thích ứng theo độ đủ của bằng chứng, không cố định số triple.

**Chi phí.** O(|E|·K·d) cho A; B và C chỉ chạy trên lân cận của top-M nhánh. Ước tính ≤ 2× SubgraphRAG.

---

## 6. Kế hoạch thí nghiệm và go/no-go

| Bước | Nội dung | Tiêu chí go | Trạng thái |
|---|---|---|---|
| **S0** | **Nhân quả** (v1 không hợp lệ, xem §2.7; v2 = 185 câu, ràng buộc đã neo): 197 câu WebQSP có ràng buộc entity, đáp án đã trong top-50 nhưng thiếu ràng buộc. Ba điều kiện: base (top-50), oracle (+ triple ràng buộc gold), control (+ cùng số triple anh em ngẫu nhiên). llama3.1 Q4, prompt SubgraphRAG, match gốc | oracle − control ≥ +5 Hit (hoặc F1) | Đã xếp hàng (screen `oracle1003`, chạy sau chain hiện tại); file trong `/relrag/runs/oracle_constraint/` |
| S1 | Nhãn: độ phủ DISC trên WebQSP **train** so với ràng buộc gold của train parse; chuẩn hóa relation ngược cho CWQ | DISC chứa gold ≥ 70% | ✅ 82.3% với DISC mềm (§2.8); còn phần chuẩn hóa relation ngược cho CWQ |
| S2 | Prototype trên subset (hạ tầng EXP-11): QR-DDE backbone + B + C | Recall ràng buộc gold @50 từ 12–37% lên ≥ 60%, ansR@50 giảm ≤ 1 | Cần code |
| S3 | Full WebQSP + CWQ (in-dist và transfer WebQSP→CWQ) | Vượt REL-RAG (P và T) về: recall ràng buộc, độ đủ bằng chứng @k, Hit/F1 downstream cùng budget (paired bootstrap), transfer CWQ conjunction | |
| S4 | GTSQA ([Graphcore/GTSQA](https://huggingface.co/datasets/Graphcore/GTSQA); subgraph gold, tới 5 seed và 5 hop, 27 kiểu cấu trúc, có split kiểu chưa thấy) | Thắng trên kiểu intersection / chưa thấy | Cần tải dữ liệu, chuyển sang format |

**Baseline bắt buộc:**
- SubgraphRAG, REL-RAG P/T (code-faithful), DDE/QR-DDE.
- **SubgraphRAG train bằng nhãn có ràng buộc gold** (nhãn oracle từ parse train). Đây là cận trên của việc "chỉ sửa nhãn", để chứng minh đóng góp nằm ở mô hình chứ không chỉ ở nhãn.
- Nếu chạy được: G-Retriever (PCST), GNN-RAG, PathISE.

**Metric:**
- Đã có: ansR, P/R/F1 theo SP/GPT/GPTp (`retrieval_report.py`).
- Thêm mới: **recall ràng buộc gold @k**, **độ đủ bằng chứng** (phủ đủ bằng chứng GPT / subgraph gold), Hit / Macro-F1 với match gốc.

---

## 7. Tính mới so với tài liệu

| Công trình | Liên quan | Khác biệt |
|---|---|---|
| SubgraphRAG (ICLR'25), REL-RAG (ICML'26), GNN-RAG (ACL Findings'25), RoG (ICLR'24) | Retriever cho KG-RAG | Supervise bằng path đáp án, đo answer recall; mù với ràng buộc (§2) |
| [PathISE](https://arxiv.org/html/2605.10791) (arXiv 05/2026) | MIL trên path yếu | Chỉ chọn path tới đáp án, không có nhánh ràng buộc hay ứng viên nhiễu |
| [Sufficient Context](https://arxiv.org/pdf/2411.06037.pdf) (ICLR'25) | Khái niệm "ngữ cảnh đủ" cho RAG văn bản | Chỉ phân tích, không học retriever; không có cấu trúc đồ thị / ứng viên nhiễu |
| GSAT (ICML'22), GIB, rationale (Lei et al. 2016) | Chọn subgraph đủ bằng IB | Bài toán phân loại đồ thị; không có câu hỏi, ứng viên nhiễu, hay RAG |
| G-Retriever (NeurIPS'24) | PCST chọn subgraph | Không học, prize theo độ tương đồng; không có tương phản ứng viên |
| STAGG (2015), query graph cho KBQA | Ràng buộc tường minh | Cần logical form / template; không phải retriever cho LLM |
| [GTSQA](https://arxiv.org/html/2511.04473) (TMLR'26), [KGQAGen](https://arxiv.org/html/2505.23495v3), [BRINK](https://arxiv.org/html/2508.08344) | Benchmark chỉ ra điểm yếu | Là benchmark; không chẩn đoán nguyên nhân ở supervision, không đề xuất phương pháp |
| [KGFR](https://arxiv.org/html/2511.04093), [GFM-RAG](https://arxiv.org/html/2502.01113v1) | Retriever tổng quát hóa chéo KG | Vẫn supervise theo path / đáp án |

---

## 8. Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| S0 cho hiệu ứng nhỏ vì LLM bù bằng tri thức tham số | Đo thêm F1 (ít đáp án thừa), thêm reasoner thứ hai; báo cáo T2 trên reasoner ẩn danh |
| Bằng chứng GPT nhiễu | Dùng ràng buộc gold từ parse WebQSP, SPARQL CWQ, subgraph gold của GTSQA |
| Ràng buộc thời gian / thứ tự cần literal (không có trong subgraph) | Ngoài phạm vi (8.6% câu WebQSP), chỉ phân tích |
| Reader ẩn danh tìm được shortcut | Kiểm tra bằng cách đổi tên ứng viên lúc test; ablation không dùng κ_x |
| CWQ: predicate "thiếu" có thể là relation ngược | Chuẩn hóa cặp relation ngược của Freebase (S1) |
| Tính mới bị nghi là "chỉ là nhãn tốt hơn" | Có baseline nhãn oracle (§6) và T1–T4 |
