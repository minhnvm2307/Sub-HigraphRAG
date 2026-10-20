# Kế hoạch nghiên cứu tuần 10-11 → 10-17: từ mẹo inference sang một mô hình

Augmentation anchor bị loại vì overfit. AER, ADR và ASE chỉ thắng rõ ở WebQSP → CWQ: ở in-distribution chúng làm giảm nhẹ, và ở CWQ → WebQSP còn thua (BENCHMARK.md). Kế hoạch này thay ba mẹo đó bằng **một mô hình trên entity graph** với bốn giả thuyết kiểm chứng được. Mỗi giả thuyết có phép chẩn đoán rẻ chạy trước, và có tiêu chí đặt trước trên **toàn bộ ma trận transfer**.

## 1. Bằng chứng hiện có

**Ai thắng ở đâu** (ansR@50; nguồn: BENCHMARK.md, run WL train CWQ ngày 10-08):

| Setting | Baseline mạnh nhất | Của mình tốt nhất | Thành phần quyết định |
|---|---|---|---|
| WebQSP (in-dist) | REL-RAG 90.6 | WL 92.6 | Relation-topic (WL); AER/ADR làm giảm 0.2–1.4 |
| WebQSP → CWQ | SubgraphRAG 70.1 | WL + ASE + aug + AER 87.4 (aug bị loại) | **AER**: composition +19.5 |
| CWQ (in-dist) | SubgraphRAG 86.2 | QR-DDE + ADR 90.9, ASE 90.3 | Mã hoá theo câu hỏi / theo anchor |
| CWQ → WebQSP | **RAPL tác giả 90.9** | RT 85.0 | Không phương pháp nào của mình thắng |

**Phân rã lỗi khi transfer WebQSP → CWQ** (DDE, ansR@50, mới đo 10-11):

| Mode | Tất cả | Composition (940 câu) | Conjunction (1,526 câu) |
|---|---|---|---|
| base | 70.1 | **54.6** | 78.1 |
| + ADR (tách theo anchor) | 72.5 | 54.5 | 82.7 |
| + ADR + giao mềm `min` | 72.8 | 54.5 | 83.4 |
| + AER | 75.8 | 74.1 | 76.8 |

Đọc bảng:
1. **Lỗi lớn nhất là composition.** Retriever train trên câu 1–2 bước không lan truyền được xa hơn. AER sửa được bằng cách chạy thêm lượt, tức là bằng chứng rằng *tầm lan truyền* (horizon) là nút thắt.
2. **Tách theo anchor giúp conjunction, nhưng phép giao đặt lên điểm z chỉ thêm 0.7.** Điểm theo từng anchor chưa được hiệu chuẩn như xác suất, nên phép giao không có ngữ nghĩa.
3. **Relation-topic (WL) là thành phần tổng quát hoá rộng nhất.** Nó là độ tương đồng text không học, nên không overfit. Ngược lại, QR-DDE (điều kiện theo câu hỏi *có học*) overfit phân phối câu hỏi của WebQSP.

**Chẩn đoán D1 (10-11, `relrag/diag_horizon.py`, `runs/diag_horizon.md`) sửa lại cách hiểu ở mục 1:**
- Ở WebQSP → CWQ, mức sụt **nằm ở câu 2 bước** (DDE 60.5, so với 86.5 khi train trên CWQ), không phải câu ≥3 bước (chỉ 171/2,831 câu, khó cả khi train trên CWQ). WebQSP train có 34.5% câu 2 bước, nên đây **không phải ngoại suy độ dài**.
- Ở CWQ → WebQSP, mức sụt nằm ở câu 0–1 bước (87.1 so với 96.2). WL tốt nhất; AER, ADR và ASE đều làm giảm.

Chia các câu CWQ 2 bước theo việc chuỗi relation SP đã xuất hiện trong nhãn train WebQSP hay chưa (ansR@50):

| Câu 2 bước | DDE | DDE + AER | WL | WL + ADR + AER | DDE train CWQ |
|---|---|---|---|---|---|
| Tổ hợp đã thấy (1,058) | 66.0 | 66.0 | 72.7 | 81.6 | 84.4 |
| **Tổ hợp mới, relation đã thấy (587)** | **49.2** | **74.8** | 53.1 | 76.0 | 88.5 |
| Có relation chưa thấy (196) | 61.4 | 77.5 | 62.6 | 79.3 | 91.6 |

Kết luận:
- Lỗi chính là **tổng quát hoá tổ hợp**: gặp tổ hợp mới thì thấp hơn 17 điểm.
- AER sửa đúng lỗi này (+25.6) vì nó **phân rã một tổ hợp 2 bước thành hai bước 1 bước đã quen** (neo lại ở entity cầu nối).
- Do đó H1 được viết lại thành giả thuyết **phân rã theo bước** (mục 4).

## 2. Khoảng trống nghiên cứu (đã đối chiếu tài liệu)

| # | Khoảng trống | Bằng chứng | Đã có ai làm? |
|---|---|---|---|
| G1 | **Tổng quát hoá tổ hợp relation cho retriever KGQA bằng chấm điểm phân rã theo bước** (D1: đây không phải ngoại suy độ dài) | Tổ hợp mới ở 2 bước 49.2, so với 66.0 cho tổ hợp đã thấy; AER +25.6 nhờ phân rã | Recurrent net cho easy→hard: [Schwarzschild 2021](https://ar5iv.labs.arxiv.org/html/2106.04537), [Bansal 2022](https://ar5iv.arxiv.org/html/2202.05826), mới chỉ có trên bài toán thuật toán. NBFNet / A*Net không kiểm tra ngoại suy độ dài ([A*Net](https://arxiv.org/pdf/2206.04798)). **Chưa có ai làm cho KG retriever** |
| G1' | Đối chiếu thêm (10-11) | — | [TransferNet](https://arxiv.org/pdf/2104.07302) (EMNLP 2021) cũng chuyển điểm entity theo điểm relation từng bước, nhưng điểm relation của mỗi bước đến từ attention riêng của bước đó (không dùng chung) và mô hình là reader trả lời, không phải retriever cho LLM. [SG-KBQA](https://arxiv.org/html/2502.12737v3) chỉ ra rằng KBQA sinh logical form có xu hướng lặp lại tổ hợp đã thấy, và sửa bằng cách đi qua schema trước. [SeedER](https://www.alphaxiv.org/abs/2605.23753) (05-2026) cho rằng dense retriever bị giới hạn trên truy vấn tổ hợp, và dùng policy RL để mở rộng. Chưa ai phân rã điểm của *retriever triple* thành tích các nhân tử một bước dùng chung, nên G1 vẫn còn trống |
| G2 | **Sửa relation mixing ngay trên entity graph, không cần line graph** | REL-RAG chỉ chứng minh cho MPNN *cộng dồn* ([REL-RAG §3](https://arxiv.org/pdf/2506.09645)). Line graph tốn gấp 83–107× cạnh, 77 MB/câu, gần 9 giờ chuẩn bị dữ liệu cho CWQ test | Thông điệp nhân theo relation (generalized Bellman-Ford) biểu diễn mỗi đường đi bằng một tích riêng ([NBFNet](https://arxiv.org/abs/2106.06935v4)), nhưng chưa ai dùng để phản biện lập luận "relation mixing" hay cho retrieval phục vụ LLM. [KGFR](https://arxiv.org/pdf/2511.04093) dùng entity graph nhưng thông điệp *cộng* (s + r) và chấm ở mức entity |
| G3 | **Giao nhiều anchor mà không cần dữ liệu train nhiều anchor** | GTSQA: *mọi* retriever thất bại với câu có ≥3 anchor, kể cả câu 1 bước; SubgraphRAG yếu nhất ở dạng "phép chiếu sau phép giao" ([GTSQA](https://arxiv.org/pdf/2511.04473)). Phép giao mềm của mình chỉ thêm 0.7 | Logic mờ không tham số trên tập mờ (GNN-QE, [Zhu 2022](https://arxiv.org/abs/2205.10128v1)), nhưng cần dạng logic có sẵn. Chưa ai dùng cho câu hỏi tự nhiên có nhiều anchor, chỉ train trên câu một anchor |
| G4 | **Giảm thiên lệch của nhãn shortest path** | Ở câu 4 bước, 72% triple đúng nằm ngoài đường ngắn nhất ([GTSQA](https://arxiv.org/pdf/2511.04473)); mọi retriever của mình train bằng nhãn SP | KGFR chỉ dùng nhãn đáp án (log-loss trên tập đáp án), không dùng đường đi |

## 3. Mô hình đề xuất: RBF-Ret (Relevance-gated recurrent Bellman-Ford retriever)

Mô hình chạy trên **entity graph**. Không phải MLP thuần mà là lan truyền lặp có điều kiện theo câu hỏi. Mỗi thành phần thay một mẹo hiện có bằng một cơ chế có ngữ nghĩa:

| Thành phần | Thay cho | Cơ chế |
|---|---|---|
| **Lan truyền lặp, trọng số dùng chung giữa các bước** (weight-tied, có "recall" giữ lại điều kiện biên; huấn luyện tăng dần kiểu Bansal) | AER (neo lại lúc inference) | Train với T = 2–3 bước, test với T lớn hơn hoặc tự dừng khi điểm hội tụ. Tầm lan truyền tăng mà không cần mẹo. Có recall nên chạy thêm bước không làm hỏng câu ngắn |
| **Thông điệp nhân theo relation**: m = s(u) ⊙ φ(z_r); φ là phép chiếu nhỏ của embedding text relation; có biến đổi theo từng bước để giữ thứ tự | Line graph của REL-RAG | Mỗi đường đi thành một tích riêng (không trộn), chi phí O(\|E\|·T), không lưu gì xuống đĩa |
| **Cổng relation-topic không học** g(r) = σ(τ · rel(q, r) + b), chỉ học τ và b | WL (giữ lại) và QR-DDE (bỏ) | Giữ đúng tín hiệu tổng quát hoá được (độ tương đồng text) mà không cho mô hình học thuộc phân phối câu hỏi |
| **Điểm theo từng anchor hiệu chuẩn như xác suất** p_a(v) = P(v nằm trên đường trả lời từ a), kết hợp bằng **t-norm tích** (giao) và max (hợp) | ADR, ASE (và aug) | Không có tham số cho phép giao, nên không cần dữ liệu nhiều anchor. **Với một anchor, mô hình trùng khít bản một anchor**, nên in-distribution không thể giảm vì thành phần này |
| **Loss tới đáp án** (log-loss trên tập đáp án, như KGFR) cộng với BCE trên nhãn SP | Nhãn SP thuần | Bớt thiên lệch theo đường ngắn nhất |

Lớp chấm triple giữ khuôn của SubgraphRAG (q, z_h, z_r, z_t, cùng trạng thái của h và t), nên vẫn retrieve triple theo budget cho LLM.

## 4. Giả thuyết và cách bác bỏ

| | Giả thuyết | Bác bỏ nếu |
|---|---|---|
| H1 (viết lại sau D1) | **Chấm điểm phân rã theo bước** (lan truyền lặp với toán tử 1 bước dùng chung trọng số; điểm của đường đi = tích các điểm từng bước) tổng quát hoá được sang **tổ hợp relation chưa thấy**, như AER nhưng nằm trong mô hình nên không làm hỏng các ô khác | Ở nhóm "tổ hợp mới, relation đã thấy" (587 câu CWQ 2 bước), không vượt DDE base 49.2 lên gần mức AER (≥ 70); hoặc nhóm "tổ hợp đã thấy" và WebQSP in-dist giảm |
| H2 | Thông điệp nhân trên entity graph phân biệt tổ hợp relation ngang line graph, với chi phí tuyến tính | Trên bài toán tổng hợp có kiểm soát (tổ hợp relation chưa thấy), nó thua line graph rõ, hoặc không hơn MPNN cộng dồn |
| H3 | Xác suất theo anchor đã hiệu chuẩn, kết hợp bằng t-norm tích, cải thiện câu ≥2 anchor mà không cần train trên câu nhiều anchor | Không hơn ADR max ở CWQ conjunction và GTSQA ≥2 anchor |
| H4 | Tổng quát hoá giữa domain đến từ cổng relation-topic không học | Bỏ cổng mà GrailQA (tập đúng) và GTSQA không giảm |
| H5 | Loss tới đáp án giảm thiên lệch SP, tăng recall ở câu dài | Ablation không đổi ở câu ≥3 bước |

**Tiêu chí thành công cuối, đặt trước:**
- **In-dist** (WebQSP, CWQ): không thua SubgraphRAG có ý nghĩa, và không thua bản tốt nhất hiện có quá 1 điểm.
- **Mọi ô transfer** (WebQSP→CWQ, CWQ→WebQSP, WebQSP→GrailQA và CWQ→GrailQA trên **tập đúng của paper**, WebQSP→GTSQA): ≥ SubgraphRAG và ≥ REL-RAG/RAPL; thắng có ý nghĩa ở đa số ô.
- **Ràng buộc:** không augmentation, không dùng thống kê của dataset đích, 2 seed.

## 5. Lịch tuần

| Ngày | Việc | Máy |
|---|---|---|
| 10-11 | Chấm xong RAPL tác giả; dọn dẹp; cập nhật BENCHMARK (run WL train CWQ). **Chẩn đoán**: D1 recall theo độ dài đường đi ở mọi ô; D2 độ hiệu chuẩn của điểm theo anchor; D3 bài toán tổng hợp cho H2 (CPU, đồ thị nhỏ) | Local, CPU nhẹ |
| 10-12 | Dựng **GrailQA tập đúng** (khoảng 995 → ~874 câu; embed GTE trên GPU thuê); cài RBF-Ret và kiểm tra: tắt cổng và T = 2 phải tái hiện DDE-lite; sanity trên bài toán tổng hợp | vast A4000 |
| 10-13 | Train RBF-Ret trên WebQSP full: T_train 2 và 3; có và không weight-tie; có và không loss tới đáp án | vast |
| 10-14 | Chấm toàn ma trận; vẽ đường recall theo T_test (2–6) cho từng số bước (H1); ablation cổng (H4) | Local và vast |
| 10-15 | Giao xác suất (H3) và hiệu chuẩn; train trên CWQ full cho ô CWQ→WebQSP và CWQ→GrailQA | vast |
| 10-16 | Seed thứ hai; đo chi phí (thời gian, bộ nhớ, đĩa) so với line graph (đã có số đo của RAPL); GTSQA | vast |
| 10-17 | Viết lại tab doc "Giải pháp & lập luận"; quyết định có chạy downstream không (OpenRouter, sẽ ước tính và chạy thử trước) | — |

**Ngân sách:** vast khoảng 20–25 giờ A4000, tức khoảng $2.5 (credit còn $4.48). OpenRouter còn khoảng $0.5 sau lần chạy RAPL. svuet: GPU đang hỏng tản nhiệt (99–103 °C), không dùng.

## 6. Rủi ro
- Cổng không học có thể quá yếu ở CWQ in-dist (QR-DDE cao hơn 3 điểm nhờ điều kiện có học). Giải pháp: chỉ học số vô hướng τ, b, và kiểm tra cả hai ô in-dist.
- Lan truyền lặp có thể "overthink" (giảm khi T lớn). Đã có giải pháp chuẩn (recall và huấn luyện tăng dần); bác bỏ H1 nếu vẫn xảy ra.
- Dựng GrailQA tập đúng phụ thuộc subgraph ODA (cần token HF, đã có).

## 7. Tiến độ (cập nhật liên tục)
- **10-11, cài đặt.**
  - `retrieve/exp_qrdde/rbf.py`: RBF-Ret, max-product Bellman-Ford theo từng anchor. Nhân tử cạnh là π(r|q) (fwd/rev), tuỳ chọn thêm nhân tử nút ψ(v|q). Giao là t-norm tích, hợp là max. Với một anchor, mô hình trùng khít bản một anchor (đã kiểm bằng assert).
  - `qrdde.py` có thêm feat `rbfonly` (điểm đường đi là logit) và `rbf` (4 feature đưa vào MLP SubgraphRAG, cộng loss phụ), các cờ `--rbf_T/--rbf_pi/--rbf_node/--rbf_aux`, và `--device cpu`.
  - `adr_probe.py`: thêm `--rbf_T` cho T lúc test và `--device`.
  - `relrag/diag_composition.py`: script cho phép tách nhóm tổ hợp của H1.
- **10-11, train trên CPU local** (`rbfonly` nhẹ, khoảng 15 ms/câu):
  - π = MLP: `runs/webqsp_rbfonly_s42`.
  - π = cổng không học, 6 tham số: `runs/webqsp_rbfgate_s42`. Dừng ở epoch 14 vì đã bão hoà, val ansR@50 cao nhất **48.6**, so với MLP ≥ 86 ở epoch 9. → Riêng tín hiệu text cố định của WL **không đủ**, π phải học có điều kiện theo câu hỏi. H4 phải kiểm ở dạng "MLP π có kèm prior hay không", không phải "chỉ có cổng".
  - π = MLP, thêm ψ: `runs/webqsp_rbfnode_s42`.
- **10-11, GrailQA tập đúng:** `data_ext/grailqa_paper/test.pkl`, 728 câu (xem EXP-37). Còn thiếu embedding.
- **10-11, peek CWQ (WebQSP→CWQ) bằng checkpoint epoch 12 của `rbfonly`, không mẹo inference** (ansR@50, `relrag/diag_composition.py`):

  | Nhóm | DDE | DDE+AER | WL+ADR+AER | **RBF-only** | DDE train CWQ (trần) |
  |---|---|---|---|---|---|
  | Tổ hợp đã thấy (1,058) | 66.0 | 66.0 | 81.6 | 75.7 | 84.4 |
  | **Tổ hợp mới, relation đã thấy (587)** | 49.2 | 74.8 | 76.0 | **83.7** | 88.5 |
  | Relation mới (196) | 61.4 | 77.5 | 79.3 | **90.3** | 91.6 |
  | Câu 1 topic (1,329) | 63.6 | — | 81.8 | **85.6** | 86.1 |
  | Câu ≥ 2 topic (1,519) | 75.7 | — | 84.4 | 75.3 | 87.1 |
  | 1 bước (810) | 94.5 | — | 94.1 | 85.7 | 92.3 |
  | Tất cả @50 / @100 / @200 | 70.0 / 79.6 / 87.7 | 75.8 / 84.8 / 91.1 | 83.2 / 89.8 / 94.7 | 80.1 / 87.2 / 94.2 | 86.7 / 91.3 / 95.1 |

  → **H1 đạt ở dạng mạnh** (nhóm tổ hợp mới 49.2 → 83.7, ngưỡng đặt trước là ≥ 70). Câu 1 topic gần chạm trần train CWQ. Hai điểm yếu là câu ≥ 2 topic và câu 1 bước.
- **10-11, các phép thử chỉ đổi inference đều thất bại (kết quả âm, ghi lại):**
  - **Soft-AND** (max-marginal của cấu trúc hội, anchor khác là ràng buộc mềm ε): nhóm ≥ 2 topic từ 75.3 xuống 70.8 / 73.3 / 73.5 (ε = 0.01 / 0.1 / 0.3). **H3 bị bác ở dạng này.**
  - **π đối xứng hai chiều:** mọi nhóm giảm (tất cả 80.1 → 74.0), vì đi ngược relation có fan-out lớn làm ngập budget.
  - **Chia budget đều theo anchor** (round-robin / max-z kiểu ADR): ≥ 2 topic 73.8 / 73.9, không giúp.
  - **Chẩn đoán lỗi** (91 câu 1 bước mà RBF thua DDE rõ): 87 câu có triple đáp án nằm hẳn dưới vị trí 50, tức không phải do hoà điểm. Nguyên nhân là π sai (vd. award_winner > directed_by cho "direct"), và thiếu bằng chứng ở mức entity/kiểu ("Hollywood studio" → Universal Studios Hollywood). Đây đúng là chỗ MLP DDE có z_t. → Nhân tử nút ψ(v|q) là phép thử quyết định.
- **10-11, kết quả chính thức của `webqsp_rbfonly_s42`** (EXP-39):
  - WebQSP: gate in-dist PASS, ansR@50 89.36.
  - WebQSP → CWQ: gate PASS so với cả SubgraphRAG (+10.2 @50) và REL-RAG (+18.4 @50), 20/20 ô hỗ trợ đều thắng. RBF@k ≈ SubgraphRAG@2k.
- **10-11, đi không quay lui (non-backtracking, `--rbf_nb`)**:
  - Chẩn đoán: anchor là hub (múi giờ, type "Country") sinh đường "nảy" r⁻¹ rồi r, ra anh em của anchor, và ngập top-50 (vd. 'Asia time_zones X' −0.51 so với đáp án −2.22).
  - Cài đặt: max-product chính xác trên walk không quay lui, giữ top-2 trạng thái mỗi nút với tag bước cuối khác nhau. Khớp brute force trên 300 đồ thị ngẫu nhiên.
  - Chỉ bật lúc test (checkpoint cũ): WebQSP ansR@50 89.26 → **90.11** (SP-R +0.8, GPT-R +0.8). CWQ transfer 80.3 → 81.1 @50, 87.5 → 88.3 @100, 93.5 → 94.2 @200. → Không có đánh đổi. Đang train lại với nb (`webqsp_rbfnb_s42`).
  - Ý nghĩa với G2: line graph tránh được quay lui vì trạng thái của nó là cạnh. Entity graph đạt được điều đó với chỉ 2 trạng thái mỗi nút.
- **10-11, nhân tử nút ψ(v|q) đặt trong lan truyền:** val ansR@50 dừng ở khoảng 81–84 (không có ψ: 88). Dừng ở epoch 21, kết quả âm.
- **10-11, nhãn rational path (GPT-4o + DeepSeek), `webqsp_rbfrat_s42`:**
  - Số triple dương ít hơn SP 3.7 lần (17k so với 63.5k).
  - WebQSP test ansR@50/100 = 89.81 / 93.97, so với SP 89.36 / 93.85.
- **10-11, kiểm chứng cơ chế bằng hybrid** (`webqsp_rbfmlp_nb_s42`: MLP entity của SubgraphRAG, **không có DDE**, cộng 4 feature đường đi RBF, train trên vast):
  - WebQSP in-dist **91.48 / 95.82 / 97.84**, cao nhất từ trước tới nay (SubgraphRAG official 90.38 / 94.42 / 96.94). Gate PASS.
  - CWQ transfer: nhóm **tổ hợp mới tụt về 51.2**, ngang WL (53.1); RBF+nb thuần được 84.0. Nhóm tổ hợp đã thấy 87.9 (RBF 76.5). Câu ≥ 2 topic **90.1** (RBF 78.1), conjunction 92.1.
  - → Đúng như dự đoán của H1: **mọi bộ chấm MLP kết hợp được quan hệ giữa đặc trưng bước-sau và relation đều học thuộc tổ hợp**, kể cả khi không có DDE. Điểm phân rã theo bước tổng quát hoá được. Hai thành phần bù nhau: MLP mạnh ở bằng chứng entity (câu conjunction, nhiều anchor), RBF mạnh ở tổ hợp.
  - Bước tiếp theo (đặt luật trước, không tinh chỉnh trên CWQ): hợp nhất hai head của cùng một mô hình bằng max-z (`heads_probe.py`), giống luật hợp nhất của ADR/AER.
- **10-11, hybrid WL + RBF + nb** (`webqsp_wlrbf_nb_s42`):
  - WebQSP 91.29 / 95.08 / 97.76.
  - CWQ: tổ hợp mới **64.5** (WL 53.1, RBF+nb 84.0). Toàn bộ 81.2 / 88.6 / 93.3.
  - → Thêm feature RBF vào MLP chỉ giảm được một phần việc học thuộc tổ hợp. Trong các dạng đã thử, chỉ điểm phân rã theo bước thuần là tổng quát hoá trọn vẹn.
- **10-11, hợp nhất hai head của hybrid bằng max-z** (`heads_probe.py`, luật đặt trước như ADR):
  - Bản `rbfmlp`: CWQ **83.1 / 90.5 / 95.3** (WL+ADR+AER: 83.2 / 89.8 / 94.7), WebQSP 91.25 / 95.59 / 97.84.
  - Đổi lại, nhóm tổ hợp mới giảm 84.0 → 66.0, câu 1 topic 86.1 → 75.7; câu ≥ 2 topic tăng 78.1 → 89.6.
  - → max-z để cho MLP lấn át ở những chỗ nó tự tin nhưng sai (tổ hợp mới). Đây không phải cách kết hợp có nguyên tắc.
- **10-11, RBF-PoE** (`--feat rbfpoe`, đang train trên vast):
  - P(bằng chứng) = P(nằm trên đường đi đáng tin từ anchor; RBF) × P(liên quan cục bộ; MLP chỉ trên text q, z_h, z_r, z_t).
  - MLP không có cờ anchor, DDE hay feature đường đi, nên về cấu trúc không mã hoá được tổ hợp. Nó chỉ tách triple hoà điểm và đem vào bằng chứng entity/kiểu. Train chung bằng một BCE trên tích.
  - Tiêu chí đặt trước: giữ được nhóm tổ hợp mới ≥ 80 (gần mức RBF+nb), đồng thời câu ≥ 2 topic hơn RBF+nb và WebQSP ≥ RBF+nb.
- **10-11, RBF-PoE (`webqsp_rbfpoe_nb_s42`): KHÔNG đạt tiêu chí đặt trước.**
  - Nhóm tổ hợp mới 82.5 (đạt ≥ 80).
  - WebQSP 88.74 / 94.70 / 97.76, thấp hơn RBF+nb (89.63) → không đạt.
  - CWQ 81.2 / 89.4 / 94.1, không hơn RBF+nb.
  - → MLP cục bộ không thấy ngữ cảnh nên không giúp được: nó không đánh giá được bước qua CVT (không có text), và kéo tích xuống.
- **10-11, giải pháp cho khoảng hụt in-dist là max-marginal của đường đi trọn vẹn (mm)**, xem EXP-39. Cấu hình chính mới: **RBF + nb (train) + mm (test)**.
  - WebQSP: 91.37 / 95.18 / 98.05.
  - CWQ: 87.1 / 92.1 / 96.0.
  - Nhóm tổ hợp mới 85.7, ≥ 2 topic 85.6, ≥ 3 bước 73.1.
  - Đang kiểm: train với mm (`webqsp_rbfnbmm_s42`); GrailQA và GTSQA với mm; train CWQ (`cwq_rbfnb_s42`) cho các ô CWQ → *.
- **10-11, đính chính về nb:** khi train lại không nb nhưng đã sửa lỗi scatter, kết quả ngang RBF+nb (xem EXP-39). Phần lợi ích trước đây gán cho nb chủ yếu đến từ việc sửa lỗi. Các đóng góp được giữ: (1) điểm phân rã theo bước, (2) max-marginal đường đi trọn vẹn (mm).
