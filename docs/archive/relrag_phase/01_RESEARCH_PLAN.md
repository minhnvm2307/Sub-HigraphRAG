# 01. Kế hoạch nghiên cứu

*Cập nhật: 2026-10-01 (kết quả M2 full data, nhãn GPT-4o/DeepSeek, dọn giai đoạn gemma). Trạng thái: ✅ xong · 🔄 đang làm · ⏳ chưa bắt đầu · ❌ bỏ.*

**Thứ tự ưu tiên hiện tại (người dùng chốt 10-01):** (1) tái hiện REL-RAG đúng cấu hình paper trước (`/relrag/run_paper.sh`); (2) quyết định số phận QR-DDE dựa trên CWQ; (3) track R tạm hoãn, người dùng tự research thêm.

Mục tiêu và giả thuyết: [`00_RESEARCH_GOAL.md`](00_RESEARCH_GOAL.md). Protocol đo: [`03_EVALUATION_PROTOCOL.md`](03_EVALUATION_PROTOCOL.md). Kết quả chi tiết của từng thí nghiệm ghi trong [`04_EXPERIMENT_LOG.md`](04_EXPERIMENT_LOG.md) theo mã `EXP-xx`.

## Tổng quan milestone

| Milestone | Nội dung | Trạng thái | Go/no-go |
|---|---|---|---|
| M0 | Chuẩn hóa baseline REL-RAG | 🔄 (nhãn đúng quy trình tác giả xong; chờ chạy `run_paper.sh`) | Baseline tái hiện được và công bằng (không bị làm yếu) |
| M1 | Phân tích điểm yếu REL-RAG | 🔄 | Có ít nhất một điểm yếu đo được và dùng được trong paper |
| M2 | QR-DDE trên full data (WebQSP + CWQ, in-distribution và transfer) | 🔄 WebQSP: không đạt (EXP-14). CWQ: QR-DDE bị ngắt ở epoch 10 nhưng val đã hơn DDE (EXP-14b) | QR-DDE ≥ REL-RAG và ≥ DDE ở R@50 (seed 42, paired bootstrap) |
| R | Nghiên cứu tài liệu và thiết kế mô hình "đột phá" (heterogeneous/multi-relational graph transformer có điều kiện theo câu hỏi) | ⏸ (khảo sát + 2 thiết kế xong; người dùng tạm hoãn để research thêm) | Có một thiết kế có lập luận, giải quyết được H1 và H4, và không dùng tham số riêng cho từng relation |
| M3 | Tổng quát hóa (claim chính của REL-RAG) | 🔄 (retrieval subset xong) | QR-DDE vượt REL-RAG khi transfer WebQSP→CWQ |
| M4 | Downstream reasoning | ⏳ | ΔF1 so với REL-RAG ≥ ngưỡng phát hiện được (~1.7) |
| M5 | Hiệu quả và biểu đồ Pareto | 🔄 | Latency ≤ 2× SubgraphRAG |
| M6 | Viết paper | ⏳ | – |

**Quy tắc chung:** mọi so sánh dùng cùng tập câu hỏi, cùng budget, cùng reasoner và cùng eval. Thí nghiệm tốn kém thì chạy trên subset trước. Chỉ chạy một job mỗi lúc, có giới hạn RAM ([`05_INFRASTRUCTURE.md`](05_INFRASTRUCTURE.md)).

## M0. Chuẩn hóa baseline REL-RAG

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Tái hiện pipeline: lọc rational path bằng gemma3 → train path/triple → inference | ✅ | EXP-01 … EXP-04 |
| Audit cài đặt (config, GCN, fp16, label, decoder) | ✅ | EXP-05: sửa decoder thành best-first |
| Reasoning @50 + hiệu chuẩn reasoner/protocol | ✅ | EXP-06, EXP-07: mình 83.8 / 72.3 so với paper 88.8 / 77.3 |
| Thử prompt reasoning gốc của tác giả để giải thích khoảng cách −5 điểm | ⏳ | `RAPL/generate_answers_llama31_8b.py`; chấm cả trích "The answer is" lẫn so khớp lỏng |
| REL-RAG với MultiDiGraph (giữ cạnh song song) | ⏳ | Tránh bị phản biện là baseline đã bị làm yếu |
| Nhãn rational path theo đúng quy trình tác giả (GPT-4o, ứng viên d..d+2) | ✅ | EXP-15: GPT-4o 901 câu + DeepSeek-V4.1-Flash 1,809 câu (phương án B); nhãn gemma (EXP-01) và mọi run gemma/SP đã xóa ngày 10-01 |
| Chốt cấu hình REL-RAG dùng để so sánh | 🔄 | Đúng paper F.1, seed 42, không self-loop như code gốc, cả path (P) và triple (T): `/relrag/run_paper.sh` → `runs/path_paper`, `runs/triple_paper` (người dùng chạy) |

## M1. Phân tích điểm yếu REL-RAG

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| E0: kích thước line graph và nguồn bùng nổ | ✅ | EXP-08: 83–107× số cạnh, 77% qua topic entity; CVT không đáng kể |
| E5: latency retriever và end-to-end | ✅ | EXP-09, EXP-10: retriever chậm 7×; end-to-end do LLM quyết định |
| Phân tích lỗi theo số hop và loại câu hỏi (CWQ: composition / conjunction / comparative / superlative) | ⏳ | Cần metadata loại câu hỏi của CWQ |
| Kiểm tra ràng buộc điểm xuất phát head = topic entity (Eq. 9) | ⏳ | Đo tỷ lệ đáp án chỉ tới được qua relation ngược chiều |
| Multi-relation giữa hai entity (H4) | ✅ | EXP-13 |

## M2. QR-DDE trên full data (WebQSP và CWQ)

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| QR-DDE so với DDE trên subset nhỏ | ✅ | EXP-11: +8.1 R@50 (WebQSP), +5.8 (CWQ) |
| Chuẩn bị record gọn full data (WebQSP train/val/test; CWQ train/val/test) | ✅ | `retrieve/exp_qrdde/prepare_full.py`; số câu khớp đúng tài liệu (WebQSP 2710/234/1559, CWQ 21943/2837/2848); CWQ train phải stream từng record ra đĩa rồi nạp lại để tránh vượt RAM (xem `05_INFRASTRUCTURE.md`) |
| **WebQSP full** (2,710 câu train): DDE, QR-DDE, seed 42; eval WebQSP test | ✅ | EXP-14: QR-DDE R@50 89.20 so với DDE 90.07 (Δ −0.87, CI [−2.02, +0.26]) → **lợi thế EXP-11 biến mất, M2 không đạt go/no-go** ở in-distribution |
| Transfer CWQ test cho checkpoint WebQSP full | ⏳ | `transfer_eval.py --runs_dir runs <run...>` (đọc feat/d/K từ `result.json`), người dùng chạy |
| Ablation trên WebQSP full: RT-DDE (điều kiện tĩnh S(q,r)=cos(z_q,z_r), không lan truyền học), QR-DDE bỏ query-conditioning (`qrdde_noq`), K ∈ {2, 3, 4}, d ∈ {32, 64, 128} | ✅ | EXP-14: query-conditioning không đo được tác dụng; K, d trong mức nhiễu; RT-DDE kém DDE có ý nghĩa (−1.58 R@50) |
| **CWQ full** (21,943 câu train dùng được): DDE, QR-DDE | 🔄 | EXP-14b. DDE xong: test R@50 86.7 (id-match), ngang SubgraphRAG chính thức. QR-DDE bị ngắt ở epoch 10; best epoch 8 trên val: ans R@50 91 so với DDE 88.7, triple R@100 89 so với 82.3. **Cần:** eval `best.pt` epoch 8 trên test (phải thêm chế độ chỉ eval cho `qrdde.py`) hoặc chạy lại tới khi dừng theo patience (~8–10 giờ); đo text-match, bootstrap, evidence recall |
| Schedule và seed train của M2 | ✅ | Giống hệt SubgraphRAG (`retrieve/configs/retriever/*.yaml`): seed 42 deterministic (`src/setup.set_seed`), `num_epochs 10000`, `patience 10` — đây là mặc định mới của `qrdde.py`. Một seed duy nhất, không chạy nhiều seed (quyết định 09-30). Lần chạy đầu (09-30) dùng nhầm 25 epoch cố định: best rơi ở epoch 21–24/25, đã dừng và lưu tại `runs_ep25/` chỉ để đối chiếu, không dùng làm kết quả |

**Go/no-go:** nếu QR-DDE full data không vượt DDE một cách có ý nghĩa (paired bootstrap trên câu hỏi test, CI 95%), chuyển trọng tâm sang mô hình của track R.

**Trạng thái go/no-go (10-01):** WebQSP không đạt, nhưng WebQSP đã bão hòa (mọi phương pháp 89–90 R@50, EXP-14). Kết luận chỉ chốt sau khi có test CWQ của QR-DDE. Nếu CWQ cũng không có ý nghĩa thì chuyển sang track R.

## R. Nghiên cứu tài liệu và thiết kế mô hình mới (song song với M2)

**Động cơ:** relation-aware / relation-topic / weighted-DDE đã bị REL-RAG bao phủ; residual hay đổi feature chỉ là tinh chỉnh. Cần một đóng góp mô hình hóa mới (xem mục 5 của [`00_RESEARCH_GOAL.md`](00_RESEARCH_GOAL.md)). Ghi kết quả vào [`06_LITERATURE_REVIEW.md`](06_LITERATURE_REVIEW.md).

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Kiểm chứng giả thuyết multi-relation (H4) bằng dữ liệu | ✅ | EXP-13: 13–14% triple, 22–29% câu; code REL-RAG mất ~6% bằng chứng |
| Phản biện H4: lỗi cài đặt hay giới hạn phương pháp? Đủ lớn làm luận điểm chính không? | ✅ | `06_LITERATURE_REVIEW.md` §6: cả hai — mất 6% bằng chứng là bug (sửa bằng MultiDiGraph), nhưng việc line graph không tạo cạnh giữa 2 relation song song là giới hạn cấu trúc thật; khuyến nghị dùng làm 1 trong 3 luận điểm hỗ trợ, không phải luận điểm độc lập |
| Tổng quan: heterogeneous graph transformer (HGT…), multi-relational GNN (R-GCN, CompGCN, RGAT), conditional message passing (NBFNet, A*Net, ULTRA, InGram), graph transformer (Graphormer, GraphGPS…), retriever cho KGQA (GNN-RAG, SubgraphRAG, REL-RAG…) | ✅ | `06_LITERATURE_REVIEW.md` §4; venue/năm xác minh qua web search, riêng RGAT không xác minh được venue (có thể chỉ là arXiv preprint) |
| Phân tích: mô hình nào xử lý được (a) relation mixing, (b) cặp entity nhiều relation, (c) tổ hợp r₁→r₂, (d) relation chưa thấy khi transfer, (e) chi phí tuyến tính | ✅ | `06_LITERATURE_REVIEW.md` §5: không paper nào khảo sát được kết hợp text-conditioning với zero tham số theo relation; ULTRA/InGram inductive nhưng theo cấu trúc đồ thị, không theo text |
| Đề xuất thiết kế mô hình + lập luận lý thuyết (so với Theorem 3.1 của REL-RAG) | ✅ (chờ duyệt) | `06_LITERATURE_REVIEW.md` §7: đã đọc trực tiếp Appendix A.4.1 của paper gốc để xác định chính xác cơ chế √R (không gian tham số O(R·d²) do W_r riêng theo relation) và lập luận vì sao 2 thiết kế đề xuất (TR-NBFNet; Relational Attention text-conditioned) tránh được nó |
| Prototype trên subset nhỏ (giống EXP-11), so với QR-DDE và REL-RAG | ⏳ (tạm hoãn) | Người dùng cần research thêm về thiết kế ở `06_LITERATURE_REVIEW.md` §7 trước khi quyết định (09-30) |

## M3. Tổng quát hóa (claim chính của REL-RAG)

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Transfer WebQSP→CWQ ở tầng retrieval cho REL-RAG (path/triple) | ✅ | EXP-12: đúng paper R@50 ~53; triple + residual 74.6 |
| Transfer cho QR-DDE và DDE (subset, rồi full data) | ✅ / ⏳ | EXP-12 (subset: DDE 56–57, QR-DDE 58–64 R@50); full data sau M2 |
| Tìm hiểu vì sao residual giúp transfer +23 R@50 (lân cận cấu trúc so với feature riêng của triple) | ⏳ | Luận điểm phân tích, không phải đóng góp chính |
| Tách kết quả CWQ theo kiểu shift của paper: ID / compositional (tổ hợp relation mới) / OOD (relation chưa thấy khi train) | ⏳ | Xây từ tập relation-path của WebQSP train |
| GrailQA (nếu lấy được subgraph) | ⏳ | Tùy chọn |

## M4. Downstream reasoning

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Cố định reasoner (llama3.1 Q4 qua ollama), prompt, eval gốc, 1,608 câu | ✅ | EXP-07 |
| QR-DDE @50 so với REL-RAG @50 và SubgraphRAG @50, paired bootstrap | ⏳ | Sau M2 |
| Transfer WebQSP→CWQ downstream | ⏳ | Sau M3 |
| Input dạng path (Table 10 của paper) cho mọi retriever | ⏳ | Áp dụng như nhau cho tất cả |

## M5. Hiệu quả

| Việc | Trạng thái |
|---|---|
| Latency retriever (median/p95), tách theo nhóm subgraph có hub | 🔄 (đã có tổng thể; chưa tách nhóm) |
| Biểu đồ Pareto latency so với R@50 / F1 cho mọi retriever | ⏳ |
| Bộ nhớ và thời gian train | ⏳ |

## M6. Viết paper
Hình chính dự kiến là Pareto latency so với chất lượng; bảng chính là ID + transfer; phần phân tích gồm E0 và ablation cơ chế.

## Rủi ro và cách xử lý

| Rủi ro | Cách xử lý |
|---|---|
| Lợi thế của QR-DDE mất đi ở full data (chỉ là hiệu ứng ít dữ liệu) | Chạy M2 sớm nhất; chuẩn bị phương án B (relation graph kiểu ULTRA) |
| Tính mới bị nghi ngờ (NBFNet/ULTRA đã có) | Đóng khung đóng góp là phân tích REL-RAG + retriever có điều kiện theo câu hỏi cho RAG + bằng chứng về tổng quát hóa và chi phí |
| Không tái hiện được số tuyệt đối của REL-RAG | Luôn báo cáo cả số của paper lẫn số tái hiện; so trong cùng pipeline |
| Phần cứng (1080 Ti, RAM ~19GB khả dụng) | Làm subset trước, dùng memmap, chạy tuần tự có giới hạn RAM |
