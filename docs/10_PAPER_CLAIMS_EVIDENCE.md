# 10. Bản đồ claim → bằng chứng cho paper (hướng docs/09)

*Tạo: 2026-10-04. Mỗi claim kèm thí nghiệm chứng minh, con số và trạng thái. Các thí nghiệm theo mã EXP trong [`04_EXPERIMENT_LOG.md`](04_EXPERIMENT_LOG.md).*

**Tên tạm:** *Where, not What: Anchor-Equivariant Retrieval for Compositional Generalization in KG-RAG.*

## 1. Câu chuyện một đoạn
- REL-RAG (ICML'26) cho rằng line graph đem lại khả năng tổng quát hóa thành phần cho retriever KG-RAG.
- Ta đo lại ở tầng retrieval: train trên WebQSP, đo trên CWQ. REL-RAG transfer **kém hơn** một MLP trên entity graph (−8.1 ansR@50). Trên câu composition nó sụp xuống 26%.
- Nguyên nhân: mọi retriever học được *mức liên quan gắn với vị trí so với anchor*. Ta chứng minh nhân quả bằng can thiệp dịch anchor: cùng câu hỏi, dời anchor 1 hop thì recall tụt 21–43 điểm.
- Giải pháp nguyên lý: **retrieval bất biến theo anchor (AER)**. Lặp một bộ chấm hop nguyên thủy trên một tập anchor thay đổi dần (thêm entity cầu nối), trộn bằng max đơn điệu, và train có tăng cường anchor.
- AER giữ nguyên WebQSP và **hơn REL-RAG +16 ansR@50 khi transfer sang CWQ**. Nguyên lý dùng được cho cả line graph (REL-RAG + neo lại +8.5).
- Lý thuyết: (1) không xác định được ở cặp (relation, vị trí) chưa thấy; (2) phân rã recall = ρ_bridge × ρ_prim; (3) cận an toàn cho phần in-distribution. Cả ba đều được kiểm chứng trên dữ liệu.

## 2. Claim → bằng chứng

| # | Claim | Bằng chứng | Số liệu | Trạng thái |
|---|---|---|---|---|
| C1 | Claim tổng quát hóa của line graph không đứng vững ở tầng retrieval | EXP-19: transfer WebQSP → CWQ | DDE 70.05 so với REL-RAG P 61.90: +8.14 [+6.22, +10.06] | ✅ |
| C2 | Lỗi tập trung ở composition (cầu nối văn bản) | EXP-19, theo loại câu và số topic | composition: REL-RAG 26.0, DDE 54.6, in-distribution 83.5; câu 1 topic: 17.8 / 52.1 / 88.5 | ✅ |
| C3 | Nguyên nhân là **liên quan bị gắn với vị trí** (nhân quả, không phải độ khó ngôn ngữ) | EXP-21 E1: can thiệp dịch anchor 1 hop trên WebQSP | Tụt: DDE −20.7, QR-DDE −43.3, **REL-RAG P −69.2** [−71.4, −67.1] (EXP-23) | ✅ |
| C4 | Mô hình học càng nhiều cấu trúc thì càng nhạy với vị trí, và độ nhạy dự đoán độ kém khi transfer | E1, EXP-19, EXP-23 | Độ nhạy: REL-RAG −69 > QR-DDE −43 > DDE −21 > AER −19; thứ tự transfer: DDE > RT-DDE > QR-DDE-noq > REL-RAG ≈ QR-DDE | ✅ |
| C5 | Tiền đề lý thuyết: relation chính không bao giờ đứng sau cầu nối văn bản trong train | Chuỗi gold của WebQSP train | 1 hop 56.2%, 2 hop qua CVT 33.1%, **2 hop qua văn bản 0.0%** | ✅ |
| C6 | Khi neo đúng chỗ, hop nguyên thủy học từ WebQSP đủ mạnh | EXP-19 §1.4: cầu nối gold | 53.3 → 86.8, so với 88.5 in-distribution | ✅ |
| C7 | AER (đã train) giữ WebQSP và vượt REL-RAG khi transfer | EXP-20 | WebQSP 89.68 (−0.92 [−2.30, +0.42], không có ý nghĩa); CWQ 77.95 (+16.05 [+14.18, +18.01]) | ✅ |
| C8 | Từng thành phần của AER đều cần thiết | EXP-21 E2 | Trộn max (thay hẳn trôi tới 68.0 ở K=3); cầu nối theo mô hình hơn ngẫu nhiên +2.62 (composition +8.2); train neo lại: 1 lượt +4.8; B=3; K=2 | ✅ |
| C9 | Lý thuyết dự đoán đúng | EXP-21 E3 | ρ_bridge ≈ 90%; trúng 79–82, trượt 43–45; Mệnh đề 3 đúng 100% | ✅ |
| C10 | Nguyên lý dùng được cho nhiều mô hình, kể cả line graph | EXP-22 E4 | REL-RAG + neo lại: CWQ +8.54 [+7.33, +9.80], composition 26.0 → 54.7 | ✅ |
| C11 | Cải thiện retrieval chuyển thành cải thiện câu trả lời | A1: CWQ, 1,000 câu, Llama-3.1-8B fp16 (vLLM), cùng prompt | **Tạm:** DDE − REL-RAG Hit +0.8 [−2.6, +4.2], không có ý nghĩa, dù retrieval +8; composition +7.3 nhưng comparative/superlative −10/−15. Chờ AER | ⚠️ 🔄 |
| C12 | AER trên nền QR-DDE (lan truyền có điều kiện + neo lại) | EXP-23 | CWQ 71.53 < AER-DDE 77.95: lan truyền có điều kiện overfit, DDE là nền tốt hơn | ✅ (kết quả âm) |
| C13 | Câu sâu hơn (≥ 3 hop) và cấu trúc chưa thấy | GTSQA | – | ⏳ cần hạ tầng |
| C14 | Chi phí | Latency | AER 30 ms/query (2 lượt) so với DDE ~11, REL-RAG ~70 | ✅ (cần đo lại cùng phần cứng) |

## 3. Điểm yếu cần chuẩn bị trước khi bị phản biện
- **"Chỉ là retrieval lặp (PullNet, SR)."** Trả lời: chẩn đoán nhân quả (C3, C5); không cần dữ liệu thành phần; trộn đơn điệu có cận (Mệnh đề 3); dùng được cho nhiều mô hình (C10); ablation (C8). Cần thêm baseline retrieval lặp *có train* (kiểu PullNet) trong cùng điều kiện.
- **"Chỉ một cặp dataset."** Cần GTSQA (C13). Nếu được thì thêm hướng ngược CWQ → WebQSP.
- **"Downstream?"** C11: kết quả tạm cho thấy chênh retrieval +8 chỉ thành Hit +0.8 với Llama-8B trên CWQ. Phải phân tích vì sao (đáp án có trong ngữ cảnh nhưng LLM vẫn sai?), và cần một reasoner mạnh hơn. Ngoài Llama-8B, nên có thêm một reasoner thứ hai.
- **"REL-RAG được tái hiện có công bằng không?"** Có: REL-RAG code-faithful (EXP-17) đạt 90.6 in-distribution, cao hơn bản cấu hình paper.
- **"Composition: AER đã train (70.0) thấp hơn probe không train (75.4)?"** Cần giải thích: train neo lại làm lượt 1 mạnh hơn, nhưng lượt 2 đóng góp ít hơn khi trộn max. Thử ablation `reanchor_p` ∈ {0.25, 0.5, 1.0}.

## Cập nhật 10-04 (EXP-24, EXP-25)
- **C11 / downstream:** pipeline đã hiệu chuẩn với file chính thức của SubgraphRAG (lệch ≤ 1 điểm).
  - WebQSP: REL-RAG P @50 = 84.95 / 70.15 (Hit / F1), **không đạt 88.8 / 77.3 như paper** và thấp hơn SubgraphRAG @100 (86.61 / 70.57). AER ngang REL-RAG.
  - Downstream transfer CWQ (dạng thô) chưa chạy.
- **C13 / GTSQA (zero-shot):** neo lại không giúp. QR-DDE tốt nhất (ansR@50 44.0 so với DDE 35.4). Câu nhiều seed (giao) sụp với mọi mô hình.
  - Phạm vi claim của AER thu hẹp lại: **chuỗi qua cầu nối**.
  - Manh mối mới: lan truyền có điều kiện theo câu hỏi tổng quát hóa sang KG khác tốt hơn.

## Cập nhật 10-05 (EXP-26)
- **C11 (downstream transfer, đúng mô hình của paper, bf16):**
  - AER − REL-RAG P = **+4.50 Hit [+1.00, +8.00]**; composition **+10.26 [+4.11, +15.84]**. ✅
  - AER − DDE = +1.10, không có ý nghĩa. ⚠️ Lợi thế downstream so với baseline entity graph cần nhiều câu hơn (chạy đủ 3,531 câu) hoặc một reasoner mạnh hơn mới phân định được.
- Hướng tiếp theo cho downstream:
  1. Chạy đủ 3,531 câu CWQ (≈ $4 trên RTX 4090, nhớ lọc phí băng thông).
  2. GPT-4o-mini @500 cho Bảng 2 khi được duyệt ngân sách.

## Cập nhật 10-05 (EXP-27, cổng retrieval)
- Theo quy tắc `03_EVALUATION_PROTOCOL.md` §0 (bản đã nới), **AER qua cổng**: WebQSP hòa với cả hai baseline; CWQ transfer thắng cả hai ở @50/100/200.
- **C7, C11 cần viết lại cho chính xác:**
  - Lợi thế chủ yếu nằm ở answer recall khi transfer. Lợi thế về bằng chứng so với SubgraphRAG chỉ khoảng +2 đến +4 SP/GPT-R.
  - Downstream so với SubgraphRAG không được kỳ vọng (AER@k < SubgraphRAG@2k), khớp với EXP-26.
- Từ nay mọi claim retrieval phải kèm bảng đủ ans, SP, GPT, GPTp ở nhiều budget.
