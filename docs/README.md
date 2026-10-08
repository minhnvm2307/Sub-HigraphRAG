# Tài liệu nghiên cứu

**Chủ đề:** retriever cho KGQA (retrieve-then-reason) có khả năng tổng quát hóa.
**Baseline:** SubgraphRAG (ICLR'25) và REL-RAG (ICML'26, line graph + bi-GCN).
**Hướng chính:** AER, Anchor-Equivariant Retrieval ([`09_…`](09_NEW_DIRECTION_ANCHOR_EQUIVARIANCE.md)).
*Cập nhật: 2026-10-05.*

## Các file chính

| File | Nội dung |
|---|---|
| [`00_RESEARCH_GOAL.md`](00_RESEARCH_GOAL.md) | Bối cảnh, câu hỏi nghiên cứu, tiêu chí thành công (§7: cổng retrieval) |
| [`02_BASELINE_REL_RAG.md`](02_BASELINE_REL_RAG.md) | Phương pháp REL-RAG và bản tái hiện |
| [`03_EVALUATION_PROTOCOL.md`](03_EVALUATION_PROTOCOL.md) | **§0: cổng retrieval trước downstream**; dataset, metric, budget, transfer |
| [`04_EXPERIMENT_LOG.md`](04_EXPERIMENT_LOG.md) | Nhật ký thí nghiệm EXP-00 … EXP-27 |
| [`05_INFRASTRUCTURE.md`](05_INFRASTRUCTURE.md) | Máy, quy tắc bộ nhớ, server (Kaggle, vast), lệnh chạy |
| [`09_NEW_DIRECTION_ANCHOR_EQUIVARIANCE.md`](09_NEW_DIRECTION_ANCHOR_EQUIVARIANCE.md) | Hướng chính AER: chẩn đoán, lý thuyết, phương pháp |
| [`10_PAPER_CLAIMS_EVIDENCE.md`](10_PAPER_CLAIMS_EVIDENCE.md) | Bản đồ claim → bằng chứng (C1–C14) |
| [`REPORT.md`](REPORT.md) | Benchmark retrieval đầy đủ (ans, SP, GPT, GPTp; nhiều budget); §3 là kết quả cổng của AER |
| [`papers/`](papers/) | Paper REL-RAG |

Báo cáo tổng hợp toàn bộ quá trình (có sơ đồ, công thức, code): [Claude Docs](https://claude.ai/code/artifact/51134e4b-fc18-41fa-a1be-36da46ce925c).

## Trạng thái (10-05)

- **AER qua cổng retrieval** (`REPORT.md` §3):
  - WebQSP: answer recall hòa với cả hai baseline.
  - CWQ transfer: answer recall thắng cả SubgraphRAG lẫn REL-RAG ở @50 / @100 / @200.
  - Lợi thế về bằng chứng (SP-R, GPT-R +2 đến +4 so với SubgraphRAG) nhỏ hơn nhiều lợi thế answer recall (+4 đến +8).
  - Hiệu quả budget so với SubgraphRAG không đạt.
- **Downstream:** AER hơn REL-RAG +4.50 Hit trên CWQ transfer (EXP-26). So với SubgraphRAG chưa có cơ sở kỳ vọng tăng, nên chưa chạy thêm.
- **Việc tiếp theo:** tăng recall bằng chứng khi transfer; mở rộng sang truy vấn dạng giao (GTSQA).
- **Quy tắc làm việc:**
  - Claude viết code, người dùng chạy job train và infer.
  - Phải lập luận được khả năng Hit/F1 tăng trước khi chạy downstream (`03` §0).
  - API trả phí phải ước tính và chạy thử trước.

## Lưu trữ

- [`archive/relrag_phase/`](archive/relrag_phase/): roadmap M0–M6 cũ (01), tổng quan tài liệu và track R (06), flow code REL-RAG (07), hướng witness đã bị loại (08), prompt session cũ.
- [`archive/subgraphrag/`](archive/subgraphrag/README.md): giai đoạn cải tiến SubgraphRAG (Phase 1–16).
