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
| [`13_RBF_LAP_LUAN_VA_HUONG_TIEP.md`](13_RBF_LAP_LUAN_VA_HUONG_TIEP.md) | RBF-Ret: lập luận, lợi / hại, phân rã lỗi QA, mức trần nhãn |
| [`14_TONG_KET_GIOI_HAN.md`](14_TONG_KET_GIOI_HAN.md) | **Tổng kết giai đoạn RBF (10-23): khó khăn, giới hạn, bài học cho hướng mới** |
| [`REPORT.md`](REPORT.md) | Benchmark retrieval đầy đủ (ans, SP, GPT, GPTp; nhiều budget); §3 là kết quả cổng của AER |
| [`papers/`](papers/) | Paper REL-RAG |

Báo cáo tổng hợp toàn bộ quá trình (có sơ đồ, công thức, code): [Claude Docs](https://claude.ai/code/artifact/51134e4b-fc18-41fa-a1be-36da46ce925c).

## Quy trình code retrieval (từ 10-20)

Chỉ giữ RT, WL, ASE, AER / ADR (lúc inference) và RBF (nb + mm). Bốn file, mỗi file một việc:

```bash
cd retrieve/exp_qrdde
python labels.py --dataset webqsp --split train --version v2          # nhãn gold-query v1 / v2 / v3 (v3 cần v2 trước)
python train_retriever.py --dataset webqsp --feat dde+rt --pos_weight effective_num --labels v2 --out runs/<run>   # train
python infer.py --run <run> --dataset cwq --modes base,aer,adr,adr_aer                                             # retrieve toàn bộ tập test
cd .. && python eval.py -d cwq -p exp_qrdde/runs/<run>/pred_cwq_base.pth -p ... --subset true                      # bảng metric
```

- `eval.py` in **mẫu số của từng nhóm metric** ở đầu bảng:
  - `--subset true` (mặc định): chỉ các câu có đáp án trong subgraph, WebQSP 1,559 / CWQ 2,848. Đây là quy ước của mọi bảng retrieval trong BENCHMARK / report.
  - `--subset false`: toàn bộ tập test, WebQSP 1,639 / CWQ 3,531. Câu không có đáp án trong subgraph tính ansR = 0.
  - Câu bị thiếu trong file luôn tính 0 và có cảnh báo, không bị bỏ âm thầm.
- **Train trên server:** `vast/remote_env.sh` (cài env) rồi `vast/remote_train.sh "<tên> | <tham số train_retriever.py>" ...`. Tune BGE: `bge_sparql_data.py` để tạo dữ liệu, `vast/remote_bge.sh` để train và chấm.
- **Lưu ý về độ lặp lại:**
  - WL + AER trên GPU không lặp lại từng bit giữa các lần chạy, vì DDE dùng phép cộng atomic. Trên CPU thì kết quả y hệt.
  - RBF chạy GPU và CPU có thể lệch ±0.2 ansR do thứ tự các triple hoà điểm.
- **File export cũ trong `runs/`:**
  - `adr_<ds>_<mode>.pth` chỉ có các câu có đáp án (subset).
  - `adr_<ds>_base_mm1_full.pth` có đủ tập test.
  - File mới do `infer.py` tạo có tên `pred_<ds>_<mode>.pth`, luôn đủ tập test.
- **Tên script cũ trong nhật ký và các tài liệu trước 10-20** (mọi số đã được kiểm lại là trùng khớp với code mới):

  | Script cũ | Thay bằng |
  |---|---|
  | `qrdde.py` | `train_retriever.py` |
  | `adr_probe.py`, `score_full.py`, `bge_bonus_export.py` | `infer.py` |
  | `sparql_labels.py`, `sparql_coverage.py`, `label_compare.py` (v1), `make_sparql_v2.py`, `make_sparql_v3.py` | `labels.py` |
  | `relrag/benchmark_all.py`, `sparql_phase_table.py`, `query_recall.py`, `gate_report.py`, `wl_adr_report.py` | `retrieve/eval.py` |
  | `vast/*.sh` | `vast/remote_{env,train,bge}.sh` |

  Đã xoá (các ablation âm hoặc hướng đã bỏ): QR-DDE, RT-DDE, các biến thể RBF (ψ, gate, poe, hybrid MLP, cmm / soft-AND, train kèm mm, không nb), nhãn rational / union, prior memory / Path+Memory, cắt theo τ / threshold, dataset ngoài (GTSQA, MetaQA, GrailQA), two-stage / listwise, HiRAG / LangGraph. Code cũ nằm ở commit `2d555d0`. Các script `.sh` cũ (bị git-ignore) được nén ở `.cleanup_backup_1020_sh.tar.gz`.

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
