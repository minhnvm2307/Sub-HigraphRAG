# Lưu trữ: giai đoạn cải tiến SubgraphRAG (Phase 1–16)

Giai đoạn này lấy **SubgraphRAG** (retriever MLP + DDE) làm baseline. Từ 09/2026 baseline chính của dự án là **REL-RAG** (xem [`../../00_RESEARCH_GOAL.md`](../../00_RESEARCH_GOAL.md)).
Các tài liệu ở đây giữ nguyên để tra cứu: những gì đã thử, kết quả, và lý do dừng. Số liệu bên dưới lấy từ [`DEVELOP_PHASE_legacy.md`](DEVELOP_PHASE_legacy.md).

## Tóm tắt theo phase

| Phase | Nội dung | Kết quả chính (WebQSP test trừ khi ghi khác) | Kết luận |
|---|---|---|---|
| 1 | Baseline SubgraphRAG (MLP + DDE) | Ans R@50 90.1, R@100 94.2. **Train WebQSP → eval CWQ**: R@50 68.5, R@100 78.7, R@400 92.9 | Mốc tham chiếu; có hiện tượng "score flattening" |
| 2 | MLP so với chỉ dùng embedding | Embedding-only R@50 64.8 (WebQSP), 57.8 (CWQ) | MLP + DDE tốt hơn nhiều |
| 3 | Phân tích score flattening, ghép HiRAG | LLM chỉ dùng ~23 triple/câu nhưng cần K ~400 để phủ; HiRAG làm giảm chất lượng | Dừng hướng HiRAG |
| 4 | Relation-topic: tiêm vào MLP + DDE có trọng số | Kết hợp cả hai: R@50 91.9 (+1.8), R@100 95.6 (+1.4); từng thành phần riêng thì không giúp | Có tăng nhỏ |
| 5 | Chuyển reasoning sang LangGraph | Chỉ refactor + monitoring | – |
| 6 | Chẩn đoán: rank đáp án, mất cân bằng nhãn, công thức DDE | Tỷ lệ nhãn dương < 1%; CWQ có ~20% câu không có path | Tài liệu: `DDE_Mathematical_Formulation.md` |
| 7 | Phân tích relation-topic | Topic-relation top-5 chỉ recall 43% relation trên shortest path | Relation retrieval là điểm nghẽn |
| 8 | Thử BGE-M3 (dense/sparse, nhiều format) | Relation Hit@5 0.68 → 0.76; ans R@50 +0.4–1.1 | Tăng nhỏ |
| 9 | Sửa công thức weighted DDE; so sánh loss | Loss tốt nhất (effective-number, BCE + margin) R@50 ~92.6–92.8 | Sửa lỗi chuẩn hóa PEConv |
| 10 | Listwise reranking | Kém | Bỏ |
| 11 | Fine-tune BGE-M3 LoRA cho relation retrieval | Relation Recall@10 dev 0.653 → 0.770 | Chưa đưa vào retriever cuối |
| 12 | Phân tích scale của feature, MLP phân cấp | Đề xuất chuẩn hóa từng block / late fusion | Chỉ dừng ở phân tích |
| 13 | Path + Memory hybrid, Inversed loss | WebQSP R@50 92.8, R@100 96.0; CWQ R@50 88.0. Reasoning (Llama3.1-8B): WebQSP Hit@1 90.1 / F1 75.8 | Cấu hình tốt nhất của giai đoạn |
| 15 | Khử nhiễu nhãn SP bằng LLM | Test R@50 +1.1, R@100 +0.4 | Tăng nhỏ |
| 16 | Thử RAPL (REL-RAG v1) | **Đã hủy**: kết quả cũ (R@100 36.9%) sai do lỗi cài đặt | Làm lại ở `04_EXPERIMENT_LOG.md` |

## Bài học mang sang giai đoạn mới
1. Các cải tiến quanh relation-topic, loss và embedding chỉ tăng recall 1–2 điểm trên WebQSP: trần của MLP + DDE ở in-distribution đã khá cao.
2. **Generalization là điểm yếu thật**: train WebQSP → eval CWQ chỉ đạt R@50 68.5 (so với ~86 khi train trên CWQ). Đây cũng chính là claim chính của REL-RAG.
3. Nhãn shortest-path nhiễu, nhưng khử nhiễu (LLM, GPT label) chỉ giúp ít.
4. Protocol eval phải cố định trước khi so sánh (id-match vs text-match, eval gốc vs corrected, cùng tập câu hỏi).

## Danh sách tài liệu lưu trữ

| File | Nội dung |
|---|---|
| `DEVELOP_PHASE_legacy.md` | Nhật ký đầy đủ Phase 1–16 |
| `01_Architecture_Baseline.md` | Kiến trúc SubgraphRAG |
| `02_Evaluation_Metrics.md` | Kết quả baseline SubgraphRAG |
| `DDE_Mathematical_Formulation.md` | Công thức DDE và relation-topic DDE |
| `METHOD_TOPIC_RELATION.md`, `TOPIC_RELATION_SYNTHESIS.md` | Các phương pháp chọn topic-relation, Path + Memory |
| `EXPERIMENT_SEMANTIC_PATH_FILTERING.md` | Kiểm toán nhãn SP và lọc path theo ngữ nghĩa |
| `PIPELINE_PRECOMPUTE_AND_EXPORT.md` | Pipeline precompute/export |
| `ABOUT_CODE.md` | Danh mục script và môi trường conda của giai đoạn này |
