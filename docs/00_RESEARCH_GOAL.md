# 00. Mục tiêu nghiên cứu

*Cập nhật: 2026-09-30 (bổ sung H4 và yêu cầu về đóng góp)*

## 1. Bối cảnh

Bài toán: **Knowledge Graph Question Answering (KGQA)** theo mô hình *retrieve-then-reason*.
Với câu hỏi *q* và subgraph *G_q* (Freebase), một **graph retriever** chọn một tập nhỏ triple làm bằng chứng. Sau đó một **LLM reasoner** sinh câu trả lời từ tập triple đó.

**Baseline chính:** REL-RAG (*Relation-Aware Retrieval-Augmented Generation for Generalizable KGQA*, ICML 2026; phiên bản đầu là RAPL, code ở `RAPL/`). Paper: [`papers/REL_RAG_Relation_Aware_Retrieval.pdf`](papers/REL_RAG_Relation_Aware_Retrieval.pdf). Chi tiết phương pháp và bản tái hiện: [`02_BASELINE_REL_RAG.md`](02_BASELINE_REL_RAG.md).

REL-RAG đưa ra ba claim:
1. **Relation mixing** là lý do GNN trên entity graph khó tổng quát hóa. Mỗi entity gộp message của mọi relation kề nó, nên tín hiệu của các relation bị trộn lẫn.
2. Biến đổi sang **line graph** (mỗi triple thành một node) giúp mô hình hóa chuyển tiếp giữa các relation một cách tường minh. Theo Theorem 3.1, cận tổng quát hóa tốt hơn √R lần trong cả ba kịch bản ID, compositional và OOD.
3. Kết quả thực nghiệm: WebQSP F1 77.3 / Hit 88.8 với Llama3.1-8B ở 50 triple (+6.8 F1 so với SubgraphRAG). **Train WebQSP → eval CWQ / GrailQA** tốt hơn biến thể raw graph (+5–10%).

Giai đoạn trước dùng SubgraphRAG làm baseline. Kết quả giai đoạn đó được lưu trong [`archive/subgraphrag/`](archive/subgraphrag/README.md), chỉ dùng để tham khảo.

## 2. Những gì bản tái hiện đã cho thấy (tính đến 09/2026)

Chi tiết trong [`02_BASELINE_REL_RAG.md`](02_BASELINE_REL_RAG.md) và [`04_EXPERIMENT_LOG.md`](04_EXPERIMENT_LOG.md).

- **Chi phí:** line graph có số cạnh gấp **83× (WebQSP) và 107× (CWQ)** graph gốc; 77% số cạnh đi qua topic entity. Retriever chậm hơn MLP của SubgraphRAG khoảng **7×** (69.5 so với 10.1 ms/query).
- **Retrieval in-distribution:** ở cùng budget, REL-RAG path-based **gần bằng nhưng không vượt** SubgraphRAG (R@50: 88.8 so với 90.4).
- **Downstream:** theo đúng protocol eval của paper, REL-RAG(P)@50 + llama3.1 đạt 83.8 Hit / 72.3 F1, so với 88.8 / 77.3 trong paper. Trong cùng pipeline, mức hơn SubgraphRAG@50 chỉ **+1.5 F1** và chưa có ý nghĩa thống kê.
- **Tín hiệu cho hướng mới:** một cơ chế lan truyền có điều kiện theo câu hỏi trên *entity graph* (**QR-DDE**) tăng recall +6–8 điểm so với DDE khi train trên subset, và chỉ tốn thêm khoảng 1 ms.

## 3. Câu hỏi nghiên cứu

> **RQ chính.** Có thể đạt được (hoặc vượt) khả năng tổng quát hóa mà REL-RAG gán cho line graph, **mà không cần line graph**, bằng message passing có điều kiện theo câu hỏi trên entity graph, với chi phí tuyến tính theo số cạnh không?

- **RQ1 (tổng quát hóa):** khi train trên WebQSP và eval trên CWQ (compositional shift) cùng các tập OOD, retriever đề xuất có vượt REL-RAG không?
- **RQ2 (cơ chế):** relation mixing có thực sự là nguyên nhân chính không? Phần lợi thế đến từ việc điều kiện hóa theo câu hỏi, từ việc lan truyền có tham số, hay chỉ từ inductive bias trong chế độ ít dữ liệu?
- **RQ3 (hiệu quả):** đánh đổi giữa recall, F1 downstream và latency (đường Pareto) so với REL-RAG và SubgraphRAG như thế nào?
- **RQ4 (downstream):** ở cùng budget và cùng reasoner, retriever mới có cải thiện Hit/F1 một cách có ý nghĩa thống kê không?
- **RQ5 (multi-relation):** mô hình hóa cặp entity như tập relation có điều kiện theo câu hỏi có cải thiện nhóm câu hỏi dính tới multi-relation (22–29% câu) và cải thiện transfer không?

## 4. Giả thuyết

| # | Giả thuyết | Cách kiểm chứng |
|---|---|---|
| H1 | Relation mixing được giải quyết bằng **conditional relational message passing** (họ NBFNet/ULTRA) trên entity graph: message nhân với trọng số w_q(r) phụ thuộc câu hỏi, nên tổ hợp r₁→r₂ được mã hóa dọc theo path mà không cần line graph | QR-DDE ≥ REL-RAG về recall in-distribution và transfer |
| H2 | REL-RAG chỉ điều kiện hóa theo câu hỏi ở phép dot product cuối cùng; retriever có điều kiện hóa ngay trong lan truyền sẽ tổng quát hóa tốt hơn | Ablation: QR-DDE có và không có query-conditioning |
| H3 | Chi phí bậc hai của line graph (tập trung ở topic entity) không cần thiết | Pareto latency/recall; phân tích theo nhóm subgraph có hub |
| H4 | Line graph (và GCN đồng nhất) không biểu diễn được **cặp entity nối bằng nhiều relation**: các triple song song (h,r₁,t), (h,r₂,t) không có cạnh nối với nhau; code gốc còn loại bỏ chúng. Một mô hình coi mỗi cạnh là **tập relation có điều kiện theo câu hỏi** (heterogeneous / multi-relational graph transformer) sẽ xử lý tốt hơn | EXP-13 (đã có: 13–14% triple, 22–29% câu bị ảnh hưởng); so sánh trên nhóm câu có multi-relation |

## 5. Yêu cầu về đóng góp (để nhắm tới hội nghị A\*)

Các tinh chỉnh dạng module (đổi feature DDE, thêm residual, relation-topic có trọng số, loss) **không đủ làm đóng góp chính**. Nhiều tinh chỉnh như vậy đã được REL-RAG bao phủ, hoặc chỉ là kỹ thuật. Đóng góp chính cần là **một cách mô hình hóa mới, có lập luận và chứng minh rõ vì sao nó giải quyết được điểm yếu của line graph**. Hai hướng ứng viên:

1. **Conditional relational message passing trên entity graph** (QR-DDE, họ NBFNet/ULTRA), mở rộng thành mô hình đầy đủ thay vì chỉ là feature cho MLP.
2. **Heterogeneous / multi-relational graph transformer có điều kiện theo câu hỏi:** mỗi cạnh là một *tập relation* giữa hai entity. Attention được điều kiện hóa theo embedding văn bản của relation và của câu hỏi, **không** dùng tham số riêng cho từng loại relation, vì như vậy sẽ mâu thuẫn với mục tiêu tổng quát hóa; xem cảnh báo ở EXP-13.

Các tinh chỉnh như residual hay self-loop chỉ đóng vai trò ablation hoặc phân tích (ví dụ: vì sao line-graph GCN transfer kém, EXP-12).

## 6. Đóng góp dự kiến (điều kiện: các milestone go/no-go trong [`archive/relrag_phase/01_RESEARCH_PLAN.md`](archive/relrag_phase/01_RESEARCH_PLAN.md) đạt)

1. **Phân tích thực nghiệm REL-RAG** trong setting open-weight có thể tái hiện: chi phí line graph, độ nhạy với protocol eval, kết quả đo lại khi cùng budget.
2. **Phương pháp:** QR-DDE, một retriever nhẹ (MLP scorer) cùng mã hóa cấu trúc có điều kiện theo câu hỏi trên entity graph.
3. **Bằng chứng:** QR-DDE đạt hoặc vượt REL-RAG về tổng quát hóa (WebQSP→CWQ) với latency gần như SubgraphRAG, kèm ablation giải thích cơ chế.

## 7. Tiêu chí thành công (để viết paper)

> **Cập nhật 2026-10-05: cổng retrieval có hiệu lực, thay cho các tiêu chí bên dưới khi có mâu thuẫn** ([`03_EVALUATION_PROTOCOL.md`](03_EVALUATION_PROTOCOL.md) §0).
> - Đóng góp được chấp nhận khi in-distribution xấp xỉ bằng baseline, **và** ở ít nhất một setting tổng quát hóa, answer recall vượt cả SubgraphRAG lẫn REL-RAG ở @50 / @100 / @200.
> - SP / GPT là metric hỗ trợ, không bắt buộc.
> - Downstream chỉ chạy khi lập luận được rằng Hit/F1 có thể tăng.
> - Kết quả với AER (`docs/REPORT.md` §3): **qua cổng**. Tuy vậy, lợi thế về bằng chứng nhỏ, và hiệu quả budget so với SubgraphRAG không đạt.

*Tiêu chí cũ (09-30), giữ lại để tham khảo:*

- In-distribution (WebQSP, full data, seed 42 như SubgraphRAG): QR-DDE R@50 ≥ REL-RAG path best-first (88.8) **và** ≥ SubgraphRAG (90.4).
- Transfer WebQSP→CWQ: QR-DDE vượt REL-RAG ở R@50/R@100, có ý nghĩa theo paired bootstrap trên câu hỏi test (CI 95%).
- Downstream (budget 50, cùng reasoner, eval gốc, ~1,600 câu): ΔF1 so với REL-RAG vượt ngưỡng phát hiện được (~1.7 F1, paired bootstrap 95%).
- Latency retriever ≤ 2× SubgraphRAG.

## 8. Ngoài phạm vi (hiện tại)
- Fine-tune LLM reasoner; retriever dạng agent/LLM (ToG, RoG…).
- GrailQA: chưa có dữ liệu subgraph trong repo, sẽ cân nhắc ở milestone M3.
