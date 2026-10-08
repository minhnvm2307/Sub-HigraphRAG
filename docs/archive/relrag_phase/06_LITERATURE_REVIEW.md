# 06. Tổng quan tài liệu: đối thủ và hướng mô hình mới

*Tạo: 2026-09-30. Cập nhật: 2026-09-30 (track R hoàn thành khảo sát tài liệu + đề xuất thiết kế, chờ duyệt trước khi prototype).*

## 1. Mục đích
Tìm một **đóng góp mô hình hóa mới** vượt qua điểm yếu của REL-RAG, thay vì tinh chỉnh module. Trọng tâm là hướng *heterogeneous / multi-relational graph transformer* có điều kiện theo câu hỏi, đối chiếu với các họ phương pháp liên quan.

## 2. Điểm yếu của REL-RAG đã có bằng chứng (đầu vào cho thiết kế)

| Điểm yếu | Bằng chứng |
|---|---|
| Chi phí bậc hai: line graph lớn gấp 83–107× số cạnh, 77% qua topic entity | EXP-08 |
| Transfer WebQSP→CWQ kém ở tầng retrieval (R@50 ~53 so với 86 in-distribution) | EXP-12 |
| Không mô hình hóa cặp entity nối bằng nhiều relation: triple song song không kề nhau trong line graph; code gốc còn loại bỏ chúng (mất ~6% bằng chứng GPT, 22–29% câu có dính multi-relation) | EXP-13 |
| Chỉ điều kiện hóa theo câu hỏi ở phép dot product cuối, không có trong lúc lan truyền | Kiến trúc (Eq. 86); EXP-11 gợi ý conditioning trong lan truyền có ích |
| Line graph có hướng chỉ có chuyển tiếp tail→head; không có tương tác head–head / tail–tail (cấu trúc của câu hỏi giao/ràng buộc) | Phân tích trong `WEB_RESEARCH.md`; **chưa được dữ liệu ủng hộ** (EXP-11: nhóm câu nhiều topic không yếu riêng) |

### 2.1. Xác minh lại Theorem 3.1 từ bản PDF gốc (mới, 2026-09-30)

Đã đọc trực tiếp `docs/papers/REL_RAG_Relation_Aware_Retrieval.pdf` (không dựa vào web search) để xác minh venue và nội dung chính xác của Theorem 3.1:

- **Venue xác nhận đúng như đã ghi:** "Proceedings of the 43rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026" — đây là main conference (một agent tìm kiếm web báo nhầm là ICML 2026 Workshop dựa trên một OpenReview listing khác; PDF gốc bác bỏ điều này).
- **Cơ chế chính xác đằng sau hệ số √R** (Appendix A.4.1, "Functional Forms of GCN Models"): hai lớp giả thuyết được định nghĩa **chính xác** như sau —
  - Entity graph: `H_G = {f_W : ||W_r||_F ≤ B với mọi r ∈ R}` — mỗi relation có **ma trận trọng số riêng** W_r ∈ R^(d×d), không gian tham số **O(R·d²)**.
  - Line graph: `H_G' = {⟨φ_r1, M·φ_r2⟩ : ||M||_F ≤ B, ||φ_r||_2 ≤ 1}` — một ma trận **dùng chung** M cộng R embedding quan hệ chiều thấp φ_r, không gian tham số O(d² + R·d) = O(d²) khi R ≤ d.
  - Cận tổng quát hóa (Theorem A.1, dùng covering number + Dudley's entropy integral) tỉ lệ với **căn bậc hai của số tham số** của mỗi lớp giả thuyết, nên ra đúng hệ số √R giữa hai lớp.

**Hệ quả quan trọng cho thiết kế (khác với cách hiểu "line graph nói chung tốt hơn entity graph nói chung"):** hệ số √R xuất phát **cụ thể** từ việc entity-graph model có **ma trận học riêng theo từng loại relation**. Một mô hình trên entity graph gốc mà **không có bất kỳ tham số học nào được đánh chỉ số theo relation** (relation chỉ vào mô hình qua một embedding văn bản cố định/tinh chỉnh nhẹ z_r, đi qua một hàm **dùng chung cho mọi relation**) sẽ có không gian tham số không phụ thuộc R — về nguyên tắc đạt cùng lớp cận tổng quát hóa O(√(d/m)) như line graph, bằng đúng kỹ thuật chứng minh mà REL-RAG dùng cho chính nó, mà **không cần** chuyển sang line graph. Đây là lập luận lý thuyết cho cả hai thiết kế ở mục 7.

## 3. Các câu hỏi cần trả lời

1. Mô hình nào xử lý được **đồng thời**: (a) relation mixing, (b) cặp entity nối bằng nhiều relation, (c) tổ hợp r₁→r₂, (d) relation chưa thấy khi transfer (inductive), (e) chi phí tuyến tính theo số cạnh?
2. HGT và các graph transformer dị thể có dùng **tham số riêng cho từng loại relation** không? Nếu có thì vướng chính lập luận √R trong Theorem 3.1 của REL-RAG. Biến thể nào thay tham số theo loại bằng điều kiện hóa theo embedding văn bản?
3. Có công trình nào coi mỗi cạnh là **tập relation** (multiplex edge, hyperedge, edge-set attention) giữa hai entity không?
4. Có công trình nào dùng graph transformer (attention toàn cục hoặc cục bộ) làm **retriever cho KGQA + LLM** không? Nó được so với REL-RAG / SubgraphRAG / GNN-RAG như thế nào?
5. Có lập luận lý thuyết nào (khả năng biểu diễn, cận tổng quát hóa) để đặt cạnh Theorem 3.1?

## 4. Danh sách ứng viên (đã khảo sát, venue/năm xác minh qua web search + đối chiếu nhiều nguồn)

| Họ | Paper | Venue/năm (đã xác minh) | Điểm cần kiểm tra |
|---|---|---|---|
| Heterogeneous graph transformer | HGT (Hu et al.) | **WWW 2020**, xác nhận | Tham số theo meta-relation (node-type, edge-type, node-type); không inductive với relation chưa thấy |
| Multi-relational GNN | R-GCN (Schlichtkrull et al.) | **ESWC 2018**, xác nhận (LNCS, pp. 593–607) | W_r riêng từng relation (basis decomposition chỉ giảm tham số, không loại bỏ phụ thuộc R) |
| | CompGCN (Vashishth et al.) | **ICLR 2020**, xác nhận | Vẫn cần embedding z_r riêng theo relation ID dù dùng chung ma trận W |
| | RGAT (Busbridge et al.) | **Không xác minh được venue** — chỉ là arXiv preprint (1904.05811), không tìm thấy OpenReview/DBLP xác nhận accept. Một số nguồn web ghi nhầm "ICLR 2019" | WIRGAT dùng attention riêng theo relation; cả hai biến thể vẫn cần tham số theo relation ID |
| Conditional message passing | NBFNet (Zhu et al.) | **NeurIPS 2021**, xác nhận | Inductive với **entity** mới nhưng KHÔNG inductive với **relation** mới (vẫn cần embedding w_q(r) học theo relation ID) — hay bị nhầm là "inductive" nói chung |
| | A*Net (Zhu et al.) | **NeurIPS 2023**, xác nhận | Kế thừa NBFNet; thêm priority function chọn ~10% cạnh/node mỗi bước → chi phí thực tế tốt hơn tuyến tính |
| | ULTRA (Galkin et al.) | **ICLR 2024**, xác nhận | Inductive thật với relation chưa thấy, nhưng theo cấu trúc đồ thị quan hệ (4 loại tương tác h2h/h2t/t2h/t2t), **không dùng text** |
| | InGram (Lee et al.) | **ICML 2023**, xác nhận (PMLR v202, pp. 18796–18809) | Inductive thật với relation chưa thấy qua affinity đồng xuất hiện, cũng **không dùng text** |
| Graph transformer | Graphormer (Ying et al.) | **NeurIPS 2021**, xác nhận | Chỉ 1 shortest-path giữa 2 node → **collapse khi có nhiều relation song song** (cùng lỗi như line graph, chỉ khác cơ chế) |
| | GraphGPS (Rampášek et al.) | **NeurIPS 2022**, xác nhận | MPNN cục bộ xử lý cạnh song song độc lập (cộng dồn), nhưng attention toàn cục không có token theo cạnh |
| | Relational Attention (Diao & Loynd) | **ICLR 2023 (spotlight)**, xác nhận | Cạnh là "công dân hạng nhất" được cập nhật mỗi layer → 2 relation trên cùng cặp entity có thể **attend lẫn nhau trực tiếp**; chưa test ở quy mô KG/Freebase |
| Retriever cho KGQA | SubgraphRAG (Li, Miao, Li) | **ICLR 2025**, xác nhận | MLP + DDE, dùng text embedding (không có W_r riêng theo relation) nhưng chấm từng triple độc lập, không có tương tác r1↔r2 |
| | GNN-RAG (Mavromatis & Karypis) | **Findings of ACL 2025**, xác nhận (preprint arXiv 2405.20139, 2024) | GNN kiểu ReaRev/NSM, vẫn thuộc nhóm có tham số theo relation |
| | KGFR (Cui, Sun, Hu, Fu) | **arXiv preprint 11/2025, chưa xác nhận venue chính thức** | Relation encode qua mô tả văn bản do LLM sinh → inductive thật với relation chưa thấy; KHÔNG xử lý multi-relation entity pair; không so sánh với REL-RAG |
| | GSR "Less is More" (Huang et al.) | **Findings of EMNLP 2024**, xác nhận | Sinh chuỗi relation dưới dạng token bởi LM nhỏ; không giải quyết multi-relation entity pair |
| | REL-RAG (Yao et al.) | **ICML 2026 (main, PMLR 306)**, xác nhận qua PDF gốc; tiền thân arXiv:2506.09645 "RAPL" (06/2025, cùng nhóm tác giả) | Baseline của dự án này |
| Tìm thêm | Không tìm được paper 2025–2026 nào phê bình trực tiếp điểm yếu multi-relation-entity-pair của line graph/REL-RAG | — | Phát hiện của EXP-13 hiện có vẻ **mới** so với tài liệu có thể tìm kiếm được (REL-RAG bản thân cũng mới, 2026) |

**Ghi chú quan trọng đã xác minh (không có trong 8 paper multi-relational/conditional-MP khảo sát ban đầu):** không paper nào trong nhóm R-GCN/CompGCN/RGAT/NBFNet/A*Net/HGT/ULTRA/InGram **kết hợp** điều kiện hóa theo text quan hệ **và** không dùng tham số riêng theo loại relation. ULTRA/InGram không dùng tham số riêng theo relation nhưng dựa vào **cấu trúc đồ thị** (không phải text) để suy ra biểu diễn quan hệ mới — nên vẫn cần relation mới xuất hiện đủ nhiều trong đồ thị mới để tính affinity/tương tác, không tổng quát hóa được chỉ từ tên/mô tả relation như QR-DDE hiện tại đang làm.

## 5. Bảng so sánh 5 tiêu chí

| Paper | (a) mixing | (b) multi-relation | (c) tổ hợp r₁→r₂ | (d) inductive với relation chưa thấy | (e) chi phí tuyến tính | Code |
|---|---|---|---|---|---|---|
| REL-RAG (line graph) | ✓ (tường minh qua cấu trúc line graph) | ✗ (EXP-13: triple song song không kề nhau) | ✓ (t→h) | ✗ (Theorem A.1 vẫn cần R embedding φ_r riêng, dù nhẹ hơn W_r) | ✗ bậc hai (83–107× cạnh) | ✓ (RAPL) |
| R-GCN | ~ (cộng dồn, không tương tác chéo) | ✓ (multigraph gốc) | ✗ | ✗ (W_r riêng từng relation — đúng nhóm bị Theorem 3.1 phê bình) | ✓ | ✓ |
| CompGCN | ~ (cộng dồn qua W dùng chung) | ✓ (multigraph gốc) | ✗ (composition là entity⊗relation, không phải path) | ✗ (vẫn cần z_r riêng theo relation ID) | ✓ | ✓ |
| RGAT | ~/✓ tùy biến thể (ARGAT mixing qua softmax chung) | ✓ (multigraph gốc) | ✗ | ✗ (attention riêng theo relation) | ✓ | ✓ (venue chưa xác minh) |
| NBFNet | ✓ (nhân tích lũy dọc path) | ✓ (multigraph gốc) | ✓ (cơ chế chính, mạnh nhất trong nhóm) | ✗ (chỉ inductive với **entity**, không với **relation**) | ✓ | ✓ |
| A*Net | ✓ (kế thừa NBFNet) | ✓ | ✓ (kế thừa NBFNet) | ✗ (kế thừa NBFNet) | ✓✓ (chỉ duyệt ~10% cạnh/bước) | ✓ |
| ULTRA | ✓ (relation-graph message passing) | ✓ | ✓ (2 tầng: relation-graph + entity-graph) | ✓ **nhưng theo cấu trúc, không theo text** | ✓ | ✓ |
| InGram | ✓ (attention qua affinity) | ✓ | ~ (chưa xác nhận có path algebra tường minh) | ✓ **nhưng theo cấu trúc, không theo text** | ✓ (chưa có big-O chính thức) | ✓ |
| HGT | ✓ (attention qua meta-relation) | ✓ (multigraph gốc) | ✗ | ✗ (tham số theo node/edge-type, đóng, không dùng text) | ✓ | ✓ |
| Graphormer | ~ (chỉ qua bias vô hướng) | ✗ (1 shortest-path, collapse khi >1 relation) | ✗ (không thiết kế cho path KG) | ✗ (embedding theo hop, không theo text) | ✓ | ✓ |
| GraphGPS | ~ (tùy module MPNN cắm vào) | ✓ (nhánh MPNN cục bộ, chỉ cộng dồn) | ✗ | phụ thuộc module cắm vào | ✓ | ✓ |
| Relational Attention | ✓ (cạnh attend cạnh) | ✓ (cạnh là công dân hạng nhất, không collapse) | ~ (chưa test trên KG) | chưa đánh giá | ✓ (chưa test ở quy mô Freebase) | không rõ |
| SubgraphRAG | ✗ (chấm từng triple độc lập) | ✗ (không có tương tác r1↔r2) | ✗ | ✓-nghiêng (dùng text embedding, không W_r) | ✓ | ✓ |
| GNN-RAG | ~ (GNN kiểu ReaRev/NSM) | ✗ | ✗ | ✗ | ✓ | ✓ |
| KGFR | ? (chưa rõ cơ chế mixing) | ✗ (chưa xử lý) | ? | ✓ (text do LLM sinh — **inductive thật + text-based**, gần nhất với mục tiêu) | ? | ✓ (chưa xác minh còn hoạt động) |
| GSR | ~ (sinh token tuần tự bởi LM) | ✗ | ~ (tuần tự qua LM, không phải path algebra) | ✓-nghiêng (LM vocabulary) | ✓ | ✓ |
| **QR-DDE (của mình, EXP-11)** | ✓ (mean-aggregate có trọng số) | ✓ (multigraph gốc, không line graph) | ~ (yếu — chỉ qua K vòng mean, chưa phải path algebra nhân tích lũy) | ✓ (w_q(r) từ z_r text, **không** W_r riêng theo relation) | ✓ | ✓ (nội bộ) |
| **Thiết kế 1 đề xuất (TR-NBFNet, mục 7.1)** | ✓ (nhân tích lũy dọc path) | ✓ (multigraph gốc) | ✓ (Bellman-Ford thật, như NBFNet) | ✓ (w_q(r) từ z_r text, không W_r riêng) | ✓ | chưa có |
| **Thiết kế 2 đề xuất (Relational Attention text-conditioned, mục 7.2)** | ✓ (cạnh attend cạnh trực tiếp) | ✓✓ (2 relation trên cùng cặp entity **attend lẫn nhau**, mạnh hơn thiết kế 1) | ~ (chưa kiểm chứng path algebra tường minh) | ✓ (mọi input quan hệ đều qua z_r text) | ✓ (hệ số lớn hơn do attention theo bậc) | chưa có |

**Kết luận đối chiếu bảng:** không paper nào khảo sát được đạt cả 5 tiêu chí đồng thời VÀ dùng text-conditioning cho relation. NBFNet/A*Net đạt (a)(b)(c)(e) nhưng không đạt (d) theo nghĩa "relation chưa thấy" (chỉ inductive entity). ULTRA/InGram đạt (a)(b)(c)(d)(e) nhưng (d) của họ là structural, không phải text-based — nghĩa là relation mới vẫn phải **xuất hiện đủ dữ liệu trong đồ thị đích**, còn QR-DDE hiện tại dùng text nên có thể suy luận về một relation mới chỉ từ tên của nó, kể cả khi nó gần như không xuất hiện trong đồ thị transfer. Đây là khoảng trống mà hai thiết kế đề xuất ở mục 7 nhắm tới lấp.

## 6. Đánh giá phản biện giả thuyết H4

*(Task B mục 1 — câu hỏi: H4 có phải điểm yếu ở mức phương pháp, hay chỉ là lỗi cài đặt? Có đủ lớn làm luận điểm chính không?)*

Dựa trên EXP-13 và phân tích cấu trúc line graph:

1. **Có hai lớp vấn đề tách biệt, cần phân biệt rõ:**
   - **Lỗi cài đặt (fixable, không phải luận điểm phương pháp):** code REL-RAG dùng `nx.DiGraph` để dựng subgraph trước khi tạo line graph — `DiGraph` chỉ cho phép 1 cạnh giữa mỗi cặp (u,v), nên khi 2 relation cùng chiều tồn tại giữa cùng một cặp entity, cạnh thứ hai **ghi đè** cạnh thứ nhất trước khi mô hình kịp thấy nó. Đây là nguồn gốc chính xác của con số "mất ~6% bằng chứng GPT" trong EXP-13. Sửa bằng `nx.MultiDiGraph` là đủ, không cần đổi kiến trúc.
   - **Giới hạn ở mức phương pháp (không sửa được bằng patch):** ngay cả khi đã sửa lỗi trên (mỗi triple có node riêng trong line graph), quy tắc tạo cạnh của line graph là "nối node A→B nếu tail(A) = head(B)" — hai triple song song (h,r₁,t) và (h,r₂,t) có tail giống nhau (t) và head giống nhau (h), nên **không thỏa điều kiện này với nhau** (trừ trường hợp suy biến t=h). Chúng chỉ *tình cờ* có chung tập hàng xóm trong line graph (vì mọi node kề h cũng kề cả 2 node-triple này), chứ không có cạnh trực tiếp để trao đổi thông tin. Đây là hạn chế **cấu trúc/tô pô** của phép biến đổi line graph, độc lập với việc code có bug hay không.

2. **Độ lớn thực nghiệm:** 12.7–14.3% triple nằm trong cặp entity đa-relation cùng chiều thật sự (không tính các cặp đảo chiều trùng lặp do Freebase — 42.9–46.2% triple, đây là redundancy khác, không phải "multi-relation" theo nghĩa mô hình cần phân biệt). Ảnh hưởng tới 22.1–28.6% câu hỏi (có bằng chứng GPT chạm vào ít nhất 1 cặp đa-relation). Đây là tỷ lệ **có thật và không nhỏ**, nhưng **chưa có ablation trực tiếp** đo được việc sửa riêng điểm này cải thiện bao nhiêu recall/F1 — EXP-13 mới dừng ở phân tích thống kê cấu trúc, chưa phải phân tích nhân quả (causal).

3. **Kết luận:** H4 **đúng một phần và có cơ sở ở mức phương pháp**, không chỉ là lỗi cài đặt — quy tắc tạo cạnh của line graph thực sự không cho phép 2 relation song song tương tác trực tiếp, và đây là một tính chất cấu trúc, không phải bug. Tuy nhiên, **một mình H4 chưa đủ lớn để làm luận điểm chính duy nhất** cho một paper A*: (a) độ lớn hiệu ứng (13–14% triple) khiêm tốn hơn nhiều so với vấn đề tổng quát hóa transfer (EXP-12: R@50 rơi từ 86 xuống 53, tức -38%) hay chi phí bậc hai (83–107×, EXP-08); (b) chưa có bằng chứng nhân quả trực tiếp nối H4 với một khoảng recall cụ thể. **Khuyến nghị:** dùng H4 làm **một trong ba luận điểm hỗ trợ** (cùng với chi phí bậc hai và transfer kém), không phải luận điểm độc lập; đồng thời đo trực tiếp bằng cách so sánh recall trên nhóm câu multi-relation-affected (EXP-13 đã định nghĩa) so với nhóm còn lại — nếu thiết kế mới thu hẹp khoảng cách với REL-RAG **nhiều hơn** trên nhóm multi-relation-affected, đó là bằng chứng nhân quả trực tiếp còn thiếu. Kế hoạch đo cụ thể ở mục 7.3 (H1/H4).

## 7. Đề xuất thiết kế mô hình (trình duyệt trước khi prototype)

Cả hai thiết kế dùng chung nguyên tắc lý thuyết ở mục 2.1: **không có tham số học nào được đánh chỉ số theo loại relation** — mọi thông tin quan hệ đi qua embedding văn bản cố định z_r (GTE, giống hệ thống hiện tại) qua một hàm dùng chung cho mọi relation. Theo đúng kỹ thuật chứng minh của Theorem A.1 (Appendix A.4.1), điều này khiến không gian tham số không phụ thuộc R, nên về nguyên tắc đạt cùng lớp cận tổng quát hóa O(√(d/m)) như line graph mà không cần chuyển đổi sang line graph — tránh được cả chi phí bậc hai (EXP-08) lẫn hạn chế multi-relation (EXP-13, mục 6).

### 7.1. Thiết kế 1 — TR-NBFNet (Text-and-question-conditioned NBFNet), rủi ro thấp

Mở rộng QR-DDE hiện tại (`retrieve/exp_qrdde/qrdde.py`) từ "mean-aggregate có trọng số" thành **path composition nhân tích lũy thật sự** kiểu Bellman-Ford tổng quát hóa (NBFNet/A*Net), trên **entity graph gốc** (không line graph):

- **Kiến trúc:**
  - Biên giới: `h^0_v = 1[v ∈ topic] · u(z_q)` (giữ nguyên như QR-DDE hiện tại).
  - Trọng số quan hệ: `w(r, q) = MLP([z_r; z_q])` — **giữ nguyên hàm hiện có của QR-DDE**, không có tham số theo relation ID, chỉ có z_r (embedding văn bản cố định) làm input.
  - Message (thay đổi so với hiện tại): thay vì `h[src] * w` rồi mean-aggregate như hiện tại, dùng phép **kết hợp nhân tích lũy** kiểu NBFNet: `m_{u→v} = h_u^{(k-1)} ⊙ w(r_{uv}, q)` (element-wise), rồi aggregate (sum/mean) qua mọi cạnh vào v — cho phép trạng thái ẩn thực sự "mang" chuỗi quan hệ đã đi qua (r₁, r₂, …) thay vì chỉ là trung bình có trọng số của các vòng.
  - K vòng lan truyền (K=3 mặc định, dùng lại lưới ablation K∈{2,3,4} đã chạy ở M2).
- **Loss:** BCE trên nhãn shortest-path/rational-path hiện có — không đổi loss (theo đúng ràng buộc: loss chỉ là ablation, không phải đóng góp chính).
- **Độ phức tạp:** O(|E|·K·d), tuyến tính theo số cạnh subgraph gốc — không có bước tạo line graph nên tránh hẳn hệ số 83–107× của EXP-08.
- **Lập luận so với Theorem 3.1/A.1:** không có ma trận W_r hay embedding φ_r nào được học riêng theo relation ID (khác cả R-GCN/CompGCN VÀ khác chính line graph của REL-RAG, vốn vẫn có R embedding φ_r riêng) — theo đúng kỹ thuật chứng minh ở mục 2.1, không gian tham số không phụ thuộc R.
- **Đo H1:** so sánh Thiết kế 1 vs QR-DDE hiện tại (mean-aggregate) vs DDE trên WebQSP/CWQ full data (đang chạy ở M2) — nếu path composition nhân tích lũy vượt QR-DDE hiện tại rõ rệt (hơn phương sai giữa seed), ủng hộ H1 ở mức tinh hơn: không chỉ "có điều kiện hóa theo câu hỏi" là đủ, mà "composition đúng kiểu path algebra" mới là yếu tố quyết định.
- **Đo H4:** dùng lại phân loại câu hỏi "multi-relation-affected" (22–29%, EXP-13) làm lát cắt đánh giá — so sánh khoảng cách recall giữa Thiết kế 1 và REL-RAG trên nhóm này so với nhóm còn lại. Nếu khoảng cách thu hẹp nhiều hơn trên nhóm multi-relation-affected, đó là bằng chứng nhân quả trực tiếp còn thiếu ở EXP-13.
- **Kế hoạch prototype (subset, theo mẫu EXP-11):** thêm `--feat trnbf` vào `qrdde.py`, dùng lại 100% hạ tầng subset 800/1500 câu đã có (`exp_qrdde/data/`), so sánh với dde/qrdde/qrdde_noq/rtdde đã có sẵn kết quả. Chi phí thấp (tái dùng hạ tầng, có thể chạy trong 1 buổi GPU).

### 7.2. Thiết kế 2 — Relational Attention với text-conditioning, rủi ro cao hơn, tham vọng hơn

Thiết kế 1 vẫn chỉ **cộng dồn** (aggregate) các message tới từng node — hai relation song song (h,r₁,t) và (h,r₂,t) đóng góp cộng tính vào trạng thái tiếp theo của node đích nhưng **không trực tiếp so sánh/tương tác với nhau**. Thiết kế 2 giải quyết triệt để hơn bằng cách phỏng theo *Relational Attention* (Diao & Loynd, ICLR 2023 — coi cạnh là "công dân hạng nhất", được cập nhật mỗi layer, có thể attend lẫn nhau), thay tham số học theo loại cạnh bằng text-conditioning:

- **Kiến trúc:**
  - Mỗi triple (cạnh) e=(h,r,t) có biểu diễn khởi tạo `z_e^0 = MLP([z_h; z_r; z_t; z_q])` — z_r vẫn là embedding văn bản cố định, không tham số theo relation ID.
  - Mỗi layer: cạnh cập nhật qua attention với **các cạnh khác cùng chung một đầu mút** (chung h hoặc chung t): `z_e^(k) = z_e^(k-1) + Attn(z_e^(k-1), {z_e'^(k-1) : e' chung đầu mút với e})`. Vì (h,r₁,t) và (h,r₂,t) chung **cả hai** đầu mút, chúng có kênh attention **trực tiếp** với nhau — giải quyết H4 ở mức mạnh nhất (so sánh/kết hợp được, không chỉ cùng tồn tại).
  - Biểu diễn entity cho bộ chấm điểm hiện có (`he[h]`, `he[t]` trong `Scorer`) lấy từ pooling các cạnh kề.
- **Loss:** BCE, không đổi.
- **Độ phức tạp:** O(|E|·deg_trung_bình·d) cho bước attention giữa cạnh-cạnh — vẫn tuyến tính với subgraph KGQA thông thường (bậc bị chặn), nhưng hằng số lớn hơn Thiết kế 1; cần đo latency thực tế so với ngưỡng ≤2× SubgraphRAG (`00_RESEARCH_GOAL.md` §7).
- **Lập luận so với Theorem 3.1/A.1:** như Thiết kế 1 — không tham số theo relation ID.
- **Mức độ mới:** theo khảo sát, *Relational Attention* mới chỉ được đánh giá trên benchmark thuật toán CLRS, **chưa từng áp dụng cho KGQA hay ở quy mô từ vựng relation kiểu Freebase** — nếu áp dụng thành công đây là đóng góp mới hơn Thiết kế 1 (Thiết kế 1 về bản chất là một biến thể của NBFNet, một kiến trúc đã có; Thiết kế 2 là một tổ hợp/thích ứng chưa ai làm cho bài toán này).
- **Kế hoạch prototype:** không tái dùng trực tiếp được vòng lặp `qrdde.py` hiện tại (cần forward pass theo cạnh, không phải theo node) — cần viết mới `--feat reltrans`. Ưu tiên: chỉ đầu tư sau khi có kết quả Thiết kế 1 trên full data + lát cắt multi-relation, để biết còn khoảng trống cần thiết kế mạnh hơn hay không.

### 7.3. Khuyến nghị thứ tự thực hiện

1. **Trước tiên:** hoàn thành M2 (đang chạy) để có baseline full-data DDE/QR-DDE chắc chắn.
2. **Sau khi được duyệt:** prototype Thiết kế 1 trên subset (rẻ, tái dùng hạ tầng, lập luận lý thuyết đã rõ) — đây là bước go/no-go tiếp theo.
3. **Chỉ nếu** Thiết kế 1 trên full data vẫn còn khoảng cách rõ rệt với REL-RAG trên nhóm câu multi-relation-affected: đầu tư Thiết kế 2.
4. Không bắt đầu code Thiết kế nào trước khi người dùng duyệt mục này.
