# Tổng Hợp Chi Tiết Các Quy Trình Precompute & Export Subgraph Trong Dự Án

> **Mục đích tài liệu**: Lưu trữ toàn diện cấu trúc, thuật toán và logic hoạt động của các script tiền xử lý đặc trưng (`precompute_*.py`) và các script trích xuất đồ thị con động (`export_*.py`) trong SubgraphRAG, nhằm phục vụ tra cứu và tái tạo khi cần, đồng thời giải phóng thư mục `retrieve/` khỏi các file thử nghiệm rời rạc.

---

## PHẦN I: CÁC TIẾN TRÌNH TÍNH TOÁN TRƯỚC ĐẶC TRƯNG (PRECOMPUTE PIPELINE)

Toàn bộ các đặc trưng quan hệ được tính toán trước và lưu trữ tại `retrieve/data_files/{dataset}/relation_topics/{feature_name}/{split}.pth` theo cấu trúc:
$$\text{Tensor kích thước } (N_{\text{relations}}, 2): \quad [\text{Cột 0: Điểm liên tục } S(r) \in [0, 1], \quad \text{Cột 1: Mặt nạ nhị phân } M(r) \in \{0, 1\}]$$

### 1. `precompute_path_memory_hybrid.py` (Phase 13 - Sản Phẩm SOTA Hiện Tại)
- **Mục tiêu**: Kết hợp tín hiệu cấu trúc chuỗi đường đi 1–2 hop với bộ nhớ câu hỏi tương đồng $k$-NN để giải quyết quan hệ ẩn/đa bước trên CWQ.
- **Công thức cốt lõi**:
  $$S_{\text{hybrid}}(r) = (1 - \lambda) \cdot S_{\text{path}}(r) + \lambda \cdot S_{\text{memory}}(r)$$
  - $S_{\text{path}}(r)$: Điểm đường đi lớn nhất chứa quan hệ $r$, suy giảm theo độ dài hop:
    $$S(p) = \text{score}(r_1) \cdot \gamma^{\text{len}(p)-1}$$ (với $\gamma = 0.5$).
  - $S_{\text{memory}}(r)$: Điểm tần suất quan hệ xuất hiện trong top-$k$ câu hỏi láng giềng gần nhất ($k=15$) qua Cosine Query Embedding (sử dụng chuẩn hóa Leave-One-Out trên train để chống rò rỉ dữ liệu).
- **Trọng số tối ưu**: $\lambda = 0.5$, $\gamma = 0.5$, $k = 15$.
- **File lưu trữ**: `data_files/{dataset}/relation_topics/path_memory_hybrid/{train,val,test}.pth`.

### 2. `precompute_bge_dense_sparse.py` (Phase 11)
- **Mục tiêu**: Tính toán điểm tương đồng giữa câu hỏi và từng quan hệ bằng mô hình đa ngôn ngữ BGE-M3 kết hợp cả biểu diễn dày đặc (Dense) và thưa thớt (Lexical Sparse - Lexical Weights).
- **Công thức**:
  $$S_{\text{dense}}(q, r) = \cos(\mathbf{e}_q, \mathbf{e}_r), \quad S_{\text{sparse}}(q, r) = \sum_{w \in q \cap r} W_q(w) \cdot W_r(w)$$
  $$S_{\text{bge}}(r) = 0.6 \cdot S_{\text{dense}}(r) + 0.4 \cdot S_{\text{sparse}}(r)$$
- **File lưu trữ**: `data_files/{dataset}/relation_topics/bge_dense_sparse/{split}.pth`.

### 3. `precompute_relation_topics.py` (Phase 4 & Phase 7)
- **Mục tiêu**: Script tạo đặc trưng quan hệ thế hệ đầu (dùng GTE-Large hoặc Contriever) đo mức độ liên quan ngữ nghĩa trực tiếp $S(q, r)$ phục vụ cơ chế RT-MLP và RT-DDE.
- **File lưu trữ**: `data_files/{dataset}/relation_topics/gte/{split}.pth`.

---

## PHẦN II: CÁC TIẾN TRÌNH TRÍCH XUẤT ĐỒ THỊ CON ĐỘNG (EXPORT PIPELINE)

Các script này nhận đầu vào là file `retrieval_result.pth` (chứa toàn bộ triples kèm điểm số dự đoán của mô hình Retriever) và áp dụng thuật toán cắt tỉa động để tạo file phục vụ giai đoạn **QA Reasoning (Phase 14)**.

### 1. `export_m34_bridge.py` (M3.4: Nucleus Top-P + Bridge Completion - Cấu Hình SOTA Đề Xuất)
- **Thuật toán**:
  1. **Bước 1 (Nucleus Top-P Selection)**:
     - Chuẩn hóa điểm số các cạnh thành phân phối xác suất: $p_i = \max(0, s_i) / \sum_j \max(0, s_j)$.
     - Tích lũy xác suất đến ngưỡng $P = 0.90$. Chọn $K$ cạnh hạt giống trong khoảng $[K_{\min}=15, K_{\max}=60]$.
  2. **Bước 2 (Bridge Completion)**:
     - Xây dựng đồ thị lân cận từ tập ứng viên.
     - Với mỗi cạnh hạt giống $(h, r, t)$ chưa liên thông trực tiếp với Topic Entity, tìm đường đi ngắn nhất có điểm số cao nhất nối từ Topic Entity đến cạnh đó.
     - Tự động chèn các "cạnh cầu nối" (bridge triples) bị thiếu để đảm bảo mạch suy luận không bị đứt gãy.
- **Hiệu quả thực nghiệm**: Tăng Exact Match lên **53.81%** (+1.46%) và Macro F1 lên **74.12%** trong khi cắt giảm **65.6% token đồ thị**.
- **File đầu ra**: `retrieval_result_m34_bridge.pth`.

### 2. `export_dynamic_subgraph.py` (M1.4: Nucleus Top-P Thuần Túy)
- **Thuật toán**: Cắt tỉa theo phân vị xác suất tích lũy $P = 0.90$ trong giới hạn $[15, 60]$ triples.
- **Đặc điểm**: Giảm trung bình từ 100 triples xuống **34.3 triples** (-65.7%), giúp LLM tăng tốc độ suy luận 2.5 lần.
- **File đầu ra**: `retrieval_result_m14_top_p.pth`.

### 3. `export_baseline_m14.py` (M1.4 Trên Mô Hình Baseline Gốc Jul26)
- **Thuật toán**: Áp dụng thuật toán Top-P ($P=0.90$) trên mô hình Baseline cũ (Retriever không có Path+Memory).
- **Mục tiêu khoa học**: Thực hiện thí nghiệm phân tách nhân tố (Factorial Ablation) để chứng minh hiệu quả vượt trội đến từ chất lượng scoring của mô hình mới chứ không phải chỉ do thuật toán cắt tỉa.

### 4. `export_fixed_top15.py` (Fixed Top-15 Cực Hạn)
- **Thuật toán**: Cắt cứng lấy đúng 15 triples có điểm cao nhất.
- **Mục tiêu**: Đo lường điểm rơi chất lượng khi ép ngân sách token xuống mức tối thiểu cực hạn (-85% triples).

---

## PHẦN III: KẾT LUẬN & TRẠNG THÁI HỆ THỐNG

- Toàn bộ dữ liệu đặc trưng đã được precompute đầy đủ trong `data_files/`.
- Toàn bộ các file kết quả trích xuất `.pth` đã nằm sẵn trong các thư mục checkpoint tương ứng (`webqsp_rt_path_memory_hybrid_inversed_Sep07-18:06:25/`).
- Do đó, các script mã nguồn `.py` này đã hoàn thành vai trò lịch sử và được dọn dẹp sạch khỏi thư mục `retrieve/` để đảm bảo codebase luôn tinh gọn, chuẩn mực.
