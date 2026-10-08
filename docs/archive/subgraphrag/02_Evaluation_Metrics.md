# 02. BASELINE EVALUATION & METRICS

## 1. Hiệu suất tổng quan (WebQSP & CWQ)
- Việc sử dụng MLP scoring cải thiện từ 25% đến 35% answer recall so với retrieval chỉ dựa trên embedding similarity (Semantic search thuần túy).
- **Top-K Coverage:** Để LLM có đủ manh mối, cần Top-K $\approx 400-500$, dù trung bình LLM chỉ dùng khoảng 23 triples hữu ích cho reasoning.

## 2. Phát hiện thực nghiệm (Empirical Findings)

### 2.1. Hiện tượng Score Flattening & Ảnh hưởng của Relation-Topic
Sự phân biệt giữa các candidate triples suy giảm nghiêm trọng khi mở rộng không gian tìm kiếm:
- **Trước đây:** Khoảng 75% triples có score tiệm cận 0 ($\le 0.01$). Sau Top-50, hầu hết các điểm số bị san phẳng (ví dụ: rank 2 = 0.002, rank 4 = 0.0000).
- **Hiện tại (Sau khi tích hợp Relation-Topic):** Phân phối điểm số (score distribution) đã có sự cải thiện rõ rệt, ít bị san phẳng ở các vị trí đầu hơn do mô hình phân biệt tốt hơn vai trò dẫn hướng của các relations tương đồng với câu hỏi.
- **Tiến độ:** Đang tiến hành chạy thử nghiệm benchmark diện rộng để đánh giá chi tiết sự thay đổi của Precision/Recall và F1-score ở các mốc K khác nhau.

### 2.2. Điểm yếu về Context & Thông tin toàn cục
Việc đánh giá từng triple $\tau$ độc lập khiến mô hình bỏ lỡ sự liên kết ngữ nghĩa tổng thể. Thử nghiệm toàn cục HiRAG (tạo Community Report thông qua Leiden/GMM clustering) đã thất bại (Hit giảm từ 91.58% xuống 64.21% và F1 giảm từ 78.54% xuống 52.93%), cho thấy nhiễu ngữ cảnh từ các tóm tắt không sát với truy vấn multi-hop.

### 2.3. Đánh giá tính Generalization (Train WebQSP -> Eval CWQ)
Mô hình giữ được khả năng tổng quát hóa khá tốt sang đồ thị chưa từng xuất hiện nhờ DDE và Semantic Embeddings, tuy nhiên Recall giảm tương đối từ 20% đến 40% tùy thuộc vào giá trị K.

## 3. Môi trường thực thi đánh giá
Mọi tập lệnh kiểm thử và đánh giá hiệu năng (chạy `eval.py`, `analyze_result.py` hay `reasoning_eval.py`) được thiết lập chạy trực tiếp trong môi trường Conda duy nhất:
```bash
conda activate subgraphrag
```
Điều này triệt tiêu hoàn toàn sự sai lệch chỉ số do không đồng bộ thư viện (ví dụ numpy/pandas) giữa các môi trường cũ.