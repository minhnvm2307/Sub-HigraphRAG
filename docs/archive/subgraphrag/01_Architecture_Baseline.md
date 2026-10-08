# 01. KNOWLEDGE GRAPH RETRIEVAL ARCHITECTURE

## 1. Bài toán tối ưu hóa cốt lõi
Mục tiêu là trích xuất đồ thị con $G_{q}^{*}$ tối đa hóa xác suất trả lời đúng $A_{q}$:
$$G_{q}^{*} = \arg\max_{G_{q} \subseteq G} P(A_{q} \mid G_{q}, q)$$

Do $P$ phụ thuộc vào LLM (không có đạo hàm trực tiếp), bài toán được xấp xỉ thông qua việc đánh giá tính độc lập của từng bộ ba (triple) $\tau = (h, r, t)$. Giả định phân rã:
$$Q(G_{q} \mid G, q) = \prod_{\tau \in G_{q}} p(\tau) \prod_{\tau \notin G_{q}} (1 - p(\tau))$$
Hệ quả: Bài toán truy xuất đồ thị con chuyển thành hàng triệu bài toán phân loại nhị phân độc lập.

## 2. Xây dựng đặc trưng (Feature Construction)
Input của mô hình MLP cho mỗi triple $\tau = (h, r, t)$ là một vector biểu diễn tổng hợp tích hợp cả đặc trưng thực thể (Entity) và thông tin dẫn hướng quan hệ (Relation-Topic):
$$z_{\tau} = [z_{q} \parallel (z_{h} + PE + DDE) \parallel [z_{r} \parallel \text{one-hot}(r \in R_q)] \parallel (z_{t} + PE + DDE)]$$

Trong đó:
- $z_{q}, z_{h}, z_{r}, z_{t}$: Embedding vector tương ứng của câu hỏi, head, relation, tail.
- **PE (Positional Encoding):** Vector one-hot xác định entity có phải là Topic Entity $T_{q}$ hay không.
- **Relation-Topic Embedding Integration:** Vector one-hot kích thước 2 biểu thị liệu quan hệ $r$ có thuộc tập quan hệ trọng tâm của câu hỏi $R_q$ (được xác định bằng cơ chế lai **Hybrid Lexical-Semantic Similarity**) hay không.
- **DDE (Directional Distance Encoding):** Nắm bắt cấu trúc không gian của đồ thị.

### 2.1. Lan truyền đặc trưng DDE (Message Passing)
Khởi tạo vector nhãn $s_{e}(0)$. Lan truyền qua $l$ vòng:
- **Hướng thuận:** Tính trung bình nhãn từ các node đi vào:
  $$s_{e(l+1)} = \text{MEAN}\{s_{e'(l)} \mid (e', \cdot, e) \in G\}$$
- **Hướng nghịch:** Tính trung bình nhãn từ các node đi ra:
  $$s_{e(r, l+1)} = \text{MEAN}\{s_{e'(r,l)} \mid (e, \cdot, e') \in G\}$$

Vector thực thể cuối cùng: $s_{e} = [s_{e}(0) \parallel s_{e}(1) \parallel \dots \parallel s_{e}(r,1) \dots]$. DDE giúp mô hình nhận biết được các node đóng vai trò cầu nối trên đường dẫn suy luận đa bước (multi-hop).

## 3. Kiến trúc mô hình phân loại (MLP Retriever)
Sử dụng Lightweight MLP với Loss Function là Binary Cross Entropy (BCE):
$$\mathcal{L} = -\sum_{\tau \in \tilde{G}_{q}} \log p_{\theta}(\tau) - \sum_{\tau \notin \tilde{G}_{q}} \log (1 - p_{\theta}(\tau))$$
Mô hình đảm bảo tốc độ thời gian thực (parallel triple-scoring) so với các kiến trúc GNN nặng nề.

## 4. Môi trường thực thi thống nhất (Unified Runtime Environment)
Để đảm bảo tính nhất quán và loại bỏ xung đột giữa các thư viện phục vụ Embedding Extraction, MLP Retriever, và LLM Reasoner, dự án sử dụng duy nhất một môi trường Conda thống nhất:
- **Conda Environment:** `subgraphrag`
- **Cấu hình chính:** Python 3.10.x, PyTorch (CUDA 12.1), PyTorch Geometric (PyG 2.5.3), vLLM (0.5.5), Transformers, NetworkX, và Wandb. (Xem chi tiết hướng dẫn cài đặt trong `AGENT_SKILL.md`).