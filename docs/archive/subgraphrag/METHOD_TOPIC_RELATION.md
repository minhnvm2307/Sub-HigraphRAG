# Topic-Relation Selection: Methods & Path+Memory Hybrid

## 1. Vấn đề: Tại sao cần chọn Topic-Relation?

Trong Knowledge Graph QA (KGQA), mỗi câu hỏi $q$ liên kết với một **Topic Entity** $e_t$ (thực thể đầu vào) và một **subgraph** $\mathcal{G}_q \subseteq \mathcal{G}$ chứa trung bình 1,500–15,000 triples. Nhiệm vụ của retriever MLP là xếp hạng tất cả triples trong $\mathcal{G}_q$ và đưa top-$K$ triples vào LLM.

**Hai thành phần đặc trưng quan trọng:**

- **DDE (Distributional Distance Encoding)**: Mã hóa khoảng cách hình học của một triple $(h, r, t)$ đến Topic Entity trong đồ thị, qua $\lceil$rounds$\rceil$ vòng lan truyền message. DDE cho biết "triple này gần hay xa topic entity".
- **Topic PE (Topic Position Encoding)**: Binary flag đánh dấu triple có chứa Topic Entity không. Tuy nhiên, flat DDE không phân biệt được nhóm quan hệ nào có xu hướng kết nối tới answer entity.

**Insight cốt lõi**: Nếu biết trước một tập nhỏ "vàng" các quan hệ liên quan câu hỏi (gọi là **Topic Relations** $\mathcal{R}_q \subseteq \mathcal{R}$), ta có thể gán **weighted DDE** $w_r \in [0.1, 1.0]$ cho DDE của mỗi triple: triples chứa quan hệ được nhận diện sẽ có feature DDE cao hơn, giúp MLP phân biệt chúng dễ hơn.

Bài toán trở thành: **Làm sao để chọn đúng các Topic Relations $\mathcal{R}_q$ trước khi chạy MLP retriever?**

---

## 2. Tiến Trình Các Phương Pháp

### 2.1 Phương pháp dựa trên Embedding đơn lẻ (Dense + Sparse)

**Ý tưởng**: Tính cosine similarity giữa embedding câu hỏi $\mathbf{e}_q$ và embedding từng quan hệ $\mathbf{e}_r$, kết hợp với BM25 sparse score:

$$S_{\text{hybrid}}(r) = 0.5 \cdot \text{sim}_{\text{dense}}(q, r) + 0.5 \cdot \text{sim}_{\text{sparse}}(q, r)$$

**Chọn quan hệ**: Fixed Top-K hoặc Cluster Ratio Cutoff — cắt ngưỡng theo tỷ lệ score của cụm liền kề.

**Hạn chế chính**:
- Đánh giá từng quan hệ **độc lập**, không mô hình hoá quan hệ phụ thuộc giữa $r_1 \to r_2$ trong câu hỏi đa bước (multi-hop).
- Trên CWQ (66% câu hỏi 2-hop), quan hệ $r_2$ hoàn toàn phụ thuộc vào $r_1$. Khi đánh giá độc lập, $r_2$ bị score thấp và bị loại trừ → Recall sập xuống 0.51 trên CWQ.

| Phương pháp | WebQSP F1 | CWQ F1 |
| :--- | :---: | :---: |
| Fixed Top-5 | 0.5081 | 0.4317 |
| Cluster Ratio 0.90 | 0.6027 | 0.4721 |
| Dense Top-5 Paths | 0.6336 | 0.5652 |

---

### 2.2 Path-Level Scoring

**Ý tưởng**: Thay vì score từng quan hệ riêng lẻ, score theo **đường đi $p = (r_1, r_2)$** từ topic entity đến một node bất kỳ:

$$S(p) = \frac{\gamma \cdot S(r_1) + S(r_2)}{\gamma + 1}, \quad \gamma = 0.5$$

Decay $\gamma$ giảm dần theo chiều sâu của path, ưu tiên các quan hệ càng gần topic entity. Sau đó chọn các quan hệ từ top-$\hat{n}$ paths theo score cao nhất (greedy cover).

**Cải thiện**: CWQ Recall tăng từ 0.5124 → 0.7066 vì $r_2$ bây giờ được đánh giá trong ngữ cảnh của path đầy đủ.

---

### 2.3 Path + Memory Hybrid ⭐ (Phương pháp hiện tại)

**Vấn đề còn lại**: Dense embedding score $S_{\text{dense}}(r)$ chỉ phản ánh similarity bề mặt ngữ nghĩa giữa câu hỏi và tên quan hệ. Nhưng **quan hệ Freebase có tên rất trừu tượng** (ví dụ: `government.governmental_jurisdiction.governing_officials` ≠ từ ngữ trong câu hỏi), khiến embedding score không đáng tin cậy.

**Intuition**: Nếu ta đã từng trả lời các câu hỏi tương tự trước đây, ta biết những quan hệ nào thường đúng cho loại câu hỏi đó. Đây chính là **episodic memory** — bộ nhớ ngắn hạn về các câu hỏi trong training set.

#### Thành phần 1: Path Dense Score $S_{\text{path}}(p)$

Từ mỗi Topic Entity, liệt kê tất cả các đường đi 1–2 hop trong subgraph. Với mỗi đường đi $p = (r_1, ..., r_k)$:

$$S_{\text{path}}(p) = \frac{\sum_{i=1}^{k} \gamma^{k-i} \cdot S_{\text{dense}}(r_i)}{\sum_{i=1}^{k} \gamma^{k-i}}, \quad \gamma = 0.5$$

Score này mang tính cấu trúc đồ thị, không bỏ qua quan hệ phụ thuộc.

#### Thành phần 2: Question Memory $M_{\text{mem}}(r | q)$

**Câu hỏi nào mới nhất gần với câu hỏi hiện tại?**

1. **Bộ nhớ huấn luyện**: Từ toàn bộ training set, lưu lại `(câu hỏi, danh sách quan hệ vàng)` cho mỗi mẫu. Đây là `rich_train.jsonl` với ~2,439 bản ghi `fit`.
2. **Similarity đo lường**: Sử dụng **TF-IDF character n-gram (3–5 grams)** cosine similarity để tìm $k=5$ câu hỏi tương tự nhất từ bộ nhớ (bỏ qua topic entity để tránh bias về tên riêng):

$$\text{sim}(q, q') = \text{cos}\big(\text{TF-IDF}_{char}(q \setminus e_t), \; \text{TF-IDF}_{char}(q' \setminus e_t')\big)$$

3. **Bầu chọn có trọng số**: Mỗi câu hỏi láng giềng $q'_i$ bầu chọn cho các quan hệ vàng của nó với trọng số tỷ lệ similarity:

$$M_{\text{mem}}(r | q) = \sum_{i=1}^{k} \frac{\text{sim}(q, q'_i)}{\sum_j \text{sim}(q, q'_j)} \cdot \mathbb{1}[r \in \mathcal{R}^+_{q'_i}]$$

4. **Ước lượng số quan hệ cần chọn $\hat{n}$**: Lấy trung bình có trọng số số lượng quan hệ vàng của $k$ láng giềng:

$$\hat{n} = \sum_{i=1}^{k} w_i \cdot |\mathcal{R}^+_{q'_i}|$$

#### Kết hợp Hybrid

Điểm path-memory fused theo mỗi đường đi $p$:

$$S_{\text{hybrid}}(p) = 0.5 \cdot S_{\text{path}}(p) + 0.5 \cdot S_{\text{mem}}(p)$$

Trong đó $S_{\text{mem}}(p) = \text{weighted\_path\_scores}(\text{paths}, M_{\text{mem}}, \gamma)$ cũng được tính theo cơ chế decay path giống $S_{\text{path}}$.

**Chọn quan hệ**: Greedy cover theo path score cho đến khi đủ $\lceil\hat{n}\rceil$ quan hệ:

$$\mathcal{R}_q = \text{greedyCover}(\text{TopPaths}, \hat{n})$$

#### Tại sao Leave-One-Out (LOO) quan trọng?

Trong quá trình precompute cache cho tập `train`, nếu không loại trừ mẫu hiện tại khỏi bộ nhớ kNN, mẫu $q_i$ sẽ **tự tìm thấy chính mình** (sim ≈ 1.0) và bỏ phiếu cho quan hệ vàng của chính nó — tạo ra "shortcut nhãn vàng" giúp MLP fit train nhưng không tổng quát hóa ra test. LOO giải quyết hoàn toàn vấn đề này bằng cách set $\text{sim}(q_i, q_i) = -1$.

#### Kết quả chọn Topic-Relation

| Phương pháp | WebQSP F1 | CWQ F1 | Recall WebQSP | Recall CWQ |
| :--- | :---: | :---: | :---: | :---: |
| Cluster Ratio 0.90 (Dense) | 0.6027 | 0.4721 | 0.7152 | 0.5124 |
| Path Dense Top-5 Paths | 0.6336 | 0.5652 | 0.7066 | 0.5563 |
| **Path+Memory Hybrid (0.5:0.5)** | **0.6111** | **0.6463** | **0.7010** | **0.7407** |
| **Path+Memory Hybrid (0.7:0.3)** | **0.6581** | **0.6364** | **0.7593** | **0.7303** |

---

## 3. Hàm Mất Mát: Từ BCE đến Inversed Loss

### Vấn đề Class Imbalance

Trong mỗi subgraph, tỷ lệ triples vàng (shortest path) chỉ chiếm ~1–5% tổng số triples. Binary Cross-Entropy (BCE) thuần sẽ bỏ qua các positive mẫu vì gradient từ 95%+ negative lấn át.

### Inversed Dynamic Weighting

$$w_{\text{pos}} = \frac{N_{\text{neg}}}{N_{\text{pos}}}$$

Trọng số này được tính **động theo từng sample** (không cố định toàn bộ dataset), cho phép mô hình thích ứng với subgraph lớn hay nhỏ.

**So sánh với Effective Num Loss** (Cui et al., $\beta=0.999$): Effective Num dùng trọng số cố định $(1-\beta)/(1-\beta^n)$ không phụ thuộc vào kích thước thực tế của đồ thị con, gây ra over-penalization trên subgraph nhỏ và under-penalization trên subgraph lớn.

| Loss | SP Recall@100 | Ans Recall@100 | GPT Recall@100 |
| :--- | :---: | :---: | :---: |
| BCE | 0.8974 | 0.9590 | 0.8770 |
| Effective Num | 0.9090 | 0.9530 | 0.8850 |
| InfoNCE | 0.8950 | 0.9550 | 0.8750 |
| **Inversed (Dynamic)** | **0.9120** | **0.9600** | **0.8920** |

---

## 4. Tóm Tắt Pipeline

```
Câu hỏi q + Topic Entity e_t
        │
        ▼
  [Path Enumeration]          [Question Memory kNN]
  1–2 hop paths từ e_t        5 câu hỏi tương tự nhất
        │                           │
        ▼                           ▼
  S_path(p)              M_mem(r | q) + n̂
        │                           │
        └──────── Hybrid 0.5:0.5 ──┘
                         │
                         ▼
              S_hybrid(p) → greedy cover → R_q (Topic Relations)
                         │
                         ▼
  [MLP Retriever] với feature: DDE × w_r + Topic PE + dense emb
  (Weighted DDE: w_r = max(S(r), 0.1) cho r ∈ R_q, else 0.1)
                         │
                         ▼
              Top-100 triples → LLM (Llama-3.1-8B)
                         │
                         ▼
                    Answer text
```
