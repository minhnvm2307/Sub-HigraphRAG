# Detail evaluate (ablation on AER, ADR, ASE)

> Sinh tự động bởi `relrag/make_benchmark_md.py` từ `relrag/runs/bench/bench_results.json` (`relrag/bench_matrix.py`), 10-07.
> Cách chấm giống `gate_report.py`: ansR = text-match đáp án với head/tail; SP-R = triple trên đường ngắn nhất (nhãn REL-RAG); GPT-R = triple bằng chứng GPT-4o, khớp chính xác (h, r, t). Trong mỗi setting, mọi phương pháp chấm trên **cùng tập câu** (có đáp án trong graph và có mặt trong mọi export của setting đó).
> Mọi số là **một seed (42)**, chưa có khoảng tin cậy. Độ nhiễu giữa seed đã đo: DDE 5.6 điểm, QR-DDE 3.6, WL 0.4 (ansR@50, WebQSP→CWQ, EXP-34).
> **Thiên lệch đã biết của SP-R / GPT-R (EXP-36):** SP-R dùng nhãn trên graph của REL-RAG; GPT-R khớp chính xác (h, r, t) theo hướng của SubgraphRAG. RAPL / REL-RAG hay trả về triple dùng relation nghịch, nên **GPT-R của RAPL / REL-RAG bị đánh giá thấp** (theo cặp entity, khoảng cách với SubgraphRAG chỉ còn khoảng 3 điểm). ansR không bị ảnh hưởng.
> AER = neo lại ở cầu nối (B = 3 triple, 2 lượt, gộp max z-score); ADR = thêm một lượt cho từng topic; ASE = mã hoá tập anchor; WL = relation-topic + weighted loss (effective-number); aug = augmentation anchor khớp phân phối CWQ train (EXP-35, **có dùng thống kê CWQ train**).

- WebQSP (train WebQSP, in-distribution): n = 1559 câu
- WebQSP → CWQ (train WebQSP, OOD / compositional): n = 2848 câu
- CWQ (train CWQ, in-distribution): n = 2848 câu
- CWQ → WebQSP (train CWQ, OOD): n = 1559 câu

## AER on different backbone (ans, sp, gpt)@50-100-500

### In distribution: WebQSP (train WebQSP, in-distribution)

| Backbone | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 | Δ ansR@50 so với backbone gốc |
|---|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) + AER | 89.72 | 94.10 | 98.85 | 81.19 | 86.75 | 96.32 | 80.68 | 86.96 | 96.93 | -0.34 |
| QR-DDE + AER | 88.71 | 93.85 | 99.04 | 83.77 | 89.37 | 97.91 | 81.88 | 87.19 | 95.97 | -0.49 |
| DDE, AER-trained (reanchor_p 0.5) + AER | 89.68 | 94.18 | 98.88 | 81.25 | 86.83 | 96.46 | 80.91 | 86.80 | 97.08 | -0.38 |
| RT-combined (no weighted loss) + AER | 91.55 | 95.74 | 99.08 | 82.41 | 88.09 | 97.08 | 82.29 | 88.26 | 97.04 | -0.24 |
| RT + weighted loss (WL) + AER | 92.27 | 96.30 | 99.27 | 83.98 | 89.73 | 97.46 | 84.54 | 89.65 | 97.42 | -0.31 |
| ASE (on DDE) + AER | 89.23 | 94.28 | 99.14 | 82.31 | 88.49 | 97.42 | 81.48 | 87.33 | 96.46 | -0.87 |
| WL + ASE + AER | 92.50 | 96.09 | 99.24 | 85.16 | 90.60 | 97.79 | 84.28 | 89.52 | 97.10 | -0.33 |
| WL + aug + AER | 90.64 | 95.28 | 99.23 | 83.65 | 89.62 | 97.49 | 84.71 | 89.32 | 97.08 | -1.24 |
| WL + ASE + aug + AER | 92.33 | 95.99 | 99.19 | 84.51 | 89.41 | 97.40 | 84.70 | 89.43 | 96.72 | -0.39 |

### OOD: WebQSP → CWQ (train WebQSP, OOD / compositional)

| Backbone | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 | Δ ansR@50 so với backbone gốc |
|---|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) + AER | 75.78 | 84.76 | 96.03 | 51.68 | 61.18 | 81.81 | 55.75 | 65.57 | 86.05 | +5.74 |
| QR-DDE + AER | 70.45 | 81.66 | 96.09 | 55.04 | 64.12 | 83.42 | 59.06 | 69.39 | 88.83 | +9.50 |
| DDE, AER-trained (reanchor_p 0.5) + AER | 77.95 | 85.18 | 96.78 | 52.01 | 61.68 | 82.35 | 56.86 | 66.30 | 87.49 | +3.17 |
| RT-combined (no weighted loss) + AER | 80.38 | 87.12 | 97.50 | 54.02 | 63.86 | 84.63 | 59.59 | 68.77 | 88.66 | +7.97 |
| RT + weighted loss (WL) + AER | 80.27 | 88.45 | 97.76 | 53.41 | 63.10 | 85.03 | 61.72 | 71.88 | 89.91 | +6.22 |
| ASE (on DDE) + AER | 74.91 | 84.56 | 96.93 | 53.49 | 62.84 | 83.29 | 54.03 | 63.97 | 84.00 | +3.42 |
| WL + ASE + AER | 77.16 | 85.91 | 97.86 | 57.32 | 67.09 | 86.18 | 61.99 | 70.93 | 89.06 | +9.23 |
| WL + aug + AER | 79.66 | 88.68 | 97.94 | 54.72 | 64.80 | 86.60 | 61.37 | 71.90 | 90.90 | +2.73 |
| WL + ASE + aug + AER | 87.38 | 92.38 | 98.44 | 64.16 | 72.54 | 89.65 | 71.73 | 79.57 | 92.59 | +6.76 |

### In distribution: CWQ (train CWQ, in-distribution)

| Backbone | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 | Δ ansR@50 so với backbone gốc |
|---|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) + AER | 86.39 | 90.80 | 98.16 | 65.14 | 73.11 | 88.26 | 71.01 | 80.20 | 94.41 | -0.29 |
| QR-DDE + AER | 89.74 | 93.57 | 98.61 | 74.99 | 80.72 | 91.16 | 79.46 | 85.52 | 94.83 | +0.05 |
| RT-combined (no weighted loss) + AER | 85.59 | 91.92 | 98.01 | 62.60 | 71.87 | 88.34 | 70.76 | 80.02 | 94.56 | -0.63 |
| ASE (on DDE) + AER | 89.36 | 93.66 | 98.38 | 74.83 | 81.04 | 91.51 | 77.82 | 84.22 | 94.02 | -0.97 |

### OOD: CWQ → WebQSP (train CWQ, OOD)

| Backbone | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 | Δ ansR@50 so với backbone gốc |
|---|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) + AER | 80.67 | 87.24 | 97.03 | 69.01 | 78.24 | 93.31 | 69.53 | 78.69 | 93.08 | -1.55 |
| QR-DDE + AER | 79.90 | 86.32 | 96.10 | 74.20 | 81.86 | 94.20 | 73.03 | 81.28 | 92.88 | -1.07 |
| RT-combined (no weighted loss) + AER | 83.56 | 90.18 | 97.47 | 70.01 | 79.55 | 94.20 | 70.76 | 79.60 | 93.57 | -1.41 |
| ASE (on DDE) + AER | 77.95 | 87.07 | 97.16 | 69.03 | 79.26 | 94.21 | 67.51 | 77.62 | 92.98 | -3.88 |

=> Cái nào tốt nhất:

- WebQSP (train WebQSP, in-distribution): ansR@50 cao nhất là **WL + ASE + AER** (92.50); AER **không giúp backbone nào** (Δ tốt nhất -0.24, trên RT-combined (no weighted loss)).
- WebQSP → CWQ (train WebQSP, OOD / compositional): ansR@50 cao nhất là **WL + ASE + aug + AER** (87.38); AER giúp nhiều nhất trên **QR-DDE** (+9.50).
- CWQ (train CWQ, in-distribution): ansR@50 cao nhất là **QR-DDE + AER** (89.74); AER giúp nhiều nhất trên **QR-DDE** (+0.05).
- CWQ → WebQSP (train CWQ, OOD): ansR@50 cao nhất là **RT-combined (no weighted loss) + AER** (83.56); AER **không giúp backbone nào** (Δ tốt nhất -1.07, trên QR-DDE).

**AER, các siêu tham số khác (đã có từ trước, EXP-23, WebQSP → CWQ, ansR@50, AER train với `reanchor_p`):**

| reanchor_p | ALL | composition | conjunction | comparative | superlative | WebQSP |
|---|---|---|---|---|---|---|
| 0.25 | 75.38 | 68.1 | 79.7 | 77.0 | 75.4 | 89.17 |
| **0.5** | **77.95** | **70.0** | 81.3 | 85.4 | 83.1 | 89.68 |
| 1.0 | 77.13 | 63.4 | 83.3 | 84.7 | 88.5 | 89.62 |

B (số triple lấy cầu nối) chỉ chạy B = 3; K (số lượt) chỉ chạy 2. Chưa có ablation B / K có nhãn đầy đủ.

## ADR on different backbone (ans, sp, gpt)@50-100-500

### In distribution: WebQSP (train WebQSP, in-distribution)

| Backbone | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 | Δ ansR@50 so với backbone gốc |
|---|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) + ADR | 89.82 | 94.29 | 98.76 | 81.85 | 87.12 | 96.41 | 80.96 | 87.03 | 97.10 | -0.24 |
| QR-DDE + ADR | 89.39 | 93.98 | 99.01 | 84.71 | 89.97 | 97.98 | 82.39 | 87.48 | 95.79 | +0.19 |
| DDE, AER-trained (reanchor_p 0.5) + ADR | 89.76 | 94.08 | 98.88 | 81.38 | 87.05 | 96.44 | 79.63 | 86.01 | 96.89 | -0.30 |
| RT-combined (no weighted loss) + ADR | 91.64 | 95.52 | 99.18 | 83.28 | 88.40 | 97.02 | 82.64 | 88.01 | 97.22 | -0.15 |
| RT + weighted loss (WL) + ADR | 92.32 | 95.92 | 99.20 | 84.50 | 89.82 | 97.40 | 84.40 | 89.60 | 97.40 | -0.27 |
| ASE (on DDE) + ADR | 89.74 | 94.72 | 99.18 | 82.69 | 89.10 | 97.23 | 80.26 | 86.78 | 96.56 | -0.36 |
| WL + ASE + ADR | 92.40 | 96.05 | 99.22 | 85.27 | 90.67 | 97.61 | 83.55 | 88.74 | 96.99 | -0.44 |
| WL + aug + ADR | 91.68 | 95.76 | 99.27 | 84.78 | 90.20 | 97.53 | 84.61 | 89.21 | 97.21 | -0.20 |
| WL + ASE + aug + ADR | 92.34 | 96.08 | 99.18 | 84.46 | 89.28 | 97.22 | 82.51 | 87.36 | 95.90 | -0.38 |

### OOD: WebQSP → CWQ (train WebQSP, OOD / compositional)

| Backbone | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 | Δ ansR@50 so với backbone gốc |
|---|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) + ADR | 72.49 | 80.52 | 94.81 | 51.64 | 60.36 | 81.27 | 56.87 | 65.96 | 85.54 | +2.44 |
| QR-DDE + ADR | 64.90 | 74.71 | 91.91 | 51.26 | 59.08 | 79.35 | 56.14 | 64.89 | 84.65 | +3.94 |
| DDE, AER-trained (reanchor_p 0.5) + ADR | 76.72 | 83.99 | 95.48 | 52.16 | 61.41 | 81.83 | 57.52 | 66.75 | 86.35 | +1.94 |
| RT-combined (no weighted loss) + ADR | 74.06 | 82.53 | 95.62 | 53.13 | 62.23 | 83.40 | 58.49 | 67.20 | 87.46 | +1.65 |
| RT + weighted loss (WL) + ADR | 77.37 | 85.37 | 96.05 | 53.06 | 62.73 | 84.71 | 61.06 | 70.60 | 89.35 | +3.32 |
| ASE (on DDE) + ADR | 76.32 | 83.74 | 96.75 | 53.72 | 63.23 | 85.71 | 58.13 | 67.08 | 87.76 | +4.82 |
| WL + ASE + ADR | 74.53 | 82.66 | 95.48 | 54.37 | 63.91 | 84.96 | 60.07 | 68.29 | 88.15 | +6.61 |
| WL + aug + ADR | 77.66 | 85.73 | 96.60 | 52.60 | 63.34 | 85.23 | 59.38 | 69.88 | 89.25 | +0.73 |
| WL + ASE + aug + ADR | 79.54 | 86.73 | 96.45 | 54.65 | 64.16 | 86.05 | 62.03 | 71.42 | 89.31 | -1.08 |

### In distribution: CWQ (train CWQ, in-distribution)

| Backbone | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 | Δ ansR@50 so với backbone gốc |
|---|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) + ADR | 87.56 | 91.89 | 98.31 | 66.44 | 74.02 | 88.40 | 73.20 | 81.04 | 94.76 | +0.89 |
| QR-DDE + ADR | 90.87 | 94.29 | 98.72 | 74.51 | 80.36 | 90.72 | 79.40 | 85.43 | 94.77 | +1.18 |
| RT-combined (no weighted loss) + ADR | 87.16 | 92.41 | 98.23 | 63.58 | 72.29 | 88.40 | 72.72 | 81.18 | 94.91 | +0.95 |
| ASE (on DDE) + ADR | 89.30 | 93.57 | 98.68 | 71.37 | 78.34 | 90.78 | 77.52 | 84.08 | 93.73 | -1.03 |

### OOD: CWQ → WebQSP (train CWQ, OOD)

| Backbone | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 | Δ ansR@50 so với backbone gốc |
|---|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) + ADR | 82.13 | 88.20 | 97.16 | 71.92 | 80.14 | 93.84 | 72.05 | 80.02 | 93.42 | -0.09 |
| QR-DDE + ADR | 81.38 | 87.34 | 96.30 | 76.57 | 83.31 | 94.69 | 74.74 | 82.19 | 93.13 | +0.41 |
| RT-combined (no weighted loss) + ADR | 85.05 | 90.85 | 97.77 | 73.13 | 80.68 | 94.84 | 73.23 | 80.83 | 94.09 | +0.09 |
| ASE (on DDE) + ADR | 81.75 | 88.48 | 97.51 | 71.66 | 80.21 | 94.49 | 70.40 | 78.55 | 93.76 | -0.08 |

=> Cái nào tốt nhất:

- WebQSP (train WebQSP, in-distribution): ansR@50 cao nhất là **WL + ASE + ADR** (92.40); ADR giúp nhiều nhất trên **QR-DDE** (+0.19).
- WebQSP → CWQ (train WebQSP, OOD / compositional): ansR@50 cao nhất là **WL + ASE + aug + ADR** (79.54); ADR giúp nhiều nhất trên **WL + ASE** (+6.61).
- CWQ (train CWQ, in-distribution): ansR@50 cao nhất là **QR-DDE + ADR** (90.87); ADR giúp nhiều nhất trên **QR-DDE** (+1.18).
- CWQ → WebQSP (train CWQ, OOD): ansR@50 cao nhất là **RT-combined (no weighted loss) + ADR** (85.05); ADR giúp nhiều nhất trên **QR-DDE** (+0.41).

## ASE on different backbone (ans, sp, gpt)@50-100-500

### In distribution: WebQSP (train WebQSP, in-distribution)

| Method | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 |
|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) | 90.07 | 94.41 | 98.88 | 81.85 | 87.09 | 96.40 | 81.02 | 86.93 | 97.06 |
| ASE (on DDE) | 90.10 | 94.92 | 99.18 | 82.99 | 89.39 | 97.37 | 80.56 | 86.92 | 96.55 |
| RT + weighted loss (WL) | 92.58 | 96.09 | 99.26 | 84.69 | 90.03 | 97.46 | 84.47 | 89.66 | 97.37 |
| WL + ASE | 92.84 | 96.13 | 99.29 | 85.59 | 90.93 | 97.74 | 84.05 | 88.89 | 97.01 |
| WL + aug | 91.88 | 96.01 | 99.27 | 84.87 | 90.33 | 97.57 | 84.76 | 89.23 | 97.21 |
| WL + ASE + aug | 92.72 | 96.21 | 99.18 | 84.62 | 89.48 | 97.30 | 82.72 | 87.38 | 95.89 |

### OOD: WebQSP → CWQ (train WebQSP, OOD / compositional)

| Method | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 |
|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) | 70.05 | 79.58 | 94.81 | 49.77 | 58.65 | 80.51 | 53.59 | 63.05 | 84.53 |
| ASE (on DDE) | 71.49 | 80.72 | 95.66 | 50.95 | 60.29 | 82.37 | 50.90 | 60.66 | 83.04 |
| RT + weighted loss (WL) | 74.04 | 83.56 | 96.31 | 50.89 | 60.58 | 83.60 | 57.65 | 67.59 | 88.22 |
| WL + ASE | 67.92 | 78.30 | 95.25 | 51.71 | 60.64 | 82.88 | 54.85 | 63.22 | 85.47 |
| WL + aug | 76.93 | 85.24 | 97.10 | 51.95 | 62.41 | 85.04 | 57.92 | 68.31 | 89.01 |
| WL + ASE + aug | 80.62 | 86.57 | 96.31 | 58.51 | 66.59 | 86.64 | 64.79 | 72.65 | 89.56 |

### In distribution: CWQ (train CWQ, in-distribution)

| Method | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 |
|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) | 86.67 | 91.31 | 98.22 | 66.79 | 74.03 | 88.10 | 72.89 | 80.88 | 94.55 |
| ASE (on DDE) | 90.34 | 94.18 | 98.72 | 75.07 | 80.89 | 91.19 | 79.11 | 84.85 | 93.89 |

### OOD: CWQ → WebQSP (train CWQ, OOD)

| Method | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 |
|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) | 82.22 | 88.15 | 97.10 | 71.66 | 79.82 | 93.73 | 71.84 | 79.75 | 93.31 |
| ASE (on DDE) | 81.83 | 88.63 | 97.57 | 71.85 | 80.34 | 94.55 | 70.54 | 78.58 | 93.74 |

=> ASE trên DDE chỉ có ích khi train trên dữ liệu nhiều anchor (CWQ). Trên WL, ASE chỉ có ích khi có augmentation (WL + ASE + aug); WL + ASE không aug kém hơn WL (EXP-35).

## Tổ hợp

## AER + ADR on different backbone (ans, sp, gpt)@50-100-500

### In distribution: WebQSP (train WebQSP, in-distribution)

| Backbone | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 | Δ ansR@50 so với backbone gốc |
|---|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) + AER + ADR | 89.53 | 93.93 | 98.79 | 81.16 | 86.77 | 96.41 | 80.63 | 86.98 | 96.97 | -0.53 |
| QR-DDE + AER + ADR | 88.97 | 93.85 | 98.92 | 83.93 | 89.41 | 97.88 | 82.18 | 87.46 | 95.98 | -0.23 |
| DDE, AER-trained (reanchor_p 0.5) + AER + ADR | 89.37 | 93.85 | 98.81 | 81.18 | 86.80 | 96.47 | 80.79 | 86.74 | 97.08 | -0.69 |
| RT-combined (no weighted loss) + AER + ADR | 91.24 | 95.46 | 99.14 | 82.39 | 88.02 | 97.11 | 82.40 | 88.29 | 97.12 | -0.56 |
| RT + weighted loss (WL) + AER + ADR | 91.92 | 95.96 | 99.21 | 83.84 | 89.46 | 97.43 | 84.63 | 89.63 | 97.44 | -0.66 |
| ASE (on DDE) + AER + ADR | 88.92 | 93.98 | 99.14 | 82.02 | 88.24 | 97.30 | 81.32 | 87.21 | 96.46 | -1.18 |
| WL + ASE + AER + ADR | 91.94 | 95.53 | 99.24 | 84.73 | 90.22 | 97.70 | 83.80 | 89.25 | 97.08 | -0.89 |
| WL + aug + AER + ADR | 90.51 | 95.03 | 99.23 | 83.55 | 89.52 | 97.48 | 84.65 | 89.32 | 97.08 | -1.38 |
| WL + ASE + aug + AER + ADR | 92.19 | 95.92 | 99.19 | 84.43 | 89.27 | 97.32 | 84.72 | 89.45 | 96.77 | -0.53 |

### OOD: WebQSP → CWQ (train WebQSP, OOD / compositional)

| Backbone | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 | Δ ansR@50 so với backbone gốc |
|---|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) + AER + ADR | 78.58 | 85.79 | 95.98 | 54.04 | 63.91 | 83.36 | 59.20 | 68.68 | 87.53 | +8.54 |
| QR-DDE + AER + ADR | 75.84 | 85.66 | 96.39 | 58.33 | 67.02 | 85.09 | 63.92 | 73.25 | 90.05 | +14.88 |
| DDE, AER-trained (reanchor_p 0.5) + AER + ADR | 79.02 | 86.38 | 96.44 | 53.75 | 63.04 | 83.56 | 59.62 | 69.09 | 88.40 | +4.23 |
| RT-combined (no weighted loss) + AER + ADR | 82.37 | 89.20 | 97.75 | 57.11 | 66.63 | 86.38 | 63.47 | 72.47 | 90.08 | +9.96 |
| RT + weighted loss (WL) + AER + ADR | 83.18 | 89.82 | 98.00 | 56.83 | 66.63 | 87.40 | 65.59 | 74.90 | 91.65 | +9.13 |
| ASE (on DDE) + AER + ADR | 80.71 | 87.75 | 97.55 | 56.66 | 66.25 | 86.80 | 61.32 | 70.46 | 88.98 | +9.22 |
| WL + ASE + AER + ADR | 82.48 | 89.72 | 98.09 | 60.06 | 69.58 | 88.20 | 66.86 | 75.40 | 91.55 | +14.56 |
| WL + aug + AER + ADR | 80.47 | 89.20 | 97.39 | 56.27 | 66.65 | 87.48 | 64.49 | 74.26 | 91.62 | +3.54 |
| WL + ASE + aug + AER + ADR | 86.52 | 92.59 | 98.63 | 63.05 | 71.76 | 90.02 | 70.81 | 79.01 | 93.02 | +5.90 |

### In distribution: CWQ (train CWQ, in-distribution)

| Backbone | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 | Δ ansR@50 so với backbone gốc |
|---|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) + AER + ADR | 87.10 | 91.83 | 98.19 | 66.34 | 73.97 | 88.72 | 72.86 | 81.47 | 94.89 | +0.43 |
| QR-DDE + AER + ADR | 90.63 | 94.35 | 98.81 | 74.36 | 80.63 | 91.46 | 79.13 | 85.63 | 95.28 | +0.94 |
| RT-combined (no weighted loss) + AER + ADR | 87.04 | 92.53 | 98.32 | 64.42 | 73.46 | 88.89 | 72.72 | 81.36 | 94.93 | +0.83 |
| ASE (on DDE) + AER + ADR | 88.33 | 92.91 | 98.47 | 71.08 | 78.77 | 91.14 | 76.45 | 83.54 | 94.06 | -2.01 |

### OOD: CWQ → WebQSP (train CWQ, OOD)

| Backbone | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 | Δ ansR@50 so với backbone gốc |
|---|---|---|---|---|---|---|---|---|---|---|
| DDE (SubgraphRAG repro) + AER + ADR | 80.65 | 87.35 | 97.15 | 69.32 | 78.60 | 93.51 | 69.85 | 79.05 | 93.21 | -1.57 |
| QR-DDE + AER + ADR | 80.22 | 86.73 | 96.10 | 74.58 | 82.21 | 94.29 | 73.37 | 81.53 | 92.98 | -0.75 |
| RT-combined (no weighted loss) + AER + ADR | 83.65 | 90.27 | 97.60 | 70.40 | 79.86 | 94.38 | 71.19 | 79.90 | 93.79 | -1.32 |
| ASE (on DDE) + AER + ADR | 77.86 | 86.94 | 97.10 | 68.86 | 79.13 | 94.17 | 67.34 | 77.58 | 93.08 | -3.97 |

=> Cái nào tốt nhất:

- WebQSP (train WebQSP, in-distribution): ansR@50 cao nhất là **WL + ASE + aug + AER + ADR** (92.19); AER + ADR **không giúp backbone nào** (Δ tốt nhất -0.23, trên QR-DDE).
- WebQSP → CWQ (train WebQSP, OOD / compositional): ansR@50 cao nhất là **WL + ASE + aug + AER + ADR** (86.52); AER + ADR giúp nhiều nhất trên **QR-DDE** (+14.88).
- CWQ (train CWQ, in-distribution): ansR@50 cao nhất là **QR-DDE + AER + ADR** (90.63); AER + ADR giúp nhiều nhất trên **QR-DDE** (+0.94).
- CWQ → WebQSP (train CWQ, OOD): ansR@50 cao nhất là **RT-combined (no weighted loss) + AER + ADR** (83.65); AER + ADR **không giúp backbone nào** (Δ tốt nhất -0.75, trên QR-DDE).

## In-Class eval (CWQ test, ansR@50 / ansR@100)

### WebQSP → CWQ (train WebQSP, OOD / compositional)

| Method | composition (940) | conjunction (1526) | comparative (199) | superlative (183) | 0–1 hop (810) | 2 hop (1867) | ≥ 3 hop (171) |
|---|---|---|---|---|---|---|---|
| SubgraphRAG (repro, train WebQSP) | 54.6 / 66.0 | 78.1 / 87.2 | 75.5 / 82.0 | 76.0 / 83.1 | 94.5 / 98.4 | 60.5 / 72.2 | 59.0 / 70.8 |
| RAPL (author ckpt, webqsp.pt) | 30.4 / 38.6 | 84.4 / 91.0 | 85.7 / 93.1 | 86.9 / 94.5 | 89.1 / 93.4 | 57.1 / 65.1 | 66.7 / 80.6 |
| REL-RAG (path_bf) (reprod) | 26.0 / 37.4 | 77.6 / 87.9 | 84.7 / 92.8 | 90.2 / 98.4 | 88.5 / 94.4 | 49.6 / 61.6 | 70.2 / 83.9 |
| DDE (SubgraphRAG repro) + AER | 74.1 / 82.5 | 76.8 / 86.6 | 76.0 / 83.0 | 75.4 / 82.5 | 92.4 / 97.5 | 70.2 / 80.4 | 58.1 / 71.3 |
| DDE (SubgraphRAG repro) + AER + ADR | 74.9 / 83.4 | 81.6 / 88.3 | 75.9 / 81.2 | 75.4 / 82.0 | 94.9 / 97.8 | 73.5 / 81.7 | 57.2 / 73.5 |
| RT + weighted loss (WL) | 58.0 / 69.9 | 80.7 / 89.5 | 87.7 / 93.0 | 85.8 / 94.0 | 95.2 / 98.3 | 65.7 / 77.5 | 65.4 / 79.5 |
| RT + weighted loss (WL) + AER + ADR | 77.1 / 84.2 | 86.0 / 92.3 | 87.2 / 93.5 | 86.9 / 94.0 | 94.1 / 98.0 | 79.7 / 86.9 | 69.6 / 83.1 |
| ASE (on DDE) | 51.5 / 65.6 | 80.2 / 86.9 | 84.7 / 92.9 | 86.9 / 93.4 | 95.0 / 98.0 | 64.1 / 75.0 | 41.3 / 61.3 |
| ASE (on DDE) + AER + ADR | 69.0 / 79.9 | 86.5 / 91.4 | 87.0 / 91.5 | 85.8 / 93.4 | 96.0 / 98.7 | 76.0 / 84.7 | 59.4 / 69.3 |
| WL + ASE + aug | 54.9 / 66.1 | 94.0 / 96.9 | 90.7 / 95.8 | 90.7 / 95.1 | 98.4 / 99.5 | 74.4 / 81.8 | 64.7 / 77.2 |
| WL + ASE + aug + AER + ADR | 77.9 / 85.2 | 91.3 / 96.4 | 87.7 / 95.4 | 89.6 / 96.2 | 97.3 / 99.4 | 83.3 / 90.4 | 70.3 / 84.3 |

### CWQ (train CWQ, in-distribution)

| Method | composition (940) | conjunction (1526) | comparative (199) | superlative (183) | 0–1 hop (810) | 2 hop (1867) | ≥ 3 hop (171) |
|---|---|---|---|---|---|---|---|
| SubgraphRAG (official ckpt) | 83.5 / 89.9 | 89.8 / 93.7 | 75.9 / 80.4 | 81.4 / 92.9 | 91.2 / 94.7 | 86.6 / 91.6 | 58.1 / 74.1 |
| RAPL (author ckpt, cwq.pt) | 60.6 / 66.8 | 90.7 / 95.3 | 88.8 / 93.8 | 92.9 / 96.7 | 93.4 / 96.9 | 76.0 / 81.4 | 73.4 / 81.9 |
| DDE (SubgraphRAG repro) + AER | 83.7 / 88.3 | 89.8 / 93.6 | 75.1 / 81.9 | 83.6 / 89.6 | 91.2 / 94.1 | 86.8 / 91.3 | 59.5 / 70.2 |
| DDE (SubgraphRAG repro) + AER + ADR | 84.5 / 89.1 | 90.9 / 94.9 | 74.8 / 83.2 | 82.0 / 89.6 | 91.6 / 95.4 | 87.5 / 92.1 | 60.9 / 71.7 |
| ASE (on DDE) | 86.6 / 92.5 | 94.5 / 96.7 | 80.7 / 85.7 | 85.8 / 91.3 | 94.5 / 97.4 | 91.1 / 95.1 | 61.8 / 68.7 |
| ASE (on DDE) + AER + ADR | 86.2 / 91.5 | 92.4 / 95.7 | 74.8 / 82.7 | 79.8 / 87.4 | 92.4 / 95.5 | 89.0 / 93.8 | 62.1 / 71.1 |

# Benchmark overall retrieval phase

> Chú ý: ASE, AER và ADR ở dưới là trên backbone DDE (SubgraphRAG repro); các tổ hợp với WL ở phần "Bổ sung".

### Dataset: WebQSP (train WebQSP, in-distribution) (1559 samples)

| Method | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 |
|---|---|---|---|---|---|---|---|---|---|
| SubGraphRAG | 90.38 | 94.42 | 98.93 | 82.26 | 87.76 | 96.69 | 81.03 | 87.42 | 98.56 |
| RAPL (author ckpt) | 89.91 | 95.16 | 99.04 | 83.71 | 89.26 | 96.03 | 64.02 | 67.93 | 76.02 |
| REL-RAG (path_bf) (reprod) | 90.60 | 94.84 | 98.83 | 85.77 | 90.77 | 97.15 | 61.92 | 65.45 | 75.58 |
| REL-RAG (P, path_free) (reprod) | 84.91 | 88.78 | 90.93 | 80.53 | 84.04 | 86.45 | 53.95 | 55.69 | 57.05 |
| REL-RAG (T) (reprod) | 75.60 | 84.82 | 96.29 | 73.85 | 83.50 | 95.37 | 49.72 | 56.39 | 70.06 |
| QR-DDE | 89.20 | 93.98 | 99.14 | 84.60 | 89.96 | 98.02 | 82.15 | 87.23 | 95.86 |
| AER | 89.72 | 94.10 | 98.85 | 81.19 | 86.75 | 96.32 | 80.68 | 86.96 | 96.93 |
| ADR | 89.82 | 94.29 | 98.76 | 81.85 | 87.12 | 96.41 | 80.96 | 87.03 | 97.10 |
| AER + ADR | 89.53 | 93.93 | 98.79 | 81.16 | 86.77 | 96.41 | 80.63 | 86.98 | 96.97 |
| RT (topic relation + weighted loss) | 92.58 | 96.09 | 99.26 | 84.69 | 90.03 | 97.46 | 84.47 | 89.66 | 97.37 |
| RT + AER | 92.27 | 96.30 | 99.27 | 83.98 | 89.73 | 97.46 | 84.54 | 89.65 | 97.42 |
| RT + ADR | 92.32 | 95.92 | 99.20 | 84.50 | 89.82 | 97.40 | 84.40 | 89.60 | 97.40 |
| RT + AER + ADR | 91.92 | 95.96 | 99.21 | 83.84 | 89.46 | 97.43 | 84.63 | 89.63 | 97.44 |
| ASE | 90.10 | 94.92 | 99.18 | 82.99 | 89.39 | 97.37 | 80.56 | 86.92 | 96.55 |
| **Bổ sung** | | | | | | | | | |
| DDE (SubgraphRAG repro) | 90.07 | 94.41 | 98.88 | 81.85 | 87.09 | 96.40 | 81.02 | 86.93 | 97.06 |
| DDE, AER-trained (reanchor_p 0.5) + AER | 89.68 | 94.18 | 98.88 | 81.25 | 86.83 | 96.46 | 80.91 | 86.80 | 97.08 |
| RT-combined (no weighted loss) | 91.79 | 95.72 | 99.12 | 83.26 | 88.42 | 96.98 | 82.58 | 87.92 | 97.14 |
| RT-combined (no weighted loss) + AER + ADR | 91.24 | 95.46 | 99.14 | 82.39 | 88.02 | 97.11 | 82.40 | 88.29 | 97.12 |
| ASE (on DDE) + AER + ADR | 88.92 | 93.98 | 99.14 | 82.02 | 88.24 | 97.30 | 81.32 | 87.21 | 96.46 |
| WL + ASE | 92.84 | 96.13 | 99.29 | 85.59 | 90.93 | 97.74 | 84.05 | 88.89 | 97.01 |
| WL + ASE + AER + ADR | 91.94 | 95.53 | 99.24 | 84.73 | 90.22 | 97.70 | 83.80 | 89.25 | 97.08 |
| WL + aug | 91.88 | 96.01 | 99.27 | 84.87 | 90.33 | 97.57 | 84.76 | 89.23 | 97.21 |
| WL + aug + AER + ADR | 90.51 | 95.03 | 99.23 | 83.55 | 89.52 | 97.48 | 84.65 | 89.32 | 97.08 |
| WL + ASE + aug | 92.72 | 96.21 | 99.18 | 84.62 | 89.48 | 97.30 | 82.72 | 87.38 | 95.89 |
| WL + ASE + aug + AER | 92.33 | 95.99 | 99.19 | 84.51 | 89.41 | 97.40 | 84.70 | 89.43 | 96.72 |
| WL + ASE + aug + ADR | 92.34 | 96.08 | 99.18 | 84.46 | 89.28 | 97.22 | 82.51 | 87.36 | 95.90 |
| WL + ASE + aug + AER + ADR | 92.19 | 95.92 | 99.19 | 84.43 | 89.27 | 97.32 | 84.72 | 89.45 | 96.77 |

### Dataset: WebQSP → CWQ (train WebQSP, OOD / compositional) (2848 samples)

| Method | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 |
|---|---|---|---|---|---|---|---|---|---|
| SubGraphRAG | 70.05 | 79.58 | 94.81 | 49.77 | 58.65 | 80.51 | 53.59 | 63.05 | 84.53 |
| RAPL (author ckpt) | 66.82 | 74.09 | 85.13 | 54.40 | 60.71 | 72.91 | 39.39 | 44.57 | 54.68 |
| REL-RAG (path_bf) (reprod) | 61.90 | 72.26 | 85.21 | 50.82 | 58.89 | 73.40 | 34.35 | 40.80 | 53.18 |
| REL-RAG (P, path_free) (reprod) | 51.24 | 60.21 | 66.32 | 42.62 | 48.98 | 54.45 | 26.53 | 30.53 | 34.00 |
| REL-RAG (T) (reprod) | 47.76 | 60.42 | 90.42 | 32.15 | 40.64 | 67.11 | 21.44 | 27.51 | 48.91 |
| QR-DDE | 60.95 | 71.29 | 91.70 | 49.51 | 57.51 | 78.83 | 53.06 | 62.06 | 84.24 |
| AER | 75.78 | 84.76 | 96.03 | 51.68 | 61.18 | 81.81 | 55.75 | 65.57 | 86.05 |
| ADR | 72.49 | 80.52 | 94.81 | 51.64 | 60.36 | 81.27 | 56.87 | 65.96 | 85.54 |
| AER + ADR | 78.58 | 85.79 | 95.98 | 54.04 | 63.91 | 83.36 | 59.20 | 68.68 | 87.53 |
| RT (topic relation + weighted loss) | 74.04 | 83.56 | 96.31 | 50.89 | 60.58 | 83.60 | 57.65 | 67.59 | 88.22 |
| RT + AER | 80.27 | 88.45 | 97.76 | 53.41 | 63.10 | 85.03 | 61.72 | 71.88 | 89.91 |
| RT + ADR | 77.37 | 85.37 | 96.05 | 53.06 | 62.73 | 84.71 | 61.06 | 70.60 | 89.35 |
| RT + AER + ADR | 83.18 | 89.82 | 98.00 | 56.83 | 66.63 | 87.40 | 65.59 | 74.90 | 91.65 |
| ASE | 71.49 | 80.72 | 95.66 | 50.95 | 60.29 | 82.37 | 50.90 | 60.66 | 83.04 |
| **Bổ sung** | | | | | | | | | |
| DDE (SubgraphRAG repro) | 70.05 | 79.58 | 94.81 | 49.77 | 58.65 | 80.51 | 53.59 | 63.05 | 84.53 |
| DDE, AER-trained (reanchor_p 0.5) + AER | 77.95 | 85.18 | 96.78 | 52.01 | 61.68 | 82.35 | 56.86 | 66.30 | 87.49 |
| RT-combined (no weighted loss) | 72.41 | 80.31 | 95.51 | 51.11 | 60.30 | 82.70 | 54.71 | 63.65 | 86.71 |
| RT-combined (no weighted loss) + AER + ADR | 82.37 | 89.20 | 97.75 | 57.11 | 66.63 | 86.38 | 63.47 | 72.47 | 90.08 |
| ASE (on DDE) + AER + ADR | 80.71 | 87.75 | 97.55 | 56.66 | 66.25 | 86.80 | 61.32 | 70.46 | 88.98 |
| WL + ASE | 67.92 | 78.30 | 95.25 | 51.71 | 60.64 | 82.88 | 54.85 | 63.22 | 85.47 |
| WL + ASE + AER + ADR | 82.48 | 89.72 | 98.09 | 60.06 | 69.58 | 88.20 | 66.86 | 75.40 | 91.55 |
| WL + aug | 76.93 | 85.24 | 97.10 | 51.95 | 62.41 | 85.04 | 57.92 | 68.31 | 89.01 |
| WL + aug + AER + ADR | 80.47 | 89.20 | 97.39 | 56.27 | 66.65 | 87.48 | 64.49 | 74.26 | 91.62 |
| WL + ASE + aug | 80.62 | 86.57 | 96.31 | 58.51 | 66.59 | 86.64 | 64.79 | 72.65 | 89.56 |
| WL + ASE + aug + AER | 87.38 | 92.38 | 98.44 | 64.16 | 72.54 | 89.65 | 71.73 | 79.57 | 92.59 |
| WL + ASE + aug + ADR | 79.54 | 86.73 | 96.45 | 54.65 | 64.16 | 86.05 | 62.03 | 71.42 | 89.31 |
| WL + ASE + aug + AER + ADR | 86.52 | 92.59 | 98.63 | 63.05 | 71.76 | 90.02 | 70.81 | 79.01 | 93.02 |

### Dataset: CWQ (train CWQ, in-distribution) (2848 samples)

| Method | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 |
|---|---|---|---|---|---|---|---|---|---|
| SubGraphRAG | 86.20 | 91.43 | 98.36 | 66.28 | 74.47 | 88.16 | 75.54 | 84.42 | 98.34 |
| RAPL (author ckpt) | 80.76 | 85.87 | 92.65 | 63.55 | 69.50 | 81.41 | 45.05 | 49.28 | 58.67 |
| REL-RAG (path_bf) (reprod) | — | — | — | — | — | — | — | — | — |
| REL-RAG (P, path_free) (reprod) | — | — | — | — | — | — | — | — | — |
| REL-RAG (T) (reprod) | — | — | — | — | — | — | — | — | — |
| QR-DDE | 89.69 | 93.17 | 98.51 | 75.71 | 80.90 | 90.62 | 79.60 | 85.24 | 94.45 |
| AER | 86.39 | 90.80 | 98.16 | 65.14 | 73.11 | 88.26 | 71.01 | 80.20 | 94.41 |
| ADR | 87.56 | 91.89 | 98.31 | 66.44 | 74.02 | 88.40 | 73.20 | 81.04 | 94.76 |
| AER + ADR | 87.10 | 91.83 | 98.19 | 66.34 | 73.97 | 88.72 | 72.86 | 81.47 | 94.89 |
| RT (topic relation + weighted loss) | — | — | — | — | — | — | — | — | — |
| RT + AER | — | — | — | — | — | — | — | — | — |
| RT + ADR | — | — | — | — | — | — | — | — | — |
| RT + AER + ADR | — | — | — | — | — | — | — | — | — |
| ASE | 90.34 | 94.18 | 98.72 | 75.07 | 80.89 | 91.19 | 79.11 | 84.85 | 93.89 |
| **Bổ sung** | | | | | | | | | |
| DDE (SubgraphRAG repro) | 86.67 | 91.31 | 98.22 | 66.79 | 74.03 | 88.10 | 72.89 | 80.88 | 94.55 |
| RT-combined (no weighted loss, stopped ep 27) | 86.21 | 91.99 | 98.15 | 64.02 | 72.93 | 88.10 | 72.30 | 81.26 | 94.83 |
| RT-combined (no weighted loss, stopped ep 27) + AER + ADR | 87.04 | 92.53 | 98.32 | 64.42 | 73.46 | 88.89 | 72.72 | 81.36 | 94.93 |
| ASE (on DDE) + AER + ADR | 88.33 | 92.91 | 98.47 | 71.08 | 78.77 | 91.14 | 76.45 | 83.54 | 94.06 |

### Dataset: CWQ → WebQSP (train CWQ, OOD) (1559 samples)

| Method | ansR@50 | ansR@100 | ansR@500 | SP-R@50 | SP-R@100 | SP-R@500 | GPT-R@50 | GPT-R@100 | GPT-R@500 |
|---|---|---|---|---|---|---|---|---|---|
| SubGraphRAG | 82.22 | 88.15 | 97.10 | 71.66 | 79.82 | 93.73 | 71.84 | 79.75 | 93.31 |
| RAPL (author ckpt) | 90.85 | 95.77 | 99.09 | 83.74 | 89.78 | 96.67 | 62.28 | 65.56 | 73.28 |
| REL-RAG (path_bf) (reprod) | — | — | — | — | — | — | — | — | — |
| REL-RAG (P, path_free) (reprod) | — | — | — | — | — | — | — | — | — |
| REL-RAG (T) (reprod) | — | — | — | — | — | — | — | — | — |
| QR-DDE | 80.97 | 87.21 | 96.30 | 76.30 | 83.04 | 94.61 | 74.50 | 81.96 | 93.06 |
| AER | 80.67 | 87.24 | 97.03 | 69.01 | 78.24 | 93.31 | 69.53 | 78.69 | 93.08 |
| ADR | 82.13 | 88.20 | 97.16 | 71.92 | 80.14 | 93.84 | 72.05 | 80.02 | 93.42 |
| AER + ADR | 80.65 | 87.35 | 97.15 | 69.32 | 78.60 | 93.51 | 69.85 | 79.05 | 93.21 |
| RT (topic relation + weighted loss) | — | — | — | — | — | — | — | — | — |
| RT + AER | — | — | — | — | — | — | — | — | — |
| RT + ADR | — | — | — | — | — | — | — | — | — |
| RT + AER + ADR | — | — | — | — | — | — | — | — | — |
| ASE | 81.83 | 88.63 | 97.57 | 71.85 | 80.34 | 94.55 | 70.54 | 78.58 | 93.74 |
| **Bổ sung** | | | | | | | | | |
| DDE (SubgraphRAG repro) | 82.22 | 88.15 | 97.10 | 71.66 | 79.82 | 93.73 | 71.84 | 79.75 | 93.31 |
| RT-combined (no weighted loss, stopped ep 27) | 84.97 | 90.76 | 97.57 | 72.79 | 80.39 | 94.66 | 72.78 | 80.54 | 93.90 |
| RT-combined (no weighted loss, stopped ep 27) + AER + ADR | 83.65 | 90.27 | 97.60 | 70.40 | 79.86 | 94.38 | 71.19 | 79.90 | 93.79 |
| ASE (on DDE) + AER + ADR | 77.86 | 86.94 | 97.10 | 68.86 | 79.13 | 94.17 | 67.34 | 77.58 | 93.08 |

Ô "—": chưa có export. Không có REL-RAG repro train trên CWQ; không có WL (RT + weighted loss) train trên CWQ, chỉ có RT-combined (không weighted loss, dừng ở epoch 27) ở mục Bổ sung.

# Benchmark QA task

> Llama-3.1-8B-Instruct, prompt `sys_icl_dc` của SubgraphRAG, chấm như paper (Hit theo eval gốc, Macro-F1 theo eval corrected, match gốc). Chỉ điền số đã chạy; mọi retriever train trên WebQSP trừ khi ghi khác.

### WebQSP (in-distribution; EXP-25, fp16 vLLM, T4)

| Method | Macro-F1 | Hit | Ghi chú |
|---|---|---|---|
| SubGraphRAG (official @100, 1,628 câu) | 70.57 | 86.61 | paper 70.5 / 86.6 |
| RAPL (author ckpt) | — | — | chưa chạy |
| REL-RAG (path_bf @50) (reprod, 1,628 câu) | 70.15 | 84.95 | paper REL-RAG 77.3 / 88.8 |
| REL-RAG (P) / (T) (reprod) | — | — | chưa chạy |
| QR-DDE | — | — | chưa chạy |
| AER (trained, @50, 1,559 câu chung) | 69.83 | 86.40 | cùng tập: SubgraphRAG @100 72.10 / 88.58; REL-RAG P 71.41 / 86.59 |
| ADR, AER + ADR, RT, RT + …, ASE | — | — | chưa chạy (retrieval in-dist ngang baseline, gate không yêu cầu) |

### WebQSP → CWQ (bf16 vLLM, RTX 4090; EXP-26 / EXP-30)

| Method | Macro-F1 | Hit | Tập câu |
|---|---|---|---|
| *SubgraphRAG official @100, train CWQ (trần)* | *47.47 / 47.06* | *58.20 / 56.87* | 1,000 đầu / 3,531 |
| SubGraphRAG (DDE repro, train WebQSP) | 35.69 | 48.50 | 1,000 đầu |
| RAPL (author ckpt) | — | — | chưa chạy |
| REL-RAG (P @50) (reprod) | 34.07 | 45.10 | 1,000 đầu |
| DDE + neo lại (không train) | 37.27 | 48.90 | 1,000 đầu |
| AER (trained) | 36.90 | 49.60 | 1,000 đầu |
| RT + AER + ADR (WL) | 40.94 / 42.84 | 51.80 / 54.18 | 1,000 đầu / 3,531 |
| ADR, AER + ADR (DDE), ASE, WL + ASE + aug | — | — | chưa chạy |

Bootstrap Hit (1,000 câu): WL + AER + ADR − REL-RAG P = +6.70 [+3.20, +10.30]; − DDE = +3.30 [+0.10, +6.60]; − AER = +2.20 [−0.90, +5.40].

# General evaluation (more dataset): GrailQA

> **Cảnh báo (EXP-37):** tập GrailQA đã chấm (897 câu từ `graliqa.json` của ODA, tập **dev**) **không phải** tập 874 câu của paper (GrailQA **train**, top-5 domain People / Biology / Computer / Religion / Digicams). Số dưới chỉ là phép thử OOD tạm, không so với paper. Tập đúng đã dựng lại được (995 câu trước lọc) nhưng chưa embed.

| WebQSP → GrailQA-ODA-dev (897 câu), ansR@50 / @100 / @200 | Tất cả | 1-hop (645) | 2-hop (219) | 3-hop (33) | zero-shot (513) |
|---|---|---|---|---|---|
| SubgraphRAG repro | 71.7 / 79.1 / 86.9 | 83.5 / 89.1 / 94.2 | 43.2 / 57.1 / 73.4 | 30.5 / 30.5 / 33.7 | 71.8 / 80.2 / 89.9 |
| REL-RAG (path_bf) repro | 70.0 / 76.5 / 79.8 | 81.5 / 87.0 / 89.6 | 42.7 / 52.4 / 58.6 | 27.4 / 30.5 / 30.5 | 68.0 / 75.6 / 78.3 |
| DDE + AER + ADR | 72.1 / 79.4 / 87.3 | 82.6 / 88.6 / 94.2 | 46.3 / 59.0 / 73.7 | 36.6 / 36.7 / 42.8 | 71.9 / 79.8 / 90.2 |
| RT + weighted loss (WL) | 83.2 / 87.0 / 91.8 | 94.0 / 96.4 / 98.1 | 59.6 / 67.8 / 79.4 | 27.6 / 31.1 / 50.7 | 85.0 / 88.9 / 93.2 |
| WL + AER + ADR | 83.4 / 89.2 / 93.0 | 93.7 / 96.3 / 98.0 | 60.3 / 76.0 / 84.9 | 36.8 / 37.4 / 47.8 | 85.0 / 90.8 / 94.4 |

**GTSQA (Wikidata, 1,622 câu) và MetaQA-3hop (4,710 câu), train WebQSP (EXP-24, chỉ họ DDE):**

| ansR@50 | GTSQA 1 / 2 / 3 / 4 hop | GTSQA unseen graph type (767) | MetaQA-3hop ansR@20 / @50 / @100 |
|---|---|---|---|
| DDE | 44.1 / 30.3 / 33.9 / 20.0 | 27.6 | 12.2 / 21.0 / 33.5 |
| DDE 2 lượt | 38.4 / 34.6 / 33.4 / 18.7 | 27.3 | 12.4 / 21.4 / 34.3 |
| AER (2 lượt) | 37.0 / 32.0 / 33.8 / 18.2 | 26.5 | 9.3 / 18.9 / 32.1 |
| QR-DDE | 50.6 / 41.4 / 42.2 / 21.3 | 34.7 | 14.9 / 21.7 / 30.8 |
| SubgraphRAG / RAPL / REL-RAG / WL / ASE | — | — | — (chưa chạy; WL cần thêm nhánh relation-topic vào `ext_eval.py`) |

### *(Lý tưởng sau này) Ma trận transfer đầy đủ* — trạng thái hiện tại

| Train ↓ / Test → | WebQSP | CWQ | GrailQA (tập paper, 874) | GTSQA (Wikidata) | MetaQA-3hop |
|---|---|---|---|---|---|
| WebQSP | ✅ retrieval (trên); QA một phần | ✅ retrieval; QA một phần | ⏳ tập đúng chưa embed (chỉ có ODA-dev) | ◐ chỉ họ DDE | ◐ chỉ họ DDE |
| CWQ | ✅ retrieval; QA chưa | ✅ retrieval; QA chỉ trần SubgraphRAG | ⏳ | ⏳ | ⏳ |
