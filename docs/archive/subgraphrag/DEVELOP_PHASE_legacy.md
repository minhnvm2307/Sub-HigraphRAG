# KNOWLEDGE GRAPH RETRIEVAL - SUBGRAPHRAG PROJECT ALIGNMENT

## 1. Project Overview
Dự án tập trung vào bài toán Knowledge Graph Question Answering (KGQA), cốt lõi là cải tiến phương pháp SubGraphRAG. Mục tiêu tối thượng là tìm ra đồ thị con $G_{q} \subseteq G$ tối ưu nhất cho câu hỏi $q$ để cung cấp cho LLM.

## Phase 1: Baseline SubgraphRAG (Lightweight MLP + DDE)

### Features:
- **Semantic**: Embedding (query, head, tail, relation)
- **Structural**: DDE (Triple quan trọng nếu gần topic entity/nằm trên path reasoning)

Input vector:
$$h_{input} = [z_q || z_h || z_t || PE_{topic} || DDE || z_{r}]$$
Trong đó:
- z_q: embedding của query/question
- z_h, z_t: head_entity/tail_entity embedding 
- PE: onehot_encode: có phải là topic entity hay không
- DDE: mã hóa khoảng cách entity đến toptic entity
- z_relation

### Results:

#### Retrievel Metrics: 
Dataset: WebQSp

  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.901                        0.828              0.801
100       0.942                        0.888              0.863
200       0.970                        0.934              0.919
400       0.987                        0.968              0.959

 K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 5       0.661                        0.530              0.491
10       0.755                        0.641              0.614
20       0.828                        0.729              0.706
30       0.863                        0.777              0.752

#### Generalization (Eval on CWQ):
- Train dataset: webqsp
- Eval dataset: cwq

 K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 5       0.293                        0.211              0.241
10       0.406                        0.293              0.326
20       0.536                        0.390              0.415
30       0.608                        0.449              0.468

  K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
 50       0.685                        0.524              0.536
100       0.787                        0.623              0.636
200       0.871                        0.736              0.735
400       0.929                        0.836              0.829 

### Reviews

Score flattening:

Examples:
```
ID: WebQTest-832_c334509bb5e02cacae1ba2e80c176499
Query: Lou Seal is the mascot for the team that last won the World Series when?
Retrieved Triples:
  1. (San Francisco Giants) --[sports.sports_team.team_mascot]--> (Lou Seal)  score=0.642407
  2. (Lou Seal) --[sports.mascot.team]--> (San Francisco Giants)  score=0.002309
  3. (Lou Seal) --[common.topic.image]--> (Lou-Seal.jpg)  score=0.000245
  4. (Lou Seal) --[common.topic.image]--> (Lou seal giants mascot)  score=0.000025
  5. (Lou-Seal.jpg) --[common.image.appears_in_topic_gallery]--> (Lou Seal)  score=0.000004
  6. (Lou Seal) --[common.topic.notable_types]--> (Mascot)  score=0.000002
  7. (San Francisco Giants) --[sports.sports_team.championships]--> (2014 World Series)  score=0.000001
  8. (Lou seal giants mascot) --[common.image.appears_in_topic_gallery]--> (Lou Seal)  score=0.000001
  9. (San Francisco Giants) --[sports.sports_team.championships]--> (2012 World Series)  score=0.000001
  10. (San Francisco Giants) --[sports.sports_team.season_record]--> (m.0dl3192)  score=0.000000
  11. (San Francisco Giants) --[sports.sports_team.league]--> (m.0crt4b6)  score=0.000000
  12. (San Francisco Giants) --[sports.sports_team.team_mascot]--> (Crazy Crab)  score=0.000000
  13. (San Francisco Giants) --[sports.sports_team.league]--> (m.0crtd80)  score=0.000000
...Từ 14 đến 400: Score = 0.0000
```

## Phase 2: Comparing MLP retrieve vs Embedding only retrieve:
Embedding only: Embedding triples to vector and get similarity score and select topk.
MLP: MLP + DDE

### Results:

#### MLP
**Evaluate trên (cwq)**
 K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
30       0.608                        0.449              0.468
40       0.650                        0.491              0.505
50       0.685                        0.524              0.536
70       0.732                        0.570              0.582

**Evaluate trên (webqsp)**
 K  ans_recall  shortest_path_triple_recall  gpt_triple_recall  
30       0.863                        0.777              0.752  
40       0.884                        0.807              0.776  
50       0.901                        0.828              0.801  
70       0.925                        0.860              0.835 


### Embedding only:

# Cwq
 K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
30       0.502                        0.328              0.334
40       0.543                        0.360              0.373
50       0.578                        0.382              0.393
70       0.629                        0.419              0.425

# Webqsp
 K  ans_recall  shortest_path_triple_recall  gpt_triple_recall
30       0.575                        0.470              0.463
40       0.613                        0.498              0.487
50       0.648                        0.522              0.507
70       0.697                        0.559              0.536

### Review:
MLP still much better though the flattening score is still high.

## Phase 3: Deep dive to analyze flattening score and improve it by merge method with HiRAG (hirachical local-global merge)

### Score analysis:
```
## Top K = 500: Phân tích kết quả reasoning của LLM (Meta-Llama3.1)
khi sử dụng 500 triples làm input

Reasoning-path coverage summary
--------------------------------------------------------------------------------------
     K     avg_cov   micro_cov    avg_prec   micro_prec   hit_rate     noise
--------------------------------------------------------------------------------------
     5      33.43%       7.33%      34.01%       34.01%     78.11%    65.99%
    10      42.30%      11.38%      26.41%       26.40%     85.17%    73.59%
    20      50.83%      16.87%      19.59%       19.58%     90.76%    80.41%
    30      55.14%      20.94%      16.22%       16.20%     92.43%    83.78%
    40      58.20%      24.16%      14.04%       14.02%     93.65%    85.96%
--------------------------------------------------------------------------------------

Reasoning-path coverage summary
--------------------------------------------------------------------------------------
     K     avg_cov   micro_cov    avg_prec   micro_prec   hit_rate     noise
--------------------------------------------------------------------------------------
    50      60.70%      27.01%      12.56%       12.54%     94.48%    87.44%
   100      68.50%      37.42%       8.70%        8.69%     96.79%    91.30%
   200      75.35%      49.23%       5.72%        5.72%     97.88%    94.28%
   300      79.15%      56.19%       4.36%        4.36%     98.40%    95.64%
   400      81.60%      60.56%       3.52%        3.53%     98.97%    96.48%
   500      83.58%      63.79%       2.98%        2.99%     99.10%    97.02%
--------------------------------------------------------------------------------------
Evaluated samples: 1558 | 
Skipped without reasoning paths: 70 | 
Skipped without scored triples: 0
```

#### Số lượng triples trung bình/câu hỏi được LLM sử dụng (useful triples)
Avg reasoning triples per sample: 23.20

#### Phân tích Rank của các entity quan trọng trong topK
Answer-entity rank stats: 
	found_rate=58.70%, 
	avg_best_rank=40.61, 
	median_best_rank=17.00, 
	avg_best_score=0.6060

Nhận xét:
Dù số lượng triples hữu ích cho LLM trả lời chỉ trung bình khoảng 23 triples nhưng cần top K ~ 400,500 để cover gần hết 

### Detailed analysis of score flattening: 
python reason/metrics/thinking_topk_analysis.py --dataset_name webqsp 

```

Coverage used_triples <-> input_triples
--------------------------------------------------------------------------------------
method       K    samples   used_cov  input_cov  hit      avg_rank
--------------------------------------------------------------------------------------
MLP          100    1558    36.99%     8.44%    96.73%      33.64
MLP          200    1558    48.50%     5.55%    97.88%      60.24
MLP          300    1558    55.58%     4.24%    98.27%      83.85
MLP          400    1558    60.29%     3.46%    98.72%     104.41
-------------------------------------------------------------------------------
EMB          100    1558    25.09%     5.79%    86.91%      39.20
EMB          200    1558    36.20%     4.18%    93.52%      72.18
EMB          300    1558    44.06%     3.39%    96.21%     103.30
EMB          400    1558    49.08%     2.84%    97.69%     128.51

Embedding similarity stats for used triples from scored_400
  triple_question: n=35082 mean=0.6206 median=0.6313
  triple_q_entity: n=35082 mean=0.7882 median=0.8338
  head_question: n=35082 mean=0.5641 median=0.6557
  head_q_entity: n=35082 mean=0.7645 median=1.0000
  tail_question: n=35082 mean=0.4538 median=0.4775
  tail_q_entity: n=35082 mean=0.5592 median=0.5842

Sample used triples with score/rank

[WebQTest-19] used=1 | what is my timezone in louisiana
  Louisiana -> location.location.time_zones -> Central Time Zone | MLP(score=0.9998, rank=1) | EMB(score=0.8217, rank=1)

[WebQTest-1083] used=3 | where president obama was born
  Barack Obama -> people.person.place_of_birth -> Honolulu | MLP(score=0.9897, rank=1) | EMB(score=0.7759, rank=3)
  Barack Obama -> influence.influence_node.influenced_by -> Craig McKenzie | MLP(score=0.0000, rank=415) | EMB(score=0.6132, rank=2483)
  Craig McKenzie -> people.person.place_of_birth -> Chicago | MLP(score=0.0002, rank=65) | EMB(score=0.7004, rank=175)

[WebQTest-131] used=9 | who inspired obama
  Barack Obama -> influence.influence_node.influenced -> Reinhold Niebuhr | MLP(score=0.8173, rank=3) | EMB(score=0.6673, rank=201)
  Barack Obama -> influence.influence_node.influenced -> Saul Alinsky | MLP(score=0.5868, rank=5) | EMB(score=0.6881, rank=74)
  Barack Obama -> influence.influence_node.influenced_by -> Craig McKenzie | MLP(score=0.0619, rank=17) | EMB(score=0.6438, rank=685)
  Barack Obama -> influence.influence_node.influenced_by -> Nipsey Russell | MLP(score=0.8069, rank=4) | EMB(score=0.6968, rank=45)
  Barack Obama -> people.person.profession -> Author | MLP(score=NA, rank=NA) | EMB(score=0.6177, rank=1344)
  Barack Obama -> people.person.profession -> Politician | MLP(score=0.0024, rank=133) | EMB(score=0.6057, rank=1621)
  Barack Obama -> people.person.profession -> Law professor | MLP(score=NA, rank=NA) | EMB(score=0.5899, rank=1991)
  Barack Obama -> people.person.profession -> Lawyer | MLP(score=0.0005, rank=494) | EMB(score=0.5890, rank=2010)
  Barack Obama -> people.person.profession -> Writer | MLP(score=0.0009, rank=279) | EMB(score=0.6159, rank=1394)
```

### Merge with HiRAG
indexing_hirag.py and main_hirag.py

#### Results:
```
Config	Hit@1	Macro F1	Exact Match	Totally Wrong	Micro F1
hi_global	69.57	56.32	43.48 (best)	30.43	0.3366
hi_bridge	73.91 (best)	58.83	34.78	21.74 (best)	0.3962
hi_bridge + MLP score	69.57	60.51 (best)	43.48 (best)	26.09	0.5225 (best)
```

```
	Hit (%)	F1 (Macro)	Recall (Macro)	Total Match (%)	Totally Wrong (%)
Hi Bridge	62.11	52.09	49.78	35.79	23.16
SubgraphRAG	91.58	78.54	80.43	60.00	3.16
Hi Bridge + Tăng top-k	61.05	52.23	50.00	37.89	26.32
Hi (bridge + global)	64.21	52.93	52.04	35.79	21.05
Tăng top-k + Hi (bridge + global)	63.16	52.20	50.73	35.79	23.16
Hi + topk=100 + SubgraphRAG	81.05	68.87	66.88	53.68	11.58
```

#### Review:
Not improved, Decided to not improve this retreival phase by combine with hirachical because of waste of token usage and time-comsuming (ich not the purpose of initial paper)


## Phase 4: Integrating Relation Topic Retrieval (Direct MLP Injection & Parallel Weighted DDE)

### Concept & Design Decisions
To address retrieval stagnation, we designed a dual-path mechanism using **Relation Topics**:
1. **Direct MLP Injection:** Re-encodes the relation topic similarity ranking into a 2D one-hot vector indicating whether a relation is in the Top-K (`[0, 1]` for yes, `[1, 0]` for no) and concatenates it directly to the triple representation.
2. **Parallel DDE Propagation:** Re-weights GNN distance encoding message-passing. The GNN runs two parallel propagations:
   - *Structural DDE (Topo DDE):* Traditional unweighted message passing.
   - *Semantic DDE:* Propagation weighted by the semantic-lexical similarity score of the relation, clamped to a minimum of 0.1 fallback to prevent blocking graph signal flow.

### Experimental Setup
Conducted systematic A/B comparisons across four experimental configurations on WebQSP dataset:
- **Baseline (Path 0):** RT_MLP: False, RT_DDE: False
- **MLP-Only (Path 1):** RT_MLP: True, RT_DDE: False
- **DDE-Only (Path 2):** RT_MLP: False, RT_DDE: True
- **Combined (Path 3):** RT_MLP: True, RT_DDE: True

### Results (WebQSP)

#### 1. Answer Recall (ANS_RECALL)
| Run / Path | RT_MLP | RT_DDE | K=50 | K=100 | K=200 | K=300 | K=400 | K=500 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Path 0)** | False | False | 0.901 | 0.942 | 0.970 | 0.982 | 0.987 | 0.989 |
| **MLP-Only (Path 1)** | True | False | 0.891 | 0.937 | 0.965 | 0.974 | 0.983 | 0.985 |
| **DDE-Only (Path 2)** | False | True | 0.887 | 0.940 | 0.969 | 0.979 | 0.986 | 0.989 |
| **Combined (Path 3)** | True | True | **0.919** | **0.956** | **0.976** | **0.986** | **0.989** | **0.991** |

#### 2. Shortest Path Recall (SHORTEST_PATH_RECALL)
| Run / Path | RT_MLP | RT_DDE | K=50 | K=100 | K=200 | K=300 | K=400 | K=500 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Path 0)** | False | False | 0.828 | 0.888 | 0.934 | 0.955 | 0.968 | 0.975 |
| **MLP-Only (Path 1)** | True | False | 0.814 | 0.876 | 0.926 | 0.948 | 0.960 | 0.968 |
| **DDE-Only (Path 2)** | False | True | 0.823 | 0.885 | 0.932 | 0.954 | 0.966 | 0.974 |
| **Combined (Path 3)** | True | True | **0.839** | **0.892** | **0.937** | **0.958** | **0.970** | **0.976** |

#### 3. GPT Triple Recall (GPT_TRIPLE_RECALL)
| Run / Path | RT_MLP | RT_DDE | K=50 | K=100 | K=200 | K=300 | K=400 | K=500 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Path 0)** | False | False | 0.801 | 0.863 | 0.919 | 0.941 | 0.959 | 0.970 |
| **MLP-Only (Path 1)** | True | False | 0.793 | 0.855 | 0.903 | 0.930 | 0.948 | 0.959 |
| **DDE-Only (Path 2)** | False | True | 0.792 | 0.859 | 0.908 | 0.937 | 0.950 | 0.963 |
| **Combined (Path 3)** | True | True | **0.817** | **0.871** | **0.924** | **0.947** | **0.960** | **0.968** |

### Key Findings
1. **The Synergy of Combined Method:** Individually, MLP-only injection or DDE-only weighting slightly decreases performance compared to baseline. However, when combined, they act synergistically to significantly improve Answer Recall, Shortest Path Recall, and GPT Triple Recall across all K values (except K=500 for GPT Triple Recall where it is slightly lower).
2. **Answer Recall Gains:** Combined model improves Answer Recall @ 100 by **+1.4%** (reaching **95.6%**) and Recall @ 50 by **+1.8%** (reaching **91.9%**).
3. **Reasoning Path Coverage:** Shortest Path Recall @ 50 improved by **+1.1%** (reaching **83.9%**).
4. **Generalization & Success:** The dual-path mechanism successfully broke through the baseline retrieval limit, offering high-fidelity subgraph candidate generation for multi-hop reasoning.

### Score Flattening Analysis (Relation-Topic vs Baseline)
To evaluate the impact of the Relation-Topic (RT) mechanism on the score flattening phenomenon, we ran comparative distribution analyses on the top 400 retrieved triples for each query:

| Metric | Baseline | RT (MLP-Only) | RT (DDE-Only) | RT (Combined) |
| :--- | :---: | :---: | :---: | :---: |
| **High Scores (> 0.2)** | 5.18% | 6.11% | 5.76% | 6.20% |
| **Medium Scores (0.01 - 0.2)** | 16.19% | 13.93% | 9.93% | 8.65% |
| **Near-zero Scores (<= 0.01)** | 78.63% | 79.97% | 84.30% | 85.15% |
| **Avg Unique Scores (per query)** | N/A | **298.33** | **328.87** | **325.44** |
| **Avg Pairwise Difference** | N/A | 0.050579 | 0.045796 | 0.052377 |
| **Avg Entropy** | N/A | 0.532266 | 0.470271 | 0.442159 |

**Key Takeaways:**
1. **Suppression of Noise:** The Near-zero scores (<= 0.01) increased from **78.63% (Baseline)** to **85.15% (Combined)**. This indicates that the Relation-Topic mechanism successfully pushes more irrelevant background triples to near-zero, acting as a cleaner semantic filter.
2. **Selective Enrichment of Signals:** The High scores (> 0.2) increased from **5.18% (Baseline)** to **6.20% (Combined)**. This shows that the Combined model successfully concentrates more probability mass on high-confidence, relevant reasoning paths.
3. **High Fine-grained Resolution:** In all RT configurations, the average number of unique scores per query is extremely high (**325.44** for Combined and **328.87** for DDE-Only, out of 400 triples). This indicates extremely high scoring resolution and minimal duplicate flat scores.




## Phase 5: LangGraph Migration, LangSmith Observability, and Metrics Refinement

This phase just refactor QA-task code with useful tools for monitoring

### Experimental Results (WebQSP - scored_100)
Evaluation of the corrected LangGraph pipeline against the baseline on the WebQSP dataset under `scored_100` mode:

| Metric | Baseline | LangGraph (Ollama:llama) | Delta |
| :--- | :---: | :---: | :---: |
| **Hit@1** | 84.80% | **89.93%** | **+5.13%** |
| **Macro F1** | 72.10% | **74.96%** | **+2.86%** |
| **Macro Precision** | 80.06% | **84.16%** | **+4.10%** |
| **Macro Recall** | 73.82% | **76.79%** | **+2.97%** |
| **Exact Match** | **51.76%** | **51.76%** | 0.00% (Identical) |
| **Totally Wrong** | 11.55% | **7.38%** | **-4.17% (Lower is better)** |
| **No-Ans Ratio** | 0.77% | 1.03% | +0.26% |

## Phase 6: Diagnostic Analysis & Mathematical Formulation

### Objectives
3 analysis tasks for paper preparation:
1. **Answer Entity Distribution**: Rank/score analysis of answer entities in retrieval_result.pth — Baseline (Jul02) vs RT-Combined (Jun30)
2. **Dataset Label Analysis**: Class imbalance assessment on WebQSP & CWQ triple_scores (shortest-path weak supervision)
3. **DDE Math Formulation**: Clean LaTeX formulation of Topic-Entity DDE and Relation-Topic DDE for scientific report

### Status: ✅ Complete

### Scripts Created
- `retrieve/analyze_answer_entity_rank.py` — Task 1 ✅
- `retrieve/analyze_label_distribution.py` — Task 2 ✅
- `docs/DDE_Mathematical_Formulation.md` — Task 3 ✅

### Task 1 Results — Answer Entity Rank Distribution (WebQSP)
| Metric | Baseline (Jul02) | RT-Combined (Jun30) | Delta |
|---|---|---|---|
| Avg # answer entities/sample | 5.12 | 5.12 | — |
| Avg best rank | 11.44 | **8.55** | **−2.89** |
| Median best rank | 1.0 | 1.0 | 0 |
| Avg score @ best rank | 0.6509 | **0.6992** | **+0.05** |
| Found rate @10 | 81.21% | **83.53%** | +2.32% |
| Found rate @50 | 89.57% | **90.91%** | +1.34% |
| Found rate @100 | 91.82% | **92.80%** | +0.98% |
| Found rate @500 | 94.33% | **94.45%** | +0.12% |
- Chart: `retrieve/answer_entity_rank_distribution.png`
- Key insight: ~81% of samples have answer entity at rank 1-10. "Not found" (~5.7%) is due to entity not being in the indexed subgraph, not poor ranking.

### Task 2 Results — Label Distribution (Class Imbalance)
| Dataset | Split | Avg Total Triples | Avg Positives | Global Pos Ratio | No-Path% |
|---|---|---|---|---|---|
| WebQSP | train | 4221 | 23.48 | **0.556%** | 4.8% |
| WebQSP | test | 4306 | 20.27 | **0.471%** | 5.2% |
| CWQ | test | 4273 | 9.44 | **0.221%** | **19.8%** |

- **Verdict: 🔴 SEVERE imbalance** on all splits — positive triples < 1% globally
- WebQSP: 65% 1-hop, 34% 2-hop paths → mostly simple questions
- CWQ: 66% 2-hop, 6% 3-4hop → harder multi-hop, ~20% no-path samples skipped
- The model currently uses unweighted BCE → may be biased to predict 0; weighted loss is a candidate improvement

### Task 3 — DDE Mathematical Formulation
- Document: `docs/DDE_Mathematical_Formulation.md`
- Covers: Topic-Entity DDE (init, forward/backward propagation, encoding), Relation-Topic Semantic DDE (hybrid scoring, edge weighting, parallel propagation), TopK MLP injection, final triple feature vector with dimension table

## Phase 7: Pre-Refactor Score Analysis (Relation-Topic Deep Dive)

Ở phase này ta sẽ phân tích dữ liệu hiện tại 1 cách toàn diện hơn để tìm ra điểm nghẽn của improvments

Hiện tại so với baseline, ta đã cải thiện kết quả retrieval_phase bằng cách thêm topic-relation feature để cho ra weighted_DDE và topk=5 relation được chọn làm topic, score để làm feature. Tiếp đến là weighted_loss function để giải quyết unbalanced dataset

### Câu hỏi đặt ra là:

1. **Relation đã thực sự đúng là topic hay không?**: 
Relation-topic là phase có sự cải thiện lớn nhất, vậy nên cần biết xem là top5 relation lấy được bằng emb search có cải thiện dược thêm không?
2. **Dữ liệu dataset có câu hỏi dạng nào, số hop ra sao, có khải thác được thêm thông tin gì không?**
3. **Cách cấu trúc feature input để đưa vào model MLP đã được feature engineering tốt chưa?**
4. **Mô hình MLP đã đủ chưa? Có nên đánh đổi latency để thêm 1 lớp reranking vào?**
5. **Loss function nào là hiệu quả nhất cho bài toán?**

### Một số insight về dữ liệu

#### A. Phân tích tổng quát: thông kê phân phối dataset (WebQSP)

```python
============================================================================================
  WEBQSP / TRAIN  |  2,848 samples  |  upper-tail cutoff P95
============================================================================================
  Metric                                 Mean <= P      P cutoff       Kept/All
  ---------------------------------- ------------- ------------- --------------
  Entities / sample                        1330.88       1995.00  2,706/2,848 
  Relations / sample                        277.53        456.65  2,705/2,848 
  Triples / sample                         3973.30       7993.60  2,705/2,848 
  Topic entities / sample                     1.00          1.00  2,797/2,848 
  Topic entities in graph / sample            1.00          1.00  2,797/2,848 
  Answer entities / sample                    3.14         30.00  2,709/2,848 
  Answer entities in graph / sample           2.82         26.00  2,706/2,848 
  Positive (label=1) triples / sample         10.03        107.30  2,705/2,848 
  Negative (label=0) triples / sample       3950.14       7961.55  2,705/2,848 
  Positive-label rate / sample              0.0030        0.0334  2,705/2,848 
  Shortest paths / sample                     8.75         90.00  2,572/2,705 
  Hops / shortest path                        1.35          2.00  2,567/2,577 

  Label balance (before outlier filtering):
    label=1 triples: 66,866 | label=0 triples: 11,955,737 | 0:1 = 178.8:1
  Shortest-path count / sample (raw, non-outlier graphs): min=0, max=752, no-path=128 (4.7%)
  Path traversal skipped 143 graph outliers (> P95 triples = 7994).

============================================================================================
  WEBQSP / TEST  |  1,639 samples  |  upper-tail cutoff P95
============================================================================================
  Metric                                 Mean <= P      P cutoff       Kept/All
  ---------------------------------- ------------- ------------- --------------
  Entities / sample                        1360.72       1995.00  1,568/1,639 
  Relations / sample                        282.69        459.00  1,558/1,639 
  Triples / sample                         4060.62       7982.10  1,557/1,639 
  Topic entities / sample                     1.00          1.00  1,612/1,639 
  Topic entities in graph / sample            1.00          1.00  1,612/1,639 
  Answer entities / sample                    3.09         29.10  1,557/1,639 
  Answer entities in graph / sample           2.90         25.00  1,559/1,639 
  Positive (label=1) triples / sample          9.79         90.00  1,558/1,639 
  Negative (label=0) triples / sample       4040.92       7951.30  1,557/1,639 
  Positive-label rate / sample              0.0031        0.0318  1,558/1,639 
  Shortest paths / sample                     8.57         74.00  1,482/1,557 
  Hops / shortest path                        1.34          2.00  1,478/1,478 

  Label balance (before outlier filtering):
    label=1 triples: 33,219 | label=0 triples: 7,024,320 | 0:1 = 211.5:1
  Shortest-path count / sample (raw, non-outlier graphs): min=0, max=888, no-path=79 (5.1%)
  Path traversal skipped 82 graph outliers (> P95 triples = 7982).
```

> Nhận xét: dữ liệu webqsp có số hop luôn là 1-2 hop cho mỗi câu hỏi, vậy nên answer entity rất gần topic entity, vậy nên khả năng cao ta có thể coi các relation nằm trên shortest-path từ topic->answer sẽ là topic-relation (groundtruth) 

#### B. Phân tích 1 số sample (so sánh triples score cao với triples còn thiếu)

Ở đây ta sẽ phân tính định tính, nhìn trực quan các query tại sao không lấy được các triple liên quan, các answer entity mặc dù chúng ở gần như vậy?

```python
====================================================================================================
  Run name: webqsp_baseline_Jul26-16:43:42
  Dataset: webqsp
  a_entity not in graph: 80  [skipped]
  Ranking misses (in graph but first answer rank > 50): 80
  Show top retrieved: 5 | Show answer-hit triples after threshold: 20
====================================================================================================

  ── ID:            WebQTest-1143
     Question       : what galileo galilei was famous for
     Topic ents     : ['Galileo Galilei']
     Ans ents       : ['Heliocentrism', 'Solar System']
     Total scored   : 500
     Top retrieved triples:
         Rank       Score  MatchedEnt              Triple                                                                In shortest path? 
       ------  ----------  ----------------------  --------------------------------------------------------------------  ------------------
            1    0.998897  -                       (Galileo Galilei) --[people.person.profession]--> (Physicist)         No                
            2    0.998806  -                       (Galileo Galilei) --[people.person.profession]--> (Astrologer)        No                
            3    0.998287  -                       (Galileo Galilei) --[people.person.profession]--> (Mathematician)     No                
            4    0.995901  -                       (Galileo Galilei) --[people.person.profession]--> (Scientist)         No                
            5    0.987167  -                       (Galileo Galilei) --[people.person.profession]--> (Astronomer)        No                
     Answer-entity triples missed after rank 50:
         Rank       Score  MatchedEnt              Triple                                                                In shortest path? 
       ------  ----------  ----------------------  --------------------------------------------------------------------  ------------------
          433    0.000000  Solar System            (Ganymede) --[astronomy.star_system_body.star_system]--> (Solar Sys…  Yes               

  ── ID:            WebQTest-873
     Question       : where does toronto get its water from
     Topic ents     : ['Toronto']
     Ans ents       : ['Ashbridges Bay Wastewater Treatment Plant']
     Total scored   : 500
     Top retrieved triples:
         Rank       Score  MatchedEnt              Triple                                                                In shortest path? 
       ------  ----------  ----------------------  --------------------------------------------------------------------  ------------------
            1    0.839751  -                       (Ontario) --[location.province.capital]--> (Toronto)                  No                
            2    0.057763  -                       (Rock dove) --[base.ecology.food_web_member.belongs_to_ecosystem_s]…  No                
            3    0.048701  -                       (Toronto) --[location.location.containedby]--> (Ontario)              No                
            4    0.041141  -                       (House sparrow) --[base.ecology.food_web_member.belongs_to_ecosyste…  No                
            5    0.019121  -                       (Toronto) --[base.ecology.ecosystem.ecosystem_members]--> (Rock dov…  No                
     Answer-entity triples missed after rank 50:
         Rank       Score  MatchedEnt              Triple                                                                In shortest path? 
       ------  ----------  ----------------------  --------------------------------------------------------------------  ------------------
          410    0.000004  Ashbridges Bay Wastew…  (Ashbridges Bay Wastewater Treatment Plant) --[location.location.co…  Yes
          484    0.000002  Ashbridges Bay Wastew…  (Ashbridges Bay Wastewater Treatment Plant) --[base.infrastructure.…  No

  ── ID:            WebQTest-780
     Question       : what is the new movie john carter about
     Topic ents     : ['John Carter']
     Ans ents       : ['Action Film', 'Adventure Film', 'Fantasy', 'Science Fiction']
     Total scored   : 500
     Top retrieved triples:
         Rank       Score  MatchedEnt              Triple                                                                In shortest path? 
       ------  ----------  ----------------------  --------------------------------------------------------------------  ------------------
            1    0.642190  -                       (John Carter) --[fictional_universe.work_of_fiction.setting]--> (He…  No                
            2    0.476679  -                       (John Carter) --[film.film.produced_by]--> (Colin Wilson)             No                
            3    0.447407  -                       (Colin Wilson) --[film.producer.film]--> (John Carter)                No                
            4    0.391175  -                       (John Carter) --[film.film.produced_by]--> (Lindsey Collins)          No                
            5    0.356232  -                       (John Carter) --[film.film.produced_by]--> (Jim Morris)               No                
     Answer-entity triples missed after rank 50:
         Rank       Score  MatchedEnt              Triple                                                                In shortest path?
       ------  ----------  ----------------------  --------------------------------------------------------------------  ------------------
          410    0.000466  Action Film             (John Carter) --[film.film.genre]--> (Action Film)                    Yes               
          434    0.000377  Adventure Film          (John Carter) --[film.film.genre]--> (Adventure Film)                 Yes               
          445    0.000324  Fantasy                 (John Carter) --[film.film.genre]--> (Fantasy)                        Yes               
             
```
> Nhận xét: Hầu hết top-ranking triple đều không nằm trên shortest-path (nhãn training) và các triple đúng lại nằm ở rank khá thấp ~top 400. Tại sao lại không under fit? Có phải do dataset khó có thể fit hay là do loss function chưa học được tính chất ranking? (tiềm năng cho các function như contrastive learning, lambda_loss, infoNCE)

#### C. Phân tích hiểu quả của phương pháp topic-relation

Phần tích hiệu quả của phương pháp retrieve topic-relation hiện tại (emb search only)

```python
=======================================================================================
  TOPIC-RELATION EVALUATION | webqsp/test | alpha=0.5 | topk=5
========================================================================================
  Evaluable samples (>=1 shortest-path relation): 1,554/1,638
  Macro precision@5: 0.2456
  Macro recall@5   : 0.4332
  Hit rate@5       : 0.6847
  MRR (first gold relation)  : 0.5265

ID: WebQTest-21
Q: who is keyshia cole dad
Chosen topic-relation:
Relation                                                          |    Score | Semantic | Lexical | In shortest path?
----------------------------------------------------------------------------------------------------------------------
fictional_universe.fictional_character.parents                    |   0.2432 |   0.4863 |  0.0000 | No
base.popstra.celebrity.infidelity_victim                          |   0.2394 |   0.4789 |  0.0000 | No
base.schemastaging.music_genre_concept.artists                    |   0.2309 |   0.4618 |  0.0000 | No
music.record_label.artist                                         |   0.2300 |   0.4600 |  0.0000 | No
music.album.featured_artists                                      |   0.2296 |   0.4593 |  0.0000 | No
Ground-truth shortest-path relation scores:
  Rank | Relation                                                          |    Score | Semantic | Lexical | Selected top-K?
------------------------------------------------------------------------------------------------------------------------------
     6 | people.person.parents                                             |   0.2280 |   0.4560 |  0.0000 | No
    63 | people.person.children                                            |   0.2000 |   0.4000 |  0.0000 | No

ID: WebQTest-22
Q: what kind government does egypt have
Chosen topic-relation:
Relation                                                          |    Score | Semantic | Lexical | In shortest path?
----------------------------------------------------------------------------------------------------------------------
government.government.government_for                              |   0.7696 |   0.5392 |  1.0000 | No
government.government_agency.government                           |   0.6525 |   0.5322 |  0.7727 | No
government.governmental_jurisdiction.government                   |   0.6324 |   0.5148 |  0.7500 | No
government.governmental_jurisdiction.government_bodies            |   0.5162 |   0.5168 |  0.5156 | No
government.governmental_jurisdiction.government_positions         |   0.5102 |   0.5048 |  0.5156 | No
Ground-truth shortest-path relation scores:
  Rank | Relation                                                          |    Score | Semantic | Lexical | Selected top-K?
------------------------------------------------------------------------------------------------------------------------------
     6 | location.country.form_of_government                               |   0.4813 |   0.5923 |  0.3704 | No
     8 | government.form_of_government.countries                           |   0.4749 |   0.6316 |  0.3182 | No
```

> Nhận xét: kết quả trên cho ta thấy là có khá ít relation thuộc shortest_path được chọn. Hiệu quả này thực sự cần cải thiện.

## Phase 8: Relation-Topic Optimization Experiments

Từ phase trước ta thấy được lexical score luôn ~0.0 và ít đóng góp vào việc retrieve đúng relation. Thậm chí lexical, keyword match còn có thể gây sai đối với vài trường hợp:
**VD: Bị nhầm matching obama vs obamabase**
Q: who inspired obama ?
Relation retrieved được lại là:
- base.obamabase.possible_cabinet_member.cabinet_position 
- base.obamabase.cabinet_member.cabinet_position 
- base.obamabase.cabinet_position.potential_cabinet_members     

### Thử nghiệm với mô hình embedding mới: BGE-M3 

Mô hình này có sẵn có chế combine: dense vector + sparse vector (token-level) có thể thay thế word-match

### Kết quả:

Kết quả test WebQSP @5:

- Dense: embedding → Dense vector
- Sparse: embedding thành các tokens
- Raw: dạng `domain.topic.relation`
- Split: dạng `domain topic relation`
- Natural: dạng `relation of topic`

```python
Ví dụ:
# Raw format:
people.person.parents  
# Split format:
people  person  parents  
# Natural format:
Parents of person
```

| Experiment | P@5 | R@5 | Hit@5 | MRR |
| --- | --- | --- | --- | --- |
| Dense (natural) | **0.2876** | **0.5099** | **0.7542** | **0.6110** |
| Sparse raw | 0.0495 | 0.0847 | 0.1782 | 0.1266 |
| Sparse split | 0.1261 | 0.2356 | 0.3855 | 0.2815 |
| Sparse natural | 0.1270 | 0.2454 | 0.3797 | 0.3067 |
| Dense + sparse raw | 0.2012 | 0.3580 | 0.6165 | 0.4131 |
| Dense + sparse split | 0.2145 | 0.4075 | 0.6210 | 0.4515 |
| Dense + sparse natural | 0.2219 | 0.4189 | 0.6358 | 0.4541 |

**Kết quả**

| Model | Format | Top-K | Precision | Recall | Hit Rate | MRR |
| --- | --- | --- | --- | --- | --- | --- |
| `gte-large-en-v1.5`  | - | 5 | 0.2456 | 0.4332 | 0.6847 | 0.5265 |
|  | - | 10 | 0.1746 | 0.5726 | 0.7921 | 0.5265 |
|  | - | 20 | 0.1166 | 0.7035 | 0.8822 | 0.5265 |
| BGE-M3 | Raw | 5 | 0.2759 | 0.4957 | 0.7471 | 0.5909 |
|  | Raw | 10 | 0.1880 | 0.6175 | 0.8288 | 0.5909 |
| BGE-M3 | Split | 5 | **0.2878** | **0.5154** | **0.7593** | **0.6088** |
|  | Split | 10 | **0.1920** | **0.6300** | **0.8430** | **0.6088** |
|  | Split | 20 | **0.1218** | **0.7349** | **0.9048** | **0.6088** |
| BGE-M3 | Natural | 5 | 0.2835 | 0.5034 | 0.7471 | 0.5977 |
|  | Natural | 10 | 0.1857 | 0.6152 | 0.8269 | 0.5977 |

So sánh **Retrieval comparison on GTE vs BGE embedding model**

#### **Answer Entity Recall**

| Run name | K_50 | K_100 | K_200 | K_400 | K_500 |
| --- | --- | --- | --- | --- | --- |
| webqsp_rt_combined_Jul26 | 0.911 | **0.954** | 0.975 | 0.988 | 0.990 |
| webqsp_rt_combined_bge_dense_sparse_Aug02 | **0.922** | 0.953 | **0.978** | **0.989** | **0.993** |

#### **Shortest Path Recall**

| Run name | K_50 | K_100 | K_200 | K_400 | K_500 |
| --- | --- | --- | --- | --- | --- |
| webqsp_rt_combined_Jul26 | 0.834 | **0.894** | **0.939** | **0.969** | 0.976 |
| webqsp_rt_combined_bge_dense_sparse_Aug02 | **0.834** | 0.890 | 0.936 | 0.968 | **0.976** |

#### **GPT-Labeled Recall**

| Run name | K_50 | K_100 | K_200 | K_400 | K_500 |
| --- | --- | --- | --- | --- | --- |
| webqsp_rt_combined_Jul26 | 0.805 | 0.868 | 0.918 | 0.960 | 0.967 |
| webqsp_rt_combined_bge_dense_sparse_Aug02 | **0.811** | **0.869** | **0.924** | **0.964** | **0.971** |

#### Nhận xét: việc chọn mô hình xịn hơn cũng không cải thiện lớn gì, và sparse vector không tốt hơn lexical là bao. Ta có thể nghĩ đến việc finetune lại bge model để thích hợp cho bài toán này

## Phase 9: Big refactory code: fix DDE format, re-evaluate weighted DDE + Weighted loss function comparison

1. Trước đó công thức DDE đang bị implement nhầm lẫn: 

**weighted DDE cũ:**
Việc message passing được khởi tạo với PEConv(agg='mean') khiến cho mẫu số chia cho tổng các neighbors quanh node đó
**Expected weighted DDE**: khi lan truyền nhiều lượt cần chia cho tổng trọng số của các neighbors (thay vì số lượng)

#### Kết quả (Biết răng phiên bản Aug03 là phiên bản mới)

> Nhận xét: phương pháp đúng công thức nhìn chung có vẻ cao hơn 1 chút về answer recall, nhưng ko khác biệt nhiều

**Answer Entity Recall**

| Cấu hình | Phiên bản | K=50 | K=100 | K=200 | K=400 | K=500 |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Aug03 | 0.906 | 0.949 | 0.973 | 0.986 | 0.988 |
| RT combined | Aug03 | 0.918 | 0.957 | 0.979 | 0.990 | 0.991 |
|  | Jul26 | 0.911 | 0.954 | 0.975 | 0.988 | 0.990 |
| RT combined + BGE dense/sparse | Aug03 | 0.917 | 0.953 | 0.972 | 0.988 | 0.990 |
|  | Aug02 | 0.922 | 0.953 | 0.978 | 0.989 | **0.993** |
| S4b weighted BCE | Aug03 | 0.920 | 0.956 | 0.977 | 0.988 | 0.990 |
|  | Jul25 | 0.917 | 0.956 | 0.978 | 0.990 | 0.992 |

**Shortest Path Recall**

| Cấu hình | Phiên bản | K=50 | K=100 | K=200 | K=400 | K=500 |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Aug03 | 0.829 | 0.885 | 0.935 | 0.966 | 0.973 |
| RT combined | Aug03 | 0.831 | 0.889 | 0.937 | 0.969 | 0.975 |
|  | Jul26 | 0.834 | 0.894 | 0.939 | 0.969 | 0.976 |
| RT combined + BGE dense/sparse | Aug03 | 0.834 | 0.892 | 0.934 | 0.968 | 0.977 |
|  | Aug02 | 0.834 | 0.890 | 0.936 | 0.968 | 0.976 |
| S4b weighted BCE | Aug03 | 0.841 | 0.899 | 0.942 | 0.973 | 0.979 |
|  | Jul25 | 0.849 | 0.904 | 0.947 | 0.977 | **0.983** |

**GPT-Labeled Recall**

| Cấu hình | Phiên bản | K=50 | K=100 | K=200 | K=400 | K=500 |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Aug03 | 0.808 | 0.865 | 0.920 | 0.963 | 0.972 |
| RT combined | Aug03 | 0.814 | 0.870 | 0.920 | 0.960 | 0.968 |
|  | Jul26 | 0.805 | 0.868 | 0.918 | 0.960 | 0.967 |
| RT combined + BGE dense/sparse | Aug03 | 0.812 | 0.868 | 0.923 | 0.959 | 0.969 |
|  | Aug02 | 0.811 | 0.869 | 0.924 | 0.964 | 0.971 |
| S4b weighted BCE | Aug03 | 0.822 | 0.868 | 0.920 | 0.961 | 0.972 |
|  | Jul25 | 0.833 | 0.884 | 0.930 | 0.962 | 0.969 |

2. Đánh giá so sánh khác phương pháp loss function ~ weighted_bce:

**Insight**: việc trước đó chọn static-weighted = 200 đạt kết quả tốt nhất là dựa vào reversed-frequency của positive-vs-negative label của tập test. Con số 200 này chỉ là fit với dataset này nên ta có thể coi như phương pháp đang có hiệu quả tốt trước đó là **reversed_freq**

Nhưng khi dùng pp weighted DDE chuẩn thì liệu loss function nào là đúng, và 1 chi tiết quan tọng là 200 chỉ là global frequency, còn có thể sử dụng batched frequency nữa:

### Kết quả mới nhất:

#### Answer Entity Recall

| Cấu hình | Phiên bản | K=50 | K=100 | K=200 | K=400 | K=500 |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Aug03 | 0.906 | 0.949 | 0.973 | 0.986 | 0.988 |
| RT combined | Aug03 | 0.918 | 0.957 | 0.979 | 0.990 | 0.991 |
|  | Jul26 | 0.911 | 0.954 | 0.975 | 0.988 | 0.990 |
| RT combined + BGE dense/sparse | Aug03 | 0.917 | 0.953 | 0.972 | 0.988 | 0.990 |
|  | Aug02 | 0.922 | 0.953 | 0.978 | 0.989 | **0.993** |
| S4.1c effective number | Aug03 | 0.926 | 0.961 | **0.982** | 0.991 | **0.993** |
|  | Jul29 | 0.918 | 0.954 | 0.977 | 0.991 | 0.992 |
| S4.4a InfoNCE | Aug03 | 0.918 | 0.956 | 0.978 | 0.990 | **0.993** |
|  | Jul29 | 0.918 | 0.951 | 0.978 | **0.993** | **0.996** |
| S4.4b hybrid BCE + margin | Aug03 | **0.928** | 0.959 | 0.978 | 0.989 | 0.990 |
|  | Jul29 | 0.926 | **0.962** | 0.980 | 0.992 | 0.994 |
| S4.5a BGE dense top-5 | Jul29 | 0.911 | 0.946 | 0.970 | 0.989 | 0.991 |
| S4b weighted BCE | Aug03 | 0.920 | 0.956 | 0.977 | 0.988 | 0.990 |
|  | Jul25 | 0.917 | 0.956 | 0.978 | 0.990 | 0.992 |
| webqsp_wl_inversed_Aug06 | NEW | 0.917 | 0.951 | 0.975 | 0.989 | 0.991 |
| webqsp_rt_shortest_path_Aug06 | SPECIAL | **0.949** | **0.979** | **0.993** | **0.999** | **1.000** |

#### Shortest Path Recall

| Cấu hình | Phiên bản | K=50 | K=100 | K=200 | K=400 | K=500 |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Aug03 | 0.829 | 0.885 | 0.935 | 0.966 | 0.973 |
| RT combined | Aug03 | 0.831 | 0.889 | 0.937 | 0.969 | 0.975 |
|  | Jul26 | 0.834 | 0.894 | 0.939 | 0.969 | 0.976 |
| RT combined + BGE dense/sparse | Aug03 | 0.834 | 0.892 | 0.934 | 0.968 | 0.977 |
|  | Aug02 | 0.834 | 0.890 | 0.936 | 0.968 | 0.976 |
| S4.1c effective number | Aug03 | 0.851 | 0.909 | 0.948 | 0.974 | 0.980 |
|  | Jul29 | **0.859** | **0.911** | **0.949** | **0.978** | **0.983** |
| S4.4a InfoNCE | Aug03 | 0.845 | 0.901 | 0.940 | 0.971 | 0.979 |
|  | Jul29 | 0.852 | 0.906 | 0.948 | 0.976 | 0.982 |
| S4.4b hybrid BCE + margin | Aug03 | 0.841 | 0.896 | 0.938 | 0.971 | 0.977 |
|  | Jul29 | 0.851 | 0.906 | 0.948 | 0.977 | 0.982 |
| S4.5a BGE dense top-5 | Jul29 | 0.850 | 0.904 | 0.948 | 0.977 | 0.981 |
| S4b weighted BCE | Aug03 | 0.841 | 0.899 | 0.942 | 0.973 | 0.979 |
|  | Jul25 | 0.849 | 0.904 | 0.947 | 0.977 | **0.983** |
| webqsp_wl_inversed_Aug06 | NEW | **0.855** | **0.911** | **0.949** | 0.977 | 0.982 |
| webqsp_rt_shortest_path_Aug06 | SPECIAL | **0.906** | **0.950** | **0.980** | **0.994** | **0.996** |

#### GPT-Labeled Recall

| Cấu hình | Phiên bản | K=50 | K=100 | K=200 | K=400 | K=500 |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Aug03 | 0.808 | 0.865 | 0.920 | 0.963 | 0.972 |
| RT combined | Aug03 | 0.814 | 0.870 | 0.920 | 0.960 | 0.968 |
|  | Jul26 | 0.805 | 0.868 | 0.918 | 0.960 | 0.967 |
| RT combined + BGE dense/sparse | Aug03 | 0.812 | 0.868 | 0.923 | 0.959 | 0.969 |
|  | Aug02 | 0.811 | 0.869 | 0.924 | 0.964 | 0.971 |
| S4.1c effective number | Aug03 | 0.833 | 0.887 | 0.932 | 0.965 | 0.970 |
|  | Jul29 | 0.832 | 0.881 | 0.928 | 0.963 | 0.969 |
| S4.4a InfoNCE | Aug03 | 0.829 | 0.884 | 0.923 | 0.961 | 0.970 |
|  | Jul29 | 0.831 | 0.882 | 0.929 | 0.961 | 0.970 |
| S4.4b hybrid BCE + margin | Aug03 | 0.829 | 0.880 | 0.923 | 0.965 | 0.971 |
|  | Jul29 | **0.835** | **0.890** | **0.937** | **0.967** | **0.974** |
| S4.5a BGE dense top-5 | Jul29 | 0.824 | 0.872 | 0.917 | 0.958 | 0.967 |
| S4b weighted BCE | Aug03 | 0.822 | 0.868 | 0.920 | 0.961 | 0.972 |
|  | Jul25 | 0.833 | 0.884 | 0.930 | 0.962 | 0.969 |
| webqsp_wl_inversed_Aug06 | NEW | **0.833** | **0.884** | **0.930** | **0.962** | 0.969 |
| webqsp_rt_shortest_path_Aug06 | SPECIAL | 0.739 | 0.787 | 0.838 | 0.888 | 0.898 |


> Nhận xét: 1 số phương pháp như effective number, infoNCE, hybrid cũng cải thiện ở 1 số mục, reversed có kết quả tốt hơn với static = 200

## Phase 10: Listwise - Reranking layer

Là phương pháp post-training MLP model, giúp top-k triples được ranking lại, cải thiện precision@K
Kết quả: listwise khá tệ, không khả thi

## Phase 11: Finetune bge-m3 model

Mục tiêu là fine-tune `BAAI/bge-m3` để retrieve relation nằm trên tất cả shortest paths. Input
encoder chỉ có question và relation text; answer/path chỉ dùng tạo nhãn và evaluate. Nhãn hiện tại
là `edge_complete_v1`, bảo toàn directed parallel edges.

Implementation dùng backbone/scoring chính thức của FlagEmbedding 1.4.0, Hugging Face Trainer và
PEFT LoRA q/k/v (`r=8`, `alpha=16`). Objective là multi-positive InfoNCE tùy biến với microbatch 1
và gradient accumulation 16. Checkpoint chỉ chứa adapter, không chứa base model.

### V1 entity-disjoint

Split mới nhóm các record theo connected component của cả topic entity và answer entity:

- fit: 2,439 records;
- dev: 271 records;
- fit/dev overlap: query ID = 0, topic entity = 0, answer entity = 0.

Kết quả apples-to-apples trên dev mới, cùng candidate pool, formatter `split`, dense scorer và
evaluator:

| Model | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|
| BGE-M3 zero-shot | 0.65301 | 0.69334 | 0.59052 |
| v1 entity-disjoint | **0.76961** | **0.80705** | **0.71339** |
| Delta | **+0.11660** | **+0.11371** | **+0.12287** |

Training tăng đều qua ba epoch: Recall@10 `0.75504 → 0.76534 → 0.76961`. Run hoàn tất 459
optimizer steps, train loss `0.29751`, peak VRAM `1.287 GiB`; best là `checkpoint-459`.

**Kết luận hiện tại:** leak-controlled WebQSP dev gate PASS. Chưa mine L5 hoặc launch v2. Bước kế
tiếp bắt buộc là đánh giá chính checkpoint entity-disjoint mới trên WebQSP test và CWQ test. Các
số test/CWQ trước đây (`0.78257` và `0.59475` Recall@10) thuộc checkpoint record-hash cũ, chỉ giữ
làm tham chiếu và không được dùng để đóng gate mới.

Chi tiết protocol, artifact và trạng thái: [PHASE-11.md](PHASE-11.md). Handoff vận hành:
[PHASE-11-SESSION-MEMORY.md](PHASE-11-SESSION-MEMORY.md).

## Phase 12: Feature engineering, Hierachical MLP

### 1. Vấn đề chênh lệch scale/dimension trong vector concat

Nhìn vào DDE_Mathematical_Formulation.md mục 5, với d=1024, K_fwd = K_bwd=3, 

|Block	| Dim	| Tỷ trọng|
|---|---|---|
|zq,zh,zr,zt (embedding thô)	| 4096	| ~98.7%|
|Topo + Sem DDE (head+tail)	| 48	| ~1.2%|
|TopK relation mask	| 2	| ~0.05%|

Đây không phải vấn đề "quá nhiều feature" theo nghĩa chung, mà là vấn đề imbalance về mặt hình học: layer Linear đầu tiên của MLP nhận gradient tỷ lệ với norm của input. Nếu bạn concat thẳng, DDE và mask gần như bị "chìm" trong nhiễu của 4096 chiều embedding trừ khi weight init + optimizer bù lại được — và thực nghiệm Phase 4 cho thấy đúng là DDE/mask một mình không giúp gì (Path 1, Path 2 đều thấp hơn baseline), chỉ khi kết hợp cả hai mới có synergy. Đây có thể là dấu hiệu của đúng hiện tượng "tín hiệu nhỏ bị lấn át" chứ không hẳn là bản chất feature yếu — cần kiểm chứng bằng cách:

Chuẩn hóa riêng từng block trước khi concat (LayerNorm hoặc mean/std per-block), rồi mới nối lại. Nếu synergy giữa mask và DDE biến mất sau khi chuẩn hóa fair hơn, nghĩa là trước đó synergy một phần do vấn đề scale chứ không phải bản chất thông tin bổ sung.
Cân nhắc kiến trúc late fusion thay vì concat phẳng: 2 nhánh nhỏ (embedding tower + structural tower: DDE+mask) rồi gate/attention hoặc concat sau khi mỗi nhánh đã được nén về cùng độ lớn. Cách này cũng giúp việc đo importance ở bước sau (mục 2) sạch hơn.

### Về ý tưởng XGBoost / permutation importance

Ý tưởng đúng, nhưng cần chỉnh lại đối tượng đo:

Đừng train XGBoost trực tiếp trên 4150 chiều raw embedding. Từng chiều embedding riêng lẻ không có ý nghĩa ngữ nghĩa (chỉ có ý nghĩa khi dot-product cả vector), nên feature importance ở mức "dimension 837 quan trọng" là vô nghĩa và bị nhiễu bởi multicollinearity giữa các chiều cùng block.
Thay vào đó, xây bảng meta-feature cho mỗi triple — khoảng 15–25 scalar đã có sẵn trong pipeline của bạn: cos(q,h), cos(q,r), cos(q,t), S_sem(r), S_lex(r), S_combined(r), mask(r), thống kê DDE (mean/max từng vòng forward/backward, topo vs sem), is_topic(h), is_topic(t), hop distance nếu có. Train XGBoost/LightGBM trên bảng này, dùng SHAP để lấy importance — nó sẽ trả lời đúng câu hỏi "nhóm tín hiệu nào (semantic sim, lexical sim, structural DDE, mask) đang thực sự kéo logit" mà không bị nhiễu bởi 4096 chiều thô.
Permutation importance nên làm ở mức block, không phải mức dim: trên MLP đã train, shuffle nguyên khối DDE (hoặc mask, hoặc z_r) trên validation set, đo sụt giảm recall@K. Rẻ, không cần train lại, và cho biết independent contribution — bổ sung tốt cho ablation ladder tốn kém.

> Tóm lại: ablation ladder (đã có) = gold standard cho "feature nào giữ lại"; SHAP trên meta-feature = cách nhanh để sinh giả thuyết trước khi chạy ablation; permutation block-level = kiểm tra nhanh không cần train lại. Ba cái bổ trợ nhau, không thay thế nhau.

---

## Phase 13: Path + Memory Hybrid & QA Reasoning Evaluation

### 1. Ý Tưởng & Đột Phá
- **Path + Memory Hybrid**: Kết hợp embedding chuỗi đường đi 1–2 hop (giải quyết quan hệ phụ thuộc trên CWQ) với bộ nhớ kNN Question Memory (0.5 Path + 0.5 Memory).
- **Chuẩn hóa Leave-One-Out (LOO)**: Triệt tiêu rò rỉ dữ liệu tự thân trên tập train, đảm bảo tính công bằng và nhất quán giữa train, val và test.
- **Tối ưu hàm mất mát Inversed Loss**: Tự động cân bằng trọng số $w_{pos} = N_{neg} / N_{pos}$ theo từng đồ thị con, tối ưu đồng thời cả Shortest Path Recall và Answer Reachability.

### 2. Kết Quả Retrieval Benchmark (Test Split)

| Dataset | Phương Pháp / Checkpoint | Ans R@50 | Ans R@100 | SP R@50 | SP R@100 | GPT R@100 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **WebQSP** | Baseline gốc (No RT, BCE) | 0.9061 | 0.9486 | 0.8287 | 0.8849 | 0.8650 |
| | Path+Memory Hybrid (BCE) | 0.9260 | 0.9590 | 0.8462 | 0.8974 | 0.8770 |
| | **Path+Memory Hybrid (Inversed)** | **0.9280** | **0.9600** | **0.8670** | **0.9120** | **0.8920** |
| **CWQ** | Baseline gốc (No RT, BCE) | 0.8593 | 0.9205 | 0.7141 | 0.8053 | 0.7950 |
| | **Path+Memory Hybrid (BCE)** | **0.8796** | **0.9244** | **0.7453** | **0.8261** | **0.8219** |

### 3. Đánh Giá Đối Đầu QA Reasoning (Llama-3.1-8B)

So sánh trực tiếp kết quả sinh câu trả lời với baseline chuẩn `Meta-Llama-3.1-8B-Instruct`:

| Dataset | Phương Pháp (Retrieval + Llama-3.1-8B) | Hit@1 (%) | Macro F1 (%) | Precision (%) | Recall (%) | Exact Match (%) | Totally Wrong (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **WebQSP** | Baseline (`Meta-Llama-3.1-8B-Instruct`) | 82.86 | 70.57 | 78.27 | 72.23 | 50.92 | 13.57 |
| *(1,639 mẫu)* | **Ours** (`Path+Memory Inversed`) | **90.06** | **75.76** | **84.53** | **77.86** | **52.53** | **7.44** |
| | *Chênh lệch ($\Delta$)* | *+7.20%* | *+5.19%* | *+6.26%* | *+5.63%* | *+1.61%* | *-6.13%* |
| **CWQ** | Baseline (`Meta-Llama-3.1-8B-Instruct`) | 68.89 | 61.39 | 64.12 | 66.03 | 48.28 | 27.14 |
| *(3,531 mẫu)* | **Ours** (`Path+Memory BCE`) | **70.47** | **63.12** | **66.13** | **67.48** | **51.09** | **25.42** |
| | *Chênh lệch ($\Delta$)* | *+1.58%* | *+1.73%* | *+2.01%* | *+1.45%* | *+2.81%* | *-1.72%* |


Kết quả huấn luyện đối đầu giữa Baseline (Raw Shortest-Path) và Ours (LLM-Denoised Labels) trên 30% tập train WebQSP (sử dụng cấu hình chuẩn SOTA: Relation-Topic path_memory_hybrid + Weighted Loss effective_num) đã hoàn tất với chiến thắng áp đảo thuộc về mô hình LLM-Denoised!

Dưới đây là bảng số liệu định lượng chi tiết:

I. BẢNG SO SÁNH HIỆU QUẢ HUẤN LUYỆN ĐỐI ĐẦU
(Đánh giá trên toàn bộ 234 mẫu Validation Set sạch)

Phương Pháp Huấn Luyện (30% WebQSP)	Best Epoch	Ans Recall@50 (%)	Ans Recall@100 (%) ⭐ (Ưu Tiên Số 1)	CleanTriple Recall@100 (%) (Nhãn Val Sạch)	Raw SP Recall@100 (%) (Nhãn SP Cũ)	Thời Gian Huấn Luyện
1. Baseline (Raw Shortest-Path)	10	84.19%	90.92%	87.02%	83.88%	367.6s (~6 phút)
2. Ours (LLM-Denoised Labels)	8	86.57% (+2.38%)	92.63% (+1.71%)	86.98% (Peak 88.72%)	80.63%	366.4s (~6 phút)

---

## Phase 15: LLM-Supervised Label Denoising & Reasoning Phase Evaluation

### 1. Kết Quả Huấn Luyện Đối Đầu & Test Set Benchmark
Huấn luyện mô hình MLP Retriever với cấu hình chuẩn SOTA: `webqsp_rt_path_memory_hybrid_effective.yaml` (Relation-Topic `path_memory_hybrid` + Structural DDE + BCE `pos_weight_mode: effective_num`, text encoder `bge-m3-v1`).

- **Validation Set (234 samples)**:
  * Baseline (Raw SP): Ans@50 `84.19%` | Ans@100 `90.92%`
  * Ours (LLM-Denoised): Ans@50 `86.57%` *(+2.38%)* | Ans@100 `92.63%` *(+1.71%)*
- **Test Set Độc Lập (1,639 samples unseen)**:
  * Baseline (Raw SP): Ans@50 `83.77%` | Ans@100 `89.27%` | Ans@200 `92.56%` | SP R@100 `85.94%`
  * Ours (LLM-Denoised): Ans@50 `84.91%` *(+1.13%)* | Ans@100 `89.70%` *(+0.43%)* | Ans@200 `92.63%` | SP R@100 `82.09%`

### 2. Phân Tích Phân Phối Điểm Số (Score Distribution) Checkpoint M14
- Checkpoint: `retrieve/webqsp_rt_path_memory_hybrid_inversed_Sep07-18:06:25` (Top $k = 400$, 650,386 triples)
- Biểu đồ: `retrieve/webqsp_rt_path_memory_hybrid_inversed_Sep07-18:06:25/score_distribution_m14.png`
- Phân khúc điểm: $\le 0.01$ (72.45%), $0.01 - 0.2$ (11.09%), $> 0.2$ (16.46%); $\le 0.0001$ (50.23%), $0.0001 - 0.5$ (36.79%), $> 0.5$ (12.98%).

### 3. Reasoning Trực Tiếp Trên Nhãn Shortest Path (SP Ground Truth)
- Input: `retrieve/runs/webqsp_sp_groundtruth/retrieval_result.pth` (trích xuất 33,219 triples SP chuẩn, trung bình 20.27 triples/sample).
- Screen Session: `reason_sp` (`reason/start_sp_screen.sh`).
- LLM: Ollama `llama3.1:latest` (`ollama:llama`), Prompt: `scored_100` (`sys_icl_dc`).
- File output dự đoán: `results/KGQA/webqsp/SubgraphRAG/ollama:llama/scored_100-sys_icl_dc-0-thres_0.0-test-custom_webqsp_sp_groundtruth-predictions.jsonl`.
- Log: `reason/eval_sp.log`.

---

## Phase 16: RAPL (REL-RAG) — ĐÃ HỦY

Kết quả và kết luận cũ của phase này (RAPL Recall@100 = 36.94%, "RAPL phụ thuộc nhãn GPT-4o", "line graph không khả thi") đã bị **xóa** vì sai do lỗi cài đặt:
eval chấm bằng `node_emb·q` bỏ `r_mlp`, feature `[q|r|h]` thay vì `[h|r|t]`, nhãn SP bị gộp một cột, `edge_cap=30` cắt 93% cạnh line graph, metric khác `retrieve/eval.py`.
Tái hiện lại đúng paper REL-RAG ở **Phase 17**.

---

---

> **Lưu trữ.** Tài liệu này là nhật ký giai đoạn SubgraphRAG (Phase 1–16), chỉ dùng để tham khảo.
> Phase 17 trở đi (tái hiện REL-RAG và các thử nghiệm cải tiến) đã chuyển sang `docs/04_EXPERIMENT_LOG.md`.

