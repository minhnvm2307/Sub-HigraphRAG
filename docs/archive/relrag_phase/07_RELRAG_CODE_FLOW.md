# 07. Flow code tái hiện REL-RAG (`/relrag/`) và đối chiếu với code gốc

*Tạo: 2026-10-01. Mục đích: để người dùng tự đọc và kiểm chứng. Số dòng tính theo code ở thời điểm tạo file.*

Code gốc của tác giả nằm ở `RAPL/` (RAPL v1): `MakeTrainData.py`, `FindRationalPath.py`, `load_post_processed_dataset.py`, `main.py`, `src/model/Trainerv3.py`, `src/dataset/{retriever_v2,PyG_dataset_disk,PostProcessedDataset}.py`. Không dùng các file `RAPL/run_rapl_sp_eval.py` và `RAPL/eval_rapl.py`, vì đó là bản lỗi của Phase 16.

## 1. Luồng tổng quát

```
retrieve/data_files/webqsp/processed/{split}.pkl  (subgraph SubgraphRAG: h/r/t id, entity, relation)
retrieve/data_files/webqsp/emb/gte-large-en-v1.5/{split}.pth  (embedding GTE fp32: q, entity, relation)
        │
        ├─(1) prepare_data.py ──────────► data/{split}.pkl   (record gọn: triple của DiGraph, shortest path, ứng viên triple đầu)
        ├─(2) make_memmap.py ───────────► data/emb_memmap/{split}/{q,ent,rel}.npy + index.pkl   (fp16, đọc bằng memmap)
        ├─(3) gpt_label.py build/sync ──► data/labels_<model>/webqsp_{split}/   (ứng viên d..d+2 top 10 → LLM chọn rational path)
        │     + ghép nhãn ──────────────► data/labels_hybrid_gpt4o_deepseek/{train,val}.jsonl   (path nhãn = dãy chỉ số node line graph)
        ├─(4) dataset.py ───────────────► line graph + feature [h|r|t] + path nhãn + triple đầu dương (dựng mỗi lần lấy mẫu)
        ├─(5) model.py ─────────────────► GCN 2 chiều → z, z_topic, zq, stop
        ├─(6) train.py ─────────────────► loss P (Eq. 8, 11) hoặc T (Eq. 12); chọn checkpoint → runs/<run>/best.pt
        ├─(7) infer.py ─────────────────► decode (path best-first hoặc xếp hạng triple) → retrieval_result_<budget>.pth
        └─(8) eval_retrieval_text.py ───► answer recall theo text-match, cùng tập đáp án với SubgraphRAG
```

## 2. Từng bước

| Bước | Hàm (file:dòng) | Làm gì | Tương ứng trong code gốc |
|---|---|---|---|
| 1 | `prepare_data.py:55 build_sample` | Giữ **một triple cho mỗi cặp (h, t)** (triple sau ghi đè triple trước, như `nx.DiGraph`); lưu `orig_triple_idx` để ánh xạ về triple gốc; ứng viên triple đầu = triple có head là topic entity; `sp_paths` = shortest path xuôi chiều (q→a) | `PyG_dataset_disk.py` (`nx.line_graph`, `topic_candidates` ở dòng 278–283), `retriever_v2._get_nx_g` |
| 2 | `make_memmap.py:17 convert` | Chuyển embedding fp32 (`train.pth` 11GB) sang mảng **fp16** đọc bằng memmap, kèm `index.pkl` {id: vị trí q, khoảng entity, khoảng relation} | Code gốc nạp cả dict fp32 vào RAM |
| 3a | `gpt_label.py:86 candidates` (+ `get_nx_g` :44, `cut_off_paths` :51, `translate_paths` :69) | Với mỗi cặp (topic, đáp án): simple path xuôi dài d..d+2 (chỉ lấy shortest path nếu có hơn 50 path hoặc d > 4), sắp theo độ dài, lấy top 10 | `retriever_v2.cut_off_paths`, `MakeTrainData.get_sorted_paths` (top_k = 10) |
| 3b | `gpt_label.py:102 authors_prompt`, `:116 dict_to_text` | Prompt: lấy **nguyên văn** bằng `ast` từ `FindRationalPath.py`, hội thoại 4 lượt, temperature 0 | `FindRationalPath.chat` |
| 3c | `gpt_label.py:124 parse_rational_paths`, `:140 match_candidates` | Parse các dòng `n. [r1, r2]` nằm sau "The rational paths are:", rồi ghép với ứng viên (bỏ nháy, không phân biệt hoa thường) | `load_post_processed_dataset._parse_rational_paths`, `PostProcessedDataset._match_candidate_paths` |
| 3d | Ghép nhãn (file `merge_labels.py` đã bị xóa khỏi repo; logic giữ trong `docs/04` EXP-15b) | Path cụ thể được ghép (tối đa 10) → đổi `triple_ids` thành node line graph qua `orig_triple_idx`, kiểm tra từng path là walk hợp lệ | `PostProcessedDataset._rebuild_path_label` (`max_paths = 10`) |
| 4 | `dataset.py:47 line_graph_edges` | Cạnh i→j khi và chỉ khi t_i == h_j | `nx.line_graph` |
| 4 | `dataset.py:102 build_sample` | x = [emb(h) \| emb(r) \| emb(t)] (3072 chiều); entity không có text (CVT) dùng vector 0 | `PyG_dataset_disk._build_line_graph` (dòng 354–356) |
| 4 | `dataset.py:58 load_label_paths` | Path nhãn từ `paths:<file>` hoặc `sp`; nếu LLM loại hết thì dùng lại shortest path. `topic_pos` = triple đầu của path nhãn + triple của topic entity có cos(relation, r*) > τ | `PostProcessedDataset._update_topic_labels` |
| 5 | `model.py:11 gcn_norm_adj`, `:43 _propagate`, `:52 forward` | GCN tự viết (chuẩn hóa giống PyG `gcn_norm`, khớp `GCNConv` tới 3.6e-7); 2 lớp; xuôi và ngược; `z_topic = r_mlp(rel) + z`; `stop = stop_emb + stop_mlp(q)` | `Trainerv3.__init__`, `_gnn_forward` (dòng 167–188), `train_step` |
| 6 | `train.py:24 topic_loss`, `:34 path_loss`, `:60 triple_loss` | P: Eq. 11 + Eq. 8 (dự đoán triple kế tiếp trong các láng giềng + STOP, chọn ngẫu nhiên một path nhãn). T: Eq. 12 với tỷ lệ âm/dương 1:5 | `Trainerv3.train_step` (dòng 190–397). Chế độ T không có trong code gốc |
| 6 | `train.py:88 validate`, `main` | Chọn checkpoint: `path_f1` (epoch > 5, F1 triple cuối) cho P, `val_loss` cho T | `Trainerv3.fit` (dòng 517–574) |
| 7 | `infer.py:82 decode_best_first` | Duyệt path theo xác suất giảm dần (cây best-first theo Eq. 8, mỗi bước mở tối đa 20 nhánh), lấy triple cho tới khi đủ budget | `Trainerv3.decoding` / `cond_evaluate` (sample) |
| 7 | `infer.py:119 rank_triples` | T: xếp hạng ⟨z_i, zq⟩, lấy top-K | – |
| 8 | `eval_retrieval_text.py:13` | R@K = \|A ∩ tên entity trong top-K triple\| / \|A\|, với A = `a_entity_in_graph` trong file SubgraphRAG | `retrieve/eval.py` |

## 3. Memmap và float16

- **Memmap** là mảng numpy nằm trên đĩa, hệ điều hành chỉ nạp vào RAM những trang đang được đọc, và có thể thu hồi khi thiếu bộ nhớ. Mình dùng memmap vì nạp nguyên dict embedding fp32 (`train.pth` 11GB, cộng với các process song song) đã hai lần làm sập server do hết RAM (09-28).
- **float16 là lưu với độ chính xác nửa (half precision), không phải lượng tử hóa kiểu int8.** Khi đưa vào model, dữ liệu được đổi lại thành fp32 (`model.py:54 .float()`). Đo trên WebQSP val (250 câu, mọi vector q, entity, relation): **cosine giữa fp32 và fp16 nhỏ nhất là 0.99999976, sai số L2 tương đối lớn nhất là 3.1e-4**. So sánh: code gốc train trong `torch.amp.autocast`, tức các phép nhân ma trận đã chạy ở fp16, nên mức sai số này không đáng kể.

## 4. Những điểm bản tái hiện KHÁC paper hoặc code gốc

| # | Điểm | Paper viết | Code gốc tác giả | Bản của mình | Mức ảnh hưởng / trạng thái |
|---|---|---|---|---|---|
| 1 | Gộp hai chiều GCN | Phần chính: "summation"; Appendix Eq. 86: MEAN | **Cộng** `h + hb` (`Trainerv3.py:188`) | **Trung bình** `0.5*(z+zb)` (`model.py:65`) | z nhỏ đi một nửa, nên logit cũng nhỏ đi một nửa (có thể học bù qua `q_mlp`). Chưa đo |
| 2 | Dropout trong GNN | "dropout 0.2" (không nói ở đâu) | **Không có** dropout giữa các lớp GCN; chỉ trong `q_mlp` và `stop_mlp` | Dropout 0.2 **giữa các lớp GCN** (`model.py:49`), cộng trong `q_mlp`/`stop_mlp` | Chưa đo |
| 3 | Khởi tạo trọng số | – | `GCNConv` của PyG: glorot, bias 0 | `nn.Linear` mặc định | Nhỏ |
| 4 | Độ chính xác số khi train | – | Mixed precision fp16 (`autocast` + `GradScaler`) | fp32 | Nhỏ |
| 5 | Batch | 10 | Mặc định CLI là **4**, lệnh README không đặt lại; batch PyG thật, loss lấy trung bình trên các bước trong batch | 10 bằng tích lũy gradient (mỗi mẫu 1 đồ thị) | Có thể ảnh hưởng tới tối ưu hóa |
| 6 | Loss triple đầu (Eq. 11) | Tổng trên **toàn bộ** V_pos và V_neg | **1 positive ngẫu nhiên + tối đa 8 negative** mỗi đồ thị (bỏ qua nếu < 3 negative) | BCE trên toàn bộ ứng viên (`train.py:24`), theo paper | Khác code gốc |
| 7 | Triple đầu dương (Eq. 10) | τ không nêu | Topic relation do **GPT gán nhãn** (`RelationPathTargeting.py`, RAPL v1) + triple đầu của path nhãn | Triple đầu của path nhãn + cos > 0.95 | Khác code gốc; paper REL-RAG không nhắc bước GPT này |
| 8 | Nhãn path train | GPT-4o, "mọi path" | Top-10 path d..d+2, GPT-4o | GPT-4o 901 câu + DeepSeek 1,809 câu (EXP-15/15b); câu bị LLM loại hết dùng lại shortest path (40 câu) | **Ảnh hưởng lớn**: −3.7 R@50 so với nhãn SP (DeepSeek lọc lỏng hơn) |
| 9 | Nhãn val | – | Rational path | DeepSeek (10-01) hoặc SP | – |
| 10 | Chọn checkpoint | "validation loss" | Epoch > 5, top-3 theo F1 triple cuối | Epoch > 5, top-1 theo F1 triple cuối (`train.py`, `--select path_f1`) | Đã sửa 10-01: +2.9 R@50 |
| 11 | Seed | – | **1** (`main.py:59`) | 42 (theo yêu cầu người dùng) | Một seed nên chưa đo được |
| 12 | Decode khi infer | Sample với STOP | Sample: 10 path từ mỗi điểm bắt đầu, độ sâu tối đa 6, thêm 5 điểm bắt đầu được sample | Best-first, độ sâu 4, mỗi bước tối đa 20 nhánh, lấy cho đủ budget (`infer.py:82`) | Chọn ở EXP-05 vì lấp đủ budget tốt hơn |
| 13 | Path nhãn không phải walk | – | Bỏ qua bước đó | `assert` (đã kiểm tra trước: mọi path hợp lệ) | Không ảnh hưởng |
| 14 | Input cho reasoner | Dạng path, prompt riêng (Table 10, `generate_answers_llama31_8b.py`) | – | Triple, prompt `sys_icl_dc` của SubgraphRAG (`run_reason.sh`) | Paper: path tốt hơn triple +1.7 F1 (Llama, CWQ) |
| 15 | Lưu embedding | – | fp32 | fp16 memmap | Không đáng kể (mục 3) |

**Những điểm khớp:** text encoder; cách dựng line graph (DiGraph, cạnh t_i == h_j); feature [h|r|t], entity không có text dùng vector 0; ứng viên triple đầu (head là topic entity); `q_mlp`, `r_mlp`, `stop_mlp`, `stop_emb`; `z_topic = r_mlp(rel) + z`; cấu trúc path loss (ngữ cảnh cộng dồn, STOP ở cuối); 2 lớp, hidden 512, lr 1e-3, 15 epoch; không self-loop; quy trình tạo ứng viên và prompt gán nhãn.
