# Chạy lại RAPL đúng code của tác giả: WebQSP → CWQ (checkpoint `webqsp.pt`)

Mục tiêu: chạy **nguyên code RAPL đã công bố** (thư mục `RAPL/`, không sửa dòng nào), dùng checkpoint của tác giả, để so với paper (REL-RAG Table 12, WebQSP→CWQ, Llama-3.1-8B, khoảng 50 triple: **64.1 Hit / 53.6 Macro-F1**).

Mọi thứ thêm vào đều nằm trong `rapl_faithful/`. Bên trong `RAPL/` chỉ có thêm symlink, thư mục rỗng và dữ liệu sinh ra.

## Những điểm buộc phải xử lý (code của tác giả vẫn giữ nguyên)

| # | Vấn đề trong bản công bố | Cách xử lý (không sửa code của tác giả) |
|---|---|---|
| 1 | `model_inference.py` load checkpoint từ đường dẫn ghi cứng `"xxx.pt"`, với `strict=False` | Symlink `RAPL/xxx.pt` → `webqsp.pt`. Đã kiểm tra: load đủ 21/21 tensor, không thiếu, không thừa key |
| 2 | Không script nào tạo dữ liệu **test** đã post-process: `load_post_processed_dataset.py` chỉ chạy `['train']` và ghi vào `post_processed_pathLabel/`, trong khi inference đọc `post_processed_pathLabel_4o_corrected/` | `s4_build_test_graphs.py` gọi đúng các class của tác giả (`KGDataset`, `ProcessedDiskDataset`), theo từng dòng của `load_post_processed_dataset.py`, chỉ đổi split thành `test` và thư mục ra thành thư mục mà inference đọc |
| 3 | Decode khởi đầu từ `topic_labels`, mà trường này dựng từ **nhãn của tập test** (rational path của GPT-4o, relation do LLM gán nhãn; cả hai đều cần đáp án) | Nhãn hybrid GPT-4o + DeepSeek chỉ có cho WebQSP train/val, không có cho CWQ test, nên **dùng nhãn SP** (`s3_make_sp_labels.py`): các chuỗi relation ngắn nhất trong chính tập đường đi ứng viên của tác giả, ghi đúng định dạng file của họ |
| 4 | `MakeTrainData.py` xử lý cả train và val của CWQ trước khi tới test | `s2_build_test_retrieval.py` chỉ chạy đúng khối test của nó (`RetrieverDataset(..., split='test', skip_no_path=False)`) |
| 5 | `emb.py` embed lại toàn bộ CWQ bằng GTE | Dùng lại embedding và subgraph SubgraphRAG/RoG đã có. Đã kiểm tra: câu hỏi, thứ tự id entity và id relation **trùng hoàn toàn** với `test_id_entity_mapping.pkl` của tác giả (3,531/3,531 câu) |
| 6 | Sinh câu trả lời bằng HF transformers ở máy local | `generate_answers_ollama.py`: cùng prompt (đọc `post_str` nguyên văn từ file của họ), cùng chat template Llama-3.1, cùng định dạng file lưu; LLM chạy bằng ollama trên svuet |

**Khác biệt không tránh được ở bước 6:**
- Ollama không có beam search. Tác giả dùng `num_beams=5`, `length_penalty=1.2`; ở đây chạy greedy (`temperature 0`).
- Model là `llama3.1:8b` bản Q4_K_M, tác giả dùng bf16.
- Các tham số được giữ: `num_predict 500`, `repeat_penalty 1.25` áp trên toàn bộ context (giống HF), `top_p 1.0`, seed cố định.

**Cần biết khi chấm:**
- Tác giả lưu `batch_decode(generate(...))`. Với model chỉ có decoder, hàm này trả về **prompt + câu trả lời**, nên file kết quả của họ chứa luôn các đường đi đã retrieve. Script mình viết lưu y hệt như vậy.
- Vì thế `score_answers.py` tách ba cách chấm: Hit trên toàn bộ text đã lưu (chỉ là cận trên), Hit chỉ trên câu trả lời, và Hit/F1 trên dòng "The answer is:".

## Nên chạy ở đâu

- **Bước 2–5:** nặng vừa phải. Bước 4 nạp `test.pth` 6.6 GB vào RAM và dựng 3,531 line graph (paper ghi khoảng 0.6 s và 100 MB mỗi graph, tức khoảng 35–40 phút). Nếu máy hiện tại không được chạy thì làm trên svuet (62 GB RAM) theo mục **A**. Nếu chạy ở local thì dùng mục **B**.
- **Bước 6 (LLM):** ollama trên svuet. Script chỉ gửi HTTP qua SSH tunnel, nên tự thân nó rất nhẹ.

### A. Chuẩn bị svuet (một lần)
```bash
# ở máy local: chép đúng cấu trúc thư mục sang svuet (khoảng 7 GB)
cd /home/minhnv/Lab/SubgraphRAG
rsync -aR --info=progress2 -e "ssh -p 8282" \
  RAPL rapl_faithful \
  retrieve/data_files/cwq/processed/test.pkl retrieve/data_files/cwq/entity_identifiers.txt \
  retrieve/data_files/cwq/emb/gte-large-en-v1.5/test.pth \
  relrag/runs/authors_ckpt/webqsp.pt relrag/runs/authors_ckpt/cwq.pt relrag/runs/authors_ckpt/cwq/test_id_entity_mapping.pkl \
  svuet@118.70.181.146:/tmp/minhnv_rag/SubgraphRAG/

# trên svuet: venv riêng (torch 2.1 cu121 chạy được với driver 570)
ssh -p 8282 svuet@118.70.181.146
~/anaconda3/envs/py310/bin/python -m venv /tmp/minhnv_rag/venv_rapl && source /tmp/minhnv_rag/venv_rapl/bin/activate
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.5.3 networkx pandas pyyaml pydantic termcolor wandb tqdm
export SGRAG_ROOT=/tmp/minhnv_rag/SubgraphRAG
```

### B. Hoặc ở local
```bash
export SGRAG_ROOT=/home/minhnv/Lab/SubgraphRAG
PY=/home/minhnv/.conda/envs/lab/bin/python     # có đủ thư viện; nhớ bọc bằng systemd-run giới hạn RAM
```

## Các bước (chạy trong `screen`; bước nào cũng chạy lại được)

```bash
cd $SGRAG_ROOT
bash rapl_faithful/setup_links.sh webqsp          # 1. symlink: dữ liệu, mapping của tác giả, RAPL/xxx.pt -> webqsp.pt
cd RAPL
python ../rapl_faithful/s2_build_test_retrieval.py   # 2. test_retrieval.pkl (code của tác giả; tính đường đi q→đáp án)
python ../rapl_faithful/s3_make_sp_labels.py         # 3. nhãn SP cho test, đúng định dạng của tác giả
python ../rapl_faithful/s4_build_test_graphs.py      # 4. line graph test + post-process (class của tác giả)
python model_inference.py --num_layers 2 --dataset_name cwq --bidirectional \
       --split test --use_stop_mlp --wandb_id webqsp2cwq          # 5. NGUYÊN script của tác giả
```

**Kiểm tra sau mỗi bước:**
- Bước 2 in ra `test samples: 3531`.
- Bước 3 in ra số câu có nhãn SP, khoảng 80% (bằng tỉ lệ câu có đáp án trong graph); số còn lại có nhãn rỗng.
- Bước 4 in `3531 3531 3531` và assert phải qua.
- Bước 5:
  - Đếm file: `ls experiments/text_from_KG/webqsp2cwq/cwq/K_800_M_1200/GCN_bidirectional_True_numLayers_2_useStopMlp_True_numHeads_4_K_3/seed_1/sample_*.txt | wc -l` phải ra 3,531. Script dùng `try/except` và **âm thầm bỏ qua** câu decode lỗi.
  - File `experiments/text_from_KG/webqsp2cwq/cwq/triplet_count_dict.txt` ghi số triple trung bình. EXP-31 đo được khoảng 67–73 triple mỗi câu trên CWQ, trong khi paper nói khoảng 50.
- Cấu hình decode do tác giả ghi cứng: `K=800, M=1200, N=3, way='normal'`, seed 1.

### 6. Sinh câu trả lời bằng ollama trên svuet

Ollama trên svuet là **service hệ thống dùng chung** (cổng 11434, model `llama3.1:8b` Q4_K_M đã được nạp sẵn). Không có quyền đổi biến môi trường của server, nên script tự chống việc nạp lại model như sau:
- **`num_ctx` cố định** cho mọi request (mặc định 8192). Một `num_ctx` khác sẽ khiến ollama nạp lại model.
- **`keep_alive: -1`** để model ở lại trong VRAM giữa các request.
- Gửi một request khởi động trước khi bắt đầu.
- **Chạy tuần tự** (`--workers 1`), vì server không bật `OLLAMA_NUM_PARALLEL`; gửi song song chỉ làm request phải xếp hàng.

```bash
# tunnel từ nơi chạy script tới ollama của svuet (bỏ qua nếu chạy ngay trên svuet: --url http://localhost:11434)
ssh -N -f -L 21434:localhost:11434 -p 8282 svuet@118.70.181.146
OUT=$SGRAG_ROOT/RAPL/experiments/text_from_KG/webqsp2cwq/cwq/K_800_M_1200/GCN_bidirectional_True_numLayers_2_useStopMlp_True_numHeads_4_K_3/seed_1
python $SGRAG_ROOT/rapl_faithful/generate_answers_ollama.py --file_dir $OUT --url http://localhost:21434 --limit 20   # chạy thử
python $SGRAG_ROOT/rapl_faithful/generate_answers_ollama.py --file_dir $OUT --url http://localhost:21434             # chạy đủ, tự bỏ qua câu đã làm
```
- Lần chạy thử in ra số giây mỗi câu và thời gian còn lại ước tính.
- Nếu thấy `WARNING ... prompt ... > num_ctx` thì prompt bị cắt: tăng `--num_ctx` (ví dụ 12288) cho **cả lần chạy**.
- Log từng câu nằm ở `$OUT/llama31_8b_generated_answers/_ollama_log.jsonl`.
- Kiểm tra model vẫn ở trong VRAM: `curl -s localhost:11434/api/ps` trên svuet (xem `context_length` có đúng `num_ctx` không). Nếu người khác dùng `num_ctx` khác, model sẽ bị nạp lại xen kẽ.

### 7. Chấm điểm
```bash
python $SGRAG_ROOT/rapl_faithful/score_answers.py --file_dir $OUT
```
- Kết quả in ra màn hình và ghi vào `$OUT/scores.json`, gồm `Hit_saved_text` (cận trên, có lẫn prompt), `Hit_answer_only`, `Hit_parsed`, `MacroF1_parsed`.
- Tính trên cả 3,531 câu; câu không có file trả lời được tính 0.
- So với paper: 64.1 Hit / 53.6 F1.

## Danh sách file mình đã tạo (không sửa file nào của tác giả)
- `rapl_faithful/setup_links.sh`: chỉ tạo symlink và thư mục.
- `rapl_faithful/s2_build_test_retrieval.py`, `s3_make_sp_labels.py`, `s4_build_test_graphs.py`: gọi class và config của tác giả.
- `rapl_faithful/generate_answers_ollama.py`: thay backend LLM, giữ prompt, template và định dạng file lưu.
- `rapl_faithful/score_answers.py`: bộ chấm (bản công bố không có).
