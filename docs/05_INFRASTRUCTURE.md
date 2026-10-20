# 05. Hạ tầng và cách chạy

*Cập nhật: 2026-10-01 (quy tắc làm việc với Claude, API trả phí, dọn dữ liệu).*

## 1. Phần cứng và giới hạn

| Tài nguyên | Giá trị | Ghi chú |
|---|---|---|
| GPU | GTX 1080 Ti 11GB (Pascal, compute 6.1) | fp16 chậm trên Pascal nên train bằng fp32 |
| RAM | 31GB, **chỉ ~19GB thực sự khả dụng** | Language server của Antigravity IDE (~6–8GB) và elasticsearch/kibana/neo4j của user khác chiếm ~12GB |
| Đĩa | Còn ~31GB (10-01, sau khi dọn) | Memmap embedding: WebQSP ~9GB, CWQ test ~6.2GB; shard CWQ ~53GB; record full data của QR-DDE ~3GB (`retrieve/exp_qrdde/data/`) |
| Server phụ | `svuet@118.70.181.146 -p 8282` (A2000) | Thường không kết nối được |

## 2. Quy tắc bắt buộc khi chạy job (sau sự cố sập server 09-28)

1. **Chỉ một job nặng mỗi lúc.** Không để train chạy chồng với LLM inference.
2. **Mọi job chạy trong cgroup có giới hạn RAM**, không cho dùng swap:
   ```bash
   systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 <lệnh>
   ```
   Mức thường dùng: job train/infer 8G, reasoner 6G, ollama 7G. Vượt giới hạn thì chỉ job đó bị kill.
3. Trước mỗi bước, chờ tới khi `MemAvailable ≥ 10GB` (đã có sẵn trong các script `run_*.sh`).
4. Embedding đọc qua **memmap fp16**, không nạp cả dict vào RAM; `num_workers=0`.
5. Job dài chạy trong `screen`; script phải chạy lại được (bỏ qua bước đã xong).
6. Xong việc thì tắt ollama (`screen -S ollama_gpu -X quit`) để trả VRAM.

## 3. LLM qua ollama
- Service hệ thống (cổng 11434) **chạy CPU-only** và không restart được vì không có sudo.
- Dùng server riêng có GPU: `~/.ollama_user/serve.sh` (cổng **11435**, model ở `~/.ollama_user/models`: `gemma3:12b`, `llama3.1:latest`, cả hai Q4_K_M).
  ```bash
  screen -dmS ollama_gpu systemd-run --user --scope -q -p MemoryMax=7G -p MemorySwapMax=0 ~/.ollama_user/serve.sh
  ```
- Reasoner: `reason/main.py` (`--ollama_url http://127.0.0.1:11435`), số context đặt qua biến môi trường `OLLAMA_NUM_CTX`.

## 4. Bố cục code

| Thư mục | Nội dung |
|---|---|
| `RAPL/` | Code gốc RAPL của tác giả (tham khảo; `run_rapl_sp_eval.py`, `eval_rapl.py` là bản cũ sai của Phase 16, không dùng) |
| `/relrag/` | **Bản tái hiện REL-RAG**: `prepare_data.py`, `gpt_label.py` (nhãn rational path theo quy trình tác giả, qua OpenRouter hoặc API DeepSeek), `merge_labels.py`, `make_memmap.py`, `dataset.py`, `model.py`, `train.py`, `infer.py`, `eval_retrieval_text.py`, `run_reason.sh`, `compare_reasoners.py`, `e2e_latency.py`, `run_paper.sh` (cấu hình paper, seed 42). Các run, nhãn và script của giai đoạn gemma đã xóa ngày 10-01 |
| `retrieve/` | Retriever SubgraphRAG (MLP + DDE) và các thử nghiệm giai đoạn cũ |
| `retrieve/exp_qrdde/` | **Phương pháp đề xuất (từ 10-20)**: `labels.py` (nhãn gold-query v1 / v2 / v3), `train_retriever.py` (DDE / RT / WL / ASE / RBF), `rbf.py`, `infer.py` (base / AER / ADR), `relscore_bge.py` + `bge_sparql_data.py` (BGE), `prepare_full.py` (record), `vast/remote_{env,train,bge}.sh`. Eval: `retrieve/eval.py` |
| `reason/` | Module reasoning của SubgraphRAG + eval (`metrics/evaluate_results.py` = eval gốc) |

## 5. Dữ liệu và artifact

| Đường dẫn | Nội dung |
|---|---|
| `retrieve/data_files/{webqsp,cwq}/processed/*.pkl` | Subgraph gốc của SubgraphRAG |
| `retrieve/data_files/{webqsp,cwq}/triple_scores/*.pth` | Nhãn shortest-path |
| `/relrag/data/{split}.pkl`, `data/cwq/test.pkl` | Record line graph gọn |
| `/relrag/data/emb_memmap/`, `data/cwq/emb_memmap/` | Embedding GTE fp16 memmap |
| `/relrag/data/labels_gpt4o/`, `labels_deepseek-flash_deepseekapi_noreason/`, `labels_hybrid_gpt4o_deepseek/` | Nhãn rational path: GPT-4o (901 câu train), DeepSeek-V4.1-Flash (2,710 câu), bản ghép dùng để train (EXP-15) |
| `/relrag/runs/<run>/` | Checkpoint (`best.pt`), `log.json`, `retrieval_*/metrics_*.txt`, `retrieval_result_*.pth` |
| `retrieve/exp_qrdde/data/`, `runs/` | Dữ liệu full, checkpoint và kết quả DDE / QR-DDE / AER (subset `data/` đã xóa 10-05; tạo lại bằng `prepare_subsets.py`) |
| `reason/scored_triples/` | Retrieval chính thức của SubgraphRAG (WebQSP, CWQ) |
| `reason/results/KGQA/webqsp/SubgraphRAG/ollama:llama/` | Dự đoán và eval của reasoner |

## 6. Lệnh thường dùng

```bash
conda activate lab          # không cài thêm hay sửa package trong env này
cd /relrag
# train REL-RAG
python train.py --regime path --labels paths:data/labels_hybrid_gpt4o_deepseek/train.jsonl --val_labels sp --seed 42 --no_self_loops --out runs/<name>
# inference (WebQSP hoặc CWQ)
python infer.py --ckpt runs/<name>/best.pt --mode path_bf --budgets 50,100,500 [--dataset cwq] [--no_self_loops]
# so sánh retrieval theo text-match
python eval_retrieval_text.py name=path/to/retrieval_result_500.pth --budgets 20,50,100,500 [--ref <scored_triples.pth>]
# thử nghiệm QR-DDE trên subset
cd ../../retrieve/exp_qrdde && ./run_exp.sh
```

## 7. Quy tắc làm việc với Claude (người dùng chốt trong session 09-30 / 10-01)

1. **Claude không tự chạy job train, inference hay data-prep nặng**, không khởi động, dừng hay chạy lại job của người dùng. Claude chỉ viết code, đưa lệnh chạy và những gì cần kiểm tra trong output. Claude vẫn được kiểm tra cú pháp, đọc log và `result.json`, và chạy script phân tích nhẹ trên file kết quả có sẵn (chỉ dùng CPU, có giới hạn RAM).
2. **Protocol train phải giống baseline gốc trước khi viết code.** Retriever kiểu SubgraphRAG: seed 42 deterministic (`src/setup.set_seed`), `num_epochs 10000`, `patience 10`, chọn theo val triple_recall@100, **một seed duy nhất**. REL-RAG: đúng App. F.1, seed 42, không self-loop như code gốc, cả hai chế độ P và T. Từng xảy ra lỗi: chạy nhầm 25 epoch cố định, phải hủy toàn bộ kết quả.
3. **Log wandb giống `retrieve/train.py`**: project `REL-RAG`; mỗi epoch log `loss`, `num_patient_epochs`, `epoch`; khi val cải thiện mới log `val/*`. Muốn tắt thì đặt `WANDB_MODE=disabled`.
4. **API trả phí** (người dùng có ngân sách hạn chế): luôn ước tính chi phí từ token thật trước, chạy thử 20 câu, người dùng xem rồi mới chạy toàn bộ, và luôn có chốt `--budget`.
5. **Trước khi xóa dữ liệu**: liệt kê từng mục và hỏi xác nhận. Kiểm tra cả config và tài liệu archive xem có kết quả quan trọng nào dựa vào nó không, vì `RAPL/` và `retrieve/data_files` nằm ngoài git nên xóa là mất hẳn.
6. **Không bác bỏ kết quả cũ của người dùng khi chưa đọc archive.** Ví dụ: relation-topic (RT combined) và weighted loss ở giai đoạn SubgraphRAG từng tăng R@50 từ 0.906 lên 0.918–0.935 (`archive/subgraphrag/TOPIC_RELATION_SYNTHESIS.md` §9).

## 8. API LLM trả phí (khóa trong `.env`, không in ra)

| Biến | Dùng cho | Ghi chú |
|---|---|---|
| `OPENROUTER_API_KEY` | GPT-4o qua OpenRouter | Credit đã hết sau 901 câu ($4.02). **OpenRouter đã gỡ mọi endpoint `:batch` của OpenAI từ 2026-08-26**: Batch API trả 400 "does not have a :batch endpoint", nên chỉ gọi được từng request với giá thường ($2.50 / $10 cho 1M token vào / ra) |
| `DEEPSEEK_API_KEY` | API DeepSeek chính chủ (`https://api.deepseek.com/chat/completions`) | Model `deepseek-flash` (= DeepSeek-V4.1-Flash). Thinking **bật mặc định** và bỏ qua `temperature`, nên phải gửi `thinking: {type: "disabled"}`. API không trả số tiền; `gpt_label.py` tự tính theo giá giờ cao điểm ($0.006 / $0.30 / $1.20 cho 1M token cache-hit / cache-miss / output). Giờ thấp điểm (ngoài 01–04h và 06–10h UTC, thứ Hai đến thứ Sáu) rẻ một nửa. Chấp nhận hai tin nhắn `user` liền nhau |

Gán nhãn rational path: `/relrag/gpt_label.py` với các lệnh `build`, `sync`, `labels`, `compare`, cùng cờ `--api`, `--model`, `--provider`, `--no_reasoning`, `--budget`. Sau đó `merge_labels.py` ghép thành nhãn train.

## 9. Dữ liệu đã xóa (không khôi phục được)

| Ngày | Đã xóa | Hệ quả |
|---|---|---|
| 09-30 | `retrieve/data_files/webqsp/emb/bge-m3-v1` (9GB) | **Các run SubgraphRAG tốt nhất cũ dùng BGE-M3 làm encoder** (`wl_effective_num_bge`, R@50 0.935; path+memory hybrid). Muốn tái hiện phải encode lại; checkpoint LoRA `checkpoint-459` không tìm thấy trên đĩa. File điểm relation-topic (`retrieve/data_files/*/relation_topics/`) vẫn còn. Các run dùng GTE (RT combined 0.918, `wl_effective_num` 0.926) vẫn tái hiện được |
| 09-30 | Dự đoán reasoning cũ (gpt-4o-mini, Meta-Llama-3.1-8B-Instruct), `RAPL/data` (Phase 16), một run REL-RAG bị crash | Không ảnh hưởng hướng hiện tại |
| 10-01 | Mọi run, nhãn, script và log của giai đoạn gemma trong `/relrag` (`path_llm*`, `triple_llm*`, `joint_llm`, `path_sp*`, `triple_sp`, `labels_gemma3_12b`, `llm_label.py`, `run_all.sh`, `run_transfer.sh`) | Số liệu cũ chỉ còn trong EXP-01 … EXP-13. REL-RAG phải chạy lại bằng `run_paper.sh` |

## 10. Server ngoài cho reasoning (10-03 / 10-04)

Code chung cho mọi server:
- `reason/llm_utils.py`: thêm backend `-m vllm:<served_name>` qua API tương thích OpenAI, URL lấy từ `VLLM_URL`, timeout 2 giờ (biến `VLLM_TIMEOUT`).
- `reason/main.py --workers N`: gửi N request đồng thời theo cửa sổ trượt nhưng ghi file theo đúng thứ tự, nên cơ chế chạy tiếp không đổi. Một câu lỗi được ghi thành dự đoán rỗng sau 3 lần thử, không làm dừng cả run.
- Script dựng server và chạy: scratchpad `remote/` (Kaggle) và `remote_a4000/`.

| Server | Phần cứng | Ghi chú dựng môi trường | Tình trạng |
|---|---|---|---|
| Kaggle (cloudflared, `~/.ssh/config.free_t4` host `freet4`) | 2× T4 15GB, sm_75, không có bf16 | `pip install vllm` (0.30, torch cu130). **Phải `export LD_LIBRARY_PATH=/usr/local/nvidia/lib64`** trong phiên SSH. Llama-3.1-8B fp16, `--tensor-parallel-size 2`, bản `unsloth/Meta-Llama-3.1-8B-Instruct` (không bị gated). Bỏ cờ `--disable-log-requests` (đã bị xóa ở 0.30) | Đã trả lại cho người dùng 10-04. A1 fp16: relrag_p và dde xong 1,000 câu; dde_reanchor 57 câu |
| A4000 (`ssh -p 8282 svuet@118.70.181.146`, host `a4000`; tài khoản dùng chung, chỉ làm trong `/tmp/minhnv_rag`) | 1× RTX A4000 16GB, sm_86; 62GB RAM; ổ `/` gần đầy, `/tmp` còn ~150GB | Driver 570 (CUDA 12.8): **torch cu130 không chạy được**. Dùng `vllm==0.11.0` (torch 2.8 cu128) + `transformers==4.57.1` + `wandb`, venv tạo từ `~/anaconda3/envs/py310` (base anaconda là Python 3.6). Llama-8B bf16 không vừa 16GB → thử `--quantization fp8` | **10-04 09:38: GPU gặp `CUDA error: unspecified launch failure` khi chạy FP8 (Marlin), sau đó `nvidia-smi` không thấy card. Cần admin reset GPU hoặc reboot** |

Tốc độ đo được (CWQ @50, `max_tokens` 4000):
- Ollama Q4 trên 1080 Ti: ~14 s/câu tuần tự; 4 slot song song chỉ nhanh ×1.57 về thông lượng token.
- vLLM fp16 trên 2× T4 (TP2), 48 request đồng thời: ~5–6 câu/phút. Nút thắt là thông lượng prompt (~400 token/s) và sinh token (~150 token/s).

### 10.1 vast.ai (10-05)
- CLI: venv riêng trong scratchpad (`vastai` 1.8.3). API key đọc từ `.env` (`VASTAI_API_KEY`), không in ra.
- **Phải lọc theo phí băng thông.** Khi search, thêm `inet_down_cost<=0.005` (tức ≤ $5/TB). Instance 54258055 tính $40/TB, nên việc tải Llama 16GB và dataset (18.7GB) tốn ~$0.73, đắt hơn cả 2 giờ GPU.
- **Template vLLM của vast tự phục vụ `Qwen/Qwen3.5-9B` khi khởi động.** Instance tạo theo cách này đã bị `exited`. Thay vào đó, tạo bằng `--image vastai/vllm:v0.30.0-cuda-13.0 --ssh --direct --onstart-cmd 'echo ...'` rồi tự bật model.
- Python của vLLM là `/venv/main/bin/python` (lấy từ shebang của `which vllm`). `python3` hệ thống không có torch.
- Cổng SSH lấy từ `vastai ssh-url <id>`, không dùng trường `ssh_port` (lệch 1).
- `destroy` cần cờ `-y`. Instance đã `stopped` vẫn tốn tiền ổ đĩa cho tới khi destroy.
- Watchdog trên máy lab (`scratchpad/vast_watchdog.sh`, screen `vastwatch`): kéo kết quả về mỗi 5 phút, destroy khi xong / quá 8 giờ / credit < $1 / script báo lỗi.
- RTX 4090 + Llama-3.1-8B bf16: KV cache ~35k token. Với CWQ @50: ~8 câu/phút.
