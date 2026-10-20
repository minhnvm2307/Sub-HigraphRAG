# Chạy lại RAPL bằng code của tác giả: những chỗ khác bản gốc, và rò rỉ nhãn ở bước decode

Setting: checkpoint `webqsp.pt` của tác giả (HF `TianJun1/RAPL`), test trên CWQ (WebQSP → CWQ), chạy trên svuet (`/workspace/svuet/minhnv_subgraphrag`).
Mốc của paper (REL-RAG, Table 12): Llama-3.1-8B, khoảng 50 triple, **64.1 Hit / 53.6 Macro-F1**.

## 1. Những chỗ khác bản gốc, xếp theo mức có thể làm lệch kết quả

| # | Khác gì so với code / cách chạy gốc | Lý do | Nguy cơ lệch benchmark |
|---|---|---|---|
| D1 | **Nhãn test dùng SP thay cho GPT-4o.** `annotated_paths_GPT4o/test/` và `topic_relation_candidates/test_results.pkl` được sinh từ đường ngắn nhất q→đáp án (`s3_make_sp_labels.py`), thay vì `FindRationalPath.py` / `RelationPathTargeting.py` (GPT-4o) | Nhãn hybrid GPT-4o + DeepSeek chỉ có cho WebQSP train/val, không có cho CWQ test. Chạy GPT-4o trên 3,531 câu test thì tốn tiền | **Cao.** Nhãn này quyết định điểm xuất phát của decode (xem §2). Nhãn SP chỉ đánh dấu các chuỗi relation ngắn nhất (tối đa 10). GPT-4o chọn các đường "hợp lý", có thể dài hơn hoặc ít hơn. Kết quả có thể cao hoặc thấp hơn bản dùng nhãn GPT-4o |
| D2 | **LLM:** ollama `llama3.1:8b` **Q4_K_M**, greedy; tác giả dùng HF bf16 với **beam 5**, `length_penalty 1.2` | Ollama không hỗ trợ beam search; không chạy LLM ở máy local | **Trung bình.** Beam 5 và bf16 thường cho câu trả lời ổn định hơn. Q4 từng làm mất khoảng 1 Hit (EXP-07). Các tham số `max_new_tokens 500`, `repetition_penalty 1.25` (áp trên toàn context), `top_p 1.0` được giữ |
| D3 | **Chấm điểm:** bản công bố **không có script chấm**, paper không mô tả cách chấm | Phải tự viết `score_answers.py` | **Cao khi so với paper.** Báo ba con số. File tác giả lưu có **cả prompt lẫn câu trả lời** (HF `generate` trả về prompt + completion), nên chấm "đáp án xuất hiện trong text" sẽ đếm cả đáp án chỉ có trong đường đi đã retrieve |
| D4 | `test.pkl` dựng bằng chính `EmbInferDataset` của tác giả từ HF `rmanluo/RoG-cwq`, nhưng **không chạy phần embedding GTE** của `emb.py`; dùng lại `test.pth` của SubgraphRAG | Tránh embed lại cả CWQ | **Thấp.** Script kiểm tra id và thứ tự entity/relation trùng 100% với bản SubgraphRAG mới dùng lại. Bộ encoder (GTE-large-en-v1.5) và code embed là cùng một bản (RAPL lấy từ SubgraphRAG) |
| D5 | Chỉ chạy **khối test** của `MakeTrainData.py` (`RetrieverDataset(split='test', skip_no_path=False)`), không chạy train/val | Không cần train/val khi chỉ đánh giá | **Không có.** Test được tính độc lập với train/val |
| D6 | Post-process test bằng `s4_build_test_graphs.py`, chép từng dòng của `load_post_processed_dataset.py`, chỉ đổi split `train` → `test` và thư mục ra `post_processed_pathLabel` → `post_processed_pathLabel_4o_corrected` | Bản công bố không có script nào tạo dữ liệu test mà `model_inference.py` đọc | **Không có**, nếu bản tác giả dùng cho paper cũng xử lý test theo cách này (không kiểm chứng được) |
| D7 | Không chạy `load_post_processed_dataset_test.py`; gọi `KGDataset(split='test')` trực tiếp | Script đó dựng val trước, mà val cần dữ liệu không dùng tới | **Không có.** Cùng class, cùng split |
| D8 | Symlink `RAPL/relrag` → `../relrag` để `model_dir = "relrag/runs/authors_ckpt/webqsp.pt"` trỏ đúng (dòng này trên svuet đã được đổi từ `"xxx.pt"`) | Không sửa file | **Không có.** Đã kiểm tra load đủ 21/21 tensor, không thiếu, không thừa key |
| D9 | Tạo thư mục rỗng `data/cwq/processed/{train,val}` | Nếu thiếu, PyG sẽ dựng lại toàn bộ line graph mỗi lần khởi tạo `KGDataset` | **Không có** |
| D11 | **Sửa lỗi lúc chạy cho `KGDataset._paths_to_triplets`** (file gốc không đổi; ghi đè trong `s4_build_test_graphs.py`): gọi nguyên hàm gốc, chỉ truyền bảng tên entity đã bỏ khoảng trắng đầu/cuối | Code gốc **dừng hẳn** ở 51/3,531 câu CWQ test có tên entity bắt đầu bằng khoảng trắng (ví dụ `'  Adventures by Disney - 5-Night…'`): hàm tách chuỗi theo `->` rồi `.strip()` từng đoạn nên không tra ra tên. Bản cùng tên trong `PostProcessedDataset` thì đã tự xử lý trường hợp này (in cảnh báo rồi dùng id dự phòng). Như vậy code công bố không chạy hết được tập test CWQ | **Thấp.** Chỉ ảnh hưởng nhãn đường đi của 51 câu, và các câu đó lấy đúng id thay vì dừng hay dùng id dự phòng. Nhưng cũng cho thấy bản tác giả dùng cho paper khác bản công bố |
| D12 | **Lưu line graph không kèm `x`, dựng lại `x` khi load** (`slim_x.py`, ghi đè `torch.save`/`torch.load` lúc chạy; `model_inference.py` vẫn chạy nguyên văn qua `run_model_inference.py`) | Tác giả ghi sẵn `x = [h_emb \| r_emb \| t_emb]` (3,072 fp32) cho mỗi triple vào mọi file, rồi bước post-process ghi thêm một bản sao: khoảng 77 MB mỗi câu, tổng khoảng 175 GB × 2. Lần chạy đầu đã làm đầy 100% ổ dùng chung của svuet. `x` hoàn toàn suy ra được từ id (h, r, t), vốn đã có trong metadata của tác giả, và từ file embedding | **Không có.** `x` được dựng lại bằng chính vòng lặp của tác giả (chép nguyên văn), từ cùng file embedding. `verify_slim.py` kiểm trên 40 câu thật (gồm 10 câu subgraph lớn nhất): **0 thuộc tính khác** sau vòng lưu/load, kể cả sau lần lưu lại thứ hai. Object có thêm vài thuộc tính `slim_*` mà không đoạn code nào của tác giả đọc tới. Dung lượng giảm từ 76.7 MB xuống 9.4 MB mỗi câu |
| D10 | Môi trường: torch 2.1.0 + cu121, PyG 2.5.3 (venv `.venv_rapl`) | Tác giả không công bố phiên bản | **Thấp.** Decode có lấy mẫu ngẫu nhiên (`torch.multinomial`, seed 1), nên khác phiên bản có thể khác vài câu |

**Giữ y nguyên của tác giả:**
- `model_inference.py` chạy nguyên văn, decode `K=800, M=1200, N=3, way='normal'`, seed 1.
- Kiến trúc GCN 2 lớp hai chiều, `use_stop_mlp`, `in_dims` 3072, `hidden` 512.
- Prompt (đọc `post_str` nguyên văn từ file của họ), chat template Llama-3.1, định dạng file lưu.

**Không giữ budget 50 triple:** decode của tác giả không có giới hạn số triple. EXP-31 đo được trung bình 67–73 triple mỗi câu trên CWQ, trong khi paper nói khoảng 50. Đây là hành vi của chính code gốc, mình không sửa.

## 2. Rò rỉ đáp án ở bước decode: `topic_labels`

### 2.1 Decode bắt đầu từ đâu
`Trainerv3.decoding(..., way='normal')` ([Trainerv3.py:1076-1117](../RAPL/src/model/Trainerv3.py#L1076-L1117)) chọn **tập nút xuất phát** (mỗi nút là một triple, vì đây là line graph) theo hai bước:

1. `topic_candidate_nodes = { nút có topic_labels == 1 }`: lấy **toàn bộ** các nút được gán nhãn là điểm xuất phát.
2. Lấy K = 800 mẫu (có hoàn lại) từ `topic_candidates`, theo softmax của độ tương đồng với câu hỏi, rồi **hợp** vào tập ở bước 1.

Từ mỗi nút xuất phát, model mở rộng một đường greedy và N = 3 đường ngẫu nhiên; cuối cùng giữ M = 1200 đường có xác suất cao nhất.

- `topic_candidates` ([PyG_dataset_disk.py:276-287](../RAPL/src/dataset/PyG_dataset_disk.py#L276-L287)) gồm mọi triple có đầu là topic entity của câu hỏi. Trường này **hợp lệ**, vì chỉ dùng thông tin mà câu hỏi cho biết.
- `topic_labels` **không hợp lệ** ở test, lý do ở mục tiếp theo.

### 2.2 `topic_labels` được dựng từ đâu
`ProcessedDiskDataset._update_topic_labels` ([PostProcessedDataset.py:164-197](../RAPL/src/dataset/PostProcessedDataset.py#L164-L197)) đặt `topic_labels = 1` cho một triple xuất phát khi:

- **(A)** relation của nó nằm trong danh sách relation do LLM gán nhãn cho câu đó (`topic_relation_candidates/<split>_results.pkl`, sinh bởi `RelationPathTargeting.py`); **hoặc**
- **(C)** nó là **bước đầu tiên của một đường đi hợp lý đã gán nhãn** (`path_label`).

`path_label` lại được dựng như sau:
1. `FindRationalPath.py` cho GPT-4o xem các **đường đi ứng viên từ topic tới đáp án**, rồi chọn ra đường hợp lý.
2. Đường ứng viên do `RetrieverDataset._extract_translated_path_and_relations` sinh bằng cách liệt kê đường đi `q_entity → a_entity` trong subgraph ([retriever_v2.py:387-395](../RAPL/src/dataset/retriever_v2.py#L387-L395)). Nghĩa là **đáp án đúng được dùng để sinh ứng viên**.
3. Đường được chọn được đối sánh với ứng viên (`_match_candidate_paths`) và đánh dấu bậc của từng triple trên đường.

Pipeline công bố tạo cả hai loại nhãn này **cho tập test**: `FindRationalPath.py` đọc `test_text_dict_list.pickle`, `RelationPathTargeting.py` xuất `test_results.json`. Muốn dựng split test thì bắt buộc phải có chúng ([PostProcessedDataset.py:47-52](../RAPL/src/dataset/PostProcessedDataset.py#L47-L52)).

### 2.3 Vì sao đây là rò rỉ
Theo (C), decode **luôn được đặt sẵn tại triple đầu tiên của đường đúng tới đáp án**. Sau đó việc mở rộng greedy chỉ cần đi tiếp vài bước. Ví dụ:
- Câu CWQ test đầu tiên: *"Lou Seal is the mascot for the team that last won the World Series when?"*, đáp án *2014 World Series*.
- Đường đúng đi `Lou Seal → sports.mascot.team → San Francisco Giants → … → 2014 World Series`.
- Triple đầu `(Lou Seal, sports.mascot.team, San Francisco Giants)` được đánh dấu xuất phát **vì nó nằm trên đường tới đáp án**, chứ không phải vì model chọn nó. Model chỉ còn phải đi nốt các bước sau.

Một retriever dùng trong thực tế không biết đáp án, nên chỉ có được bước 2 (K mẫu theo độ tương đồng). Bước 1 là thông tin mà test không được phép có.

### 2.4 Ảnh hưởng đo được (EXP-31, checkpoint của tác giả, decode của tác giả)

| Checkpoint → test | Không rò rỉ (chỉ K mẫu) | Có rò rỉ (thêm điểm xuất phát từ shortest path) | Chênh ansR |
|---|---|---|---|
| `webqsp.pt` → WebQSP | 91.11 | 95.91 | +4.8 |
| `webqsp.pt` → CWQ (OOD) | 66.88 | 72.95 | **+6.1** |
| `cwq.pt` → CWQ | 72.16 | 76.18 | +4.0 |

Lần chạy này **có** rò rỉ, theo đúng code gốc, nhưng nguồn nhãn là SP (D1). Vì vậy đây là con số "như code công bố, với nhãn SP", không phải một retriever không rò rỉ.

## 3. Kết quả lần chạy này

**Retrieval (bước 1–5, xong 10-10 23:32):**
- `model_inference.py` của tác giả xuất đủ **3,531/3,531** file, không câu nào bị `try/except` bỏ qua.
- Decode trung bình 2.63 s/câu.
- **Trung bình 123.5 triple mỗi câu** (`triplet_count_dict.txt`). Paper nói setting Llama-8B dùng khoảng 50 triple; bản tái hiện decode của tác giả ở EXP-31 (code của mình) ra 67–73. Như vậy code công bố đưa cho LLM **nhiều gấp khoảng 2.5 lần** số triple mà paper mô tả. Trong 20 câu thử, có câu dài tới 8,192 token prompt.
- Thời gian: dựng line graph 3.5 giờ, post-process 1 giờ, inference 2.6 giờ, đều một luồng (xem phần thảo luận chi phí).

**Bước LLM (bước 6): đang tạm dừng vì GPU svuet hỏng tản nhiệt.**
- Khi chạy thử 20 câu đo được 70.8 s/câu, tốc độ sinh chỉ khoảng 4–10 token/s.
- Nguyên nhân: GPU A4000 ở **99–103 °C** (kể cả khi không chạy gì), quạt 100%, xung nhịp 105–210 MHz trên mức tối đa 2,100, throttle `0x68` (quá nhiệt phần cứng và phần mềm). Với tốc độ này, 3,531 câu cần khoảng 69 giờ.
- 22 câu trả lời đã sinh được giữ lại (`llama31_8b_generated_answers/`). Script chạy tiếp được từ chỗ dừng.
- Khi chạy tiếp cần tăng `num_ctx` lên 16,384, vì có prompt chạm 8,192 token và bị cắt.

**Bước LLM chạy lại trên OpenRouter (10-11):**
- `meta-llama/llama-3.1-8b-instruct`, nhà cung cấp ghim cố định DeepInfra **fp8**, greedy, 3,531/3,531 câu.
- Tổng **$0.20** (7.35M token vào, 1.32M token ra). 952 câu (27%) dừng ở trần 500 token.
- Script `generate_answers_openrouter.py`. Câu trả lời lưu ở `rapl_faithful/runs/webqsp2cwq/llama31_8b_generated_answers/`.
- 22 câu chạy trước bằng ollama Q4 được tách ra `…_ollama_q4_partial/` và **không** dùng để chấm.

| Cách chấm (3,531 câu CWQ test) | Hit | Macro-F1 |
|---|---|---|
| Đáp án xuất hiện ở bất kỳ đâu trong file đã lưu (prompt + trả lời, đúng thứ tác giả lưu) | **70.6** | – |
| Đáp án xuất hiện trong phần trả lời của model | **48.1** | – |
| Dòng "The answer is: …" | 29.0 | 24.9 |
| *Paper REL-RAG, Table 12 (WebQSP → CWQ, Llama-3.1-8B)* | *64.1* | *53.6* |

**Đọc kết quả:**
- Chạy đúng code công bố, kèm nhãn test kiểu SP (D1), thì Hit tính trên câu trả lời của model là **48.1**, thấp hơn paper 16 điểm. Con số này khớp với phép chiếu ở EXP-31: Hit chiếu từ retrieval của chính tác giả là 46.5–49.1.
- Con số 64.1 của paper **chỉ đạt được nếu chấm cả prompt** (70.6). Mà file tác giả lưu đúng là có lẫn prompt. Vì không có script chấm của tác giả, đây là cách giải thích hợp lý nhất, nhưng chưa chứng minh được.
- Ba yếu tố còn có thể kéo con số xuống so với setup của tác giả: greedy thay cho beam 5 (D2), fp8 thay cho bf16, và nhãn SP thay cho GPT-4o (D1). Theo các mức đo được ở EXP-07 và EXP-31, ba yếu tố này khó bù được khoảng chênh 16 điểm.
- Retrieval đưa cho LLM trung bình **123.5 triple/câu**, không phải khoảng 50 như paper mô tả.

**Chấm lại độc lập (10-11)**, từ chính các file `sample_*.txt` và câu trả lời của run này:
- Ba con số QA ở trên được tái tạo y hệt: 70.60 / 48.09 / 29.03 / F1 24.87. Tỉ lệ có dòng "The answer is" là 80.3%.
- **Retrieval (có leak, nhãn SP).** Tính trên 2,848 câu có đáp án trong graph:
  - ansR **71.60**, ansHit 73.07.
  - Khớp với bản decode tái hiện có leak ở EXP-31 (72.95). Bản không leak ở EXP-31 là 66.88.
  - Số triple khác nhau mỗi câu: trung bình 86.1, median 31, p90 243, max 1,070. Con số 123.5 của tác giả đếm lặp các triple dùng chung giữa các path.
- **Hit theo tập con:**
  - Trên 2,848 câu có đáp án trong graph: answer-only 54.21, parsed 31.60.
  - Trên 1,000 câu đầu (cùng tập với EXP-26): answer-only 49.90, parsed 32.20.
- **Utilization:**
  - P(trả lời đúng | đáp án có trong context) = 62.3% (n = 2,104).
  - P(trả lời đúng | không có) = 27.2% (n = 1,427). Lý do: kiến thức sẵn có của Llama, cộng với việc chuỗi trả lời dài kiểu chain-of-thought dễ chứa đáp án theo kiểu chuỗi con.
