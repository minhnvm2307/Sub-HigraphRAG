# 18. Đánh vào đâu sau RBF: từ "xếp hạng triple" sang "kiểm chứng ứng viên"

*Tạo: 2026-10-22. Căn cứ: bảng trần nhãn v1/v2 của bạn, bản tổng kết 10-23, sáu panel phân tích RBF, `17`. Ràng buộc giữ nguyên: không gọi LLM ở tầng retrieval.*

---

## 0. Trả lời ngắn

**Đánh vào bước chọn ứng viên ở cuối chain, không đánh vào xếp hạng triple nữa.**

Lý do nằm ngay trong bảng trần của bạn. Trên CWQ:

| | Hit / F1 | Chênh |
|---|---|---|
| RBF v2 hiện tại | 65.4 / 55.4 | |
| Oracle: đưa đúng **chain** gold (nhãn v1) | 67.8 / 58.2 | **+2.4** so với RBF: toàn bộ phần còn lại của bài toán *xếp hạng* |
| Oracle: đưa đúng chain **và chỉ các điểm cuối đúng** (nhãn v2) | 76.5 / 66.1 | **+8.7** so với v1: giá trị của việc *chọn ứng viên* |

Retriever đã cách oracle chain 2.4 Hit — dưới mức nhiễu đổi server (2–3 Hit) mà chính bạn đo. **Mọi cải tiến xếp hạng triple từ đây sẽ không đo được.** Khoảng v1 → v2 thì gấp 3.6 lần và nằm trên mức nhiễu. Đó là chỗ duy nhất còn điểm mà không cần dữ liệu mới và không cần reasoner mạnh hơn.

![Ngân sách điểm trên CWQ theo từng đòn bẩy]({{artifact:art_08c12b2d-3e37-42e1-89b4-a6aa687a622a}})

**Hình 1.** Số điểm Hit còn lấy được trên CWQ, theo từng nguồn, lấy từ bản tổng kết 10-23 (§2.2, §3) và bảng trần nhãn. Dải xám là mức nhiễu đo được khi đổi server vLLM với cùng model và cùng retrieval. Khoảng "chọn điểm cuối" là cận trên vì nhãn v2 có rò đáp án; phần đạt được thật phải đo bằng thí nghiệm E0 ở §3.

Ba nguồn bằng chứng độc lập cùng chỉ vào chỗ này:

1. **Bảng trần:** v1 → v2 = +8.7 Hit CWQ, +1.6 Hit / +3.1 F1 WebQSP.
2. **`17` §3 (kiểm định trên 110 câu):** đặc trưng dự đoán được LLM sai là **số ứng viên có tên** (p = 0.008), không phải hạng của triple đáp án (p = 0.77). LLM có bằng chứng đúng nhưng bị các ứng viên sai trông hợp lý dẫn đi.
3. **Giới hạn nội tại của RBF mà bạn tự ghi:** *π(r|q) không nhìn entity, nên các chain song song cùng relation có điểm bằng nhau.* Đây không phải lỗi cần vá trong RBF — nó là **tính chất làm RBF tổng quát hoá tổ hợp được**. Nhưng nó có nghĩa RBF, về mặt cấu trúc, **không thể** phân biệt Nelson Rockefeller với Richard Nixon trên cùng chain `vice_president`. Phải có một thành phần khác làm việc đó.

---

## 1. Panel (b) và (c) nói gì

| | WebQSP (ID) | CWQ (OOD) |
|---|---|---|
| (c) chain gold đứng hạng 1 | 78% | **43%** |
| (c) chain gold trong top-3 | ~94% | 75% |
| (c) chain gold trong top-5 | ~96% | **86%** |
| (b) đường tốt nhất dẫn tới đáp án (τ = 0) | 78% | 43% |

Trên CWQ, RBF chọn đúng chain ở hạng 1 chưa tới một nửa số câu, nhưng có chain đúng trong top-5 ở 86% số câu. Nghĩa là **tập top-5 chain đã chứa câu trả lời; cái thiếu là cách chọn trong tập đó**. Và tiêu chí chọn tốt nhất không nằm ở chain mà ở **điểm cuối của chain**: một chain đúng sẽ dẫn tới ứng viên có đúng kiểu, nối được với các entity khác trong câu hỏi; một chain sai thường dẫn tới ứng viên không thoả ràng buộc nào.

Đây là ý cốt lõi: **kiểm chứng chain bằng điểm cuối của nó**. Nó sửa cùng lúc hai việc — chọn chain (43% → ?) và chọn ứng viên trong chain (v1 → v2).

Panel (e) cho cùng kết luận từ hướng khác: ở cùng ansR, cấu hình nào ít ứng viên có tên hơn thì đứng cao hơn. Ứng viên có tên là thứ cần tối thiểu hoá, không phải số triple.

---

## 2. Mô hình đề xuất: RBF + bộ kiểm chứng ứng viên

### 2.1 Đổi đơn vị của bài toán

| | Hôm nay | Đề xuất |
|---|---|---|
| Đơn vị retriever xuất ra | top-k **triple** | top-k **chain** → tập **ứng viên** ở cuối mỗi chain → ứng viên **qua kiểm chứng** + bằng chứng của nó |
| Ai phân biệt các ứng viên anh em | LLM, khi đọc 50 triple | Một bộ chấm nhỏ, token-free, trước khi gửi cho LLM |
| Câu nhiều anchor | max qua các anchor (hợp) | **giao tập** ứng viên (chính xác, không cần hiệu chuẩn) |
| LLM nhận gì | 50 triple xếp theo điểm | mỗi chain: các ứng viên còn lại + triple ràng buộc chứng minh chúng |

Đây là hướng "đổi đơn vị của bài toán" mà §3.4 của bản tổng kết đã nêu, nhưng **không cần sinh chương trình**: chain là chương trình (RBF đã sinh), còn việc thực thi là đi theo chain trong subgraph — thứ `bind_to_subgraph` đã làm.

### 2.2 Ba thành phần

**(a) Sinh chain — giữ nguyên RBF.** Top-k chain theo điểm max-product, k = 5 (bao 86% CWQ, 96% WebQSP). Không đụng vào π. Toàn bộ khả năng tổng quát hoá tổ hợp đã chứng minh được giữ lại.

**(b) Thực thi chain.** Với mỗi chain, đi theo từng predicate từ anchor trong subgraph, thu tập điểm cuối có tên. Đây là deterministic, không học gì.

**(c) Bộ chấm ứng viên — thành phần mới duy nhất.** Với mỗi ứng viên c ở cuối chain π, tính một điểm s(c | q, π) từ **đặc trưng không phụ thuộc vào relation của chain**:

| Đặc trưng | Vì sao nó tổng quát hoá |
|---|---|
| c có nối (≤ 2 hop) với một entity khác được nhắc trong câu hỏi không, và nối qua relation nào (text) | Đây là phép giao của câu conjunction, làm ở mức tập — chính xác, khác soft-AND đã thất bại ở mức điểm triple |
| cos(câu hỏi, text của `notable_types` của c) | Kiểu đáp án ("trường đại học", "thành phố") nói trong câu hỏi; ràng buộc kiểu có mặt 93–98% trong subgraph |
| cos(câu hỏi, (r, o)) lớn nhất trên các cạnh 1-hop của c ra ngoài chain | Ràng buộc giá trị (bằng cử nhân, vai diễn) |
| Điểm của chain π từ RBF (đông cứng) | Mang thông tin chain vào mà không cho bộ chấm học lại nó |
| Số anh em cùng chain | Để bộ chấm biết khi nào không cần lọc (chain chỉ có 1 điểm cuối) |

Train bằng **softmax listwise giữa các anh em cùng một chain**, nhãn dương là đáp án gold — đúng thông tin mà nhãn v2 dùng, nhưng giờ nó được *học* chứ không phải *đưa thẳng vào context*. Chỉ dùng train split.

### 2.3 Vì sao không lặp lại các thất bại cũ

Bạn đã thử đưa thông tin mức entity vào RBF ba lần và cả ba đều thất bại. Thiết kế trên được dựng để tránh đúng ba lý do đó:

| Lần thử cũ | Vì sao thất bại | Khác biệt ở đây |
|---|---|---|
| ψ(v\|q) đặt **trong** lan truyền (`11` §7) | val ansR@50 81–84 so với 88: nhân tử nút làm hỏng phân rã theo bước | Bộ chấm chạy **sau** khi đã chọn chain, chỉ ở điểm cuối. π không bị đụng tới |
| Hybrid MLP + RBF (EXP-39) | "mọi bộ chấm MLP kết hợp được đặc trưng bước-sau với relation đều học thuộc tổ hợp": nhóm tổ hợp mới tụt 84.0 → 51.2 | Bộ chấm **không thấy relation của chain**, nên không có gì để học thuộc. Chỉ phân biệt *giữa các anh em*, không thể đổi chain nào thắng |
| RBF-PoE: MLP cục bộ trên (q, h, r, t) | MLP không có ngữ cảnh nên không đánh giá được bước qua CVT, kéo tích xuống | Bộ chấm nhìn **lân cận của ứng viên** (kiểu, liên kết với entity khác), không nhìn triple riêng lẻ; và nó không nhân vào điểm triple |
| Soft-AND giữa các anchor (`12` §5, hai lần) | Điểm theo anchor không hiệu chuẩn, nên phép giao trên điểm không có nghĩa | Giao **tập**, không giao điểm. Không có gì cần hiệu chuẩn |

Rủi ro còn lại cần nói thẳng: nếu bộ chấm ứng viên học được tương quan giữa *kiểu đáp án* và *domain câu hỏi* của WebQSP thì nó vẫn có thể overfit khi transfer. Đó là lý do đặc trưng kiểu được đưa vào dưới dạng cos với text (không học embedding kiểu riêng), giống cách WL tổng quát hoá được nhờ không học.

### 2.4 Context mới cho LLM

Thay vì 50 triple xếp theo điểm, mỗi chain còn sống được trình bày thành một khối:

```
[chain 1, điểm 0.91] Gerald Ford → government.us_president.vice_president → ?
    ✓ Nelson Rockefeller   (bằng chứng: … )
[chain 2, điểm 0.40] …
```

Ứng viên bị loại **không xuất hiện**. Đây là thứ trực tiếp giảm "số ứng viên có tên" — đặc trưng duy nhất có p < 0.01 trong `17` §3.

---

## 3. Kiểm trần trước khi xây — đúng bài học số 2 của bản tổng kết

**E0 — trần thật của hướng này, không cần train.** Nhãn v2 có rò đáp án, nên +8.7 là cận trên. Muốn biết phần đạt được thật, dựng **nhãn v1.5**:

> chain gold (như v1), nhưng chỉ giữ các ứng viên thoả **các ràng buộc gold có mặt trong subgraph** (entity / type / value — loại 93–98% có mặt), bỏ hết ràng buộc literal và `ORDER BY` (loại có mặt 0%).

Rồi chạy QA với context v1.5. Ba kết cục:

| v1.5 − v1 trên CWQ | Đọc là | Làm gì |
|---|---|---|
| ≥ 4 Hit | Phần lớn khoảng v1 → v2 đạt được bằng thông tin có sẵn trong subgraph | Xây bộ chấm ứng viên (E1) |
| 2–4 Hit | Có, nhưng nằm sát mức nhiễu | Chỉ làm nếu cố định được server và chạy lặp |
| < 2 Hit | Khoảng v1 → v2 gần như toàn bộ là literal / rò rỉ | **Bỏ hướng này**, chuyển sang bổ sung dữ liệu KG (§5) |

Chi phí: một lượt QA CWQ + một lượt WebQSP. Code đã có: `labels.py` (v1/v2) + `bind_to_subgraph` (đã tách loại ràng buộc).

**E0b — cùng thí nghiệm nhưng với chain của RBF** (top-5 chain dự đoán + lọc ứng viên bằng ràng buộc gold có mặt). Cho biết trần khi chain là chain thật của mô hình chứ không phải chain gold. Khoảng E0 − E0b là phần chain ranking còn mất.

**E0c — miễn phí, không cần LLM: mở rộng tập đáp án trên WebQSP.** Khoảng F1 WebQSP −4.2 so với paper chủ yếu do LLM chỉ liệt kê trung bình 3.6 trong 8.8 đáp án gold. Lấy prediction đã có, với mỗi đáp án LLM chọn, tìm chain chứa nó trong context, rồi thêm các anh em cùng chain vào tập trả lời. Chấm F1 lại. Không tốn tiền. Nó cho biết có bao nhiêu F1 nằm ở khâu liệt kê. (Cảnh báo: mở rộng mù sẽ kéo cả anh em sai vào, đúng lỗi §1 — nên đây là *cận*, và phiên bản dùng được phải đi qua bộ chấm ứng viên.)

---

## 4. Thứ tự đề xuất

| # | Việc | Chi phí | Quyết định gì |
|---|---|---|---|
| 1 | **E0c** mở rộng tập đáp án trên prediction có sẵn | 0, CPU | Bao nhiêu F1 WebQSP nằm ở khâu liệt kê |
| 2 | **E0** QA với context nhãn v1.5, CWQ + WebQSP | 1 lượt QA | Hướng ứng viên có đáng làm không (bảng §3) |
| 3 | **E0b** như trên với top-5 chain của RBF | 1 lượt QA | Trần khi dùng chain thật |
| 4 | Cố định **một** endpoint QA, chạy lặp một cấu hình toàn tập → sàn nhiễu | 2 lượt QA | Không có số này thì E1 không diễn giải được |
| 5 | **E1** train bộ chấm ứng viên (chỉ train split, listwise giữa anh em) | < 1 giờ GPU; MLP nhỏ trên đặc trưng tính sẵn | — |
| 6 | **E2** context dạng khối theo chain + ứng viên đã lọc; báo (ansR, ansHit, # ứng viên có tên, QA) | 1 lượt QA mỗi dataset | Con số chính |
| 7 | **E3** giao tập ứng viên cho câu nhiều anchor; đo riêng trên CWQ conjunction và GTSQA ≥ 2 seed | 1 lượt QA | Đóng H3 bằng cách khác soft-AND |

Bước 1–3 trả lời câu "nếu thành phần này hoàn hảo thì QA lên được bao nhiêu" **trước khi viết dòng code nào của mô hình mới**.

---

## 5. Nguồn điểm lớn hơn nhưng không nằm trong tầm token-free + dữ liệu hiện tại

| Nguồn | Điểm CWQ | Cần gì | Nhận xét |
|---|---|---|---|
| Đáp án ngoài subgraph | ~13 | Truy vấn Freebase để mở thêm 1 hop từ entity cầu nối | Lớn nhất, nhưng là thay đổi **benchmark**: mọi baseline phải được cấp cùng subgraph mở rộng thì mới so công bằng. Đây là một đóng góp dữ liệu, không phải phương pháp |
| Ràng buộc literal | ~8 | Lấy literal Freebase của entity trong subgraph, rồi thực thi ràng buộc thời gian / so sánh bằng chương trình | Ghép tự nhiên với hướng ứng viên: khi ứng viên đã là một tập rõ ràng, thực thi `ORDER BY` trên tập đó là một phép sắp xếp. Nhưng cần hạ tầng Freebase (dump vài trăm GB) |
| Reasoner lặp ở câu khó | 3–4 | Reasoner lớn hơn | Ngoài phạm vi |

Đề nghị: **đừng làm hai nguồn đầu trong paper này.** Nhưng **viết phân tích trần** (19.3% câu CWQ không có đáp án trong subgraph, literal có mặt 0%, câu có ràng buộc literal Hit 50.5 so với 68.6) thành một mục riêng. Đó là phát hiện về benchmark mà cả SubgraphRAG lẫn REL-RAG đều không báo, và nó giải thích vì sao mọi phương pháp đều kẹt quanh 65–68 Hit trên CWQ.

---

## 6. Những việc nên ngừng

- **Ngừng cải tiến xếp hạng triple.** Trần còn 2.4 Hit trên CWQ, dưới mức nhiễu. Path rerank, dynamic cutoff, gate, chọn checkpoint đều đã cho kết quả trong mức nhiễu.
- **Ngừng so với số paper REL-RAG.** Bản tổng kết đã nói đúng: không tái hiện được, nên không phải tiêu chí. Phép so có giá trị là cùng protocol, có CI.
- **Ngừng chạy biến thể QA trên tập câu chọn lọc** (110 / 167 câu) mà không có đối chứng chạy lặp và McNemar.

---

## 7. Tính mới (kiểm nhanh arXiv 10-22, theo từ khoá — không đầy đủ)

| Công trình | Gần ở đâu | Khác ở đâu |
|---|---|---|
| STAGG (Yih et al., 2015) và họ semantic parsing dựa trên query graph | Sinh chain rồi gắn ràng buộc vào biến đáp án | Là bộ trả lời cuối, train và test cùng phân phối; không phải retriever cho LLM, không đo transfer |
| GNN-RAG (Mavromatis & Karypis, 2024) | GNN chấm ứng viên rồi lấy đường đi tới chúng làm context | Ứng viên chấm bằng GNN trên cả đồ thị (học relation), nên chịu đúng lỗi học thuộc tổ hợp mà EXP-39 đo; REL-RAG Table 12 cho GNN-RAG thua ở transfer |
| [Relink (2601.07192)](https://arxiv.org/abs/2601.07192) | Gọi đúng tên lỗi: distractor "liên quan nhưng gây hiểu sai" trong GraphRAG | Dùng LLM để đánh giá ứng viên; KG dựng từ văn bản |
| [ROG (2602.02382)](https://arxiv.org/abs/2602.02382) | Phân rã truy vấn thành phép chiếu / giao, cache tập đáp án trung gian | Truy vấn FOL đã cho sẵn, không phải câu hỏi tự nhiên; suy luận bằng LLM |

Chỗ còn trống: **một retriever token-free, train chỉ trên tập nguồn, tách "chọn chain" (tổng quát hoá tổ hợp, không nhìn entity) khỏi "kiểm chứng ứng viên" (nhìn entity, không nhìn relation của chain)**, và chứng minh sự tách đó là thứ giữ được transfer. Câu chuyện của paper khi đó gói lại được mọi thứ bạn đã đo: RBF là nửa đầu, nhãn SPARQL cung cấp giám sát cho nửa sau, và kết quả `17` §3 là động cơ thực nghiệm.

Lưu ý: tìm kiếm của tôi dựa trên từ khoá, không thay được một lượt đọc kỹ các bài KBQA 2025–2026 trước khi viết phần related work.

# 20. Kiến trúc mới sau RBF: khảo sát và đề xuất
Tạo: 2026-10-23. Căn cứ: EXP-39…45, bảng trần nhãn, bản tổng kết 10-23, và khảo sát arXiv (danh sách ở §6). Ràng buộc: không gọi LLM ở tầng retrieval; reasoner cố định (Llama-3.1-8B bf16 hoặc GPT-4o-mini theo bảng của paper).

0. Kiến trúc mới phải sửa nút thắt nào
Bảng dưới chỉ ghi những nút thắt đã đo được, kèm số điểm còn lấy được. Một kiến trúc chỉ đáng xây nếu nó nhắm vào một dòng ở trạng thái "mở".

#	Nút thắt	Bằng chứng	Điểm còn lấy được	Trạng thái
N1	Xếp hạng chain khi transfer	chain gold đứng hạng 1 ở 57% câu (ID) nhưng chỉ 43% (OOD); ô WebQSP→CWQ đạt 60.5, trần v1 là 67.8	~7.3 Hit ở ô transfer	mở
N2	Đáp án nằm ngoài subgraph	683/3,531 câu CWQ, trong đó 606 là composition; câu 2 bước ngữ nghĩa đi qua CVT dài 3–4 hop topo, nên subgraph 2 hop cắt mất	đến ~13 Hit CWQ	mở, cần dump
N3	Chọn giữa các ứng viên anh em	EXP-44: áp ràng buộc gold hoàn hảo trong subgraph chỉ được +0.5 F1 CWQ; LLM chọn tốt hơn g khoảng 4 lần	~0.5	đóng (subgraph không có thông tin để làm)
N4	Ràng buộc literal / thứ tự	1,801 ràng buộc ?v trên CWQ, có mặt trong subgraph 0%	~8 Hit CWQ	mở, cần dump
N5	Câu nhiều anchor	GTSQA: 1 seed đạt 71.2, 4 seed chỉ 27.6	(benchmark ngoài)	mở, ưu tiên thấp
Mọi đề xuất dưới đây đều giữ nguyên phân rã theo bước của RBF, tức nhân tử một bước π dùng chung. Đây là thứ duy nhất đã được chứng minh là tổng quát hoá được tổ hợp: nhóm tổ hợp mới đạt 83.7–85.7 so với 49.2 của DDE. Ngược lại, mọi lần cho bộ chấm kết hợp relation với ngữ cảnh entity đều làm nó học thuộc tổ hợp (hybrid: 84 → 51). Vì vậy các kiến trúc mới chỉ được làm giàu đầu vào của π hoặc đổi không gian mà π chạy trên đó, không được đổi phép phân rã.

1. Đề xuất chính — RBF-A*: retrieval trên KG mở với heuristic hoàn tất được học (N2, phụ N1)
1.1 Quan sát cốt lõi
mm (max-marginal của đường đi trọn vẹn) hiện chấm mỗi triple bằng

score(x) = g(x) + h(x)

trong đó g là log-xác suất tốt nhất của tiền tố từ anchor tới x, và h là log-xác suất tốt nhất của phần hoàn tất từ x tới một entity có tên. Đây đúng là hàm ưu tiên f = g + h của A*. Trên subgraph cố định, h được tính chính xác bằng lượt ngược, nên mm là A* với heuristic hoàn hảo. Điều này giải thích vì sao mm cho +1.7 WebQSP và +5.3 CWQ mà không cần train lại.

Trên KG mở thì không tính được h chính xác, vì phần hoàn tất nằm ngoài vùng đã mở. A*Net (2206.04798) đã cho thấy một hàm ưu tiên được học, chỉ giữ khoảng 10% nút và cạnh ở mỗi vòng, vẫn đạt SOTA trên KG cỡ triệu entity. Đề xuất ghép hai điều đó lại:

RBF-A*: g vẫn là max-product của π (không đổi), còn h là một heuristic hoàn tất được học ĥ(v | q, π_prefix). Ở mỗi vòng, chỉ mở tiếp frontier có f = g + ĥ thuộc top-B, rồi lấy thêm các cạnh của những nút đó từ KG.

1.2 Vì sao đây là đóng góp, không phải tinh chỉnh
Lý thuyết nối được với Mệnh đề 2 ở doc 09. Recall khi tổng quát hoá tổ hợp ≥ Π ρ_i, với ρ_i là xác suất giữ được cầu nối gold ở bước i. Trên KG mở, ρ_i trở thành xác suất cầu nối gold nằm trong frontier top-B, nên nó bị chặn trực tiếp bởi chất lượng của ĥ. Có thể phát biểu rằng nếu ĥ chỉ sai một hằng số so với h thật thì A* vẫn giữ cầu nối gold, đây là dạng mềm của điều kiện admissible.
Giữ nguyên phân rã. ĥ không chấm relation của câu hỏi; nó chỉ ước lượng "từ nút này còn đi tiếp được tới một entity có tên với xác suất cao bao nhiêu". Nên π không bị đụng tới, và nhóm tổ hợp mới không thể tụt vì ĥ.
Đúng chỗ REL-RAG yếu nhất. Câu composition của REL-RAG đạt 26.0 ansR, và 606/683 câu CWQ không có đáp án trong subgraph là câu composition.
1.3 Thiết kế
Thành phần	Thiết kế	Vì sao
Bước ngữ nghĩa	CVT không tính là một bước, đi qua CVT trong cùng một vòng	Câu 2 bước ngữ nghĩa qua CVT bị subgraph 2 hop cắt; doc 14 §2.2
Frontier	top-B theo f = g + ĥ, B nhỏ (ví dụ 20)	A*Net: 10% nút đã đủ
Hub	giới hạn bậc (chỉ lấy tối đa d cạnh cho mỗi cặp (nút, relation) theo π)	KGFR làm điều tương tự với Asymmetric Progressive Propagation (2511.04093); nb của bạn đã cho thấy đường "nảy" qua hub là lỗi chính
Nhiều anchor	khởi động từ mọi anchor cùng lúc, như MStar (2407.10430)	Phần N5 lấy được miễn phí
ĥ	MLP nhỏ trên [z_q; z_type(v); bậc ra của v theo relation có π cao; độ sâu còn lại]	Không nhìn chuỗi relation đã đi, nên không học thuộc được tổ hợp
Train ĥ	trên subgraph cố định, target là h thật do lượt ngược của mm tính sẵn	Nhãn miễn phí: không cần dữ liệu mới để train ĥ
Dòng cuối là điểm mạnh nhất về chi phí. h thật đã có trên mọi subgraph train, nên ĥ là một bài toán hồi quy có giám sát đầy đủ, không cần nhãn mới, không cần LLM.

1.4 Kiểm trần trước khi xây (CPU, sau khi đọc dump)
Trong 606 câu composition thiếu đáp án: bao nhiêu câu có đáp án nằm trong 1 bước ngữ nghĩa từ một entity đã có trong subgraph? Đây là trần của RBF-A* với một vòng mở rộng.
Với các câu đó, cầu nối gold (lấy từ SPARQL) có nằm trong top-B của RBF ở vòng hiện tại không? Đây là ρ_bridge thực nghiệm.
Trần QA ≈ (1) × (2) × P(Hit | đáp án trong context) ≈ (1) × (2) × 0.77.
Nếu (1) × (2) < 3% số câu CWQ thì dừng hướng này.

1.5 Rủi ro
So sánh không cùng điều kiện: paper dùng subgraph cố định. Phải báo hai setting, và cấp cùng graph mở rộng cho baseline (SubgraphRAG nhận thêm triple dễ dàng).
Hạ tầng: dump Freebase, gỡ MID theo cấu trúc (đã có cách), máy còn 48 GB trống.
Nếu ĥ kém, frontier bỏ mất cầu nối, và kết quả không tốt hơn subgraph cố định. Kiểm (2) ở trên để biết trước.
2. Đề xuất phụ, làm được ngay — π có điều kiện theo đồ thị relation (N1)
2.1 Vấn đề
π(r|q) = σ(MLP[z_q; z_r; z_q⊙z_r; lex; cos]) biểu diễn relation chỉ bằng text của chính nó. Khi transfer, relation hoặc tổ hợp relation chưa thấy chỉ được nhận diện qua độ tương đồng text. Chain gold đứng hạng 1 chỉ ở 43% câu OOD, so với 57% ID.

2.2 Họ mô hình liên quan
ULTRA (2310.04562): biểu diễn relation như một hàm của cách nó tương tác với các relation khác, bằng message passing trên đồ thị relation có 4 loại cạnh (head–head, head–tail, tail–head, tail–tail). Cách này không phụ thuộc vào ID của relation, nên chuyển được sang KG có bộ relation khác hẳn; zero-shot trên 57 KG thường ngang hoặc hơn mô hình train riêng.
Double equivariance (2302.01313) chứng minh tính bất biến này là cần cho tổng quát hoá sang relation mới, nhưng không đủ.
SEMMA (2505.20422) cho thấy khi bộ relation lúc test hoàn toàn chưa thấy, mô hình chỉ dùng cấu trúc thì sụp, còn ghép text + đồ thị relation thì hiệu quả gấp đôi. Đây đúng là trường hợp WebQSP→GrailQA của bạn.
MERIT (2610.15638, 10/2026): thay message passing trên đồ thị relation bằng transformer trên tên relation, các motif chỉ là prior mềm; nhanh hơn TRIX 6 lần, ngang SOTA zero-shot.
2.3 Đề xuất
π(r | q, R_q): z_r được thay bằng z̃_r = z_r + RelGNN(z_r; R_q), trong đó R_q là đồ thị relation dựng từ chính subgraph của câu hỏi, với 4 loại cạnh như ULTRA, và text relation là feature ban đầu như SEMMA.

π vẫn là nhân tử một bước dùng chung, nên phân rã được giữ. Thứ thay đổi là π biết thêm rằng r thường đi sau một CVT, rằng r có tail cùng kiểu với kiểu đáp án mà câu hỏi nhắc tới, rằng r và r' thường xuất hiện cùng nhau. Vì đồ thị relation là của từng câu hỏi chứ không phải của tập train, nó không mang được thống kê tổ hợp nào từ WebQSP sang CWQ.

2.4 Kiểm trước, miễn phí
Trên file retrieval OOD đã có, tách các câu mà chain gold không đứng hạng 1 theo hai nhóm: (a) relation gold có trong WebQSP train, (b) relation gold chưa thấy. Nếu phần lớn lỗi nằm ở (a), tức relation đã thấy nhưng bị xếp sai trong ngữ cảnh mới, thì đồ thị relation có cơ sở để giúp. Nếu lỗi nằm ở (b), SEMMA dự đoán phần text quan trọng hơn, và BGE prior đã chiếm phần đó.

2.5 Tiêu chí, chi phí
Bác bỏ nếu: chain gold hạng 1 trên OOD không tăng ≥ 5 điểm (43 → ≥ 48), hoặc nhóm tổ hợp mới tụt dưới 80.
Một lần train WebQSP (RelGNN 2 lớp trên vài chục relation mỗi câu, rẻ), chấm trên CWQ và GrailQA. Không cần dump, làm được ngay.
3. Thành phần cho N4 — đầu thực thi literal (sau khi có dump)
Đây không phải kiến trúc mới, mà là một toán tử tất định gắn vào cuối RBF:

Một bộ phân loại nhỏ, train bằng SPARQL của train split, đoán (predicate literal, phép toán ∈ {argmax, argmin, lọc năm, lọc "hiện tại"}) từ câu hỏi và từ tập predicate literal có trên các ứng viên.
Phép toán được thực thi trên tập điểm cuối của chain mà RBF chọn. Chỉ triple literal thắng được đưa vào context.
Về ý tưởng, đây là phép "chiếu rồi lọc" trên tập mờ của GNN-QE (2205.10128), nhưng áp lên literal. Giá trị của nó là lấp đúng phần oracle +7.3 F1 mà EXP-44 chỉ ra nằm ở literal/order. Phải kiểm trần trước bằng cách áp ràng buộc literal gold lên nhóm anh em.

4. Baseline bắt buộc phải thêm nếu claim tổng quát hoá
Họ graph foundation model cho retrieval đã phát triển nhanh trong 2025–2026, và reviewer sẽ hỏi về nó:

Mô hình	Vì sao phải so
GFM-RAG (2502.01113)	GNN 8M tham số, retrieval không cần LLM, pretrain trên 60 KG, claim zero-shot. Cùng loại "token-free" với bạn, và có checkpoint công bố
KGFR (2511.04093)	Retriever nền tảng cho KGQA, zero-shot sang KG mới, có lan truyền tiệm tiến hạn chế hub
GFM-Retriever (2603.07179)	Chọn subgraph tối thiểu và đủ bằng Information Bottleneck, không cần nhãn; gần với hướng C ở doc 19
Nếu không so với GFM-RAG thì claim "token-free retriever tổng quát hoá tốt nhất" rất dễ bị bác. Ngược lại, nếu RBF hơn GFM-RAG trên GrailQA / CWQ transfer khi GFM-RAG đã pretrain trên 60 KG còn RBF chỉ train trên WebQSP, đó là một kết quả mạnh.

5. Thứ tự đề xuất
#	Việc	Chi phí	Quyết định gì
1	Kiểm §2.4 trên file OOD có sẵn	0	Có làm π theo đồ thị relation không
2	Chạy GFM-RAG (checkpoint công bố) trên WebQSP→CWQ và GrailQA, đo retrieval	CPU/GPU nhẹ	Baseline bắt buộc; đồng thời biết họ GFM mạnh tới đâu
3	π theo đồ thị relation: 1 lần train, chấm ma trận transfer	~1 giờ GPU	N1
4	Đọc dump Freebase, gỡ MID, chạy hai phép kiểm trần §1.4 và §3	CPU vài giờ	Có làm RBF-A* / literal không
5	RBF-A* (train ĥ trên h thật, miễn phí)	1–2 giờ GPU	N2, đóng góp chính
6	QA trên các ô đã vượt sàn nhiễu	API	Số cuối
Câu chuyện của paper nếu 3 và 5 đạt: Một retriever phân rã theo bước, không gọi LLM, tổng quát hoá tổ hợp nhờ π dùng chung (RBF), chuyển được sang domain mới nhờ biểu diễn relation có điều kiện theo đồ thị relation của câu hỏi, và vượt khỏi giới hạn của subgraph dựng sẵn nhờ tìm kiếm A* với heuristic hoàn tất học được từ chính phép max-marginal của nó. Cả ba thành phần đều nhắm vào một nút thắt đã đo, và mỗi thành phần có một phép thử bác bỏ đặt trước.

6. Tài liệu đã đọc (arXiv, khảo sát 10-23)
arXiv	Tên	Dùng ở
2206.04798	A*Net: path-based reasoning có thể mở rộng	§1
2205.15319	AdaProp: lan truyền thích nghi, lấy mẫu tăng dần	§1 (họ liên quan)
2407.10430	MStar: lan truyền tiệm tiến từ nhiều điểm xuất phát	§1.3
2511.04093	KGFR: retriever nền tảng cho KGQA	§1.3, §4
2310.04562	ULTRA: foundation model cho suy luận trên KG	§2
2302.01313	Double equivariance cho relation mới	§2
2505.20422	SEMMA: KGFM có ngữ nghĩa	§2
2610.15638	MERIT: KGFM với relation transformer	§2
2205.10128	GNN-QE: truy vấn logic trên tập mờ	§3
2502.01113	GFM-RAG	§4
2603.07179	GFM-Retriever: subgraph tối thiểu và đủ	§4
2509.24276	G-reasoner: GFM 34M tham số cho suy luận trên graph	§4 (họ liên quan)
Khảo sát này tìm theo từ khoá, nên chưa thay được một lượt đọc kỹ toàn văn A*Net, ULTRA và GFM-RAG trước khi viết phần related work.