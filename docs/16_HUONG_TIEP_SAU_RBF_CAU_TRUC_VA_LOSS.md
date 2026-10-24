# 16. Sau RBF: nhân tử cấu trúc, loss khớp với inference, và phân bổ budget

*Tạo: 2026-10-24. Nhánh: `exp/rbf-paper-formalism`. Căn cứ: docs/11–15, EXP-39/43/44/45, bảng trần nhãn, và khảo sát 10-24 ở §7. Ràng buộc giữ nguyên: không gọi LLM ở tầng retrieval, không dùng thống kê tập đích.*

---

## 0. Đính chính một luật dừng, vì nó quyết định toàn bộ tài liệu này

docs/14 §6 (10-22) viết: *"ngừng cải tiến xếp hạng triple"*, vì khoảng cách tới trần oracle chain trên CWQ còn **2.4 Hit**, "dưới mức nhiễu 2–3 Hit".

**EXP-44 (10-23) đã thay thế ước lượng nhiễu đó.** Đo hai lượt trên *cùng một endpoint*: SD bootstrap của hiệu hai lượt là **0.34 (WebQSP) / 0.43 (CWQ)**, và kết luận ghi rõ *"mức 2–3 điểm trước đây là do đổi server"*.

Vậy 2.4 Hit ≈ **5σ** nếu cố định endpoint. Luật dừng được đặt trên một con số mà chính nhóm thí nghiệm hôm sau đã bác. Khoảng còn lại **đo được**, với điều kiện: một endpoint duy nhất, chạy lặp, McNemar. Mọi thí nghiệm dưới đây bắt buộc theo giao thức đó.

---

## 1. Mệnh đề 5: trần precision, và vì sao nó nên là một đóng góp chứ không phải một nhược điểm

Nhãn v2 chỉ giữ các trail của chain gold **kết thúc đúng ở đáp án gold**. Hai trail cùng chain khác nhau **chỉ ở identity của entity cuối**. Vì $\pi$ chỉ phụ thuộc $(q, r, d)$:

> **Mệnh đề 5 (trần precision của chấm điểm phân rã theo bước).** Với $\pi$ phụ thuộc duy nhất $(q, r, d)$, mọi hai walk cùng dãy $(r_i, d_i)$ nhận **cùng một điểm**. Do đó bộ chấm không phân biệt được các trail cùng chain khác điểm cuối, và trần của nó là nhãn **v1** (chain gold, mọi điểm cuối), không phải **v2**.
>
> *Chứng minh.* Trực tiếp từ $S(\omega) = \sum_i \log \pi(r_i, d_i \mid q)$: hàm điểm là một hàm của dãy nhãn bước, hằng trên lớp tương đương các walk cùng dãy nhãn. $\square$

Số liệu khớp chính xác:

| CWQ | Hit / F1 | Đọc là |
|---|---|---|
| RBF v2 hiện tại | 65.4 / 55.4 | |
| Trần v1 (chain gold, mọi điểm cuối) | 67.8 / 58.2 | **trần của Mệnh đề 5**, còn 2.4 Hit |
| Trần v2 (chain gold, chỉ điểm cuối đúng) | 76.5 / 66.1 | **+8.7 Hit, bất khả thi về nguyên lý** cho lớp mô hình này |

**Mệnh đề 1 và Mệnh đề 5 là hai mặt của một đồng xu.** Chính tính chất cho systematicity (không tham số nào đánh chỉ số theo relation hay cặp relation) là tính chất chặn trần precision. Cặp mệnh đề này giải thích **mọi** kết quả âm của giai đoạn RBF bằng một câu, thay vì sáu lần "thử rồi thất bại":

| Lần thử | Nó cố vượt | Nó mất |
|---|---|---|
| $\psi(v \mid q)$ trong lan truyền | Mệnh đề 5 | val 88 → 81–84 |
| Hybrid MLP + RBF (không DDE) | Mệnh đề 5 | tổ hợp mới 84 → 51.2 |
| RBF-PoE (MLP cục bộ) | Mệnh đề 5 | WebQSP 89.6 → 88.7 |
| max-z giữa hai head | Mệnh đề 5 | tổ hợp mới 84 → 66, 1-topic 86 → 76 |
| soft-AND trên điểm anchor | — (hiệu chuẩn) | ≥2 topic 75.3 → 70.8 |

Và EXP-44 cho biết **khoảng v1→v2 cũng không lấy được bằng bất kỳ mô hình nào trên dữ liệu này**: áp ràng buộc gold hoàn hảo có mặt trong subgraph chỉ được **+0.5 F1**; một bộ chấm học được (AUC 0.774) **thua chính LLM 4:1**. Phần phân biệt nằm ở **literal / order**, có mặt trong subgraph **0%**.

**Kết luận chiến lược:** đừng đuổi theo v1→v2 trên dữ liệu hiện tại. Ba việc còn lại, theo đúng thứ tự: (a) lấy 2.4 Hit tới trần v1 — §2 và §4; (b) khớp loss với inference — §3; (c) biến chính trade-off thành đóng góp lý thuyết — §1 này.

---

## 2. Nhân tử cấu trúc: thêm thông tin về thứ tự mà **không** mất Mệnh đề 1

### 2.1 Vì sao DDE/hybrid thất bại — và đây là một giả thuyết kiểm chứng được, chưa phải kết luận

Hybrid `rbfmlp_nb` **không có DDE** mà vẫn tụt 84 → 51.2. Nên distance encoding **không** phải nguyên nhân. Giả thuyết: MLP nhận $\mathbf{z}_h$ của entity cầu nối, và $\mathbf{z}_h$ **tiết lộ kiểu range của relation bước trước**, nên $(\mathbf{z}_h, \mathbf{z}_r)$ hoạt động như một cặp relation.

Giả thuyết này **chưa được kiểm**. Phép kiểm ở §5 (E0) là một probe tuyến tính dự đoán relation bước trước từ $\mathbf{z}_h$. Nếu AUC cao, bảng ablation cơ chế của paper chuyển từ *mô tả* 84→51 sang *giải thích* được nó.

### 2.2 Nguyên tắc thiết kế

> Một nhân tử được phép phụ thuộc cấu trúc, miễn là số **bit** nó biết về bước trước là $O(1)$ và **không nhận dạng được relation**.

| Tín hiệu về bước trước | Lượng tin | An toàn với Mệnh đề 1? |
|---|---|---|
| hướng của bước trước | 1 bit | ✅ |
| chỉ số hop | $\log T$ bit | ✅ |
| lớp của nút (named / CVT) | 1 bit | ✅ |
| $\mathbf{z}_h$ của entity cầu nối | 1024 chiều, tiết lộ kiểu | ❌ |

### 2.3 Bốn nhân tử ứng viên

| Nhân tử | Tham số | Trạng thái |
|---|---|---|
| **Bigram hướng** $\psi(d_{i-1}, d_i)$ + bias bước đầu $\chi(d_1)$ | **6** | ✅ **đã cài** (`rbf.py --dir_bigram`), DP đã kiểm brute-force |
| Bias theo hop $\beta_t$ | $T{+}1$ | chưa cài; cùng $\varphi$ cho một phân phối độ dài đúng nghĩa |
| Lớp nút $\chi(\mathrm{class}(u), \mathrm{class}(v))$ | 4 | chưa cài; mm đã làm một nửa (chỉ dừng ở named) |
| Khoảng cách tới anchor **khác**, chỉ ở bước cuối | 0 | chưa cài; phép giao ở mức cấu trúc, khác soft-AND trên điểm (đã bị bác) |

Bigram hướng gần như **miễn phí**: trạng thái non-backtracking **đã** mang tag bước cuối $(r, d)$, nên hướng đã có sẵn. Chi phí thêm là trạng thái top-2 **theo từng hướng** (8 số/nút thay vì 4).

Trực giác nó bắt được: một chain gold qua CVT có **mẫu hướng chuẩn** (`anchor →xuôi→ CVT →xuôi→ đáp án`), còn đường "nảy" và đường đi lạc thường đổi hướng theo mẫu khác. `nb` chỉ cấm *một* dạng đổi hướng; $\psi$ **học** toàn bộ bảng $2\times2$.

### 2.4 Công thức

$$
\mathrm{base}(u, d) = \max\Big(\underbrace{\mathrm{start}(u) + \chi(d)}_{\text{bước đầu}},\;\; \max_{d'} \big[\underbrace{\mathrm{excl}(u, d', (r, \bar d))}_{\text{top-2 trong nhóm } d'} + \psi(d', d)\big]\Big)
$$

$$
\mathrm{cand}(e, d) = \mathrm{base}(\mathrm{src}_d(e), d) + \log \pi(r_e, d \mid q)
$$

Vì $\psi$ chỉ phụ thuộc **hướng** của bước trước, và một bước vào hướng $d$ chỉ loại **đúng một** tag $(r_e, \bar d)$ (tag này có hướng $\bar d$, nên chỉ chạm nhóm $d' = \bar d$), trạng thái đủ là: với mỗi nút và mỗi hướng bước cuối, (điểm tốt nhất, tag của nó, điểm tốt nhất với tag khác, tag đó).

> **Mệnh đề 6.** Quy hoạch động trên là **chính xác** cho max-product trên walk không quay lui có nhân tử bigram hướng, với 8 số mỗi nút.
> **Kiểm chứng:** `retrieve/exp_qrdde/analysis/ref_semirings.py`, claim **C-A** (dạng theo lớp) và **C-A2** (dạng tích luỹ mà `rbf.py` cài). 500 đồ thị ngẫu nhiên, $T = 4$: sai số so với brute-force **1.78e-15**.

**Giới hạn đã biết của bản cài hiện tại:** $\psi$ chỉ áp ở **lan truyền tiền tố**. `best_completion` (mm) vẫn chấm phần nối dài **không có** $\psi$, nên bật `--dir_bigram` cùng `mm` sẽ trộn một tiền tố có $\psi$ với một hậu tố không có. **Chạy ablation bigram hướng với mm tắt** để so sạch; mở rộng $\psi$ vào phần nối dài là việc tiếp theo. Điều này đã ghi trong docstring của `rbf.py`.

---

## 3. Loss: hiện đang lệch pha với inference

Inference là **max-product trên walk**. Loss là **BCE trên từng triple độc lập**. Đây là một lệch pha kinh điển của structured prediction, và nó giải thích hai triệu chứng đã đo: điểm không hiệu chuẩn được giữa các câu (docs/13 ghi ở cột *hại*), và anh em cùng chain không bị phạt.

Hai cách sửa, **cùng dùng đúng cái DP đã có**:

### 3.1 CRF trên walk (ưu tiên cao hơn)

Đổi nửa vành $(\max, +) \to (\mathrm{logsumexp}, +)$, **cùng một đồ thị nhân tử**:

$$
\log Z(q) = \log \sum_{\omega \in \Omega^{\le T}} \exp S(\omega), \qquad \mathcal{L}_{\text{CRF}} = -\log \frac{\sum_{\omega \ni \text{gold}} \exp S(\omega)}{Z(q)}
$$

Horizon hữu hạn nên tổng trên walk là **tractable, chính xác**, không cần xấp xỉ.

Loại trừ theo tag trong nửa vành cộng là một phép **trừ trong không gian khối lượng**: $\sum_{\tau \ne \tau_0} M_v(\tau) = \mathrm{tot}_v - M_v(\tau_0)$. Nên trạng thái là khối lượng nhóm theo $(\text{nút}, \text{relation}, \text{hướng})$ — nhiều nhất $2|\mathcal{E}|$ phần tử, **không** phải $O(|\mathcal{V}||\mathcal{R}|)$.

> **Mệnh đề 7.** Sum-product trên **cùng tập walk không quay lui** (có cả $\psi$) tính được chính xác với bộ nhớ $O(|\mathcal{E}|)$, cho hàm phân hoạch $Z$ chính xác.
> **Kiểm chứng:** `ref_semirings.py` claim **C-B**. 500 đồ thị, $T = 4$: sai số **1.9e-10**.

Lợi ích kép: $Z$ **chuẩn hoá điểm giữa các câu**, nên **ngưỡng toàn cục mới có nghĩa** — thứ mọi thí nghiệm budget động trước đây thiếu. Lưu ý dòng thời gian: cắt ngưỡng $\tau$ là **10-17**, dynamic cutoff **10-22**, nhân tử dừng $\varphi$ **10-23**. Mọi kết luận "cắt động nằm trong nhiễu" đều đo trên mô hình **chưa có khái niệm kết thúc và chưa chuẩn hoá**.

### 3.2 Structured margin (hard negative chính xác, không phải sampling)

$$
\mathcal{L}_{\text{margin}} = \max\big(0,\; m + S(\hat\omega) - S(\omega^\star)\big), \qquad \hat\omega = \arg\max_{\omega \notin \text{gold}} S(\omega)
$$

$\omega^\star$ lấy từ nhãn v2; $\hat\omega$ tính bằng cùng DP có loại trừ. Đây là dạng sắc của ý "negative sampling kiểu RAPL": thay vì **lấy mẫu** negative, ta lấy **đúng negative tệ nhất**.

### 3.3 Cạm bẫy phải tránh: false negative

RAPL `path_loss` thực chất **đã là** contrastive giữa anh em (softmax trên out-neighbour + STOP mỗi bước). Nhưng nếu port thẳng, ta nhập luôn điều kiện-hoá-theo-ngữ-cảnh tích luỹ — chỗ học thuộc tổ hợp.

Và quan trọng hơn: **anh em cùng chain thường cũng là gold.** WebQSP có **8.8 đáp án/câu**, còn LLM chỉ liệt kê 3.6 — F1 đang mất chủ yếu ở khâu liệt kê. Phạt "anh em không phải golden" một cách mù là train trên false negative và sẽ làm F1 WebQSP **tệ hơn**. Đúng cái [HaSa](https://arxiv.org/pdf/2305.10563) cảnh báo cho KG.

**Quy tắc:** negative là *"điểm cuối không thuộc **tập** đáp án gold"*, không phải *"không phải đáp án mà LLM chọn"*, và không phải *"không nằm trong nhãn v2"* (v2 chỉ giữ một số trail).

---

## 4. Phân bổ budget: nhiễu top-k là lỗi **chọn tập**, không phải lỗi chấm điểm

Chẩn đoán đã có (docs/12 §19–20, 110 câu WebQSP mà SubgraphRAG đúng / RBF sai):

| | RBF | SubgraphRAG |
|---|---|---|
| chain gold đầy đủ trong context | **88%** | 65% |
| đa dạng relation | 18.6 | **25.6** |
| **số ứng viên có tên** | **51** | 37 |
| triple chạm topic | 47% | **64%** |

Và đặc trưng duy nhất dự đoán được LLM sai là **số ứng viên có tên** ($p = 0.008$); **hạng của triple đáp án thì không** ($p = 0.77$).

Cơ chế là **hệ quả trực tiếp của Mệnh đề 5**: anh em cùng $(h, r)$ có điểm **bằng nhau chính xác**, nên một fan-out `member` × 80 của một CVT chiếm trọn budget. Đây là tính chất cấu trúc, cùng loại với thiên lệch tiền tố mà mm đã sửa, nên phải sửa bằng ràng buộc cấu trúc.

Bản hậu xử lý (`diverse`: tối đa 8 triple/relation, bỏ cặp entity trùng, bù từ top-500) đã đo: **106 so với 90** trên tập 167 câu, và xoá khoảng thua Hit trên toàn WebQSP — nhưng $p = 0.06 / 0.44$, **chưa có ý nghĩa**. Bước tiếp mà docs/12 §19 tự viết ra và chưa ai làm: đưa vào **khâu chọn**.

**Phát biểu bài toán.** Thay vì lấy top-$k$ theo điểm, cực đại một hàm phủ **đơn điệu, submodular**:

$$
F(S) = \sum_{g \in \mathcal{G}} \max_{e \in S,\, \mathrm{group}(e) = g} \mathrm{gain}(e), \qquad |S| \le k
$$

> **Mệnh đề 8.** $F$ đơn điệu và submodular, nên greedy đạt $(1 - 1/e)$ của tối ưu.
> **Kiểm chứng:** `ref_semirings.py` claim **C-C**, greedy vs vét cạn, 200 instance.

$\mathrm{gain}$ dùng **hạng nghịch đảo** $1/(1+\mathrm{rank})$ vì điểm RBF là log-xác suất **không chuẩn hoá** (mọi giá trị âm, thang đổi theo câu) — đúng vấn đề mà §3.1 sửa. Nhóm: $(h, r)$ (fan-out hoà điểm), relation, hoặc cặp entity.

Tính chất quan trọng: khi mỗi triple là một nhóm riêng, `cover` **suy biến về đúng top-$k$** (đã assert trong test). Nên nó không thể làm hại ở những câu không có fan-out.

**Và đây là chỗ $\varphi$ mới cash out được.** Pilot EXP-45 thua −4.1 Hit trên CWQ ID với lý do ghi rõ: số ứng viên tăng 14.7 → 19.8. $\varphi$ xếp hạng tốt hơn rồi đổ thêm ứng viên vào context. **§4 và §3 là một vấn đề**: không giữ được số ứng viên thì không thu được phần lời của $\varphi$.

---

## 5. Thí nghiệm, kèm tiêu chí bác bỏ đặt trước

Giao thức bắt buộc cho mọi ô QA: **một endpoint duy nhất**, 2 lượt, McNemar, báo cả SD giữa hai lượt.

| # | Thí nghiệm | Lệnh | Chi phí | **Bác bỏ nếu** |
|---|---|---|---|---|
| **E0** | Probe: dự đoán relation bước trước từ $\mathbf{z}_h$ của entity cầu nối | cần viết | CPU <1h | AUC < 0.65 → giả thuyết §2.1 sai, phải tìm kênh rò rỉ khác trước khi claim |
| **E1** | Giải phẫu context: tie rate, fan-out, ứng viên có tên | `analysis/context_anatomy.py` | ~0, CPU | `largest_hr_fanout` trung bình < 3 → fan-out không phải nút thắt, bỏ §4 |
| **E2** | Chọn tập submodular trên pred có sẵn | `analysis/budget_select.py --mode cover` | ~0, CPU | ansR@k giảm > 0.5, **hoặc** ứng viên có tên không giảm ≥ 20% |
| **E3** | Quét horizon $T \in [2,6]$, $L \in [1,3]$ | `--rbf_T/--rbf_L` lúc train, hoặc sửa args rồi `infer.py` | ~0 (test) | recall phẳng theo $T$ → horizon không phải nút thắt, bỏ dynamic stop |
| **E4** | Bigram hướng, **mm tắt** | `train_retriever.py --dir_bigram` | ~1h GPU | tổ hợp mới tụt dưới 80, **hoặc** chain hạng 1 OOD không tăng ≥ 3 điểm |
| **E5** | CRF trên walk + ngưỡng toàn cục | cần cài (spec ở §3.1, DP đã kiểm) | ~2h GPU | ansR@50 giảm, **hoặc** ngưỡng vẫn chỉ có lợi trên WebQSP như 10-17 |
| **E6** | Structured margin, negative = điểm cuối ∉ tập đáp án gold | cần cài (§3.2) | ~1h GPU | F1 WebQSP giảm (dấu hiệu false negative) |
| **E7** | QA cho cấu hình tốt nhất của E2/E4/E5 | `prompt_lab.py` | 2 lượt API | **chỉ chạy sau** khi E1–E6 xong; chạy lẻ thì không diễn giải được |

Thứ tự: **E1 → E3 → E2** trước (cả ba gần như miễn phí và cùng kiểm một giả thuyết: *nút thắt còn lại là phân bổ budget, không phải chất lượng điểm*). Nếu cả ba không nhúc nhích thì giả thuyết sai, dừng sớm và rẻ. **E0** chạy song song vì nó chỉ phục vụ phần lập luận.

---

## 6. Code đã có trong nhánh này

| File | Việc | Trạng thái kiểm chứng |
|---|---|---|
| `analysis/ref_semirings.py` | Bản tham chiếu numpy + brute force cho C-A, C-A2, C-B, C-C | ✅ **chạy và pass** (500 đồ thị, $T{=}4$; sai số 1.8e-15 / 1.9e-10) |
| `rbf.py` | `--dir_bigram` (`propagate_nb_dirbigram`), `T`/`L` thành tham số; nhánh cũ **không đổi** khi tắt cờ | ⚠️ parse OK; **chưa chạy** (container không có torch) → chạy `check_rbf_vs_ref.py` trước khi train |
| `analysis/check_rbf_vs_ref.py` | Đối chiếu torch ↔ bản tham chiếu, cả nhánh cũ và nhánh mới | ⚠️ chưa chạy (cần torch) |
| `analysis/context_anatomy.py` | E1 | ✅ logic lõi đã test (fixture tổng hợp); ⚠️ phần `torch.load` chưa chạy |
| `analysis/budget_select.py` | E2 (`topk` / `cap` / `cover`) | ✅ logic lõi đã test, gồm tính suy biến về top-$k$; ⚠️ phần I/O chưa chạy |
| `train_retriever.py` | `--dir_bigram`, `--rbf_T`, `--rbf_L` | ⚠️ parse OK, chưa chạy |

**Chưa cài:** CRF trên walk (§3.1) và structured margin (§3.2) — spec và DP đã kiểm, phần còn lại là dựng $\omega^\star$ có thứ tự từ nhãn v2, việc này tôi không làm được mà không có dữ liệu để đối chiếu. Ba nhân tử cấu trúc còn lại ở §2.3 cũng chưa cài.

Lệnh chạy trước khi train bất cứ thứ gì:

```bash
cd retrieve/exp_qrdde/analysis
python ref_semirings.py --graphs 500 --T 4          # không cần torch
python check_rbf_vs_ref.py --graphs 200 --T 3       # cần torch; phải in "both agree"
```

---

## 7. Khảo sát 10-24: lan truyền thông tin trên KG, và chỗ trống

Tìm theo từ khoá, **không** thay được một lượt đọc toàn văn. Hai host `arxiv.org` và `aclanthology.org` bị chặn trong session dựng tài liệu này, nên phần dưới dựa trên abstract/snippet, **chưa phải bản đầy đủ**.

| Công trình | Lan truyền kiểu gì | Khác RBF ở đâu |
|---|---|---|
| [NBFNet](https://arxiv.org/pdf/2106.06935) | Bellman–Ford tổng quát, $\oplus$ = sum/PNA, trạng thái là vector | Cộng dồn trộn đường đi; embedding theo **id** relation nên không chuyển KG |
| [A*Net](https://arxiv.org/pdf/2206.04798) | NBFNet + hàm ưu tiên học được, mở ~10% nút | Vẫn cộng dồn; tiết kiệm bằng **tỉa nhánh** (xấp xỉ), RBF tiết kiệm bằng **trạng thái vô hướng** (chính xác) |
| [AdaProp](https://arxiv.org/html/2205.15319) (KDD'23, [code](https://github.com/LARS-research/AdaProp)) | Lấy mẫu lân cận thích nghi bằng Gumbel top-$k$, incremental sampling tuyến tính theo bước | Giải bài toán **nở số entity** ở $L > 5$; **không** có dừng học được. Đây là baseline đúng cho E3 nếu horizon hoá ra là nút thắt |
| RED-GNN | Lan truyền tiệm tiến quanh entity truy vấn | Cùng họ; cùng điểm nghẽn độ sâu |
| [ULTRA](https://arxiv.org/abs/2310.04562) / [SEMMA](https://arxiv.org/pdf/2505.20422) / MERIT | Biểu diễn relation **bằng cách nó tương tác với relation khác** (đồ thị relation 4 loại cạnh) | Đây là đề xuất §2 của docs/14 và vẫn là hướng đúng cho N1: 94% phần mất khi transfer nằm ở nhóm "relation **và** chain đều đã thấy" (68.3 vs 85.9), tức **không** phải relation lạ |
| [GNN-QE](https://arxiv.org/pdf/2205.10128) / [AnyCQ](https://arxiv.org/abs/2409.13959) / FuzzQE | Phân rã truy vấn thành phép toán trên **tập mờ**; giao bằng t-norm (AnyCQ: Gödel/min) | RBF dùng t-norm **tích** cho `inter`; soft-AND kiểu min đã bị bác ở mức điểm. Hướng còn lại là giao **tập** (§2.3 dòng 4), không phải giao điểm |
| [KGFR](https://arxiv.org/pdf/2511.04093) | Conditional message passing nhiều góc nhìn, hạn chế hub tiệm tiến | Baseline bắt buộc nếu claim tổng quát hoá; cùng loại "retriever nền tảng, token-free" |
| [REL-RAG](https://arxiv.org/pdf/2506.09645) | bi-GCN trên **line graph** + decode path tự hồi quy có token STOP | Nguồn của $\varphi$; chi phí $\sum_v \deg(v)^2$ |
| [Think Parallax / ParallaxRAG](https://arxiv.org/abs/2510.15552) (ACL'26) | Tách câu hỏi và KG thành **view theo từng attention head** (head chuyên hoá theo hop); MLP dùng chung trên $(\text{query view}, h, r, t)$; gate theo câu hỏi; KL tới phân phối shortest-path; ràng buộc đa dạng giữa head | WebQSP 86.85/71.73, CWQ **58.41/48.33** — **thua** REL-RAG ở cả hai tập và thua RBF 7 điểm Hit trên CWQ. Vào ACL bằng **quan sát cơ chế + phân tích**, không bằng bảng số. Và họ chọn transfer **đổi domain** (BioASQ), **không** đổi tổ hợp |
| [FBV-KGQA](https://aclanthology.org/2026.emnlp-main.297/) (EMNLP'26) | "Forward-backward verification" cho đường suy luận | **Cần đọc**: tên gợi đúng cấu trúc $g + h$ của mm. Tôi không lấy được method details, host bị chặn |

**Hai chỗ trống tôi tìm mà không thấy ai lấp:**
1. **CRF / max-margin trên walk cho retriever KGQA.** Machinery là sách giáo khoa (forward algorithm, structured perceptron), nhưng tìm theo từ khoá không ra công trình nào áp cho retrieval trên KG. §3 là chỗ này.
2. **Nhân tử cấu trúc relation-agnostic bậc hai** (bigram hướng, lớp nút) cho tổng quát hoá tổ hợp. Tìm "second-order relation-agnostic" không ra. §2 là chỗ này.

**Bài học từ ParallaxRAG, áp cho bản thảo:** số của REL-RAG **không** phải điều kiện xuất bản — một paper ACL'26 cùng chủ đề còn thua cả RBF trên CWQ. Trục chính nên là **chẩn đoán D1 + Mệnh đề 1/5 + ablation cơ chế**, và phương pháp là hệ quả. Thêm nữa, ParallaxRAG chấm bằng MLP trên $(h, r, t)$ **và** chuyên hoá **theo từng hop** — cả hai kênh học thuộc tổ hợp mà EXP-39 đã đo. Lý thuyết của chúng ta vì thế **tiên đoán** nó sụp ở ô compositional shift. Nếu họ công bố code (tôi chưa xác nhận được), chạy họ trên WebQSP→CWQ và trên nhóm 587 câu "tổ hợp mới" là một thí nghiệm rất mạnh: dùng một phương pháp ACL làm **instance xác nhận** cho chẩn đoán của mình.

---

## 8. Việc **không** nên làm

- Đưa $\mathbf{z}_h / \mathbf{z}_t$ vào bộ chấm dưới bất kỳ dạng nào (4 lần, 4 lần mất Mệnh đề 1).
- Đuổi theo khoảng v1→v2 trên dữ liệu hiện tại (EXP-44 đã đóng bằng số đo: +0.5 F1).
- Bộ chấm ứng viên thay lựa chọn của LLM (thua LLM 4:1).
- Chạy QA lẻ, đổi endpoint giữa các lượt, hoặc so trên tập câu chọn lọc mà không có đối chứng chạy lặp và McNemar.
