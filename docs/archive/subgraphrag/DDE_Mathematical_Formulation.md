# DDE Mathematical Formulation — SubgraphRAG

This document presents the formal mathematical description of the **Dual DDE (Distance-aware Degree Encoding)** mechanism used in SubgraphRAG, covering both the baseline **Topic-Entity DDE** and the proposed **Relation-Topic Semantic DDE**.

All formulas are grounded directly in the implementation at `retrieve/src/model/retriever.py`.

---

## Notation

| Symbol | Description |
|--------|-------------|
| $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | Knowledge Graph subgraph for question $q$ |
| $\mathcal{T} \subseteq \mathcal{V}$ | Set of topic entities (question entities) |
| $\mathcal{R}$ | Set of unique relations in $\mathcal{G}$ |
| $e = (h, r, t)$ | A triple: head entity, relation, tail entity |
| $\mathcal{N}^+(v)$ | Out-neighbors of $v$ (forward direction) |
| $\mathcal{N}^-(v)$ | In-neighbors of $v$ (reverse direction) |
| $\mathbf{z}_q \in \mathbb{R}^d$ | Query embedding |
| $\mathbf{z}_r \in \mathbb{R}^d$ | Relation $r$ text embedding |
| $\mathbf{e}_v \in \mathbb{R}^d$ | Entity $v$ text embedding |
| $K_{fwd}, K_{bwd}$ | Number of forward / backward DDE rounds |
| $\alpha \in [0,1]$ | Lexical weight in hybrid similarity score |

---

## 1. Topic-Entity DDE (Baseline)

The purpose of DDE is to encode, for each entity $v$, how far it is from the topic entities $\mathcal{T}$ in both forward and backward graph traversal directions. This acts as a **structural proximity signal** for the triple scoring MLP.

### 1.1 Initialization

Each entity is initialized with a 2-dimensional one-hot vector indicating whether it is a topic entity:

$$\mathbf{h}_v^{(0)} = \mathrm{onehot}\!\left(\mathbf{1}[v \in \mathcal{T}]\right) \in \{0,1\}^2$$

### 1.2 Forward Propagation

For $k = 1, \ldots, K_{fwd}$, each entity aggregates the mean signal from its out-neighbors:

$$\mathbf{h}_v^{(k)} = \frac{1}{|\mathcal{N}^+(v)|} \sum_{u \in \mathcal{N}^+(v)} \mathbf{h}_u^{(k-1)}$$

Each round captures entities reachable within $k$ hops **from** topic entities.

### 1.3 Backward Propagation

For $k = 1, \ldots, K_{bwd}$, the same aggregation is applied on the reversed graph:

$$\mathbf{h}_v^{(k),\mathrm{rev}} = \frac{1}{|\mathcal{N}^-(v)|} \sum_{u \in \mathcal{N}^-(v)} \mathbf{h}_u^{(k-1),\mathrm{rev}}$$

Each round captures entities that can **reach** topic entities within $k$ hops.

### 1.4 DDE Encoding

The full DDE representation for entity $v$ is the concatenation of all round outputs:

$$\mathbf{d}_v^{\mathrm{topo}} = \left[\mathbf{h}_v^{(1)},\ \ldots,\ \mathbf{h}_v^{(K_{fwd})},\ \mathbf{h}_v^{(1),\mathrm{rev}},\ \ldots,\ \mathbf{h}_v^{(K_{bwd}),\mathrm{rev}}\right] \in \mathbb{R}^{2(K_{fwd}+K_{bwd})}$$

---

## 2. Relation-Topic Semantic DDE (Proposed)

The limitation of the baseline DDE is that all edges are treated equally during propagation. Irrelevant relations (e.g., `common.topic.image`) can propagate as strongly as semantically relevant ones. We address this with **Relation-Topic weighting**.

### 2.1 Semantic Similarity

The cosine similarity between the query embedding and each relation embedding:

$$S_{\mathrm{sem}}(r) = \frac{\mathbf{z}_q \cdot \mathbf{z}_r}{\|\mathbf{z}_q\| \cdot \|\mathbf{z}_r\|}$$

### 2.2 Lexical Similarity

Freebase relation names are dot-separated hierarchical strings (e.g., `people.person.place_of_birth`). We compute a token-level overlap score with position-aware weighting:

$$S_{\mathrm{lex}}(r) = \frac{\displaystyle\sum_{p \in \mathrm{parts}(r)} w_p \cdot \max_{t \in \mathcal{Q}} \,\sigma(t,\, \mathrm{tokens}(p))}{\displaystyle\sum_{p \in \mathrm{parts}(r)} w_p}$$

where:
- $\mathrm{parts}(r)$ = dot-split segments of relation name
- $w_p$ = position weight: $1.0$ (last segment), $0.5$ (second-to-last), $0.2$ (others)
- $\mathcal{Q}$ = token set of question $q$ after stop-word removal
- $\sigma(t, p)$ = token similarity: $1.0$ (exact), $0.9$ (synonym / prefix), $0.0$ (no match)

### 2.3 Hybrid Combined Score

$$S_{\mathrm{combined}}(r) = \alpha \cdot S_{\mathrm{lex}}(r) + (1 - \alpha) \cdot S_{\mathrm{sem}}(r), \quad \alpha = 0.5$$

### 2.4 Edge Weight with Minimum Fallback

To prevent blocking structural signal propagation entirely on low-similarity relations:

$$w_e = \max\!\left(S_{\mathrm{combined}}(r_e),\ 0.1\right), \quad \forall\, e = (h, r_e, t) \in \mathcal{E}$$

The fallback of $0.1$ ensures every edge retains at least $10\%$ of its propagation capacity.

### 2.5 Semantic DDE Propagation

The same DDE architecture is run in parallel with edge-weighted message passing:

**Forward:**
$$\mathbf{h}_v^{(k),\mathrm{sem}} = \frac{\displaystyle\sum_{u \in \mathcal{N}^+(v)} w_{(u \to v)} \cdot \mathbf{h}_u^{(k-1),\mathrm{sem}}}{\displaystyle\sum_{u \in \mathcal{N}^+(v)} w_{(u \to v)}}$$

**Backward** (on reversed graph, same weights):
$$\mathbf{h}_v^{(k),\mathrm{sem\text{-}rev}} = \frac{\displaystyle\sum_{u \in \mathcal{N}^-(v)} w_{(v \to u)} \cdot \mathbf{h}_u^{(k-1),\mathrm{sem\text{-}rev}}}{\displaystyle\sum_{u \in \mathcal{N}^-(v)} w_{(v \to u)}}$$

**Semantic DDE encoding:**
$$\mathbf{d}_v^{\mathrm{sem}} = \left[\mathbf{h}_v^{(1),\mathrm{sem}},\ \ldots,\ \mathbf{h}_v^{(K_{fwd}),\mathrm{sem}},\ \mathbf{h}_v^{(1),\mathrm{sem\text{-}rev}},\ \ldots,\ \mathbf{h}_v^{(K_{bwd}),\mathrm{sem\text{-}rev}}\right] \in \mathbb{R}^{2(K_{fwd}+K_{bwd})}$$

---

## 3. Top-K Relation Mask (MLP Injection)

In addition to DDE, we directly inject a binary relation-relevance signal into the triple classifier:

$$\mathcal{R}^* = \mathrm{TopK}_{r \in \mathcal{R}}\, S_{\mathrm{combined}}(r), \quad |\mathcal{R}^*| = K_r$$

$$\mathbf{o}_r^{\mathrm{TopK}} = \mathrm{onehot}\!\left(\mathbf{1}[r \in \mathcal{R}^*]\right) \in \{0,1\}^2$$

Default: $K_r = 5$.

---

## 4. Final Triple Representation

The combined entity DDE (topological + semantic) for entity $v$:

$$\mathbf{d}_v^{\mathrm{RT}} = \left[\mathbf{d}_v^{\mathrm{topo}},\ \mathbf{d}_v^{\mathrm{sem}}\right] \in \mathbb{R}^{4(K_{fwd}+K_{bwd})}$$

The complete feature vector for triple $e = (h, r, t)$ fed to the classifier MLP:

$$\mathbf{x}_e = \left[\mathbf{z}_q,\ \mathbf{e}_h,\ \mathbf{z}_r,\ \mathbf{e}_t,\ \mathbf{d}_h^{\mathrm{RT}},\ \mathbf{d}_t^{\mathrm{RT}},\ \mathbf{o}_r^{\mathrm{TopK}}\right]$$

$$\dim(\mathbf{x}_e) = 4d + 4(K_{fwd}+K_{bwd}) \cdot 2 + 2 + 2$$

The scoring MLP is:

$$s_e = \mathrm{MLP}(\mathbf{x}_e) = W_2\,\sigma(W_1\,\mathbf{x}_e + b_1) + b_2 \in \mathbb{R}$$

where $\sigma$ is ReLU.

---

## 5. Summary: Baseline vs. Proposed Feature Dimensions

| Component | Baseline | RT-Combined |
|---|---|---|
| Query emb $\mathbf{z}_q$ | $d$ | $d$ |
| Head emb $\mathbf{e}_h$ + topic PE | $d + 2$ | $d + 2$ |
| Relation emb $\mathbf{z}_r$ | $d$ | $d$ |
| Tail emb $\mathbf{e}_t$ + topic PE | $d + 2$ | $d + 2$ |
| Topo DDE (head + tail) | $4(K_{fwd}+K_{bwd})$ | $4(K_{fwd}+K_{bwd})$ |
| **Sem DDE (head + tail)** | — | $4(K_{fwd}+K_{bwd})$ |
| **TopK relation mask** | — | $2$ |
| **Total** | $4d + 4 + 4(K_{fwd}+K_{bwd})$ | $4d + 4 + 8(K_{fwd}+K_{bwd}) + 2$ |

With $d=1024$, $K_{fwd}=K_{bwd}=3$: Baseline = **4124-dim**, RT-Combined = **4150-dim**.
