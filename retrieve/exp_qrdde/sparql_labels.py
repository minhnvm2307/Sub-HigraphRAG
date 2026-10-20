"""Trich nhan huan luyen tu SPARQL gold cua WebQSP / CWQ.

Thay cho nhan yeu "shortest path topic -> answer", SPARQL cho dung bon thu:
  1. chain          : chuoi predicate tu topic entity toi bien dap an (InferentialChain)
  2. mediator_vars  : bien trung gian tren chain = nut CVT (dinh nghia chinh xac, khong doan bang regex m.*)
  3. constraints    : cac pattern KHONG nam tren chain -> bang chung rang buoc (witness)
  4. order_by       : predicate dung de sap xep cho cau superlative / thoi gian

Khong goi LLM. Chay duoc tren CPU.

Dung:
    python sparql_labels.py --dataset cwq  --in ComplexWebQuestions_train.json --out cwq_train_labels.jsonl
    python sparql_labels.py --dataset webqsp --in WebQSP.train.json --out webqsp_train_labels.jsonl
"""
from __future__ import annotations
import json, re, argparse, collections
from typing import Any

MID = re.compile(r"^m\.[0-9a-z_]+$")


# ----------------------------------------------------------------- parsing
def _split_where(sparql: str) -> tuple[list[str], list[str]]:
    """Tach phan than WHERE thanh cac menh de, co dem ngoac nhon long nhau."""
    i = sparql.find("WHERE")
    if i < 0:
        return [], ["<khong tim thay WHERE>"]
    body = sparql[sparql.find("{", i) + 1:]
    depth, buf, clauses = 0, [], []
    for ch in body:
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                break
            depth -= 1
        buf.append(ch)
    body = "".join(buf)
    # FILTER / OPTIONAL / UNION khong co dau cham ket thuc, nen phai an chung ra truoc,
    # neu khong chung se dinh vao triple pattern ngay sau.
    KW = ("FILTER", "OPTIONAL", "UNION", "MINUS")
    out, filters, cur = [], [], []
    i, n = 0, len(body)
    while i < n:
        up = body[i:i+8].upper()
        kw = next((k for k in KW if up.startswith(k)), None)
        if kw and (i == 0 or not body[i-1].isalnum()):
            j = i + len(kw)
            while j < n and body[j] not in "({":
                j += 1
            if j < n:
                opener, closer = ("(", ")") if body[j] == "(" else ("{", "}")
                d, k = 0, j
                while k < n:
                    if body[k] == opener:
                        d += 1
                    elif body[k] == closer:
                        d -= 1
                        if d == 0:
                            break
                    k += 1
                filters.append(" ".join(body[i:k+1].split()))
                i = k + 1
                continue
        # dau cham ket thuc menh de = dau cham co khoang trang hai ben;
        # khong duoc tach theo moi dau cham vi predicate Freebase day dau cham
        if (body[i] == "."
                and (i == 0 or body[i-1].isspace())
                and (i+1 >= n or body[i+1].isspace())):
            out.append("".join(cur).strip()); cur = []
        else:
            cur.append(body[i])
        i += 1
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return [c for c in out if c], filters


def parse_sparql(sparql: str) -> dict[str, Any]:
    """-> {select_var, patterns:[(s,p,o)], filters:[str], order_by:{...}|None, limit, unparsed:[str]}"""
    m = re.search(r"SELECT\s+(?:DISTINCT\s+)?(\?\w+)", sparql)
    select_var = m.group(1) if m else None

    order_by = None
    mo = re.search(r"ORDER\s+BY\s+(DESC|ASC)?\s*\(?\s*(?:xsd:\w+\s*\()?\s*(\?\w+)", sparql, re.I)
    if mo:
        order_by = {"direction": (mo.group(1) or "ASC").upper(), "var": mo.group(2)}
    ml = re.search(r"LIMIT\s+(\d+)", sparql, re.I)
    limit = int(ml.group(1)) if ml else None

    clauses, filters = _split_where(sparql)
    patterns, unparsed = [], []
    for c in clauses:
        cs = " ".join(c.split())
        if not cs:
            continue
        toks = cs.split()
        if len(toks) != 3:
            unparsed.append(cs)
            continue
        s, p, o = (t[3:] if t.startswith("ns:") else t for t in toks)
        patterns.append((s, p, o))
    return {"select_var": select_var, "patterns": patterns, "filters": filters,
            "order_by": order_by, "limit": limit, "unparsed": unparsed}


# ------------------------------------------------------------------ labels
def _is_var(x: str) -> bool:
    return x.startswith("?")


TYPE_PREDS = ("common.topic.notable_types", "type.object.type")


def extract_label(parsed: dict[str, Any], topic_mids: list[str] | None = None) -> dict[str, Any]:
    """Chia pattern thanh chain (anchor -> dap an) va constraint (phan con lai).

    topic_mids: topic entity that su cua cau hoi (truong `q_entity` trong record cua
    SubgraphRAG). Nen truyen vao. Neu khong co, ham dung hiristic: bo cac hang la
    doi tuong cua predicate kieu, roi chon chain DAI NHAT lam chain chinh -- vi trong
    CWQ composition, entity duoc nhac ten nam xa ?x, con rang buoc kieu/gia tri nam ke ?x.
    """
    pats, ans = parsed["patterns"], parsed["select_var"]
    if not pats or ans is None:
        return {"ok": False, "reason": "khong co pattern hoac khong co bien SELECT"}

    if topic_mids:
        anchors = [t for t in topic_mids if any(t in (s, o) for s, p, o in pats)]
        anchor_src = "dataset"
    else:
        type_objs = {o for s, p, o in pats if p in TYPE_PREDS}
        anchors = sorted({x for s, p, o in pats for x in (s, o)
                          if MID.match(x) and x not in type_objs})
        anchor_src = "heuristic"
    if not anchors:
        return {"ok": False, "reason": "khong co anchor entity (ns:m.*) trong WHERE"}

    adj = collections.defaultdict(list)                       # node -> [(node_khac, chi_so_pattern)]
    for i, (s, p, o) in enumerate(pats):
        adj[s].append((o, i)); adj[o].append((s, i))

    def chain_from(t: str) -> list[int] | None:
        prev, seen = {t: None}, {t}
        dq = collections.deque([t])
        while dq:
            u = dq.popleft()
            if u == ans:
                break
            for v, pi in adj[u]:
                if v not in seen:
                    seen.add(v); prev[v] = (u, pi); dq.append(v)
        if ans not in prev:
            return None
        idxs, cur = [], ans
        while prev[cur] is not None:
            u, pi = prev[cur]; idxs.append(pi); cur = u
        return idxs[::-1]

    found = [(t, c) for t in anchors if (c := chain_from(t)) is not None]
    if not found:
        return {"ok": False, "reason": "khong co duong tu anchor toi bien dap an trong SPARQL"}
    found.sort(key=lambda tc: -len(tc[1]))                    # chain dai nhat lam chain chinh
    topic, chain_idx = found[0]

    def to_hops(t: str, idxs: list[int]) -> list[dict[str, str]]:
        hops, cur = [], t
        for pi in idxs:
            s, p, o = pats[pi]
            fwd = (s == cur)
            hops.append({"predicate": p, "direction": "fwd" if fwd else "rev",
                         "from": cur, "to": o if fwd else s})
            cur = o if fwd else s
        return hops

    chain = to_hops(topic, chain_idx)
    other_chains = [{"anchor": t, "chain": to_hops(t, c)} for t, c in found[1:]]

    chain_nodes = [topic] + [h["to"] for h in chain]
    # CHU Y: o muc pattern khong phan biet duoc CVT voi entity co ten -- ca hai deu la bien.
    # Viec phan biet lam o bind_to_subgraph, bang bang ten entity (giong rang buoc cua mm).
    intermediates = [n for n in chain_nodes[1:-1] if _is_var(n)]

    chain_set = set(chain_idx)
    constraints = []
    for i, (s, p, o) in enumerate(pats):
        if i in chain_set:
            continue
        attach = [n for n in (s, o) if n in chain_nodes]
        constraints.append({"predicate": p, "subject": s, "object": o,
                            "attaches_to": attach,
                            "kind": ("type" if p.endswith("notable_types") or p == "type.object.type"
                                     else "order_key" if parsed["order_by"] and
                                          parsed["order_by"]["var"] in (s, o)
                                     else "value" if MID.match(o)
                                     else "other")})
    ob = parsed["order_by"]
    if ob:
        ob = dict(ob, predicate=next((c["predicate"] for c in constraints
                                      if ob["var"] in (c["subject"], c["object"])), None))
    return {"ok": True, "topic": topic, "anchor_source": anchor_src,
            "other_anchor_chains": other_chains, "answer_var": ans,
            "chain": chain, "n_topological_hops": len(chain),
            "intermediate_vars": intermediates,
            "constraints": constraints, "order_by": ob, "limit": parsed["limit"],
            "n_filters": len(parsed["filters"]), "unparsed": parsed["unparsed"]}


# ------------------------------------------------------- bind vao subgraph
def bind_to_subgraph(label: dict[str, Any], triples: list[tuple[str, str, str]],
                     named_ids: set[str] | None = None) -> dict[str, Any]:
    """Khop chain va constraint cua SPARQL vao subgraph cua cau hoi.

    triples: danh sach (head_id, relation, tail_id) cua subgraph (dung id Freebase).
    Tra ve cac triple duoc danh nhan duong, tach theo vai tro.
    Day la nhan thay the cho shortest-path: no la *mo hinh hoa dung cua cau hoi*,
    khong phai duong ngan nhat tinh co noi topic voi dap an.
    """
    by_sp = collections.defaultdict(list)                     # (subject, predicate) -> [object]
    by_po = collections.defaultdict(list)                     # (predicate, object)  -> [subject]
    for h, r, t in triples:
        by_sp[(h, r)].append(t); by_po[(r, t)].append(h)

    frontier = {label["topic"]: []}                           # node -> cac triple da di qua
    for hop in label["chain"]:
        nxt = {}
        for node, trail in frontier.items():
            if hop["direction"] == "fwd":
                for t in by_sp.get((node, hop["predicate"]), []):
                    nxt[t] = trail + [(node, hop["predicate"], t)]
            else:
                for h in by_po.get((hop["predicate"], node), []):
                    nxt[h] = trail + [(h, hop["predicate"], node)]
        frontier = nxt
        if not frontier:
            break

    chain_triples = {tr for trail in frontier.values() for tr in trail}
    chain_nodes = set(frontier) | {label["topic"]} | {tr[0] for tr in chain_triples} \
                                | {tr[2] for tr in chain_triples}

    constraint_triples = set()
    for c in label["constraints"]:
        for n in chain_nodes:
            for t in by_sp.get((n, c["predicate"]), []):
                if not MID.match(c["object"]) or c["object"] == t:
                    constraint_triples.add((n, c["predicate"], t))
            for h in by_po.get((c["predicate"], n), []):
                constraint_triples.add((h, c["predicate"], n))

    # Buoc ngu nghia: nut trung gian KHONG co ten (CVT) khong tinh la mot buoc.
    # named_ids = tap id entity co text trong record cua cau hoi.
    n_semantic = None
    if named_ids is not None and frontier:
        trail = next(iter(frontier.values()))
        cur, inter = label["topic"], []
        for h, r, t in trail:
            cur = t if h == cur else h
            inter.append(cur)
        inter = inter[:-1]                                    # bo nut dap an
        n_semantic = len(trail) - sum(1 for nid in inter if nid not in named_ids)

    return {"chain_triples": sorted(chain_triples),
            "constraint_triples": sorted(constraint_triples - chain_triples),
            "n_answer_candidates": len(frontier),
            "n_topological_hops": len(label["chain"]),
            "n_semantic_hops": n_semantic,
            "chain_complete": bool(frontier)}


# --------------------------------------------------------------------- cli
def iter_questions(path: str, dataset: str):
    data = json.load(open(path, encoding="utf-8"))
    if dataset == "cwq":
        for r in data:
            yield r["ID"], r.get("question", ""), r["sparql"]
    else:                                                      # webqsp
        for r in data["Questions"]:
            for p in r.get("Parses", []):
                if p.get("Sparql"):
                    yield r["QuestionId"], r.get("RawQuestion", ""), p["Sparql"]
                    break


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cwq", "webqsp"], required=True)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    n_ok = n = 0
    reasons = collections.Counter()
    with open(a.out, "w", encoding="utf-8") as f:
        for qid, question, sparql in iter_questions(a.inp, a.dataset):
            lab = extract_label(parse_sparql(sparql))
            n += 1; n_ok += bool(lab.get("ok"))
            if not lab.get("ok"):
                reasons[lab["reason"]] += 1
            f.write(json.dumps({"id": qid, "question": question, **lab},
                               ensure_ascii=False) + "\n")
    print(f"{n_ok}/{n} cau trich duoc chain ({n_ok/max(n,1):.1%})")
    for r, c in reasons.most_common():
        print(f"  {c:>6}  {r}")


if __name__ == "__main__":
    main()
