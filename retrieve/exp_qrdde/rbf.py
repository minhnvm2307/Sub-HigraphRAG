"""RBF-Ret core (docs/11_KE_HOACH_TUAN_1011.md): relevance-gated, step-factorised Bellman-Ford over the entity graph.

One-hop operator (shared by every step, so a k-hop path score is a product of k one-hop scores):
    pi(r | q) = sigmoid(MLP([z_q ; z_r ; z_q * z_r ; prior_r]))   two outputs: forward (h -> t) and reverse (t -> h)
    prior_r  = [lexical(q, r), cos(z_q, z_r)]                       the non-learned relation-topic signal of WL
Per anchor a, max-product propagation in log space for T steps (most reliable path, generalised Bellman-Ford):
    log p_a^0 = 0 at a, -inf elsewhere
    log p_a^{t+1}(v) = max(log p_a^t(v), max_{(u,r,v)} log p_a^t(u) + log pi_f(r), max_{(v,r,u)} log p_a^t(u) + log pi_r(r))
Triple scores (log), for every triple e = (h, r, t):
    path_a(e)  = max(log p_a(h) + log pi_f(r), log p_a(t) + log pi_r(r))      e lies on a reliable path from a
    union(e)   = max_a path_a(e)
    inter(e)   = max_a [ path_a(e) + S(far end) - log p_a(far end) ],  S(v) = sum_b log p_b(v)   (product t-norm:
                 the far endpoint is also reached from every other anchor)
    final(e)   = logaddexp(union(e), inter(e))
With one anchor S - log p_a = 0, so inter = union and final = union + log 2: the ranking is exactly the single-anchor
model (in-distribution questions with one topic are unaffected by the set operator by construction).
inter only lifts triples whose far end is itself reached from the other anchors (the last hop), and by at most log 2.
soft_and(e) (inference only, eps) is the max-marginal of the best conjunctive structure through e:
    J_a(v)    = sum_{b != a} max(log p_b(v), log eps)          other anchors as soft constraints (eps: a constraint may
                                                                be unmet / a spurious topic; eps -> 0 strict AND,
                                                                eps = 1 union)
    C_a(u)    = max_v [ best path score u -> v + J_a(v) ]     backward Bellman-Ford from J_a, T steps
    soft_and(e) = max_a max(log p_a(h) + log pi_f(r) + C_a(t), log p_a(t) + log pi_r(r) + C_a(h))
One anchor: J = 0, C = 0, soft_and = union.
No parameter belongs to the set operators; only pi is learned.
Node factor (node=True): a second one-step factor psi(v | q) = sigmoid(MLP([z_q ; z_v ; z_q * z_v])) on every reached
entity (CVT / non-text nodes share one learned logit, anchors get 1), so the model is max-product inference on a
factor graph with edge potentials pi(r | q) and node potentials psi(v | q):
    log p_a^{t+1}(v) = max(..., max_u log p_a^t(u) + log pi(r) + log psi(v))
psi sees one entity and the question only, never a composition, so step-factorisation is kept; it separates triples
that pi alone ties (same relation out of the same node) and carries type / constraint evidence.
Non-backtracking (nb=True): a walk may not take relation r forwards right after taking it backwards, or the reverse
(t' -r^-1-> h -r-> t visits a sibling of h; h' -r-> t -r^-1-> h visits a co-parent). These "bounces" score as the
product of two high factors and flood the budget from hub anchors (time zones, type nodes). Exact max-product over
non-backtracking walks on the entity graph: each node keeps its best score with the tag of its last step (2r forward,
2r+1 reverse) and its best score with a *different* last-step tag (Viterbi top-2 with distinct back-pointers); a step
that would bounce reads the second one. A line graph gets this for free (its states are edges, O(|E|) states and
up to O(sum deg^2) transitions); here it costs 2 states per node.
Complete-path max-marginal (mm=True): the triple score above is a *prefix* score, so it never increases along a path and
the first hop of a relevant relation always outranks its continuation. Through Freebase mediator (CVT) nodes, which are
never answers, this floods the budget with anchor -> CVT hops ("member" x 80) before any CVT -> answer hop. With mm the
score of e is the best *complete* walk through e: forward messages to e, then the best continuation after e that ends at a
text entity (a structural, parameter-free terminal constraint: walks may stop only at text entities):
    B^0(v) = 0 if v is a text entity else -inf;  B^{k+1}(u) = max(B^k(u), max_{steps u -> v} log pi + B^k(v))
    path_f(e) += B(t), path_r(e) += B(h)                (exact non-backtracking version with nb, top-2 first-step tags)
A text-entity tail keeps its prefix score (stopping is free); a CVT tail gets the score of its best completion.
pi modes: 'mlp'  = the MLP above;
          'gate' = sigmoid(tau . prior_r + b) per direction (6 parameters): the non-learned WL relation-topic signal
                   only, nothing that can memorise the training question distribution (docs/11 H4).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

NEG = -1e4


def smax(src, index, n):
    """scatter_max over dim 1 whose empty slots are NEG (torch_scatter fills them with 0, which would make every node
    without a message as good as an anchor; bug in the non-nb passes before 10-11 noon, see EXP-39)"""
    out, arg = scatter_max(src, index, dim=1, dim_size=n)
    return out.masked_fill(arg == src.shape[1], NEG)


class RBF(nn.Module):
    def __init__(self, emb=1024, hidden=256, T=3, prior=True, pi='mlp', node=False, nb=False, mm=False, Tb=2, extra_prior=0):
        super().__init__()
        self.T, self.prior, self.mode, self.nb, self.mm, self.Tb = T, prior or pi == 'gate', pi, nb, mm, Tb
        self.extra_prior = extra_prior        # extra per-relation prior columns in x['rel_prior'] (e.g. Path+Memory: 2)
        if node:
            self.psi = nn.Sequential(nn.Linear(3 * emb, hidden), nn.ReLU(), nn.Linear(hidden, 1))
            self.psi_nt = nn.Parameter(torch.zeros(1))
        if pi == 'gate':
            self.pi = nn.Linear(2 + extra_prior, 2)
        else:
            k = 3 * emb + (2 if prior else 0) + extra_prior
            self.pi = nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def relation_logits(self, x):
        q = x['q'].expand(len(x['rel']), -1)
        if self.prior:
            cos = F.cosine_similarity(x['rel'], q, dim=-1)
            prior = torch.stack([x['lex'], cos], 1)
            if self.extra_prior:
                prior = torch.cat([prior, x['rel_prior']], 1)
        if self.mode == 'gate':
            return self.pi(prior)
        feats = [q, x['rel'], q * x['rel']] + ([prior] if self.prior else [])
        return self.pi(torch.cat(feats, 1))                      # [n_rel, 2]: forward, reverse

    def forward(self, x, T=None, eps=None):
        """returns dict of per-triple log-scores (union, inter, final) and per-relation logits"""
        T = T or self.T
        n, h, t, r = x['n_ent'], x['h'], x['t'], x['r']
        logit = self.relation_logits(x)
        lpf, lpr = F.logsigmoid(logit[:, 0])[r], F.logsigmoid(logit[:, 1])[r]           # per triple
        anchors = x['topic'][:, 1].nonzero().reshape(-1)
        if hasattr(self, 'psi'):
            q = x['q'].expand(len(x['ent']), -1)
            lv = torch.cat([self.psi(torch.cat([q, x['ent'], q * x['ent']], 1)).view(-1),
                            self.psi_nt.expand(n - len(x['ent']))])
            lv = F.logsigmoid(lv).clone(); lv[anchors] = 0.0                              # log psi(v)
            lpf, lpr = lpf + lv[t], lpr + lv[h]                    # entering t (forward step) / h (reverse step)
        A = max(len(anchors), 1)
        lp = torch.full((A, n), NEG, device=h.device)
        if len(anchors):
            lp[torch.arange(len(anchors), device=h.device), anchors] = 0.0
        if self.nb:
            path_f, path_r, lp = self._nb_paths(lp, h, t, r, lpf, lpr, n, T)
        else:
            for _ in range(T):
                fw = smax(lp[:, h] + lpf, t, n)
                bw = smax(lp[:, t] + lpr, h, n)
                lp = torch.maximum(lp, torch.maximum(fw, bw)).clamp(min=NEG)
            path_f, path_r = lp[:, h] + lpf, lp[:, t] + lpr       # [A, E]: reached at h then step forward, or at t then back
        if self.mm:
            bf, br = self._completion(h, t, r, lpf, lpr, n, len(x['ent']))
            path_f, path_r = (path_f + bf).clamp(min=NEG), (path_r + br).clamp(min=NEG)
        path = torch.maximum(path_f, path_r)
        union = path.max(0).values
        S = lp.sum(0)                                             # [n]
        inter_f = path_f + S[t] - lp[:, t]                        # far end of a forward step is t
        inter_r = path_r + S[h] - lp[:, h]
        inter = torch.maximum(inter_f, inter_r).max(0).values
        final = torch.logaddexp(union, inter)
        out = {'union': union, 'inter': inter, 'final': final, 'rel_logit': logit, 'node_logp': lp, 'per_anchor': path}
        if eps is not None:
            soft = lp.clamp(min=math.log(eps))
            J = soft.sum(0, keepdim=True) - soft                     # [A, n]: sum over the other anchors
            C = J.clone()
            for _ in range(T):                                        # backward: u -> v along a forward or reverse step
                via_f = smax(C[:, t] + lpf, h, n)                             # u = h steps forward to t
                via_r = smax(C[:, h] + lpr, t, n)                             # u = t steps back to h
                C = torch.maximum(C, torch.maximum(via_f, via_r)).clamp(min=NEG)
            out['soft_and'] = torch.maximum(path_f + C[:, t], path_r + C[:, h]).max(0).values
        return out


def _top2_merge(v1, g1, v2, g2, cv, ctag, dst, n):
    """new (best, its tag, best with another tag, its tag) per node from the old state and candidate steps
    cv [A, C] arriving at dst [C] with tags ctag [C]"""
    m1, arg = scatter_max(cv, dst, dim=1, dim_size=n)
    has = arg < cv.shape[1]
    m1 = m1.masked_fill(~has, NEG)
    t1 = torch.where(has, ctag[arg.clamp(max=cv.shape[1] - 1)], torch.full_like(arg, -2))
    take = m1 > v1
    nv1, ng1 = torch.where(take, m1, v1), torch.where(take, t1, g1)
    cand2 = cv.masked_fill(ctag[None, :] == ng1[:, dst], NEG)
    m2, arg2 = scatter_max(cand2, dst, dim=1, dim_size=n)
    has2 = arg2 < cv.shape[1]
    m2 = m2.masked_fill(~has2, NEG)
    t2 = torch.where(has2, ctag[arg2.clamp(max=cv.shape[1] - 1)], torch.full_like(arg2, -2))
    o1 = torch.where(g1 != ng1, v1, torch.full_like(v1, NEG))      # old best survives as 2nd if its tag differs
    o2 = torch.where(g2 != ng1, v2, torch.full_like(v2, NEG))
    nv2, ng2 = m2, t2
    for ov, og in ((o1, g1), (o2, g2)):
        better = ov > nv2
        nv2, ng2 = torch.where(better, ov, nv2), torch.where(better, og, ng2)
    return nv1.clamp(min=NEG), ng1, nv2.clamp(min=NEG), ng2


def _excl(v1, g1, v2, idx, tag):
    """best score at nodes idx [E] over walks whose last step is not tag [E]"""
    return torch.where(g1[:, idx] == tag[None, :], v2[:, idx], v1[:, idx])


def _nb_paths(self, lp, h, t, r, lpf, lpr, n, T):
    v1, v2 = lp, torch.full_like(lp, NEG)
    g1 = torch.full(lp.shape, -1, dtype=torch.long, device=lp.device); g2 = torch.full_like(g1, -2)
    fwd, rev = 2 * r, 2 * r + 1                                    # tag of a forward / reverse step over r
    dst, ctag = torch.cat([t, h]), torch.cat([fwd, rev])
    for _ in range(T):
        cf = _excl(v1, g1, v2, h, rev) + lpf                       # forward h -> t, not right after reverse r into h
        cr = _excl(v1, g1, v2, t, fwd) + lpr                       # reverse t -> h, not right after forward r into t
        v1, g1, v2, g2 = _top2_merge(v1, g1, v2, g2, torch.cat([cf, cr], 1), ctag, dst, n)
    return _excl(v1, g1, v2, h, rev) + lpf, _excl(v1, g1, v2, t, fwd) + lpr, v1


RBF._nb_paths = _nb_paths


def _completion(self, h, t, r, lpf, lpr, n, n_text, init=None):
    """log score of the best continuation after a forward step into t (bf) / a reverse step into h (br) that ends at a
    text entity; anchor-independent, so computed once ([1, E]). init [B, n] = an end bonus per (row, end node), for the
    constraint-aware variant (cmm_scores, one row per anchor); stopping stays allowed only at text entities."""
    if init is None:
        v1 = torch.full((1, n), NEG, device=h.device); v1[:, :n_text] = 0.0      # stopping is allowed at text entities
    else:
        v1 = torch.full_like(init, NEG); v1[:, :n_text] = init[:, :n_text]
    if not self.nb:
        for _ in range(self.Tb):
            cf = smax(v1[:, t] + lpf, h, n)                                   # u = h continues forward to t
            cr = smax(v1[:, h] + lpr, t, n)                                   # u = t continues back to h
            v1 = torch.maximum(v1, torch.maximum(cf, cr)).clamp(min=NEG)
        return v1[:, t], v1[:, h]
    g1 = torch.full(v1.shape, -1, dtype=torch.long, device=h.device); v2 = torch.full_like(v1, NEG); g2 = torch.full_like(g1, -2)
    fwd, rev = 2 * r, 2 * r + 1                                    # tag = the continuation's first step
    dst, ctag = torch.cat([h, t]), torch.cat([fwd, rev])
    for _ in range(self.Tb):
        cf = _excl(v1, g1, v2, t, rev) + lpf                       # u = h steps forward to t; t must not then step back on r
        cr = _excl(v1, g1, v2, h, fwd) + lpr                       # u = t steps back to h; h must not then step forward on r
        v1, g1, v2, g2 = _top2_merge(v1, g1, v2, g2, torch.cat([cf, cr], 1), ctag, dst, n)
    return _excl(v1, g1, v2, t, rev), _excl(v1, g1, v2, h, fwd)


RBF._completion = _completion


@torch.no_grad()
def cmm_scores(self, x, cons, eps_a=1.0, eps_c=0.1, T=None):
    """constraint-aware complete-path max-marginal (inference only). The walk from anchor a must end at a text entity
    v (as mm), and v gets a soft end bonus for being reached from every OTHER question entity b:
        J_a(v) = sum_{other anchors b} max(log p_b(v), log eps_a) + sum_{constraints c} max(log p_c(v), log eps_c)
    (constraints = entities named in the question that are not topics: walks are run from them, but they never start
    a retrieved path). eps = the probability that a constraint is unmet / spurious; eps_a = eps_c = 1 -> exactly mm.
    Returns per-triple log scores (max over anchors)."""
    T = T or self.T
    n, h, t, r = x['n_ent'], x['h'], x['t'], x['r']
    logit = self.relation_logits(x)
    lpf, lpr = F.logsigmoid(logit[:, 0])[r], F.logsigmoid(logit[:, 1])[r]
    anchors = x['topic'][:, 1].nonzero().reshape(-1); na = len(anchors)
    starts = torch.cat([anchors, cons.to(anchors.device)]) if len(cons) else anchors
    lp = torch.full((max(len(starts), 1), n), NEG, device=h.device)
    if len(starts):
        lp[torch.arange(len(starts), device=h.device), starts] = 0.0
    if self.nb:
        path_f, path_r, lp = self._nb_paths(lp, h, t, r, lpf, lpr, n, T)
    else:
        for _ in range(T):
            lp = torch.maximum(lp, torch.maximum(smax(lp[:, h] + lpf, t, n), smax(lp[:, t] + lpr, h, n))).clamp(min=NEG)
        path_f, path_r = lp[:, h] + lpf, lp[:, t] + lpr
    soft = torch.cat([lp[:na].clamp(min=math.log(eps_a)), lp[na:].clamp(min=math.log(eps_c))])
    J = soft.sum(0, keepdim=True) - soft
    bf, br = self._completion(h, t, r, lpf, lpr, n, len(x['ent']), init=J[:max(na, 1)])
    path = torch.maximum((path_f[:max(na, 1)] + bf).clamp(min=NEG), (path_r[:max(na, 1)] + br).clamp(min=NEG))
    return path.max(0).values


RBF.cmm_scores = cmm_scores


def log_to_logit(lm):
    """logit of a probability given as log m (m <= 1)"""
    lm = lm.clamp(max=-1e-6)
    return lm - torch.log(-torch.expm1(lm))
