"""RBF-Ret: relevance-gated, step-factorised Bellman-Ford over the entity graph (docs/11, docs/13).

The only learned part is a ONE-HOP relation factor shared by every step, so the score of a k-hop path is the product
of k one-hop scores and generalises to compositions never seen in training:
    pi(r | q) = sigmoid(MLP([z_q ; z_r ; z_q * z_r ; lexical(q, r) ; cos(z_q, z_r) ; extra prior]))
two outputs per relation: a forward step (h -> t) and a reverse step (t -> h). The extra prior is optional
(--rel_prior bge / bge2: the fine-tuned BGE cosine of the question with the relation, 2 columns).

Scoring, per anchor (topic entity) a, in log space:
  1. propagation (max-product, T steps): log p_a(v) = score of the most reliable walk from a to v.
     Walks are NON-BACKTRACKING: a walk may not take relation r forwards right after taking it backwards (or the
     reverse). Such "bounces" (sibling -> parent -> sibling) score as the product of two high factors and flood the
     budget from hub anchors. Exact max-product over non-backtracking walks keeps, per node, the best score with the
     tag of its last step and the best score with a different last-step tag (Viterbi top-2).
  2. triple score: path_a(e) = best walk from a that ends by crossing e = (h, r, t), either direction.
  3. complete-path max-marginal (mm, switched on at TEST time): the score of e becomes the best COMPLETE walk through
     e, i.e. path_a(e) + the best continuation after e that stops at a text entity (walks may only stop at text
     entities, never at a mediator/CVT node). Without it every "anchor -> CVT" first hop outranks its useful
     continuation "CVT -> answer".
  4. several anchors: union(e) = max_a path_a(e); inter(e) = the same path when its far end is also reached from the
     other anchors (product t-norm); final(e) = logaddexp(union, inter). With one anchor final = union + log 2.
The final score is used as a logit (log_to_logit) and trained with BCE on the triple labels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max

NEG = -1e4                       # log of "unreachable"
COMPLETION_STEPS = 2             # mm: longest continuation after a triple


class RBF(nn.Module):
    def __init__(self, emb=1024, hidden=256, T=3, mm=False, extra_prior=0, pi_input='full', stop=False,
                 dir_bigram=False, L=COMPLETION_STEPS):
        """emb: text-embedding size; T: propagation steps; mm: complete-path scoring (set True at test time);
        extra_prior: number of extra per-relation prior columns read from x['rel_prior'] (BGE: 2);
        pi_input: ablation of the inputs of pi ('full', 'no_lex': drop lexical(q, r), 'no_prod': drop z_q * z_r,
        'no_lex_prod': drop both); stop: add the stop factor phi (see below); L: completion steps of mm;
        dir_bigram: add the direction-bigram factor psi (see below)"""
        super().__init__()
        self.T, self.mm, self.extra_prior, self.L = T, mm, extra_prior, L
        self.use_lex, self.use_prod = 'lex' not in pi_input, 'prod' not in pi_input
        n_in = (3 if self.use_prod else 2) * emb + 1 + int(self.use_lex) + extra_prior
        self.pi = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        # stop factor phi(stop | q, r): probability that a walk whose LAST step is relation r (forward / reverse) ends
        # there, i.e. that r is the question's answer relation. Same inputs as pi, so it still depends on (q, r) only.
        # Used by mm at test time: a complete walk pays log phi of its last step instead of stopping for free.
        self.stop = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, 2)) if stop else None
        # direction bigram psi(d_prev, d_cur) and first-step bias chi(d_1): 6 parameters in total, SHARED by every
        # relation. They carry composition-ORDER information (a gold chain through a CVT has a canonical direction
        # pattern) while no parameter is indexed by a relation or a relation pair, so the step-factorised scoring
        # stays systematic: the scorer learns nothing it could index by "r1 then r2".
        # Reference implementation + brute-force test: analysis/ref_semirings.py (claims C-A, C-A2).
        # NOTE: psi is applied in the PREFIX propagation only. best_completion (mm) still scores its continuation
        # without psi, so a dir_bigram run with mm=True mixes a psi-aware prefix with a psi-free suffix. Run the
        # dir_bigram ablation with mm off for a clean comparison; extending psi into the completion is follow-up work.
        self.dir_bigram = dir_bigram
        if dir_bigram:
            self.psi = nn.Parameter(torch.zeros(2, 2))
            self.chi = nn.Parameter(torch.zeros(2))

    def relation_inputs(self, x):
        q = x['q'].expand(len(x['rel']), -1)
        parts = [q, x['rel']] + ([q * x['rel']] if self.use_prod else [])
        parts += ([x['lex'][:, None]] if self.use_lex else []) + [F.cosine_similarity(x['rel'], q, dim=-1)[:, None]]
        if self.extra_prior:
            parts.append(x['rel_prior'])
        return torch.cat(parts, 1)

    def relation_logits(self, x):
        """[n_relations, 2] logits of pi(r | q): column 0 = forward step, column 1 = reverse step"""
        return self.pi(self.relation_inputs(x))

    def stop_logits(self, x):
        """[n_relations, 2] logits of phi(stop | q, r) for a last step forward / reverse"""
        return self.stop(self.relation_inputs(x))

    def forward(self, x):
        """-> final log-score of every triple [n_triples]"""
        n, h, t, r = x['n_ent'], x['h'], x['t'], x['r']
        logit = self.relation_logits(x)
        log_pi_fwd, log_pi_rev = F.logsigmoid(logit[:, 0])[r], F.logsigmoid(logit[:, 1])[r]     # per triple
        anchors = x['topic'][:, 1].nonzero().reshape(-1)
        log_p = torch.full((max(len(anchors), 1), n), NEG, device=h.device)                    # [anchors, nodes]
        if len(anchors):
            log_p[torch.arange(len(anchors), device=h.device), anchors] = 0.0
        if self.dir_bigram:
            path_fwd, path_rev, log_p = propagate_nb_dirbigram(log_p, h, t, r, log_pi_fwd, log_pi_rev, n, self.T,
                                                               self.psi, self.chi)
        else:
            path_fwd, path_rev, log_p = propagate_non_backtracking(log_p, h, t, r, log_pi_fwd, log_pi_rev, n, self.T)
        if self.mm:
            log_stop = F.logsigmoid(self.stop_logits(x)) if self.stop is not None else None
            after_fwd, after_rev = best_completion(h, t, r, log_pi_fwd, log_pi_rev, n, n_text=len(x['ent']),
                                                   steps=self.L,
                                                   log_stop_fwd=None if log_stop is None else log_stop[:, 0][r],
                                                   log_stop_rev=None if log_stop is None else log_stop[:, 1][r])
            path_fwd, path_rev = (path_fwd + after_fwd).clamp(min=NEG), (path_rev + after_rev).clamp(min=NEG)
        path = torch.maximum(path_fwd, path_rev)                                                # [anchors, triples]
        union = path.max(0).values
        reached_by_all = log_p.sum(0)                                                           # [nodes]
        inter = torch.maximum(path_fwd + reached_by_all[t] - log_p[:, t],
                              path_rev + reached_by_all[h] - log_p[:, h]).max(0).values
        return torch.logaddexp(union, inter)


# ------------------------------------------------------------------ non-backtracking max-product (Viterbi top-2)
def _top2_merge(best, best_tag, second, second_tag, candidates, candidate_tags, destinations, n):
    """Update each node's (best score, its last-step tag, best score with another tag, its tag) with candidate steps
    `candidates` [A, C] arriving at `destinations` [C] with tags `candidate_tags` [C]."""
    m1, arg = scatter_max(candidates, destinations, dim=1, dim_size=n)
    has = arg < candidates.shape[1]
    m1 = m1.masked_fill(~has, NEG)
    t1 = torch.where(has, candidate_tags[arg.clamp(max=candidates.shape[1] - 1)], torch.full_like(arg, -2))
    take = m1 > best
    new_best, new_tag = torch.where(take, m1, best), torch.where(take, t1, best_tag)
    others = candidates.masked_fill(candidate_tags[None, :] == new_tag[:, destinations], NEG)
    m2, arg2 = scatter_max(others, destinations, dim=1, dim_size=n)
    has2 = arg2 < candidates.shape[1]
    m2 = m2.masked_fill(~has2, NEG)
    t2 = torch.where(has2, candidate_tags[arg2.clamp(max=candidates.shape[1] - 1)], torch.full_like(arg2, -2))
    old_best = torch.where(best_tag != new_tag, best, torch.full_like(best, NEG))   # old best survives as 2nd
    old_second = torch.where(second_tag != new_tag, second, torch.full_like(second, NEG))
    new_second, new_second_tag = m2, t2
    for value, tag in ((old_best, best_tag), (old_second, second_tag)):
        better = value > new_second
        new_second, new_second_tag = torch.where(better, value, new_second), torch.where(better, tag, new_second_tag)
    return new_best.clamp(min=NEG), new_tag, new_second.clamp(min=NEG), new_second_tag


def _best_without_tag(best, best_tag, second, nodes, tag):
    """best score at `nodes` [E] over walks whose last step is not `tag` [E]"""
    return torch.where(best_tag[:, nodes] == tag[None, :], second[:, nodes], best[:, nodes])


def propagate_non_backtracking(log_p, h, t, r, log_pi_fwd, log_pi_rev, n, T):
    """T steps of non-backtracking max-product from the anchor rows of log_p [A, n].
    -> (score of crossing each triple forwards [A, E], backwards [A, E], node scores [A, n])"""
    best, second = log_p, torch.full_like(log_p, NEG)
    best_tag = torch.full(log_p.shape, -1, dtype=torch.long, device=log_p.device)
    second_tag = torch.full_like(best_tag, -2)
    tag_fwd, tag_rev = 2 * r, 2 * r + 1                       # tag of a forward / reverse step over relation r
    destinations, tags = torch.cat([t, h]), torch.cat([tag_fwd, tag_rev])
    for _ in range(T):
        step_fwd = _best_without_tag(best, best_tag, second, h, tag_rev) + log_pi_fwd   # h -> t, not right after t -> h on r
        step_rev = _best_without_tag(best, best_tag, second, t, tag_fwd) + log_pi_rev   # t -> h, not right after h -> t on r
        best, best_tag, second, second_tag = _top2_merge(best, best_tag, second, second_tag,
                                                         torch.cat([step_fwd, step_rev], 1), tags, destinations, n)
    return (_best_without_tag(best, best_tag, second, h, tag_rev) + log_pi_fwd,
            _best_without_tag(best, best_tag, second, t, tag_fwd) + log_pi_rev, best)


def propagate_nb_dirbigram(log_p, h, t, r, log_pi_fwd, log_pi_rev, n, T, psi, chi):
    """T steps of non-backtracking max-product WITH the direction-bigram factor psi and first-step bias chi.

    The state is a per-direction Viterbi top-2: for each node and each LAST-STEP DIRECTION d, the best score and
    the best score with a different last-step tag (8 scalars per node instead of 4). This is exactly what the
    factor needs and no more: a step into direction d must exclude the single tag (r, 1 - d), and psi depends on
    the previous DIRECTION only, so

        base(src, d) = max( start(src) + chi[d],  max over d_prev [ excl(src, d_prev, (r, 1 - d)) + psi[d_prev, d] ] )

    covers first steps and later steps with one expression. `log_p` supplies the boundary condition (0 at the
    anchors) and is kept as the constant `start` row. Verified against brute force in analysis/ref_semirings.py
    (claim C-A2); keep the two in sync.

    -> (score of crossing each triple forwards [A, E], backwards [A, E], node scores [A, n])
    """
    start = log_p                                                     # [A, n], 0 at the anchor of each row
    shape = log_p.shape
    best = [torch.full(shape, NEG, device=log_p.device) for _ in range(2)]        # per last-step direction
    second = [torch.full(shape, NEG, device=log_p.device) for _ in range(2)]
    best_tag = [torch.full(shape, -1, dtype=torch.long, device=log_p.device) for _ in range(2)]
    second_tag = [torch.full(shape, -2, dtype=torch.long, device=log_p.device) for _ in range(2)]
    tag_fwd, tag_rev = 2 * r, 2 * r + 1
    log_pi = (log_pi_fwd, log_pi_rev)
    source = (h, t)                                                   # a step into direction d starts here
    destination = (t, h)
    forbidden = (tag_rev, tag_fwd)                                    # the tag a step into direction d may not follow

    def base(d):
        """[A, E]: best predecessor value at the source of a step into direction d, psi and chi included"""
        nodes = source[d]
        out = start[:, nodes] + chi[d]
        for d_prev in range(2):
            prev = _best_without_tag(best[d_prev], best_tag[d_prev], second[d_prev], nodes, forbidden[d])
            out = torch.maximum(out, prev + psi[d_prev, d])
        return out

    for _ in range(T):
        candidates = [base(d) + log_pi[d] for d in range(2)]           # computed before any update: one synchronous round
        for d in range(2):
            tags = tag_fwd if d == 0 else tag_rev
            best[d], best_tag[d], second[d], second_tag[d] = _top2_merge(
                best[d], best_tag[d], second[d], second_tag[d],
                candidates[d], tags, destination[d], n)
    node = torch.maximum(torch.maximum(best[0], best[1]), start)
    return base(0) + log_pi_fwd, base(1) + log_pi_rev, node


def best_completion(h, t, r, log_pi_fwd, log_pi_rev, n, n_text, log_stop_fwd=None, log_stop_rev=None,
                    steps=COMPLETION_STEPS):
    """log score of the best non-backtracking continuation (<= COMPLETION_STEPS steps) after a forward step into t /
    a reverse step into h that stops at a text entity (nodes 0..n_text-1); anchor-independent -> [1, E] each.
    Without a stop factor, stopping at a text entity is free. With one (log_stop_* per triple = log phi of the step
    over that triple), a walk may stop right after a step into a text entity and pays log phi of that step."""
    best = torch.full((1, n), NEG, device=h.device)
    if log_stop_fwd is None:
        best[:, :n_text] = 0.0                                # stopping at a text entity is free
        stop_fwd = stop_rev = None
    else:                                                     # stopping right after the step over the triple
        stop_fwd = torch.where(t < n_text, log_stop_fwd, torch.full_like(log_stop_fwd, NEG))[None, :]
        stop_rev = torch.where(h < n_text, log_stop_rev, torch.full_like(log_stop_rev, NEG))[None, :]
    best_tag = torch.full(best.shape, -1, dtype=torch.long, device=h.device)
    second, second_tag = torch.full_like(best, NEG), torch.full_like(best_tag, -2)
    tag_fwd, tag_rev = 2 * r, 2 * r + 1                       # tag = the continuation's first step
    destinations, tags = torch.cat([h, t]), torch.cat([tag_fwd, tag_rev])
    after = lambda node, tag, stop: (_best_without_tag(best, best_tag, second, node, tag) if stop is None
                                     else torch.maximum(_best_without_tag(best, best_tag, second, node, tag), stop))
    for _ in range(steps):
        step_fwd = after(t, tag_rev, stop_fwd) + log_pi_fwd   # u = h continues forward to t (and stops there or goes on)
        step_rev = after(h, tag_fwd, stop_rev) + log_pi_rev   # u = t continues back to h
        best, best_tag, second, second_tag = _top2_merge(best, best_tag, second, second_tag,
                                                         torch.cat([step_fwd, step_rev], 1), tags, destinations, n)
    return after(t, tag_rev, stop_fwd), after(h, tag_fwd, stop_rev)


def log_to_logit(log_m):
    """logit of a probability given as log m (m <= 1)"""
    log_m = log_m.clamp(max=-1e-6)
    return log_m - torch.log(-torch.expm1(log_m))
