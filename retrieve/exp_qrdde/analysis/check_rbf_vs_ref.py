"""Cross-check the torch propagation in rbf.py against the brute-force-verified numpy reference.

Two directions, both on random graphs:
  1. `propagate_non_backtracking` (the published path) with psi = 0, chi = 0 must equal the reference. This
     validates ref_semirings.py against code whose results are already reported, so the reference is not just
     self-consistent.
  2. `propagate_nb_dirbigram` with random psi, chi must equal `compressed_max_product_accum`, which claim C-A2
     checks against brute force. This is the only verification the new torch code gets, so run it before training.

Needs torch and torch_scatter (not installed in every environment; the numpy reference alone needs neither).

  python check_rbf_vs_ref.py --graphs 200 --T 3
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rbf import NEG, propagate_nb_dirbigram, propagate_non_backtracking   # noqa: E402

from ref_semirings import compressed_max_product_accum, random_factors, random_graph   # noqa: E402


def torch_nodes(h, r, t, n_ent, anchors, T, log_pi, psi=None, chi=None):
    """node scores [A, n] from the torch implementation"""
    h_t, r_t, t_t = (torch.from_numpy(np.asarray(a, dtype=np.int64)) for a in (h, r, t))
    log_pi_t = torch.from_numpy(log_pi).double()
    log_p = torch.full((len(anchors), n_ent), float(NEG), dtype=torch.double)
    for row, a in enumerate(anchors):
        log_p[row, a] = 0.0
    fwd, rev = log_pi_t[:, 0][r_t], log_pi_t[:, 1][r_t]
    if psi is None:
        _, _, node = propagate_non_backtracking(log_p, h_t, t_t, r_t, fwd, rev, n_ent, T)
    else:
        _, _, node = propagate_nb_dirbigram(log_p, h_t, t_t, r_t, fwd, rev, n_ent, T,
                                            torch.from_numpy(psi).double(), torch.from_numpy(chi).double())
    return node.numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--graphs', type=int, default=200)
    ap.add_argument('--T', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    worst_plain, worst_bigram, n = 0.0, 0.0, 0
    for _ in range(a.graphs):
        h, r, t, n_ent, n_rel, _ = random_graph(rng)
        if len(h) == 0:
            continue
        n += 1
        log_pi, psi, chi = random_factors(rng, n_rel)
        anchor = int(rng.integers(0, n_ent))
        zero = np.zeros((2, 2)), np.zeros(2)

        want = compressed_max_product_accum(h, r, t, n_ent, anchor, a.T, log_pi, *zero)
        got = torch_nodes(h, r, t, n_ent, [anchor], a.T, log_pi)[0]
        mask = want > NEG / 2
        if mask.any():
            worst_plain = max(worst_plain, float(np.abs(want[mask] - got[mask]).max()))

        want = compressed_max_product_accum(h, r, t, n_ent, anchor, a.T, log_pi, psi, chi)
        got = torch_nodes(h, r, t, n_ent, [anchor], a.T, log_pi, psi, chi)[0]
        mask = want > NEG / 2
        if mask.any():
            worst_bigram = max(worst_bigram, float(np.abs(want[mask] - got[mask]).max()))

    print(f'{n} graphs, T={a.T}')
    print(f'  published propagate_non_backtracking  vs reference (psi = chi = 0): max diff {worst_plain:.3e}')
    print(f'  new propagate_nb_dirbigram            vs reference (random psi, chi): max diff {worst_bigram:.3e}')
    bad = [name for name, err in (('plain', worst_plain), ('dir_bigram', worst_bigram)) if err > 1e-9]
    print('MISMATCH in: ' + ', '.join(bad) if bad else 'both agree with the reference.')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
