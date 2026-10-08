"""REL-RAG retriever: bidirectional 2-layer GCN on the line graph (paper Eq. 86, App. B, F.1).

Heads (q_mlp, r_mlp, stop_mlp, stop_emb) follow the original RAPL `Trainerv3`.
GCN propagation uses torch.sparse.mm so memory is O(N*F) instead of O(E*F).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gcn_norm_adj(edge_index, n, self_loops, device):
    """Sparse A_hat[dst, src] with PyG gcn_norm semantics (degree counted at the target)."""
    src, dst = edge_index[0].to(device), edge_index[1].to(device)
    if self_loops:
        loop = torch.arange(n, device=device)
        src, dst = torch.cat([src, loop]), torch.cat([dst, loop])
    w = torch.ones(src.numel(), device=device)
    deg = torch.zeros(n, device=device).scatter_add_(0, dst, w)
    dis = deg.pow(-0.5)
    dis[torch.isinf(dis)] = 0
    w = dis[src] * w * dis[dst]
    return torch.sparse_coo_tensor(torch.stack([dst, src]), w, (n, n)).coalesce()


class RelRagRetriever(nn.Module):
    def __init__(self, in_dims=3072, hidden_dims=512, out_dims=512, num_layers=2, dropout=0.2,
                 self_loops=True, bidirectional=True, emb_dim=1024, residual=False, combine='mean', gnn_dropout=True):
        super().__init__()
        self.self_loops, self.bidirectional, self.dropout = self_loops, bidirectional, dropout
        # combine: 'mean' = paper App. Eq. 86; 'sum' = paper main text and the authors' Trainerv3._gnn_forward (h + hb).
        # gnn_dropout: dropout between GCN layers; the authors' code has none there (only in q_mlp / stop_mlp).
        self.combine, self.gnn_dropout = combine, gnn_dropout
        dims = [in_dims] + [hidden_dims] * (num_layers - 1) + [out_dims]
        self.fwd = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers))
        if bidirectional:
            self.bwd = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(num_layers))
        self.q_mlp = nn.Sequential(nn.Linear(emb_dim, 512), nn.ReLU(), nn.Dropout(dropout), nn.Linear(512, out_dims))
        self.r_mlp = nn.Sequential(nn.Linear(emb_dim, 512), nn.ReLU(), nn.Linear(512, out_dims))
        self.stop_mlp = nn.Sequential(nn.Linear(emb_dim, hidden_dims), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(hidden_dims, out_dims))
        self.stop_emb = nn.Parameter(torch.randn(out_dims))
        self.emb_dim = emb_dim
        # improvement: skip connection so a triple keeps its own [h|r|t] signal despite 1/deg normalisation
        self.res = nn.Linear(in_dims, out_dims) if residual else None

    def _propagate(self, layers, x, adj):
        h = x
        for i, lin in enumerate(layers):
            mm = torch.sparse.mm if adj.is_sparse else torch.mm
            h = mm(adj, F.linear(h, lin.weight)) + lin.bias  # A_hat (X W) + b
            if i != len(layers) - 1:
                h = F.relu(h)
                if self.gnn_dropout:
                    h = F.dropout(h, self.dropout, self.training)
        return h

    def forward(self, s, device):
        """Returns dict with node reps z, topic reps, question rep and conditional stop rep."""
        x = s.x.to(device, non_blocking=True).float()
        n = x.shape[0]
        adj_f = gcn_norm_adj(s.adj_fwd, n, self.self_loops, device)
        # cuSPARSE crashes (illegal memory access) on tiny graphs, e.g. a 1-triple CWQ subgraph: use dense
        tiny = n < 32
        if tiny:
            adj_f = adj_f.to_dense()
        z = self._propagate(self.fwd, x, adj_f)
        if self.bidirectional:
            adj_b = gcn_norm_adj(s.adj_fwd.flip(0), n, self.self_loops, device)
            adj_b = adj_b.to_dense() if tiny else adj_b
            zb = self._propagate(self.bwd, x, adj_b)
            z = z + zb if self.combine == 'sum' else 0.5 * (z + zb)
        if self.res is not None:
            z = z + self.res(x)
        q = s.q.to(device).float()
        rel = x[:, self.emb_dim:2 * self.emb_dim]
        return {
            'z': z,
            'z_topic': self.r_mlp(rel) + z,                    # as in Trainerv3 topic scoring
            'zq': self.q_mlp(q)[0],
            'stop': self.stop_emb + self.stop_mlp(q)[0],
        }
