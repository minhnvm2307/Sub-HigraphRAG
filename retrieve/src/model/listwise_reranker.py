"""Listwise-only components used by the stage-2 reranking experiment."""

import torch
import torch.nn as nn


def apply_listwise_residual(pw_logits, top_n_indices, delta, rank_margin=1e-4):
    """Apply a listwise residual while keeping pointwise top-N above the tail."""
    pw_logits = pw_logits.reshape(-1)
    delta = delta.reshape(-1)
    if top_n_indices.ndim != 1 or len(top_n_indices) != len(delta):
        raise ValueError("top_n_indices and delta must be aligned 1D tensors")
    if len(top_n_indices) == 0:
        return pw_logits

    reranked_top_n = pw_logits[top_n_indices] + delta
    if len(top_n_indices) < len(pw_logits):
        top_n_mask = torch.zeros_like(pw_logits, dtype=torch.bool)
        top_n_mask[top_n_indices] = True
        tail_max = pw_logits[~top_n_mask].max()
        required_offset = tail_max + rank_margin - reranked_top_n.min()
        # This offset does not alter ordering inside top-N. It only preserves
        # the candidate-set boundary required by two-stage retrieval.
        reranked_top_n = reranked_top_n + required_offset.clamp_min(0).detach()

    final_logits = pw_logits.clone()
    final_logits[top_n_indices] = reranked_top_n
    return final_logits


class ListwiseReranker(nn.Module):
    """Set-wise residual scorer for a frozen pointwise top-N pool."""

    def __init__(
        self,
        in_dim,
        d_model=256,
        num_heads=4,
        num_layers=2,
        dim_feedforward=1024,
        dropout=0.1,
    ):
        super().__init__()
        if d_model < 1 or dim_feedforward < 1 or num_layers < 1:
            raise ValueError("listwise dimensions and num_layers must be positive")
        if num_heads < 1 or d_model % num_heads != 0:
            raise ValueError("listwise.d_model must be divisible by listwise.num_heads")

        self.input_projection = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # A query-wide bias has no ranking value and encourages collapse.
        self.delta_head = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.delta_head.weight)

    def forward(self, triple_features):
        h_local = self.input_projection(triple_features)
        h_attn = self.transformer(h_local.unsqueeze(0)).squeeze(0)
        raw_delta = self.delta_head(h_attn + h_local)
        # Restrict this module to relative reranking corrections.
        return raw_delta - raw_delta.mean(dim=0, keepdim=True)
