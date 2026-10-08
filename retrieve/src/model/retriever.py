import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing

class PEConv(MessagePassing):
    def __init__(self, eps=1e-12):
        super().__init__(aggr="add")
        self.eps = eps

    def forward(self, edge_index, x, edge_weight=None):
        num_edges = edge_index.size(1)

        if edge_weight is None:
            edge_weight = x.new_ones(num_edges)
        else:
            edge_weight = edge_weight.to(device=x.device, dtype=x.dtype)

        if edge_weight.dim() != 1 or edge_weight.numel() != num_edges:
            raise ValueError(
                "edge_weight must be a 1D tensor with one value per edge; "
                f"received shape {tuple(edge_weight.shape)} for {num_edges} edges"
            )

        weighted_sum = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            size=(x.size(0), x.size(0)),
        )

        # edge_index[1] contains destination nodes under PyG's default
        # source-to-target message flow.
        destination = edge_index[1]
        weight_sum = x.new_zeros(x.size(0))
        weight_sum.index_add_(0, destination, edge_weight)

        # Nodes without incoming weight keep the zero output produced by the
        # additive aggregation. The mask also avoids division by zero.
        output = torch.zeros_like(weighted_sum)
        valid = weight_sum.abs() > self.eps
        output[valid] = weighted_sum[valid] / weight_sum[valid].unsqueeze(-1)
        return output

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.unsqueeze(-1)


class DDE(nn.Module):
    def __init__(self, num_rounds, num_reverse_rounds):
        super().__init__()

        self.layers = nn.ModuleList(PEConv() for _ in range(num_rounds))
        self.reverse_layers = nn.ModuleList(
            PEConv() for _ in range(num_reverse_rounds)
        )

    def forward(
        self,
        topic_entity_one_hot,
        edge_index,
        reverse_edge_index,
        edge_weight=None,
    ):
        if edge_weight is None:
            edge_weight = topic_entity_one_hot.new_ones(edge_index.size(1))

        result_list = []

        h_pe = topic_entity_one_hot
        for layer in self.layers:
            h_pe = layer(edge_index, h_pe, edge_weight=edge_weight)
            result_list.append(h_pe)

        h_pe_rev = topic_entity_one_hot
        for layer in self.reverse_layers:
            # reverse_edge_index is created by swapping the endpoints without
            # reordering edges, so the same edge_weight remains aligned.
            h_pe_rev = layer(
                reverse_edge_index,
                h_pe_rev,
                edge_weight=edge_weight,
            )
            result_list.append(h_pe_rev)

        return result_list


class Retriever(nn.Module):
    def __init__(
        self,
        emb_size,
        topic_pe,
        DDE_kwargs,
        relation_topic_source="gte",
        relation_score_mode="none",
        use_relation_topic_mask=False,
        dde_mode="baseline",
        weighted_dde_floor=0.1,
        **kwargs,
    ):
        super().__init__()
        if 'listwise' in kwargs:
            raise ValueError(
                "Retriever is pointwise-only. Build ListwiseReranker separately "
                "inside the two-stage entrypoints."
            )

        self.non_text_entity_emb = nn.Embedding(1, emb_size)
        self.topic_pe = topic_pe
        self.dde = DDE(**DDE_kwargs)
        if relation_score_mode not in {"semantic", "lexical", "hybrid", "none"}:
            raise ValueError(f"Unknown relation_score_mode: {relation_score_mode}")
        if dde_mode not in {"baseline", "weighted", "both"}:
            raise ValueError(f"Unknown dde_mode: {dde_mode}")
        if weighted_dde_floor < 0:
            raise ValueError("weighted_dde_floor must be non-negative")
        if (dde_mode in {"weighted", "both"} or use_relation_topic_mask) and (
            relation_score_mode == "none" and relation_topic_source != "groundtruth"
        ):
            raise ValueError(
                "A relation score is required by weighted/both DDE or a relation-topic mask"
            )

        self.relation_score_mode = relation_score_mode
        self.relation_topic_source = relation_topic_source
        self.use_relation_topic_mask = use_relation_topic_mask
        self.dde_mode = dde_mode
        self.weighted_dde_floor = weighted_dde_floor

        pred_in_size = 4 * emb_size
        if topic_pe:
            # Two topic-PE channels for both the head and tail entity.
            pred_in_size += 2 * 2

        dde_rounds = DDE_kwargs["num_rounds"] + DDE_kwargs["num_reverse_rounds"]

        if dde_mode in {"baseline", "both"}:
            pred_in_size += 2 * 2 * dde_rounds

        if dde_mode in {"weighted", "both"}:
            pred_in_size += 2 * 2 * dde_rounds

        if use_relation_topic_mask:
            pred_in_size += 2

        self.pred_in_size = pred_in_size
        self.pred = nn.Sequential(
            nn.Linear(pred_in_size, emb_size),
            nn.ReLU(),
            nn.Linear(emb_size, 1),
        )

    def _validate_relation_topic_features(self, relation_topic_features):
        relation_features_enabled = self.dde_mode in {"weighted", "both"} or self.use_relation_topic_mask
        features_missing = (
            relation_topic_features is None
            or relation_topic_features.numel() == 0
        )
        if relation_features_enabled and features_missing:
            raise ValueError(
                "relation_topic_features must be non-empty when "
                "dde_mode is weighted/both or use_relation_topic_mask is enabled"
            )

    def forward(
        self,
        h_id_tensor,
        r_id_tensor,
        t_id_tensor,
        q_emb,
        entity_embs,
        num_non_text_entities,
        relation_embs,
        topic_entity_one_hot,
        relation_topic_features=None,
        return_features=False,
    ):
        device = entity_embs.device
        self._validate_relation_topic_features(relation_topic_features)

        non_text_embs = self.non_text_entity_emb(
            torch.zeros(1, dtype=torch.long, device=device)
        ).expand(num_non_text_entities, -1)
        h_e = torch.cat([entity_embs, non_text_embs], dim=0)

        h_e_list = [h_e]
        if self.topic_pe:
            h_e_list.append(topic_entity_one_hot)

        edge_index = torch.stack([h_id_tensor, t_id_tensor], dim=0)
        reverse_edge_index = torch.stack([t_id_tensor, h_id_tensor], dim=0)

        # 1. Unweighted DDE. With all weights equal to one, the corrected
        # weighted mean reduces exactly to the ordinary neighbor mean.
        if self.dde_mode in {"baseline", "both"}:
            dde_list = self.dde(
                topic_entity_one_hot,
                edge_index,
                reverse_edge_index,
            )
            h_e_list.extend(dde_list)

        # 2. Relation-weighted DDE.
        if self.dde_mode in {"weighted", "both"}:
            edge_rt_features = relation_topic_features[r_id_tensor]
            # relation_topic_features is deliberately canonical: [score, mask].
            # groundtruth supplies scores 1.0 (on shortest path) and floor
            # (otherwise), while learned sources supply their selected score.
            edge_weight = torch.clamp(
                edge_rt_features[:, 0], min=self.weighted_dde_floor
            )
            dde_sem_list = self.dde(
                topic_entity_one_hot,
                edge_index,
                reverse_edge_index,
                edge_weight=edge_weight,
            )
            h_e_list.extend(dde_sem_list)

        h_e = torch.cat(h_e_list, dim=1)
        h_r = relation_embs[r_id_tensor]

        concat_list = [
            q_emb.expand(len(h_r), -1),
            h_e[h_id_tensor],
            h_r,
            h_e[t_id_tensor],
        ]

        # 3. Onehot relation-topic mask
        if self.use_relation_topic_mask:
            edge_rt_features = relation_topic_features[r_id_tensor]
            topk_mask = edge_rt_features[:, 1].long()
            edge_rt_one_hot = F.one_hot(topk_mask, num_classes=2).to(h_e.dtype)
            concat_list.append(edge_rt_one_hot)

        h_triple = torch.cat(concat_list, dim=1)
        pw_logits = self.pred(h_triple)

        if return_features:
            return pw_logits, h_triple
        return pw_logits
