import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class RetrieverLoss(nn.Module):
    """
    Comprehensive Loss Module for Subgraph Retriever to handle:
    1. Imbalanced label learning (Weighted BCE, Sqrt, Log, Effective Number of Samples, Focal Loss).
    2. Pairwise Learning-to-Rank (Margin Ranking Loss with Random or Hard Negative Mining).
    3. Listwise / Contrastive Learning (InfoNCE Loss).
    4. Hybrid Combination (BCE / Focal + Margin Ranking / InfoNCE).
    """
    def __init__(self, loss_config=None, **kwargs):
        super().__init__()
        loss_config = loss_config or {}
        
        self.loss_type = loss_config.get('type', 'bce')  # 'bce', 'focal', 'margin', 'infonce', 'hybrid'
        self.pos_weight_mode = loss_config.get('pos_weight_mode', 'static') # 'static', 'inverse_freq', 'sqrt', 'log', 'effective_num', 'none'
        if self.pos_weight_mode is None:
            self.pos_weight_mode = 'none'
        self.static_pos_weight = loss_config.get('pos_weight', 200.0)
        self.beta = loss_config.get('beta', 0.999) # For effective_num
        
        # Focal loss hyperparams
        self.focal_gamma = loss_config.get('focal_gamma', 2.0)
        
        # Margin ranking loss hyperparams
        self.margin = loss_config.get('margin', 0.2)
        self.negative_mining = loss_config.get('negative_mining', 'all') # 'all', 'hard', 'random'
        self.num_negatives = loss_config.get('num_negatives', 10)
        
        # InfoNCE hyperparams
        self.temperature = loss_config.get('temperature', 0.1)
        
        # Hybrid weights
        self.ranking_weight = loss_config.get('ranking_weight', 1.0)
        self.cls_weight = loss_config.get('cls_weight', 1.0)

        valid_types = {'bce', 'focal', 'margin', 'infonce', 'hybrid'}
        valid_weight_modes = {'static', 'inversed', 'sqrt', 'log', 'effective_num', 'none'}
        valid_mining = {'all', 'hard', 'random'}
        if self.loss_type not in valid_types:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        if self.pos_weight_mode not in valid_weight_modes:
            raise ValueError(f"Unknown pos_weight_mode: {self.pos_weight_mode}")
        if self.negative_mining not in valid_mining:
            raise ValueError(f"Unknown negative_mining: {self.negative_mining}")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= self.beta < 1:
            raise ValueError("beta must be in [0, 1)")

    def _compute_pos_weight(self, targets, device):
        if self.pos_weight_mode == 'none' or (self.pos_weight_mode == 'static' and self.static_pos_weight is None):
            return None

        pos_mask = (targets == 1.0)
        neg_mask = (targets == 0.0)
        n_pos = pos_mask.sum().item()
        n_neg = neg_mask.sum().item()
        
        if n_pos == 0:
            return torch.tensor(1.0, device=device)
            
        ratio = float(n_neg) / float(n_pos)
        
        if self.pos_weight_mode == 'static':
            if self.static_pos_weight is None:
                return None
            return torch.tensor(float(self.static_pos_weight), device=device)
        elif self.pos_weight_mode == 'inversed':
            return torch.tensor(ratio, device=device)
        elif self.pos_weight_mode == 'sqrt':
            return torch.tensor(math.sqrt(ratio), device=device)
        elif self.pos_weight_mode == 'log':
            return torch.tensor(math.log1p(ratio), device=device)
        elif self.pos_weight_mode == 'effective_num':
            # E_n = (1 - beta^n) / (1 - beta) -> w_pos / w_neg
            eff_pos = (1.0 - self.beta ** n_pos) / (1.0 - self.beta)
            eff_neg = (1.0 - self.beta ** n_neg) / (1.0 - self.beta)
            w_p = eff_neg / (eff_pos + 1e-8)
            return torch.tensor(float(w_p), device=device)
        raise RuntimeError(f"Unhandled pos_weight_mode: {self.pos_weight_mode}")

    def _bce_loss(self, logits, targets, pos_weight):
        if pos_weight is None:
            return F.binary_cross_entropy_with_logits(logits, targets)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

    def _focal_loss(self, logits, targets, pos_weight):
        p = torch.sigmoid(logits)
        bce_raw = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # p_t: probability of true class
        p_t = targets * p + (1.0 - targets) * (1.0 - p)
        focal_factor = (1.0 - p_t) ** self.focal_gamma
        
        # Class weighting
        if pos_weight is not None:
            weight = targets * pos_weight + (1.0 - targets)
            loss = weight * focal_factor * bce_raw
        else:
            loss = focal_factor * bce_raw
            
        return loss.mean()

    def _margin_ranking_loss(self, logits, targets):
        pos_mask = (targets.squeeze(-1) == 1.0)
        neg_mask = (targets.squeeze(-1) == 0.0)
        
        pos_logits = logits.squeeze(-1)[pos_mask]
        neg_logits = logits.squeeze(-1)[neg_mask]
        
        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        if self.negative_mining == 'hard':
            # Top-K highest scoring negatives
            k = min(self.num_negatives * len(pos_logits), len(neg_logits))
            neg_logits, _ = torch.topk(neg_logits, k=k)
        elif self.negative_mining == 'random':
            k = min(self.num_negatives * len(pos_logits), len(neg_logits))
            perm = torch.randperm(len(neg_logits), device=logits.device)[:k]
            neg_logits = neg_logits[perm]
            
        # Pairwise differences: s_pos - s_neg
        diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0) # (num_pos, num_neg)
        loss = F.relu(self.margin - diff)
        return loss.mean()

    def _infonce_loss(self, logits, targets):
        pos_mask = (targets.squeeze(-1) == 1.0)
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        scaled_logits = logits.squeeze(-1) / self.temperature
        log_sum_exp = torch.logsumexp(scaled_logits, dim=0)
        
        pos_scaled_logits = scaled_logits[pos_mask]
        loss = log_sum_exp - pos_scaled_logits
        return loss.mean()

    def forward(self, logits, targets):
        """
        logits: (N, 1)
        targets: (N, 1)
        """
        device = logits.device
        pos_weight = self._compute_pos_weight(targets, device)
        
        if self.loss_type == 'bce':
            return self._bce_loss(logits, targets, pos_weight)
        elif self.loss_type == 'focal':
            return self._focal_loss(logits, targets, pos_weight)
        elif self.loss_type == 'margin':
            return self._margin_ranking_loss(logits, targets)
        elif self.loss_type == 'infonce':
            return self._infonce_loss(logits, targets)
        elif self.loss_type == 'hybrid':
            cls_l = self._focal_loss(logits, targets, pos_weight) if self.focal_gamma > 0 else self._bce_loss(logits, targets, pos_weight)
            rank_l = self._margin_ranking_loss(logits, targets)
            return self.cls_weight * cls_l + self.ranking_weight * rank_l
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
