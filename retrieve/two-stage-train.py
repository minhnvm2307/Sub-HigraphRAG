"""Train a residual listwise reranker on a frozen pointwise checkpoint.

The pointwise model chooses a fixed top-N candidate pool.  Only the listwise
head is optimized, so no gradient can change candidate selection.
"""

import argparse
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config.retriever import load_two_stage_yaml, strip_listwise
from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.loss import RetrieverLoss
from src.model.listwise_reranker import ListwiseReranker, apply_listwise_residual
from src.model.retriever import Retriever
from src.setup import prepare_sample, seed_worker, set_seed


def validate_stage_config(stage_config, checkpoint_config):
    if stage_config['dataset'] != checkpoint_config['dataset']:
        raise ValueError('Stage-2 dataset settings must match the pointwise checkpoint.')

    stage_retriever = dict(stage_config['retriever'])
    checkpoint_retriever = dict(checkpoint_config['retriever'])
    stage_retriever.pop('listwise', None)
    checkpoint_retriever.pop('listwise', None)
    if stage_retriever != checkpoint_retriever:
        raise ValueError(
            'Stage-2 pointwise retriever settings differ from the checkpoint. '
            'Use a stage config with the same pointwise architecture/features.'
        )


def build_loss(config):
    loss_config = config.get('loss')
    if loss_config is not None:
        return RetrieverLoss(loss_config)
    pos_weight = config['retriever'].get('pos_weight')
    return RetrieverLoss({
        'type': 'bce',
        'pos_weight_mode': 'static' if pos_weight is not None else 'none',
        'pos_weight': pos_weight,
    })


@torch.no_grad()
def pointwise_candidates(pointwise_model, device, sample, top_n):
    prepared = prepare_sample(device, sample)
    (h_id, r_id, t_id, q_emb, entity_embs, num_non_text_entities,
     relation_embs, topic_pe, targets, answer_ids, relation_features) = prepared
    if len(h_id) == 0:
        return None

    pw_logits, triple_features = pointwise_model(
        h_id, r_id, t_id, q_emb, entity_embs, num_non_text_entities,
        relation_embs, topic_pe, relation_features, return_features=True,
    )
    pw_logits = pw_logits.reshape(-1)
    indices = torch.topk(pw_logits, k=min(top_n, len(pw_logits))).indices
    return {
        'prepared': prepared,
        'pw_logits': pw_logits,
        'top_indices': indices,
        'top_features': triple_features[indices],
    }


def metrics_for_order(order, prepared, k_list):
    h_id, _, t_id, _, entity_embs, num_non_text_entities, _, _, targets, answer_ids, _ = prepared
    order = order.cpu()
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(len(order))
    target_ids = targets.nonzero().reshape(-1)
    if len(target_ids) == 0:
        return None

    values = {}
    num_entities = len(entity_embs) + num_non_text_entities
    h_id, t_id = h_id.cpu(), t_id.cpu()
    for k in k_list:
        selected = ranks < k
        values[f'triple_recall@{k}'] = (ranks[target_ids] < k).float().mean().item()
        entity_mask = torch.zeros(num_entities)
        entity_mask[h_id[selected]] = 1.0
        entity_mask[t_id[selected]] = 1.0
        values[f'ans_recall@{k}'] = entity_mask[answer_ids].mean().item()
    return values


def candidate_coverage(top_indices, prepared):
    h_id, _, t_id, _, entity_embs, num_non_text_entities, _, _, targets, answer_ids, _ = prepared
    top_indices = top_indices.cpu()
    target_ids = targets.nonzero().reshape(-1)
    if len(target_ids) == 0:
        return None

    selected = torch.zeros(len(targets), dtype=torch.bool)
    selected[top_indices] = True
    entity_mask = torch.zeros(len(entity_embs) + num_non_text_entities)
    entity_mask[h_id.cpu()[selected]] = 1.0
    entity_mask[t_id.cpu()[selected]] = 1.0
    return {
        'candidate_triple_recall': selected[target_ids].float().mean().item(),
        'candidate_ans_recall': entity_mask[answer_ids].mean().item(),
    }


def promotion_metrics(pointwise_order, listwise_order, prepared, k_list):
    """Measure gold triples that cross each top-k boundary after reranking."""
    targets = prepared[8]
    target_ids = targets.nonzero().reshape(-1)
    if len(target_ids) == 0:
        return None

    pointwise_ranks = torch.empty_like(pointwise_order)
    pointwise_ranks[pointwise_order] = torch.arange(len(pointwise_order))
    listwise_ranks = torch.empty_like(listwise_order)
    listwise_ranks[listwise_order] = torch.arange(len(listwise_order))
    values = {}
    for k in k_list:
        pointwise_in_k = pointwise_ranks[target_ids] < k
        listwise_in_k = listwise_ranks[target_ids] < k
        values[f'promoted_gold@{k}'] = ((~pointwise_in_k) & listwise_in_k).float().mean().item()
        values[f'demoted_gold@{k}'] = (pointwise_in_k & (~listwise_in_k)).float().mean().item()
    return values


@torch.no_grad()
def eval_epoch(config, device, data_loader, pointwise_model, listwise_model):
    pointwise_model.eval()
    listwise_model.eval()
    metrics = defaultdict(list)
    listwise_cfg = config['retriever']['listwise']
    top_n = listwise_cfg['top_n']
    margin = listwise_cfg['rank_margin']

    for sample in tqdm(data_loader, desc='Evaluating stage 2'):
        candidate = pointwise_candidates(pointwise_model, device, sample, top_n)
        if candidate is None:
            continue
        prepared = candidate['prepared']
        coverage = candidate_coverage(candidate['top_indices'], prepared)
        if coverage is None:
            continue
        for key, value in coverage.items():
            metrics[key].append(value)

        delta = listwise_model(candidate['top_features']).reshape(-1)
        metrics['listwise_delta_std'].append(delta.std().item())
        final_logits = apply_listwise_residual(
            candidate['pw_logits'], candidate['top_indices'], delta, margin)
        pointwise_order = torch.argsort(candidate['pw_logits'], descending=True).cpu()
        listwise_order = torch.argsort(final_logits, descending=True).cpu()
        values = metrics_for_order(pointwise_order, prepared, config['eval']['k_list'])
        for key, value in values.items():
            metrics[f'pointwise_{key}'].append(value)
        values = metrics_for_order(listwise_order, prepared, config['eval']['k_list'])
        for key, value in values.items():
            metrics[f'listwise_{key}'].append(value)
        values = promotion_metrics(pointwise_order, listwise_order, prepared, config['eval']['k_list'])
        for key, value in values.items():
            metrics[f'listwise_{key}'].append(value)

    return {key: float(np.mean(values)) for key, values in metrics.items() if values}


def train_epoch(config, device, data_loader, pointwise_model, listwise_model, optimizer, loss_fn):
    pointwise_model.eval()
    listwise_model.train()
    top_n = config['retriever']['listwise']['top_n']
    loss_sum = 0.0
    delta_std_sum = 0.0
    updated_samples = 0
    skipped_no_positive = 0

    for sample in tqdm(data_loader, desc='Training stage 2'):
        candidate = pointwise_candidates(pointwise_model, device, sample, top_n)
        if candidate is None:
            continue
        targets = candidate['prepared'][8].to(device)[candidate['top_indices']].unsqueeze(-1)
        # A pool with no positive triple provides no within-pool reranking
        # signal. It is counted, but cannot be repaired by stage 2.
        if not bool(targets.any()):
            skipped_no_positive += 1
            continue

        delta = listwise_model(candidate['top_features'])
        delta_std_sum += delta.std().item()
        logits = candidate['pw_logits'][candidate['top_indices']].unsqueeze(-1) + delta
        loss = loss_fn(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        updated_samples += 1

    return {
        'loss': loss_sum / max(updated_samples, 1),
        'stage2/delta_std': delta_std_sum / max(updated_samples, 1),
        'stage2/updated_samples': updated_samples,
        'stage2/skipped_no_positive_pool': skipped_no_positive,
    }


def main(args):
    stage_config = load_two_stage_yaml(args.config_file)
    if not stage_config['retriever']['listwise']['enabled']:
        raise ValueError('two-stage-train.py requires retriever.listwise.enabled: true.')

    checkpoint = torch.load(args.pointwise_cpt, map_location='cpu', weights_only=False)
    if 'config' not in checkpoint or 'model_state_dict' not in checkpoint:
        raise ValueError('Expected a standard pointwise checkpoint with config and model_state_dict.')
    checkpoint_config = checkpoint['config']
    validate_stage_config(stage_config, checkpoint_config)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed = stage_config['env']['seed']
    set_seed(seed, deterministic=stage_config['env']['deterministic'])
    torch.set_num_threads(stage_config['env']['num_threads'])

    dataset_config = strip_listwise(checkpoint_config)
    train_set = RetrieverDataset(config=dataset_config, split='train')
    val_set = RetrieverDataset(config=dataset_config, split='val')
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=collate_retriever,
                              generator=generator, worker_init_fn=seed_worker)
    val_loader = DataLoader(val_set, batch_size=1, collate_fn=collate_retriever)

    emb_size = train_set[0]['q_emb'].shape[-1]
    pointwise_model = Retriever(emb_size, **dataset_config['retriever']).to(device)
    pointwise_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    pointwise_model.requires_grad_(False)
    pointwise_model.eval()

    listwise_cfg = stage_config['retriever']['listwise']
    listwise_model = ListwiseReranker(
        in_dim=pointwise_model.pred_in_size,
        d_model=listwise_cfg['d_model'], num_heads=listwise_cfg['num_heads'],
        num_layers=listwise_cfg['num_layers'], dim_feedforward=listwise_cfg['dim_feedforward'],
        dropout=listwise_cfg['dropout'],
    ).to(device)
    optimizer = Adam(listwise_model.parameters(), **stage_config['optimizer'])
    loss_fn = build_loss(stage_config).to(device)

    timestamp = time.strftime('%b%d-%H:%M:%S', time.gmtime())
    exp_name = f"{stage_config['train']['save_prefix']}_{timestamp}"
    os.makedirs(exp_name, exist_ok=True)
    wandb.init(project=stage_config['dataset']['name'], name=exp_name,
               config=pd.json_normalize(stage_config, sep='/').to_dict(orient='records')[0])

    best_metric = float('-inf')
    patience = 0
    for epoch in range(stage_config['train']['num_epochs']):
        train_log = train_epoch(stage_config, device, train_loader, pointwise_model,
                                listwise_model, optimizer, loss_fn)
        val_metrics = eval_epoch(stage_config, device, val_loader, pointwise_model, listwise_model)
        metric_key = f"listwise_{listwise_cfg['selection_metric']}"
        if metric_key not in val_metrics:
            raise RuntimeError(
                f'Missing validation metric {metric_key}. Ensure listwise.selection_metric '
                'uses ans_recall@K or triple_recall@K for a K listed in eval.k_list.'
            )

        current_metric = val_metrics[metric_key]
        patience += 1
        if current_metric > best_metric:
            best_metric = current_metric
            patience = 0
            torch.save({
                'two_stage': True,
                'config': stage_config,
                'pointwise_config': checkpoint_config,
                'pointwise_model_state_dict': pointwise_model.state_dict(),
                'listwise_model_state_dict': listwise_model.state_dict(),
                'pointwise_checkpoint': os.path.abspath(args.pointwise_cpt),
                'epoch': epoch,
                'best_val_metric': best_metric,
                'seed': seed,
            }, os.path.join(exp_name, 'cpt.pth'))

        wandb.log({'epoch': epoch, 'val/selection_metric': current_metric,
                   'val/best_selection_metric': best_metric, **train_log,
                   **{f'val/{k}': v for k, v in val_metrics.items()}})
        if patience >= stage_config['train']['patience']:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage-2 listwise reranker training')
    parser.add_argument('-d', '--dataset', required=True, choices=['webqsp', 'cwq'])
    parser.add_argument('--config_file', required=True)
    parser.add_argument('--pointwise_cpt', required=True, help='Converged pointwise cpt.pth')
    main(parser.parse_args())
