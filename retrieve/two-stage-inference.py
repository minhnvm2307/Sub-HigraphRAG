"""Inference for a checkpoint produced by two-stage-train.py."""

import argparse
import os

import torch
from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.config.retriever import strip_listwise
from src.model.listwise_reranker import ListwiseReranker, apply_listwise_residual
from src.model.retriever import Retriever
from src.setup import prepare_sample, set_seed


@torch.no_grad()
def main(args):
    checkpoint = torch.load(args.path, map_location='cpu', weights_only=False)
    required = {'two_stage', 'config', 'pointwise_config', 'pointwise_model_state_dict',
                'listwise_model_state_dict'}
    if not checkpoint.get('two_stage') or not required.issubset(checkpoint):
        raise ValueError('Expected a checkpoint created by two-stage-train.py.')

    stage_config = checkpoint['config']
    base_config = strip_listwise(checkpoint['pointwise_config'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(stage_config['env']['seed'], deterministic=stage_config['env']['deterministic'])
    torch.set_num_threads(stage_config['env']['num_threads'])

    infer_set = RetrieverDataset(config=base_config, split='test', skip_no_path=False)
    emb_size = infer_set[0]['q_emb'].shape[-1]
    pointwise_model = Retriever(emb_size, **base_config['retriever']).to(device)
    pointwise_model.load_state_dict(checkpoint['pointwise_model_state_dict'], strict=True)
    pointwise_model.eval()

    listwise_cfg = stage_config['retriever']['listwise']
    listwise_model = ListwiseReranker(
        in_dim=pointwise_model.pred_in_size,
        d_model=listwise_cfg['d_model'], num_heads=listwise_cfg['num_heads'],
        num_layers=listwise_cfg['num_layers'], dim_feedforward=listwise_cfg['dim_feedforward'],
        dropout=listwise_cfg['dropout'],
    ).to(device)
    listwise_model.load_state_dict(checkpoint['listwise_model_state_dict'], strict=True)
    listwise_model.eval()

    pred_dict = {}
    top_n = listwise_cfg['top_n']
    margin = listwise_cfg['rank_margin']
    for raw_sample in tqdm(infer_set, desc='Two-stage inference'):
        sample = collate_retriever([raw_sample])
        (h_id, r_id, t_id, q_emb, entity_embs, num_non_text_entities,
         relation_embs, topic_pe, _, _, relation_features) = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']
        scored_triples = []
        target_triples = []
        if len(h_id):
            pw_logits, triple_features = pointwise_model(
                h_id, r_id, t_id, q_emb, entity_embs, num_non_text_entities,
                relation_embs, topic_pe, relation_features, return_features=True)
            pw_logits = pw_logits.reshape(-1)
            top_indices = torch.topk(pw_logits, k=min(top_n, len(pw_logits))).indices
            delta = listwise_model(triple_features[top_indices]).reshape(-1)
            final_logits = apply_listwise_residual(pw_logits, top_indices, delta, margin)
            results = torch.topk(final_logits, k=min(args.max_K, len(final_logits)))

            for triple_id, logit in zip(results.indices.cpu().tolist(), results.values.cpu().tolist()):
                scored_triples.append((
                    entity_list[h_id[triple_id].item()],
                    relation_list[r_id[triple_id].item()],
                    entity_list[t_id[triple_id].item()],
                    float(torch.sigmoid(torch.tensor(logit))),
                ))

            for triple_id in raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist():
                target_triples.append((
                    entity_list[h_id[triple_id].item()],
                    relation_list[r_id[triple_id].item()],
                    entity_list[t_id[triple_id].item()],
                ))

        pred_dict[raw_sample['id']] = {
            'question': raw_sample['question'],
            'scored_triples': scored_triples,
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [entity_list[i] for i in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[i] for i in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_triples,
        }

    output = os.path.join(os.path.dirname(args.path), 'retrieval_result.pth')
    torch.save(pred_dict, output)
    print(f'Saved two-stage retrieval output to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-stage listwise inference')
    parser.add_argument('-p', '--path', required=True, help='Stage-2 cpt.pth')
    parser.add_argument('--max_K', type=int, default=500)
    main(parser.parse_args())
