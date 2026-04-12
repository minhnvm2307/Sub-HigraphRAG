import os
import torch

from tqdm import tqdm

from src.dataset.retriever import RetrieverDataset, collate_retriever
from src.model.retriever import Retriever
from src.setup import set_seed, prepare_sample
from src.config.retriever import load_yaml

@torch.no_grad()
def main(args):
    device = torch.device(f'cuda:0')
    
    cpt = torch.load(args.path, map_location='cpu')
    config = cpt['config']
    # config_file = f'configs/retriever/cwq.yaml'
    # config = load_yaml(config_file)
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])
    
    infer_set = RetrieverDataset(
        config=config, split='test', skip_no_path=False)
    
    emb_size = infer_set[0]['q_emb'].shape[-1]
    model = Retriever(emb_size, **config['retriever']).to(device)
    model.load_state_dict(cpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    pred_dict = dict()
    for i in tqdm(range(len(infer_set))):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']
        top_K_triples = []
        target_relevant_triples = []

        if len(h_id_tensor) != 0:
            pred_triple_logits = model(
                h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,
                num_non_text_entities, relation_embs, topic_entity_one_hot)
            pred_triple_scores = torch.sigmoid(pred_triple_logits).reshape(-1)
            top_K_results = torch.topk(pred_triple_scores, 
                                       min(args.max_K, len(pred_triple_scores)))
            top_K_scores = top_K_results.values.cpu().tolist()
            top_K_triple_IDs = top_K_results.indices.cpu().tolist()

            for j, triple_id in enumerate(top_K_triple_IDs):
                top_K_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                    top_K_scores[j]
                ))

            target_relevant_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
            for triple_id in target_relevant_triple_ids:
                target_relevant_triples.append((
                    entity_list[h_id_tensor[triple_id].item()],
                    relation_list[r_id_tensor[triple_id].item()],
                    entity_list[t_id_tensor[triple_id].item()],
                ))

        sample_dict = {
            'question': raw_sample['question'],
            'scored_triples': top_K_triples,
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples
        }
        
        pred_dict[raw_sample['id']] = sample_dict

    root_path = os.path.dirname(args.path)
    torch.save(pred_dict, os.path.join(root_path, 'retrieval_result.pth'))

# Inference using embedding-based scoring (without the retriever model)
@torch.no_grad()
def main_test(args):
    device = torch.device('cuda:0')

    cpt = torch.load(args.path, map_location='cpu')
    # config = cpt['config']
    config_file = f'configs/retriever/cwq.yaml'
    config = load_yaml(config_file)
    set_seed(config['env']['seed'])
    torch.set_num_threads(config['env']['num_threads'])

    infer_set = RetrieverDataset(
        config=config, split='test', skip_no_path=False)

    pred_dict = dict()

    for i in tqdm(range(len(infer_set))):
        raw_sample = infer_set[i]
        sample = collate_retriever([raw_sample])

        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, a_entity_id_list = prepare_sample(device, sample)

        entity_list = raw_sample['text_entity_list'] + raw_sample['non_text_entity_list']
        relation_list = raw_sample['relation_list']

        top_K_triples = []
        target_relevant_triples = []

        # Verify dimensions match for text entities only.
        # entity_embs stores embeddings for text_entity_list; non-text entities need separate handling.
        assert entity_embs.shape[0] == len(raw_sample['text_entity_list']), \
            f"Text entity embedding count mismatch: {entity_embs.shape[0]} vs {len(raw_sample['text_entity_list'])}"
        assert relation_embs.shape[0] == len(relation_list), \
            f"Relation embedding count mismatch: {relation_embs.shape[0]} vs {len(relation_list)}"
        
        if len(h_id_tensor) != 0:
            # ===== EMBEDDING-BASED SCORING =====
            emb_size = entity_embs.shape[1]
            if num_non_text_entities > 0:
                non_text_weight = cpt['model_state_dict'].get('non_text_entity_emb.weight', None)
                if non_text_weight is not None:
                    non_text_emb = non_text_weight.to(device=device, dtype=entity_embs.dtype).expand(num_non_text_entities, -1)
                else:
                    non_text_emb = torch.zeros(num_non_text_entities, emb_size, device=device, dtype=entity_embs.dtype)
                full_entity_embs = torch.cat([entity_embs, non_text_emb], dim=0)
            else:
                full_entity_embs = entity_embs

            # Make dtype consistent for cosine similarity computation.
            score_dtype = torch.float32
            q_emb = q_emb.to(dtype=score_dtype)
            relation_embs = relation_embs.to(dtype=score_dtype)
            full_entity_embs = full_entity_embs.to(dtype=score_dtype)

            # Get safe indices (clamp to valid range to prevent out-of-bounds access)
            max_ent = full_entity_embs.shape[0]
            max_rel = relation_embs.shape[0]

            # Clamp indices to valid ranges
            h_safe = h_id_tensor.clamp(0, max_ent - 1)  # head entity indices
            t_safe = t_id_tensor.clamp(0, max_ent - 1)  # tail entity indices
            r_safe = r_id_tensor.clamp(0, max_rel - 1)  # relation indices

            # Index embeddings: each row corresponds to one triple
            h_emb = full_entity_embs[h_safe]      # shape (N_triples, emb_dim)
            t_emb = full_entity_embs[t_safe]      # shape (N_triples, emb_dim)
            r_emb = relation_embs[r_safe]    # shape (N_triples, emb_dim)

            # TransE-style composition: h + r + t
            triple_emb = h_emb + r_emb + t_emb  # shape (N_triples, emb_dim)

            # Cosine similarity with query embedding
            q_norm = torch.nn.functional.normalize(q_emb, dim=-1)           # (1, emb_dim)
            triple_norm = torch.nn.functional.normalize(triple_emb, dim=-1) # (N_triples, emb_dim)

            # Compute scores: cosine_sim(triple_emb, query)
            scores = torch.matmul(triple_norm, q_norm.T).reshape(-1)  # shape (N_triples,)

            # ===== Top-K Retrieval =====
            top_K_results = torch.topk(scores, min(args.max_K, len(scores)))
            top_K_scores = top_K_results.values.cpu().tolist()
            top_K_triple_IDs = top_K_results.indices.cpu().tolist()

            for j, triple_id in enumerate(top_K_triple_IDs):
                # Use ORIGINAL indices (not clamped) to retrieve entity/relation names
                h_id = h_id_tensor[triple_id].item()
                r_id = r_id_tensor[triple_id].item()
                t_id = t_id_tensor[triple_id].item()
                
                top_K_triples.append((
                    entity_list[h_id],
                    relation_list[r_id],
                    entity_list[t_id],
                    top_K_scores[j]
                ))

            # ===== Ground Truth Triples =====
            target_relevant_triple_ids = raw_sample['target_triple_probs'].nonzero().reshape(-1).tolist()
            for triple_id in target_relevant_triple_ids:
                h_id = h_id_tensor[triple_id].item()
                r_id = r_id_tensor[triple_id].item()
                t_id = t_id_tensor[triple_id].item()
                
                target_relevant_triples.append((
                    entity_list[h_id],
                    relation_list[r_id],
                    entity_list[t_id],
                ))

        sample_dict = {
            'question': raw_sample['question'],
            'scored_triples': top_K_triples,
            'q_entity': raw_sample['q_entity'],
            'q_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['q_entity_id_list']],
            'a_entity': raw_sample['a_entity'],
            'a_entity_in_graph': [entity_list[e_id] for e_id in raw_sample['a_entity_id_list']],
            'max_path_length': raw_sample['max_path_length'],
            'target_relevant_triples': target_relevant_triples
        }

        pred_dict[raw_sample['id']] = sample_dict

    # ===== Save Results =====
    root_path = os.path.dirname(args.path)
    save_path = os.path.join(root_path, 'retrieval_result_embedding.pth')
    torch.save(pred_dict, save_path)

    print(f"✓ Saved embedding-based retrieval to: {save_path}")

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='Path to a saved model checkpoint, e.g., webqsp_Nov08-01:14:47/cpt.pth')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional retriever config YAML. If omitted, uses checkpoint config.')
    parser.add_argument('--max_K', type=int, default=500,
                        help='K in top-K triple retrieval')
    args = parser.parse_args()
    
    main_test(args)
