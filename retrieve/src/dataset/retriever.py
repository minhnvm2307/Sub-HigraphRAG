from collections import OrderedDict
import json
import networkx as nx
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
import string
import re

from torch.utils.data import Sampler
from tqdm import tqdm


class ShardedEmbDict:
    """Lazy-loading dict-like wrapper for sharded embeddings with LRU cache & mmap."""

    def __init__(self, shard_dir: str, index: dict, max_cache_shards: int = 12):
        self._shard_dir = shard_dir
        self._index = index
        self._max_cache = max_cache_shards
        self._cache = OrderedDict()

    def _load_shard(self, shard_file: str):
        if shard_file in self._cache:
            self._cache.move_to_end(shard_file)
            return self._cache[shard_file]

        shard_path = os.path.join(self._shard_dir, shard_file)
        data = torch.load(shard_path, map_location="cpu", mmap=True)
        if len(self._cache) >= self._max_cache:
            self._cache.popitem(last=False)
        self._cache[shard_file] = data
        return data

    def __contains__(self, key):
        return str(key) in self._index

    def __getitem__(self, key):
        str_key = str(key)
        shard_file = self._index[str_key]
        cache_data = self._load_shard(shard_file)
        if str_key in cache_data:
            return cache_data[str_key]
        try:
            return cache_data[int(str_key)]
        except (KeyError, ValueError):
            raise KeyError(key)


class ShardAwareSampler(Sampler):
    """Sampler that groups sample indices by shard file and randomizes both.

    Ensures sequential access within each shard (drastically reducing disk I/O from 9.6TB to 41GB/epoch)
    while maintaining overall stochastic training order across epochs.
    """

    def __init__(self, dataset, generator=None, shuffle=True):
        self.dataset = dataset
        self.generator = generator
        self.shuffle = shuffle

        emb_dict = getattr(dataset, 'emb_dict', None)
        if isinstance(emb_dict, ShardedEmbDict):
            shard_to_indices = {}
            for idx, sample in enumerate(dataset.processed_dict_list):
                s_id = str(sample['id'])
                shard_file = emb_dict._index.get(s_id, "default")
                if shard_file not in shard_to_indices:
                    shard_to_indices[shard_file] = []
                shard_to_indices[shard_file].append(idx)
            self.shard_groups = list(shard_to_indices.values())
        else:
            self.shard_groups = [[i for i in range(len(dataset))]]

    def __iter__(self):
        if not self.shuffle:
            for group in self.shard_groups:
                yield from group
            return

        num_shards = len(self.shard_groups)
        shard_order = torch.randperm(num_shards, generator=self.generator).tolist()

        for s_idx in shard_order:
            group = self.shard_groups[s_idx]
            if len(group) > 1:
                perm = torch.randperm(len(group), generator=self.generator).tolist()
                for p in perm:
                    yield group[p]
            else:
                yield group[0]

    def __len__(self):
        return len(self.dataset)

STOPWORDS = {
    'what', 'which', 'who', 'whom', 'where', 'when', 'why', 'how',
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
    'should', 'now', 'is', 'was', 'were', 'are', 'has', 'have', 'had', 'do', 'does', 'did'
}

SYNONYM_GROUPS = [
    {'speak', 'spoken', 'speech', 'language', 'languages'},
    {'born', 'birth', 'originated', 'origin'},
    {'died', 'death', 'deceased', 'burial', 'grave'},
    {'written', 'write', 'author', 'book', 'writer', 'subjects'},
    {'acted', 'actor', 'actress', 'starring', 'play', 'plays', 'cast', 'role', 'roles'},
    {'produced', 'producer'},
    {'directed', 'director'},
    {'created', 'creator', 'creative'},
    {'founded', 'founder', 'founding'},
    {'located', 'location', 'place', 'containedby', 'contains', 'adjoins', 'adjoin'},
    {'member', 'membership', 'participant', 'participated'},
    {'owned', 'owner', 'proprietor'},
    {'married', 'marriage', 'spouse', 'wife', 'husband'},
    {'parent', 'parents', 'children', 'child', 'son', 'daughter', 'father', 'mother'},
    {'nationality', 'citizen', 'country', 'nation'}
]

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [word for word in text.split() if word not in STOPWORDS]

def get_token_similarity(q_token, r_token):
    if q_token == r_token:
        return 1.0
    for group in SYNONYM_GROUPS:
        if q_token in group and r_token in group:
            return 0.9
    min_len = min(len(q_token), len(r_token))
    if min_len >= 4:
        if q_token[:min_len] == r_token[:min_len]:
            return 0.9
        if q_token[:4] == r_token[:4]:
            return 0.8
    return 0.0

def compute_lexical_score(question_tokens, relation_name):
    parts = relation_name.split('.')
    num_parts = len(parts)
    
    total_score = 0.0
    total_weight = 0.0
    
    for idx, part in enumerate(parts):
        if idx == num_parts - 1:
            weight = 1.0
        elif idx == num_parts - 2:
            weight = 0.5
        else:
            weight = 0.2
            
        part_tokens = part.split('_')
        for r_tok in part_tokens:
            if len(r_tok) < 2:
                continue
            r_tok_lower = r_tok.lower()
            if r_tok_lower in STOPWORDS:
                continue
            best_tok_score = 0.0
            for q_tok in question_tokens:
                score = get_token_similarity(q_tok, r_tok_lower)
                if score > best_tok_score:
                    best_tok_score = score
            total_score += best_tok_score * weight
            total_weight += weight
            
    if total_weight == 0.0:
        return 0.0
    return total_score / total_weight

class RetrieverDataset:
    def __init__(
        self,
        config,
        split,
        skip_no_path=True
    ):
        self.config = config
        # Load pre-processed data.
        dataset_name = config['dataset']['name']
        processed_dict_list = self._load_processed(dataset_name, split)

        # Extract directed shortest paths from topic entities to answer
        # entities or vice versa as weak supervision signals for triple scoring.
        triple_score_dict = self._get_triple_scores(
            dataset_name, split, processed_dict_list)

        # Load pre-computed embeddings.
        self.emb_dict = self._load_emb(
            dataset_name, config['dataset']['text_encoder_name'], split)

        self.bge_relation_topics = self._load_bge_relation_topics(dataset_name, split)
        self.relation_topic_cache = self._load_relation_topic_cache(
            dataset_name, config['dataset']['text_encoder_name'], split)

        # Put everything together.
        self._assembly(
            processed_dict_list, triple_score_dict, self.emb_dict, skip_no_path)

    def _load_bge_relation_topics(self, dataset_name, split):
        retriever_config = self.config['retriever']
        source = retriever_config.get('relation_topic_source', 'gte')
        if source in {'gte', 'path_memory_hybrid'}:
            return None
        if source != 'bge':
            raise ValueError(f"Unknown relation_topic_source: {source}")

        format_name = retriever_config.get('relation_topic_bge_format')
        topk = retriever_config.get('topk_rt')
        if not format_name:
            raise ValueError(
                "bge requires relation_topic_bge_format "
            )

        path = os.path.join('data_files', dataset_name, 'relation_topics', 'bge-m3',
                            f'{split}_{format_name}_top{topk}.pth')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing BGE relation-topic cache: {path}. Create it in bge-m3-analysis with "
                f"python bge_emb.py -d {dataset_name} --split {split} --format {format_name} --topk {topk}")
        cache = torch.load(path, map_location='cpu', weights_only=False)
        if cache.get('format') != format_name or cache.get('topk') != topk:
            raise ValueError(f"BGE relation-topic cache metadata does not match config: {path}")
        score_mode = retriever_config.get('relation_score_mode', 'none')
        required_scores = {
            'semantic': {'dense_scores'},
            'lexical': {'sparse_scores'},
            'hybrid': {'dense_scores', 'sparse_scores'},
            'none': set(),
        }
        if score_mode not in required_scores:
            raise ValueError(f"Unknown relation_score_mode: {score_mode}")
        missing = required_scores[score_mode] - set(cache)
        if missing:
            raise ValueError(f"BGE cache lacks required score(s) {sorted(missing)}: {path}")
        return cache

    def _load_relation_topic_cache(self, dataset_name, text_encoder_name, split):
        """Load an optional precomputed relation-topic feature cache."""
        retriever_config = self.config['retriever']
        source = retriever_config.get('relation_topic_source', 'gte')
        if source == 'path_memory_hybrid':
            cache_path = os.path.join(
                'data_files', dataset_name, 'relation_topics', 'path_memory_hybrid', f'{split}.pth'
            )
            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    f"Configured path_memory_hybrid cache does not exist: {cache_path}. "
                    "Run precompute_path_memory_hybrid.py first."
                )
            features = torch.load(cache_path, map_location='cpu', weights_only=False)
            print(f'[rt-cache] Loaded {len(features):,} samples from {cache_path}')
            return features

        cache_spec = retriever_config.get('relation_topic_cache')
        if not cache_spec or source != 'gte':
            return None

        score_mode = retriever_config.get('relation_score_mode', 'none')
        alpha = retriever_config.get('relation_score_alpha', 0.5)
        topk_rt = retriever_config.get('topk_rt')
        use_mask = retriever_config.get('use_relation_topic_mask', False)
        if cache_spec == 'auto':
            cache_path = os.path.join(
                'data_files', dataset_name, 'relation_topics', text_encoder_name,
                f'{split}_{score_mode}_alpha{alpha:g}_top{topk_rt}.pth')
        else:
            cache_path = cache_spec.format(
                dataset=dataset_name,
                encoder=text_encoder_name,
                split=split,
                score_mode=score_mode,
                alpha=alpha,
                topk=topk_rt,
            )

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Configured relation-topic cache does not exist: {cache_path}. "
                "Run precompute_relation_topics.py first."
            )

        payload = torch.load(cache_path, map_location='cpu', weights_only=False)
        metadata = payload.get('metadata', {})
        expected = {
            'dataset': dataset_name,
            'text_encoder_name': text_encoder_name,
            'split': split,
            'relation_topic_source': 'gte',
            'relation_score_mode': score_mode,
            'relation_score_alpha': float(alpha),
            'use_relation_topic_mask': bool(use_mask),
            'topk_rt': topk_rt,
            'feature_layout': ['score', 'mask'],
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(
                    f"Relation-topic cache metadata mismatch for {cache_path}: "
                    f"{key}={metadata.get(key)!r}, expected {value!r}"
                )
        features = payload.get('features')
        if not isinstance(features, dict):
            raise ValueError(f"Invalid relation-topic cache payload: {cache_path}")
        print(f'[rt-cache] Loaded {len(features):,} samples from {cache_path}')
        return features

    def _load_processed(
        self,
        dataset_name,
        split
    ):
        processed_file = os.path.join(
            f'data_files/{dataset_name}/processed/{split}.pkl')
        with open(processed_file, 'rb') as f:
            return pickle.load(f)

    def _get_triple_scores(
        self,
        dataset_name,
        split,
        processed_dict_list
    ):
        save_dir = os.path.join('data_files', dataset_name, 'triple_scores')
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f'{split}.pth')

        if os.path.exists(save_file):
            return torch.load(save_file)

        triple_score_dict = dict()
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            triple_scores_i, max_path_length_i = self._extract_paths_and_score(
                sample_i)

            triple_score_dict[sample_i_id] = {
                'triple_scores': triple_scores_i,
                'max_path_length': max_path_length_i
            }

        torch.save(triple_score_dict, save_file)
        
        return triple_score_dict

    def _extract_paths_and_score(
        self,
        sample
    ):
        nx_g = self._get_nx_g(
            sample['h_id_list'],
            sample['r_id_list'],
            sample['t_id_list']
        )

        # Each raw path is a list of entity IDs.
        path_list_ = []
        for q_entity_id in sample['q_entity_id_list']:
            for a_entity_id in sample['a_entity_id_list']:
                paths_q_a = self._shortest_path(nx_g, q_entity_id, a_entity_id)
                if len(paths_q_a) > 0:
                    path_list_.extend(paths_q_a)

        if len(path_list_) == 0:
            max_path_length = None
        else:
            max_path_length = 0

        # Each processed path is a list of triple IDs.
        path_list = []

        for path in path_list_:
            num_triples_path = len(path) - 1
            max_path_length = max(max_path_length, num_triples_path)
            triples_path = []

            for i in range(num_triples_path):
                h_id_i = path[i]
                t_id_i = path[i+1]
                triple_id_i_list = [
                    nx_g[h_id_i][t_id_i]['triple_id']
                ]              
                triples_path.append(triple_id_i_list)

            path_list.append(triples_path)

        num_triples = len(sample['h_id_list'])
        triple_scores = self._score_triples(
            path_list,
            num_triples
        )
        
        return triple_scores, max_path_length

    def _get_nx_g(
        self,
        h_id_list,
        r_id_list,
        t_id_list
    ):
        nx_g = nx.DiGraph()
        num_triples = len(h_id_list)
        for i in range(num_triples):
            h_i = h_id_list[i]
            r_i = r_id_list[i]
            t_i = t_id_list[i]
            nx_g.add_edge(h_i, t_i, triple_id=i, relation_id=r_i)

        return nx_g

    def _shortest_path(
        self,
        nx_g,
        q_entity_id,
        a_entity_id
    ):
        try:
            forward_paths = list(nx.all_shortest_paths(nx_g, q_entity_id, a_entity_id))
        except:
            forward_paths = []
        
        try:
            backward_paths = list(nx.all_shortest_paths(nx_g, a_entity_id, q_entity_id))
        except:
            backward_paths = []
        
        full_paths = forward_paths + backward_paths
        if (len(forward_paths) == 0) or (len(backward_paths) == 0):
            return full_paths
        
        min_path_len = min([len(path) for path in full_paths])
        refined_paths = []
        for path in full_paths:
            if len(path) == min_path_len:
                refined_paths.append(path)
        
        return refined_paths

    def _score_triples(
        self,
        path_list,
        num_triples
    ):
        triple_scores = torch.zeros(num_triples)
        
        for path in path_list:
            for triple_id_list in path:
                triple_scores[triple_id_list] = 1.

        return triple_scores

    def _load_emb(
        self,
        dataset_name,
        text_encoder_name,
        split
    ):
        emb_dir = f'data_files/{dataset_name}/emb/{text_encoder_name}'
        index_path = os.path.join(emb_dir, 'shard_index.json')

        if os.path.exists(index_path):
            print(f'[emb] Loading via shard index: {index_path}')
            with open(index_path, 'r') as f:
                index = json.load(f)
            return ShardedEmbDict(emb_dir, index)

        # Fallback: load merged file (requires enough RAM)
        file_path = os.path.join(emb_dir, f'{split}.pth')
        print(f'[emb] Loading merged file: {file_path}')
        return torch.load(file_path, map_location='cpu')

    @staticmethod
    def _select_relation_score(score_mode, semantic, lexical, alpha, num_relations):
        """Return the canonical relation score used by DDE and selection."""
        if score_mode == 'semantic':
            return semantic
        if score_mode == 'lexical':
            return lexical
        if score_mode == 'hybrid':
            return alpha * lexical + (1.0 - alpha) * semantic
        if score_mode == 'none':
            return torch.zeros(num_relations, dtype=torch.float32)
        raise ValueError(f"Unknown relation_score_mode: {score_mode}")

    @staticmethod
    def _calibrate_relation_score(
        score: torch.Tensor,
        calibration_mode: str = 'none',
        platt_a: float = 9.77,
        platt_b: float = -5.20,
        power_gamma: float = 1.0,
    ) -> torch.Tensor:
        """Calibrate continuous relation scores to expand contrast between Gold and Noise."""
        if calibration_mode == 'platt':
            return torch.sigmoid(platt_a * score + platt_b)
        elif calibration_mode == 'power':
            return torch.clamp(score, min=0.0) ** power_gamma
        elif calibration_mode == 'none':
            return score
        else:
            raise ValueError(f"Unknown relation_score_calibration: {calibration_mode}")

    @staticmethod
    def _select_relation_topic_mask(
        score: torch.Tensor,
        topk_rt: int | None = None,
        mask_mode: str = 'topk',
        cluster_ratio: float = 0.90,
        min_k: int = 2,
        max_k: int = 15,
    ) -> torch.Tensor:
        """Adaptive cluster or Top-k selection for relation-topic sources."""
        mask = torch.zeros_like(score)
        if score.numel() == 0:
            return mask
        if mask_mode == 'cluster':
            order = torch.argsort(score, descending=True)
            sorted_scores = score[order]
            cutoff = min(min_k, len(sorted_scores))
            for i in range(min_k - 1, min(len(sorted_scores) - 1, max_k)):
                if sorted_scores[i] > 0 and (sorted_scores[i + 1] / sorted_scores[i]) >= cluster_ratio:
                    cutoff = i + 2
                else:
                    break
            mask[order[:cutoff]] = 1.0
            return mask
        else:
            if not isinstance(topk_rt, int) or topk_rt < 1:
                raise ValueError("topk_rt must be a positive int when using a relation-topic mask")
            mask[torch.topk(score, min(topk_rt, score.numel())).indices] = 1.0
            return mask

    def _assembly(
        self,
        processed_dict_list,
        triple_score_dict,
        emb_dict,
        skip_no_path,
    ):
        self.processed_dict_list = []

        num_relevant_triples = []
        num_skipped = 0
        for i in tqdm(range(len(processed_dict_list))):
            sample_i = processed_dict_list[i]
            sample_i_id = sample_i['id']
            assert sample_i_id in triple_score_dict

            triple_score_i = triple_score_dict[sample_i_id]['triple_scores']
            max_path_length_i = triple_score_dict[sample_i_id]['max_path_length']

            num_relevant_triples_i = len(triple_score_i.nonzero())
            num_relevant_triples.append(num_relevant_triples_i)

            sample_i['target_triple_probs'] = triple_score_i
            sample_i['max_path_length'] = max_path_length_i

            if skip_no_path and (max_path_length_i in [None, 0]):
                num_skipped += 1
                continue

            sample_i['a_entity'] = list(set(sample_i['a_entity']))
            sample_i['a_entity_id_list'] = list(set(sample_i['a_entity_id_list']))

            # PE for topic entities.
            num_entities_i = len(sample_i['text_entity_list']) + len(sample_i['non_text_entity_list'])
            topic_entity_mask = torch.zeros(num_entities_i)
            topic_entity_mask[sample_i['q_entity_id_list']] = 1.
            topic_entity_one_hot = F.one_hot(topic_entity_mask.long(), num_classes=2)
            sample_i['topic_entity_one_hot'] = topic_entity_one_hot.float()

            # Canonical relation-topic feature layout: [score, mask].  The
            # model no longer needs to know whether a score came from GTE,
            # BGE, lexical overlap, or a ground-truth oracle.
            num_relations = len(sample_i['relation_list'])
            retriever_config = self.config['retriever']
            source = retriever_config.get('relation_topic_source', 'gte')
            score_mode = retriever_config.get('relation_score_mode', 'none')
            score_alpha = retriever_config.get('relation_score_alpha', 0.5)
            topk_rt = retriever_config.get('topk_rt')
            use_mask = retriever_config.get('use_relation_topic_mask', False)
            calib_mode = retriever_config.get('relation_score_calibration', 'none')
            power_gamma = retriever_config.get('relation_score_power', 1.0)
            mask_mode = retriever_config.get('relation_topic_mask_mode', 'topk')
            cluster_ratio = retriever_config.get('cluster_ratio', 0.90)
            min_k = retriever_config.get('cluster_min_k', 2)
            max_k = retriever_config.get('cluster_max_k', 15)

            if source == 'bge':
                def bge_score(key):
                    values = self.bge_relation_topics[key].get(sample_i_id)
                    if values is None or values.ndim != 1 or len(values) != num_relations:
                        raise ValueError(f"Invalid BGE {key} for sample: {sample_i_id}")
                    values = values.to(torch.float32)
                    if not torch.isfinite(values).all():
                        raise ValueError(f"BGE {key} contains NaN/Inf: {sample_i_id}")
                    return values

                semantic = bge_score('dense_scores') if score_mode in {'semantic', 'hybrid'} else None
                lexical = bge_score('sparse_scores') if score_mode in {'lexical', 'hybrid'} else None
                raw_score = self._select_relation_score(score_mode, semantic, lexical, score_alpha, num_relations)
                score = self._calibrate_relation_score(raw_score, calib_mode, power_gamma=power_gamma)
                mask = (
                    self._select_relation_topic_mask(
                        score, topk_rt, mask_mode=mask_mode,
                        cluster_ratio=cluster_ratio, min_k=min_k, max_k=max_k
                    )
                    if use_mask else torch.zeros(num_relations)
                )
                relation_topic_features = torch.stack([score, mask], dim=1)
            elif source in {'gte', 'path_memory_hybrid'}:
                cached_features = None
                if self.relation_topic_cache is not None:
                    cached_features = self.relation_topic_cache.get(str(sample_i_id))
                    if cached_features is None:
                        if source == 'path_memory_hybrid':
                            cached_features = torch.zeros((num_relations, 2), dtype=torch.float32)
                        else:
                            raise KeyError(
                                f"Missing relation-topic cache entry for sample {sample_i_id}"
                            )
                    if (
                        not isinstance(cached_features, torch.Tensor)
                        or cached_features.ndim != 2
                        or cached_features.shape != (num_relations, 2)
                        or not torch.isfinite(cached_features).all()
                    ):
                        raise ValueError(
                            f"Invalid cached relation-topic features for sample {sample_i_id}"
                        )
                    raw_score = cached_features[:, 0].to(torch.float32)
                    score = self._calibrate_relation_score(raw_score, calib_mode, power_gamma=power_gamma)
                    if mask_mode == 'cluster' or calib_mode != 'none' or (use_mask and topk_rt != 5):
                        mask = (
                            self._select_relation_topic_mask(
                                score, topk_rt, mask_mode=mask_mode,
                                cluster_ratio=cluster_ratio, min_k=min_k, max_k=max_k
                            )
                            if use_mask else torch.zeros(num_relations)
                        )
                    else:
                        mask = cached_features[:, 1].to(torch.float32) if use_mask else torch.zeros(num_relations)
                    relation_topic_features = torch.stack([score, mask], dim=1)
                else:
                    emb_i = emb_dict[sample_i_id]
                    semantic = None
                    lexical = None
                    if score_mode in {'semantic', 'hybrid'}:
                        q_emb_i = emb_i['q_emb'].to(torch.float32)  # shape: (1, 1024)
                        rel_embs_i = emb_i['relation_embs'].to(torch.float32) # shape: (num_relations, 1024)
                        semantic = torch.matmul(rel_embs_i, q_emb_i.T).squeeze(1)
                    if score_mode in {'lexical', 'hybrid'}:
                        lexical = torch.tensor(
                            [compute_lexical_score(clean_text(sample_i['question']), rel)
                             for rel in sample_i['relation_list']], dtype=torch.float32
                        )
                    raw_score = self._select_relation_score(score_mode, semantic, lexical, score_alpha, num_relations)
                    score = self._calibrate_relation_score(raw_score, calib_mode, power_gamma=power_gamma)
                    mask = (
                        self._select_relation_topic_mask(
                            score, topk_rt, mask_mode=mask_mode,
                            cluster_ratio=cluster_ratio, min_k=min_k, max_k=max_k
                        )
                        if use_mask else torch.zeros(num_relations)
                    )
                    relation_topic_features = torch.stack([score, mask], dim=1)
            else:
                raise ValueError(f"Unknown relation_topic_source: {source}")
            sample_i['relation_topic_features'] = relation_topic_features

            self.processed_dict_list.append(sample_i)

        median_num_relevant = int(np.median(num_relevant_triples))
        mean_num_relevant = int(np.mean(num_relevant_triples))
        max_num_relevant = int(np.max(num_relevant_triples))

        print(f'# skipped samples: {num_skipped}')
        print(f'# relevant triples | median: {median_num_relevant} | mean: {mean_num_relevant} | max: {max_num_relevant}')
    def __len__(self):
        return len(self.processed_dict_list)

    def __getitem__(self, i):
        sample = dict(self.processed_dict_list[i])
        sample.update(self.emb_dict[sample['id']])
        return sample

    def get_sampler(self, generator=None, shuffle=True):
        if isinstance(self.emb_dict, ShardedEmbDict):
            return ShardAwareSampler(self, generator=generator, shuffle=shuffle)
        return None

def collate_retriever(data):
    sample = data[0]
    
    h_id_list = sample['h_id_list']
    h_id_tensor = torch.tensor(h_id_list)
    
    r_id_list = sample['r_id_list']
    r_id_tensor = torch.tensor(r_id_list)
    
    t_id_list = sample['t_id_list']
    t_id_tensor = torch.tensor(t_id_list)
    
    num_non_text_entities = len(sample['non_text_entity_list'])
    
    # Dynamically determine the feature dimension to avoid hardcoding 3
    rel_feat = sample.get('relation_topic_features', None)
    if rel_feat is not None and rel_feat.shape[0] > 0:
        num_features = rel_feat.shape[1]
    else:
        num_features = 3

    return h_id_tensor, r_id_tensor, t_id_tensor, sample['q_emb'],\
        sample['entity_embs'], num_non_text_entities, sample['relation_embs'],\
        sample['topic_entity_one_hot'], sample['target_triple_probs'], sample['a_entity_id_list'],\
        sample.get('relation_topic_features', torch.zeros(0, num_features))
