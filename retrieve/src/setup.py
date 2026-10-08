import numpy as np
import os
import random
import torch

def set_seed(seed, deterministic=True):
    """Seed all RNGs and optionally require deterministic PyTorch operations."""
    seed = int(seed)
    # New config to get consistent results across runs, even with CUDA.
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic)

def seed_worker(worker_id):
    """Seed a DataLoader worker from its PyTorch-provided initial seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_sample(device, sample):
    if len(sample) == 11:
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, a_entity_id_list, relation_topic_features = sample
        relation_topic_features = relation_topic_features.to(device)
    else:
        h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
            num_non_text_entities, relation_embs, topic_entity_one_hot,\
            target_triple_probs, a_entity_id_list = sample
        relation_topic_features = torch.zeros(0, 3).to(device)

    h_id_tensor = h_id_tensor.to(device)
    r_id_tensor = r_id_tensor.to(device)
    t_id_tensor = t_id_tensor.to(device)
    q_emb = q_emb.to(device)
    entity_embs = entity_embs.to(device)
    relation_embs = relation_embs.to(device)
    topic_entity_one_hot = topic_entity_one_hot.to(device)
    
    return h_id_tensor, r_id_tensor, t_id_tensor, q_emb, entity_embs,\
        num_non_text_entities, relation_embs, topic_entity_one_hot,\
        target_triple_probs, a_entity_id_list, relation_topic_features
