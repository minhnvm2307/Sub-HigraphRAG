import os
import torch

from datasets import load_dataset
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset

def get_emb(subset, text_encoder, save_dir, final_file, shard_size=200):
    os.makedirs(save_dir, exist_ok=True)

    # 🔥 detect shard đã có
    existing = sorted([
        f for f in os.listdir(save_dir)
        if f.startswith("part_") and f.endswith(".pt")
    ])

    if existing:
        last_shard = max([int(f.split("_")[1].split(".")[0]) for f in existing])
        shard_id = last_shard + 1
        start_idx = shard_id * shard_size
        print(f"🔁 Resume from shard {shard_id}, index {start_idx}")
    else:
        shard_id = 0
        start_idx = 0
        print("🚀 Start from scratch")

    emb_dict = {}

    for i in tqdm(range(start_idx, len(subset))):
        id, q_text, text_entity_list, relation_list = subset[i]
        
        q_emb, entity_embs, relation_embs = text_encoder(
            q_text, text_entity_list, relation_list)

        emb_dict[id] = {
            'q_emb': q_emb.detach().cpu().half(),
            'entity_embs': entity_embs.detach().cpu().half(),
            'relation_embs': relation_embs.detach().cpu().half()
        }

        # save shard
        if (i + 1) % shard_size == 0:
            shard_path = os.path.join(save_dir, f"part_{shard_id}.pt")
            torch.save(emb_dict, shard_path)
            emb_dict = {}
            shard_id += 1

    # save phần còn lại
    if emb_dict:
        shard_path = os.path.join(save_dir, f"part_{shard_id}.pt")
        torch.save(emb_dict, shard_path)

    print("✅ All shards done, start merging...")
    merge_shards(save_dir, final_file)

def merge_shards(save_dir, final_file):
    import os
    import torch

    all_files = sorted([
        f for f in os.listdir(save_dir) if f.startswith("part_")
    ])

    final_dict = {}

    for f in tqdm(all_files, desc="Merging shards"):
        shard_path = os.path.join(save_dir, f)
        shard_data = torch.load(shard_path)
        final_dict.update(shard_data)

    torch.save(final_dict, final_file)
    print(f"✅ Final file saved to: {final_file}")

def main(args):
    # Modify the config file for advanced settings and extensions.
    config_file = f'configs/emb/gte-large-en-v1.5/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    torch.set_num_threads(config['env']['num_threads'])

    if args.dataset == 'cwq':
        input_file = os.path.join('rmanluo', 'RoG-cwq')
    else:
        input_file = os.path.join('ml1996', 'webqsp')

    train_set = load_dataset(input_file, split='train')
    val_set = load_dataset(input_file, split='validation')
    test_set = load_dataset(input_file, split='test')
    
    entity_identifiers = []
    with open(config['entity_identifier_file'], 'r') as f:
        for line in f:
            entity_identifiers.append(line.strip())
    entity_identifiers = set(entity_identifiers)
    
    save_dir = f'data_files/{args.dataset}/processed'
    os.makedirs(save_dir, exist_ok=True)

    train_set = EmbInferDataset(
        train_set,
        entity_identifiers,
        os.path.join(save_dir, 'train.pkl'))

    val_set = EmbInferDataset(
        val_set,
        entity_identifiers,
        os.path.join(save_dir, 'val.pkl'))

    test_set = EmbInferDataset(
        test_set,
        entity_identifiers,
        os.path.join(save_dir, 'test.pkl'),
        skip_no_topic=False,
        skip_no_ans=False)
    
    device = torch.device('cuda:0')
    
    text_encoder_name = config['text_encoder']['name']
    if text_encoder_name == 'gte-large-en-v1.5':
        from src.model.text_encoders import GTELargeEN
        text_encoder = GTELargeEN(device)
    else:
        raise NotImplementedError(text_encoder_name)
    
    emb_save_dir = f'data_files/{args.dataset}/emb/{text_encoder_name}'
    os.makedirs(emb_save_dir, exist_ok=True)
    
    get_emb(train_set, text_encoder, os.path.join(emb_save_dir, 'train.pth'))
    get_emb(val_set, text_encoder, os.path.join(emb_save_dir, 'val.pth'))
    get_emb(test_set, text_encoder, os.path.join(emb_save_dir, 'test.pth'))
    # get_emb(test_set, text_encoder, emb_save_dir, os.path.join(emb_save_dir, 'test.pth'))

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    args = parser.parse_args()
    
    main(args)
