import json
import os
import torch

from datasets import load_dataset
from tqdm import tqdm

from src.config.emb import load_yaml
from src.dataset.emb import EmbInferDataset

def build_shard_index(emb_dir):
    """Build shard_index.json mapping sample_id -> relative shard path."""
    index = {}
    shards_base = os.path.join(emb_dir, 'shards')
    if not os.path.exists(shards_base):
        return index

    for root, _, files in os.walk(shards_base):
        for f in sorted(files):
            if f.startswith("part_") and f.endswith(".pt"):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, emb_dir)
                try:
                    shard_data = torch.load(full_path, map_location="cpu")
                    for k in shard_data.keys():
                        index[str(k)] = rel_path
                except Exception as e:
                    print(f"Warning reading {full_path}: {e}")

    index_path = os.path.join(emb_dir, 'shard_index.json')
    with open(index_path, 'w') as f:
        json.dump(index, f)
    print(f"✅ Generated shard index ({len(index):,} samples) at: {index_path}")
    return index

def get_emb(subset, text_encoder, save_dir, final_file, shard_size=200, skip_merge=False):
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(final_file) and not skip_merge:
        print(f"✅ Merged file already exists at: {final_file}. Skipping.")
        return

    # 🔥 detect shard đã có
    existing = sorted([
        f for f in os.listdir(save_dir)
        if f.startswith("part_") and f.endswith(".pt")
    ], key=lambda x: int(x.split("_")[1].split(".")[0]))

    if existing:
        last_shard = max([int(f.split("_")[1].split(".")[0]) for f in existing])
        shard_id = last_shard + 1
        start_idx = shard_id * shard_size
        print(f"🔁 Resume from shard {shard_id}, index {start_idx} / {len(subset)}")
    else:
        shard_id = 0
        start_idx = 0
        print(f"🚀 Start from scratch (total: {len(subset)} samples)")

    emb_dict = {}

    for i in tqdm(range(start_idx, len(subset)), desc=os.path.basename(save_dir)):
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

    print("✅ All shards for this split done.")
    if not skip_merge:
        merge_shards_streaming(save_dir, final_file)
    else:
        print(f"⏭️  Shards saved in {save_dir}. Merging skipped (--no-merge).")

def merge_shards_streaming(save_dir, final_file):
    """Merge shards into a single .pth file."""
    all_files = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("part_") and f.endswith(".pt")],
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    merged = {}
    for f in tqdm(all_files, desc="Streaming merge"):
        shard_path = os.path.join(save_dir, f)
        shard_data = torch.load(shard_path, map_location="cpu")
        merged.update(shard_data)
        del shard_data

    torch.save(merged, final_file)
    del merged
    print(f"✅ Final file saved to: {final_file}")

def main(args):
    config_file = f'configs/emb/gte-large-en-v1.5/{args.dataset}.yaml'
    config = load_yaml(config_file)
    
    torch.set_num_threads(config['env']['num_threads'])

    if args.dataset == 'cwq':
        input_file = os.path.join('rmanluo', 'RoG-cwq')
    else:
        input_file = os.path.join('ml1996', 'webqsp')

    entity_identifiers = []
    with open(config['entity_identifier_file'], 'r') as f:
        for line in f:
            entity_identifiers.append(line.strip())
    entity_identifiers = set(entity_identifiers)
    
    save_dir = f'data_files/{args.dataset}/processed'
    os.makedirs(save_dir, exist_ok=True)

    splits_to_run = ['train', 'val', 'test'] if args.split == 'all' else [args.split]
    
    device = torch.device('cuda:0')
    text_encoder_name = config['text_encoder']['name']
    if text_encoder_name == 'gte-large-en-v1.5':
        from src.model.text_encoders import GTELargeEN
        text_encoder = GTELargeEN(device)
    else:
        raise NotImplementedError(text_encoder_name)
    
    emb_save_dir = f'data_files/{args.dataset}/emb/{text_encoder_name}'
    os.makedirs(emb_save_dir, exist_ok=True)

    for split_name in splits_to_run:
        print(f"\n=======================================================")
        print(f" Processing {args.dataset} [{split_name}]")
        print(f"=======================================================")
        
        raw_set = load_dataset(input_file, split='validation' if split_name == 'val' else split_name)
        if split_name == 'test':
            subset = EmbInferDataset(
                raw_set,
                entity_identifiers,
                os.path.join(save_dir, 'test.pkl'),
                skip_no_topic=False,
                skip_no_ans=False)
        else:
            subset = EmbInferDataset(
                raw_set,
                entity_identifiers,
                os.path.join(save_dir, f'{split_name}.pkl'))

        split_shard_dir = os.path.join(emb_save_dir, 'shards', split_name)
        final_file = os.path.join(emb_save_dir, f'{split_name}.pth')
        
        get_emb(subset, text_encoder, split_shard_dir, final_file,
                shard_size=args.shard_size, skip_merge=args.no_merge)

    if args.no_merge:
        print("\nBuilding global shard index for lazy loading...")
        build_shard_index(emb_save_dir)

if __name__ == '__main__':
    from argparse import ArgumentParser
    
    parser = ArgumentParser('Text Embedding Pre-Computation for Retrieval')
    parser.add_argument('-d', '--dataset', type=str, required=True, 
                        choices=['webqsp', 'cwq'], help='Dataset name')
    parser.add_argument('--split', type=str, default='all',
                        choices=['all', 'train', 'val', 'test'], help='Split to run')
    parser.add_argument('--shard_size', type=int, default=200,
                        help='Number of samples per shard')
    parser.add_argument('--no-merge', action='store_true',
                        help='Skip merging shards into a single monolithic file. Generates shard_index.json for lazy loading.')
    args = parser.parse_args()
    
    main(args)
