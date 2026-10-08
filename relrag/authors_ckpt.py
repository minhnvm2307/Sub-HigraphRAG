"""Convert the authors' released RAPL checkpoints (HF TianJun1/RAPL: webqsp.pt, cwq.pt; Trainerv3 state_dict) into
RelRagRetriever checkpoints, so infer.py can run them unchanged (EXP-31, docs/02 §2.1 P6).

Trainerv3 uses PyG GCNConv(add_self_loops=False) with h + hb and no dropout between GCN layers, i.e. the
code-faithful config (no_self_loops, combine=sum, gnn_dropout=0). GCNConv keeps the bias outside `lin`.
  python authors_ckpt.py            # -> runs/authors_ckpt/{webqsp,cwq}_model/{best.pt,args.json}
"""
import json, os
import torch

D = 'runs/authors_ckpt'
MAP = {'convs': 'fwd', 'convs_back': 'bwd'}

for name in ['webqsp', 'cwq']:
    src = torch.load(f'{D}/{name}.pt', map_location='cpu', weights_only=False)
    dst = {}
    for k, v in src.items():
        head = k.split('.')[0]
        if head in MAP:                                   # convs.0.lin.weight / convs.0.bias -> fwd.0.weight / fwd.0.bias
            i = k.split('.')[1]
            dst[f'{MAP[head]}.{i}.{k.split(".")[-1]}'] = v
        else:                                             # q_mlp, r_mlp, stop_mlp, stop_emb: same layout
            dst[k] = v
    out = f'{D}/{name}_model'
    os.makedirs(out, exist_ok=True)
    torch.save(dst, f'{out}/best.pt')
    json.dump({'source': f'HF TianJun1/RAPL {name}.pt', 'no_self_loops': True, 'combine': 'sum', 'gnn_dropout': 0,
               'residual': False, 'no_bidirectional': False}, open(f'{out}/args.json', 'w'), indent=1)
    print(name, len(src), '->', len(dst), 'tensors ->', out)
