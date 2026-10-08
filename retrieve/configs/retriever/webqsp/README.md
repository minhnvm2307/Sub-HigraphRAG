# How to config?

## 1. Retriever config

```yaml
retriever:
  topic_pe: true | false # Included topic_entity_onehot?
  relation_topic_source: 'gte" | 'bge' | 'groundtruth' # Choose emb model for retrieve relation topic, grountruth~choose direct relation from shortest-path
  relation_score_mode: 'semantic" | 'lexical' | 'hybrid' # How score feature constructed? Apply for both gte and bge model
  relation_score_alpha: 0.5 # Fraction of lexical score
  use_relation_topic_mask: true | false # Included onehot feature of relation topic?
  topk_rt: 5 # How many relation chosen to be topic
  dde_mode: 'baseline" | 'weighted' | 'both' # How DDE feature constructed? both mean concat
  weighted_dde_floor: 0.1 # Floor score of non-topic relation when DDE
  DDE_kwargs:
    num_rounds: 2
    num_reverse_rounds: 2
```

## 2. Loss function config

```yaml
loss:
  type: 'bce" | 'margin' | 'focal' | 'hybrid' | 'infonce' | 'none'
  pos_weight_mode: 'static" | 'inversed' | 'log' | 'sqrt'
  pos_weight: 200.0
  **kwargs: depend-on-type
```