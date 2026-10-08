"""Configuration schemas with strict pointwise / two-stage separation."""

import copy

import pydantic
import yaml

from .base import EnvYaml


class DatasetYaml(pydantic.BaseModel):
    name: str
    text_encoder_name: str

class DDEYaml(pydantic.BaseModel):
    num_rounds: int
    num_reverse_rounds: int

class RetrieverYaml(pydantic.BaseModel):
    """Pointwise retriever settings only."""

    topic_pe: bool
    DDE_kwargs: DDEYaml
    relation_topic_source: str = 'gte'
    relation_score_mode: str = 'none'
    relation_score_alpha: float = 0.5
    use_relation_topic_mask: bool = False
    topk_rt: int | None = None
    dde_mode: str = 'baseline'
    # Advanced calibration and adaptive clustering options
    relation_score_calibration: str = 'none'  # 'none', 'platt', 'power'
    relation_score_power: float = 1.0
    relation_topic_mask_mode: str = 'topk'  # 'topk', 'cluster'
    cluster_ratio: float = 0.90
    cluster_min_k: int = 2
    cluster_max_k: int = 15

    # These two fields identify an existing on-disk BGE cache. They are not
    # score/feature modes: dense vs. sparse is selected by relation_score_mode.
    relation_topic_bge_format: str | None = None
    # Optional precomputed GTE relation-topic feature cache. ``auto`` resolves
    # to the canonical path under data_files/<dataset>/relation_topics.
    relation_topic_cache: str | None = None
    pos_weight: float | None = None



class ListwiseStageYaml(pydantic.BaseModel):
    """Settings used exclusively by two-stage-train/inference."""

    enabled: bool
    top_n: int = 500
    num_heads: int = 4
    num_layers: int = 2
    d_model: int = 256
    dim_feedforward: int = 1024
    dropout: float = 0.1
    rank_margin: float = 1e-4
    selection_metric: str = 'ans_recall@100'


class TwoStageRetrieverYaml(RetrieverYaml):
    listwise: ListwiseStageYaml

class OptimizerYaml(pydantic.BaseModel):
    lr: float

class EvalYaml(pydantic.BaseModel):
    k_list: str

class RetrieverExpYaml(pydantic.BaseModel):
    num_epochs: int
    patience: int
    save_prefix: str

class RetrieverTrainYaml(pydantic.BaseModel):
    env: EnvYaml
    dataset: DatasetYaml
    retriever: RetrieverYaml
    loss: dict | None = None
    optimizer: OptimizerYaml
    eval: EvalYaml
    train: RetrieverExpYaml


class TwoStageTrainYaml(RetrieverTrainYaml):
    retriever: TwoStageRetrieverYaml


def _read_yaml(config_file):
    with open(config_file) as handle:
        yaml_data = yaml.load(handle, Loader=yaml.loader.SafeLoader)
    if yaml_data.pop('task', None) != 'retriever':
        raise ValueError(f"Config '{config_file}' must declare task: 'retriever'.")
    loss = yaml_data.get('loss')
    if loss is not None and (not isinstance(loss, dict) or not loss):
        raise ValueError(
            f"Config '{config_file}' contains an empty or invalid loss section: {loss}."
        )
    return yaml_data


def _dump_config(schema, yaml_data):
    config = schema(**yaml_data).model_dump()
    config['eval']['k_list'] = [int(k) for k in config['eval']['k_list'].split(',')]
    return config


def load_yaml(config_file):
    """Load a pointwise-only config for train.py, inference.py, and indexing."""
    yaml_data = _read_yaml(config_file)
    if 'listwise' in yaml_data.get('retriever', {}):
        raise ValueError(
            "This is a two-stage listwise config. Pointwise entrypoints only accept "
            "pointwise YAML; use two-stage-train.py or two-stage-inference.py."
        )
    return _dump_config(RetrieverTrainYaml, yaml_data)


def load_two_stage_yaml(config_file):
    """Load a stage-2-only config. It must explicitly declare listwise settings."""
    yaml_data = _read_yaml(config_file)
    if 'listwise' not in yaml_data.get('retriever', {}):
        raise ValueError(
            "Two-stage configs must declare retriever.listwise explicitly."
        )
    return _dump_config(TwoStageTrainYaml, yaml_data)


def strip_listwise(config):
    """Return the exact pointwise portion of a two-stage or checkpoint config."""
    pointwise = copy.deepcopy(config)
    pointwise['retriever'].pop('listwise', None)
    return pointwise
