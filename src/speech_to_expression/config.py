import os
from pydantic import BaseModel, PositiveInt

import typing
if typing.TYPE_CHECKING:
    pass


class NoiseDecoderConfig(BaseModel):
    head_num: int = 4
    hidden_dim: int = 1024
    layer_num: int = 15
    norm_type: str = "layer_norm"


class DiffusionConfig(BaseModel):

    model: str
    sample_mode: str
    estimate_mode: str
    loss_type: str
    noise_schedule_mode: str
    noise_decoder_config: NoiseDecoderConfig
    train_timesteps_num: int = 1000


class ShortTermConfig(BaseModel):
    prev_window: int = 3
    next_window: int = 6

    head_num: int = 2


class LongTermConfig(BaseModel):
    prev_window: int = 90
    next_window: int = 60

    head_num: int = 2
    layer_num: int = 3


class SpeechToExpressionConfig(BaseModel):
    model: str
    embed_dim: int = 768
    time_embed_dim: int = 100
    output_dim: int = 31
    lr: float = 1e-5

    st: ShortTermConfig
    lt: LongTermConfig
    diffusion: DiffusionConfig


def load_from_yaml(yaml_path: str):
    import yaml
    with open(yaml_path) as f:
        dat = yaml.safe_load(f)
    return SpeechToExpressionConfig(**dat)
