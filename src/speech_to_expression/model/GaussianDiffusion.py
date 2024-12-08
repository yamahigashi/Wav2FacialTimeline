import torch
import torch.nn as nn
import diffusers.schedulers

from .. import config as conf
from . import (
    ShortTermTemporalModule,  # noqa: F401
    LongTermTemporalModule,  # noqa: F401
    NoiseDecoder,  # noqa: F401
)

import typing
if typing.TYPE_CHECKING:

    from typing import (
        Tuple,  # noqa: F401
        Dict,  # noqa: F401
        Optional,  # noqa: F401
    )
    from jaxtyping import (
        Float,  # noqa: F401
        Array,  # noqa: F401
        Int,  # noqa: F401
    )
    from dataset import (
        Batch,  # noqa: F401
    )

    Label = Float[Array, "batch_size", "output_dim"]
    FeatBatch1st = Float[Array, "batch", "seq_len", "embed_dim"]
    MaskSeq1st = Float[Array, "seq_len", "batch"]
    MaskBatch1st = Float[Array, "batch", "seq_len"]


class GaussianDiffusion(nn.Module):
    """Gaussian Diffusion model. Forwarding through the module returns diffusion reversal scalar loss tensor.
    Input:
        x: tensor of shape (N, img_channels, *img_size)
        y: tensor of shape (N)
    Output:
        scalar loss tensor
    """
    def __init__(
        self,
        config: conf.DiffusionConfig,
        embed_dim: int,
        output_dim: int,
        time_embed_dim: int,
        short_term_module: ShortTermTemporalModule.ShortTermTemporalModule,
        long_term_module: LongTermTemporalModule.LongTermTemporalModule,
        noise_decoder_module: NoiseDecoder.NoiseDecoder,
        train: bool = True,
    ):

        super().__init__()

        self.train_timesteps_num = config.train_timesteps_num
        self.time_embed_dim = time_embed_dim

        self.timestep_embedding = nn.Embedding(self.train_timesteps_num, self.time_embed_dim)
        self.short_term_module = short_term_module
        self.long_term_module = long_term_module

        self.model = noise_decoder_module

        self.scheduler = diffusers.schedulers.DDPMScheduler(
            num_train_timesteps=config.train_timesteps_num,
            variance_type="fixed_large",
            clip_sample=False,
            timestep_spacing="trailing",
        )

    def forward(
        self,
        past_frames,
        noisy_x,
        ts,
        st_features,
        lt_features,
        tgt_key_padding_mask,
        st_key_padding_mask,
        lt_key_padding_mask
    ):
     
        # Ensure ts is within the valid range
        assert ts.max() < self.train_timesteps_num, "ts contains indices out of bounds for the embedding layer."
        timestep_emb = self.timestep_embedding(ts)
        st_latent = self.short_term_module(st_features, st_key_padding_mask, timestep_emb)
        lt_latent = self.long_term_module(lt_features, lt_key_padding_mask, timestep_emb)

        estimated = self.model(
            past_frames,
            noisy_x,
            st_latent,
            lt_latent,
            tgt_key_padding_mask
        )

        return estimated

    def generate(
        self,
        past_frames,
        st_encoded,
        lt_encoded,
        past_frame_masks,
        frame_masks,
        global_frame_masks,
        current_short_frame,
        current_long_frame,
        num_inference_steps=30
    ):
        batch_size = past_frames.size(0)
        device = past_frames.device

        # 推論用のスケジューラを設定
        self.scheduler.set_timesteps(
            num_inference_steps,
            device=device
        )

        # 初期入力をノイズから開始
        x = torch.randn_like(past_frames[:, -1, :]).unsqueeze(1).to(device)

        for t in self.scheduler.timesteps:
            ts = t.expand(batch_size).to(device)

            timestep_emb = self.timestep_embedding(ts)
            st_latent = self.short_term_module(st_encoded, frame_masks, timestep_emb)
            lt_latent = self.long_term_module(lt_encoded, global_frame_masks, timestep_emb)

            # ノイズの予測
            model_output = self.model(
                past_frames,
                x,
                st_latent,
                lt_latent,
                past_frame_masks
            )

            # 前のサンプルを計算
            x = self.scheduler.step(
                model_output,
                t,
                x
            ).prev_sample

        return x
