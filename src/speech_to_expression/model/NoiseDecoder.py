import torch
import torch.nn as nn

import utils

import typing
if typing.TYPE_CHECKING:
    import config as conf  # noqa: F401
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


class NoiseDecoder(nn.Module):

    def __init__(self, config, output_dim, embed_dim):
        # type: (conf.NoiseDecoderConfig, int, int) -> None
        """Initialize the NoiseDecoder model."""

        super().__init__()
        self.hidden_dim = config.hidden_dim
        num_heads = utils.find_closest_divisible_num_heads(self.hidden_dim, config.head_num)

        # 入力次元を hidden_dim に射影する層
        self.input_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.tgt_proj = nn.Linear(output_dim, self.hidden_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu",
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.layer_num
        )

        self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, past_frames, noisy_x, st_latent, lt_latent, tgt_key_padding_mask):
        """
        Args:
            past_frames: (batch, num_past_frames, output_dim)
                過去フレームの特徴列
            noisy_x: (batch, output_dim)
                ノイズが付与された予測対象フレーム
            st_latent: (seq_len_s, batch, embed_dim)
            lt_latent: (seq_len_l, batch, embed_dim)

        Returns:
            output: (batch, output_dim) 推定されたノイズ
        """

        batch_size = past_frames.size(0)  # noqa: F841
        num_past_frames = past_frames.size(1)  # noqa: F841

        # 過去フレーム列の末尾に noisy_x (予測対象フレーム) を追加
        # tgt: (batch, num_past_frames+1, output_dim)
        tgt = torch.cat([past_frames, noisy_x], dim=1)
        new_mask_column = torch.zeros((tgt.size(0), 1), dtype=torch.bool, device=tgt.device)
        tgt_key_padding_mask = torch.cat(
            [tgt_key_padding_mask, new_mask_column],  # (batch, num_past_frames+1)
            dim=1
        )

        # ターゲット系列を hidden_dim に射影
        tgt = self.tgt_proj(tgt)  # (batch, num_past_frames+1, hidden_dim)

        # メモリ（キー・バリュー）は短期・長期特徴の結合
        kv_features = torch.cat([st_latent, lt_latent], dim=1)  # (batch, seq_len, embed_dim)
        memory = self.input_proj(kv_features)                  # (batch, seq_len, hidden_dim)

        # Transformer Decoderの実行
        # tgt: (batch, num_past_frames+1, hidden_dim)
        # memory: (batch, seq_len, hidden_dim)
        decoder_output = self.transformer_decoder(
            tgt=tgt,
            memory=memory,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        decoder_output = self.layer_norm(decoder_output)  # (batch, num_past_frames+1, hidden_dim)

        # 最後のトークン位置が予測対象
        predicted = decoder_output[:, -1, :]   # (batch, hidden_dim)
        output = self.output_layer(predicted)  # (batch, output_dim)

        return output
