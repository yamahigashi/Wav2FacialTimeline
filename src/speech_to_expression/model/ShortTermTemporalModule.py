import torch
import torch.nn as nn

import utils

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


class ShortTermTemporalModule(nn.Module):
    """Short-term Temporal Module"""

    def __init__(self, embed_dim, num_heads):
        # type: (int, int) -> None
        """Initialize the ShortTermTemporalModule

        Args:
            embed_dim (int): The input feature dimension + the time embedding dimension
            num_heads (int): The number of heads in the multi-head attention
        """
        super(ShortTermTemporalModule, self).__init__()
        num_heads = utils.find_closest_divisible_num_heads(embed_dim, num_heads)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, st_features, st_key_padding_mask, timestep_emb):
        # type: (FeatBatch1st, MaskBatch1st, Float[Array, "batch"]) -> FeatBatch1st
        """Forward pass of the ShortTermTemporalModule

        Args:
            st_features (torch.Tensor): The input frame features
        """

        batch_size, seq_len, _ = st_features.size()
        timestep_emb_expanded = timestep_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
        st_features = torch.cat([st_features, timestep_emb_expanded], dim=-1)

        attn_output, _ = self.attention(
            st_features, st_features, st_features,
            key_padding_mask=st_key_padding_mask
        )
        output = self.layer_norm(st_features + attn_output)
        return output
