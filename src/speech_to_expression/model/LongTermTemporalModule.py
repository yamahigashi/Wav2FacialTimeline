import torch
import torch.nn as nn

from .. import utils

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
        PastFramesBatch,  # noqa: F401
        FeatBatch, # noqa: F401
        MaskBatch, # noqa: F401
        LabelBatch, # noqa: F401
    )


class LongTermTemporalModule(nn.Module):
    """Long-term Temporal Module"""

    def __init__(self, embed_dim, num_heads, num_layers):
        # type: (int, int, int) -> None
        """Initialize the LongTermTemporalModule

        Args:
            embed_dim (int): The input feature dimension + the time embedding dimension
            num_heads (int): The number of attention heads in the multi-head attention
            num_layers (int): The number of transformer encoder layers
        """
        super(LongTermTemporalModule, self).__init__()

        num_heads = utils.find_closest_divisible_num_heads(embed_dim, num_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, lt_features, src_key_padding_mask, timestep_emb):
        # type: (FeatBatch, MaskBatch, Float[Array, "batch"]) -> FeatBatch
        """Forward pass of the LongTermTemporalModule

        The input features should be in the sequence-first format, Transfomer expects the input to be
        (sequence_length, batch_size, embed_dim). The output is also in the same format. Then it is
        converted back to the batch-first format.

        Args:
            lt_features (torch.Tensor): The input global frame features
            src_key_padding_mask (torch.Tensor): The key padding mask for the input features

        Returns:
            torch.Tensor: Output of the transformer encoder
        """

        batch_size, seq_len, _ = lt_features.size()
        timestep_emb_expanded = timestep_emb.unsqueeze(1).expand(batch_size, seq_len, -1)
        lt_features = torch.cat([lt_features, timestep_emb_expanded], dim=-1)

        attn_output = self.transformer_encoder(
            lt_features,
            src_key_padding_mask=src_key_padding_mask
        )
        
        output = self.layer_norm(attn_output)
        
        return output
