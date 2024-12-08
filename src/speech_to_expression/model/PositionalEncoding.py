import torch
import torch.nn as nn

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


class PositionalEncoding(nn.Module):
    """Positional Encoding Module"""

    def __init__(self, embed_dim, max_len):
        # type: (int, int) -> None
        super(PositionalEncoding, self).__init__()

        self.max_len = max_len

        # Create positional encoding (seq_len, embed_dim)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices

        # Register as buffer: pe is (max_len, embed_dim)
        self.register_buffer("pe", pe)

        # Temporal encoding (1) for past, (2) for present, (3) for future
        self.temporal_embeddings = nn.Embedding(3, embed_dim)

    def forward(self, x, current_frame_idx):
        # type: (torch.Tensor, torch.Tensor) -> torch.Tensor
        """
        Args:
            x (torch.Tensor): 入力テンソル（形状： (batch_size, seq_len, embed_dim)）
            current_frame_idx (torch.Tensor): (batch_size,) 各バッチの現在フレームのインデックス

        Returns:
            torch.Tensor: 位置エンコーディングと時間的エンコーディングが付加された入力テンソル
                          (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.size()
        device = x.device
        assert seq_len <= self.max_len

        # Positional encodingを展開: peは(max_len, embed_dim)で定義済み
        # ここでseq_len分取り出し、(batch_size, seq_len, embed_dim)に拡張
        pe = self.pe[:seq_len, :].to(device)              # (seq_len, embed_dim)
        pe = pe.unsqueeze(0).expand(batch_size, seq_len, embed_dim)  # (batch_size, seq_len, embed_dim)

        # Temporal indicesの計算
        # frame_indices: (batch_size, seq_len) 各サンプルで0からseq_len-1まで
        frame_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        frame_indices = frame_indices.expand(batch_size, seq_len)  # (batch_size, seq_len)

        # current_frame_idx: (batch_size,) -> (batch_size, seq_len)
        current_frame_idx_expanded = current_frame_idx.unsqueeze(1).expand(batch_size, seq_len)

        # temporal_indicesを計算 (0: past, 1: present, 2: future)
        temporal_indices = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        temporal_indices[frame_indices < current_frame_idx_expanded] = 0  # past
        temporal_indices[frame_indices == current_frame_idx_expanded] = 1  # present
        temporal_indices[frame_indices > current_frame_idx_expanded] = 2  # future

        # temporal_encodings: (batch_size, seq_len, embed_dim)
        temporal_encodings = self.temporal_embeddings(temporal_indices)

        # x, pe, temporal_encodingsを加算
        x = x + pe + temporal_encodings

        # nan or infがあるか確認
        if not torch.isfinite(x).all():
            print("Found NaN or infinite values in the output of PositionalEncoding")
        return x
