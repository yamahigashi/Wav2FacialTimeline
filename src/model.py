import dataclasses

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

import typing
if typing.TYPE_CHECKING:
    from typing import (
        Tuple  # noqa: F401
    )
    from jaxtyping import (
        Float,  # noqa: F401
        Array,  # noqa: F401
        Int,  # noqa: F401
    )
    from dataset import (
        Batch,  # noqa: F401
    )

MAX_WINDOW_SIZE = 600


@dataclasses.dataclass
class HyperParameters:
    """Hyperparameters for the SpeechToExpressionModel"""

    embed_dim: int = 768
    output_dim: int = 31
    lr: float = 1e-3

    stm_prev_window: int = 3
    stm_next_window: int = 6
    ltm_prev_window: int = 90
    ltm_next_window: int = 60

    # ShortTermTemporalModule
    stm_heads: int = 8

    # LongTermTemporalModule
    ltm_heads: int = 8
    ltm_layers: int = 8

    # FeatureAttentionProcessor
    attn_heads: int = 8
    attn_layers: int = 8

    # MultiFrameAttentionAggregator
    agg_heads: int = 8

    # DiffusionModel
    diff_steps: int = 100  # DiffusionModel steps
    diff_beta_start: float = 0.0001
    diff_beta_end: float = 0.02


class ShortTermTemporalModule(nn.Module):
    """Short-term Temporal Module"""

    def __init__(self, embed_dim, num_heads):
        # type: (int, int) -> None
        """Initialize the ShortTermTemporalModule

        Args:
            embed_dim (int): The input feature dimension
            num_heads (int): The number of heads in the multi-head attention
        """
        super(ShortTermTemporalModule, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, frame_features, key_padding_mask):
        # type: (Float[Array, "seq_len", "batch", "embed_dim"], Float[Array, "batch", "seq_len"]) -> Float[Array, "seq_len", "batch", "embed_dim"]
        """Forward pass of the ShortTermTemporalModule

        Args:
            frame_features (torch.Tensor): The input frame features
        """
        attn_output, _ = self.attention(
            frame_features, frame_features, frame_features,
            key_padding_mask=key_padding_mask
        )
        output = self.layer_norm(frame_features + attn_output)
        return output


class LongTermTemporalModule(nn.Module):
    """Long-term Temporal Module"""

    def __init__(self, embed_dim, num_heads, num_layers):
        # type: (int, int, int) -> None
        """Initialize the LongTermTemporalModule

        Args:
            embed_dim (int): The input feature dimension
            num_heads (int): The number of attention heads in the multi-head attention
            num_layers (int): The number of transformer encoder layers
        """
        super(LongTermTemporalModule, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, global_frame_features, src_key_padding_mask):
        # type: (Float[Array, "seq_len", "batch", "embed_dim"], Float[Array, "batch", "seq_len"]) -> Float[Array, "seq_len", "batch", "embed_dim"]
        """Forward pass of the LongTermTemporalModule

        The input features should be in the sequence-first format, Transfomer expects the input to be
        (sequence_length, batch_size, embed_dim). The output is also in the same format. Then it is
        converted back to the batch-first format.

        Args:
            global_frame_features (torch.Tensor): The input global frame features
            src_key_padding_mask (torch.Tensor): The key padding mask for the input features

        Returns:
            torch.Tensor: Output of the transformer encoder
        """

        attn_output = self.transformer_encoder(
            global_frame_features,
            src_key_padding_mask=src_key_padding_mask
        )
        
        output = self.layer_norm(attn_output)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional Encoding Module"""

    def __init__(self, embed_dim, max_len):
        # type: (int, int) -> None

        super(PositionalEncoding, self).__init__()

        # Embeddings layer for relative positional encoding
        self.relative_position_embeddings = nn.Embedding(2 * max_len + 1, embed_dim)

        # Embeddings layer for temporal encoding, 3 values: past, present, future
        self.temporal_embeddings = nn.Embedding(3, embed_dim)

        self.max_len = max_len
        
        # Precompute relative position matrix once (for fixed seq_len)
        relative_positions = self._precompute_relative_positions(max_len)
        self.register_buffer("relative_positions", relative_positions)
       
        # FIXME: This buffered tensor causes a CUDA RuntimeError:
        # Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
        # Precompute embeddings for these relative positions
        # precomputed_relative_embeddings = self.relative_position_embeddings(self.relative_positions)
        # self.register_buffer("precomputed_relative_embeddings", precomputed_relative_embeddings)

    def _precompute_relative_positions(self, max_len):
        """Precompute the relative positions matrix for fixed max_len."""
        positions = torch.arange(max_len, dtype=torch.long)
        relative_positions = positions.view(-1, 1) - positions.view(1, -1)
        relative_positions += (max_len - 1)  # Shift to positive values

        return relative_positions

    def forward(self, x, current_frame_idx):
        # type: (Float[Array, "seq_len", "batch", "embed_dim"], Int[Array, "batch"]) -> Float[Array, "seq_len", "batch", "embed_dim"]
        """
        Forward pass for combined relative positional encoding and temporal encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, embed_dim)
            current_frame_idx (int): The index of the current frame in the sequence (center of past and future).
            
        Returns:
            torch.Tensor: The input tensor with relative positional encodings and temporal encodings added.
        """
        seq_len, batch_size, embed_dim = x.size()

        # 1. Encode relative positions an truncate to the current sequence length
        relative_position_encodings = self.relative_position_embeddings(self.relative_positions[:seq_len, :seq_len])
        relative_position_encodings = relative_position_encodings.unsqueeze(1).expand(-1, batch_size, -1, -1)
        relative_position_encodings = relative_position_encodings.sum(dim=2)
        if seq_len > self.max_len:
            # expand the relative position encodings to the current sequence length
            relative_position_encodings = nn.functional.pad(
                relative_position_encodings,
                (0, 0, seq_len - self.max_len, 0),
                mode="constant",
            )

        # 2. Encode temporal positions
        temporal_indices = torch.zeros((seq_len, batch_size), device=x.device, dtype=torch.long)  # initialize temporal indices to 0 (current frame)

        # Create a matrix representing the frame indices (seq_len, 1)
        frame_index_matrix = torch.arange(seq_len, device=x.device).unsqueeze(1)  # type: Int[Array, "seq_len", 1]
        
        # Use broadcasting to compare with current_frame_idx and assign 0 (past) 1 (current) 2 (future)
        past_mask = frame_index_matrix < current_frame_idx.unsqueeze(0)
        current_mask = frame_index_matrix == current_frame_idx.unsqueeze(0)
        future_mask = frame_index_matrix > current_frame_idx.unsqueeze(0)
        
        # Set past and future indices
        temporal_indices[past_mask] = 0
        temporal_indices[current_mask] = 1
        temporal_indices[future_mask] = 2
        # Retrieve temporal encodings based on temporal indices (seq_len, batch_size, embed_dim)
        temporal_encoding = self.temporal_embeddings(temporal_indices)  # Temporal embeddings are looked up using batched indices

        # 3. Combine relative position encodings and temporal encodings
        x = x + temporal_encoding + relative_position_encodings

        return x


class FeatureAttentionProcessor(nn.Module):

    def __init__(self, embed_dim, num_heads, num_layers):
        super(FeatureAttentionProcessor, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_features):
        # type: (Float[Array, "seq_len", "batch", "embed_dim"]) -> Float[Array, "seq_len", "batch", "embed_dim"]

        attn_output = self.transformer_encoder(input_features)
        output = self.layer_norm(attn_output)
        return output


class MultiFrameAttentionAggregator(nn.Module):

    def __init__(self, embed_dim, num_heads):
        # type: (int, int) -> None

        super(MultiFrameAttentionAggregator, self).__init__()

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, attn_output):
        # type: (Float[Array, "seq_len", "batch", "embed_dim"]) -> Float[Array, "batch", "embed_dim"]

        # attn_output: (seq_len, batch_size, embed_dim)
        # attn_weights: (batch_size, seq_len, seq_len)
        attn_output, attn_weights = self.multihead_attn(attn_output, attn_output, attn_output)
        permuted_output = attn_output.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)
        attn_scores = attn_weights.mean(dim=-1)  # (batch_size, seq_len)
        attn_scores = attn_scores.unsqueeze(-1)  # (batch_size, seq_len, 1)
        output = (permuted_output * attn_scores).sum(dim=1)  # (batch_size, embed_dim)

        norm_output = self.layer_norm(output)
        return norm_output


class DimensionalityReducer(nn.Module):

    def __init__(self, embed_dim, output_dim):
        super(DimensionalityReducer, self).__init__()
        self.projection = nn.Linear(embed_dim, output_dim)
        self.gelu = nn.GELU()

    def forward(self, aggregated_features):
        # type: (Float[Array, "batch", "embed_dim"]) -> Float[Array, "batch", "output_dim"]

        output = self.projection(aggregated_features)  # (batch_size, output_dim)
        output = self.gelu(output)  # 非線形活性化を適用
        return output


class SpeechToExpressionModel(pl.LightningModule):
    """Temporal Diffusion Speaker Model"""

    def __init__(self, hparams):
        # type: (HyperParameters) -> None
        """Initialize the SpeechToExpressionModel

        Args:
            embed_dim (int): The input feature dimension
            num_heads (int): The number of heads in the multi-head attention
            num_steps (int): The number of diffusion steps
            num_layers (int): The number of transformer encoder layers
            lr (float): The learning rate for training the model
        """
        super(SpeechToExpressionModel, self).__init__()

        short_term_window = hparams.stm_prev_window + hparams.stm_next_window + 1
        long_term_window = hparams.ltm_prev_window + hparams.ltm_next_window + 1
        
        self.short_term_module = ShortTermTemporalModule(hparams.embed_dim, hparams.stm_heads)
        self.long_term_module = LongTermTemporalModule(hparams.embed_dim, hparams.ltm_heads, hparams.ltm_layers)
        self.short_positional_encoding = PositionalEncoding(hparams.embed_dim, short_term_window)
        self.long_positional_encoding = PositionalEncoding(hparams.embed_dim, long_term_window)
        self.attn_processor = FeatureAttentionProcessor(hparams.embed_dim, hparams.attn_heads, hparams.attn_layers)
        self.aggregator = MultiFrameAttentionAggregator(hparams.embed_dim, hparams.agg_heads)
        self.reducer = DimensionalityReducer(hparams.embed_dim, hparams.output_dim)
        self.lr = hparams.lr

        self.memory_weights = nn.Parameter(torch.ones(2))

        self.save_hyperparameters()

    def forward(
            self,
            frame_features,
            global_frame_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
            speaker_labels=None,
            inference=False
    ):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor|None, bool) -> torch.Tensor
        """Forward pass of the SpeechToExpressionModel.

        First, the short-term temporal attention module is applied to the frame features.
        Then, the long-term temporal attention module is applied to the global frame features.
        The outputs of the short-term and long-term modules are concatenated and passed to the
        biased conditional self-attention module. Finally, the output is passed through the diffusion
        model to denoise and refine the output.

        The bias factor in the biased conditional self-attention module is conditioned on the input condition.

        Args:
            frame_features (torch.Tensor): The input frame features
            global_frame_features (torch.Tensor): The input global frame features
            condition (torch.Tensor): The input condition

        Returns:
            torch.Tensor: The output features
        """
        batch_size, _, embed_dim = frame_features.size()

        # Permute the input features to the correct shape
        # The nn.Transformer expects the input to be sequence-first data. Then,
        # (batch_size, seq_len, embed_dim) -> （(sequence_length, batch_size, embed_dim)）
        frame_features = frame_features.permute(1, 0, 2)
        global_frame_features = global_frame_features.permute(1, 0, 2)

        short_term_output = self.short_term_module(frame_features, frame_masks)
        long_term_output = self.long_term_module(global_frame_features, global_frame_masks)

        weights = torch.softmax(self.memory_weights, dim=0)
        weighted_short_term = weights[0] * short_term_output
        weighted_long_term = weights[1] * long_term_output
        encoded_short_term = self.short_positional_encoding(weighted_short_term, current_short_frame)
        encoded_long_term = self.long_positional_encoding(weighted_long_term, current_long_frame)
        combined_features = torch.cat((encoded_short_term, encoded_long_term), dim=0)

        attention_output = self.attn_processor(combined_features)
        aggregated_features = self.aggregator(attention_output)
        final_output = self.reducer(aggregated_features)

        return final_output

    def training_step(self, batch, batch_idx):
        # type: (Batch, int) -> torch.Tensor
        """Training step of the SpeechToExpressionModel

        Lightning calls this method during the training loop.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A batch of input data
            batch_idx (int): The index of the batch

        Returns:
            torch.Tensor: The loss value for the training step
        """

        (
            frame_features,
            global_frame_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
            labels
        ) = batch

        outputs = self(
            frame_features,
            global_frame_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame
        ) # type: Float[Array, "batch_size", "embed_dim"]
        labels = labels.expand(outputs.shape[0], -1)  # extend to batch dimension

        # Calculate the mean squared error loss
        loss = nn.functional.mse_loss(outputs, labels)

        weights = torch.softmax(self.memory_weights, dim=0)

        # loss += emo_loss
        self.log("train_loss", loss)
        self.log("short_term_weight", weights[0])
        self.log("long_term_weight", weights[1])

        return loss

    def validation_step(self, batch, batch_idx):
        # type: (Batch, int) -> None
        """Validation step of the SpeechToExpressionModel
        Lightning calls this method during the validation loop.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A batch of input data
            batch_idx (int): The index of the batch
        """

        (
            frame_features,
            global_frame_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
            labels
        ) = batch

        outputs = self(
            frame_features,
            global_frame_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame
        ) # type: Float[Array, "batch_size", "embed_dim"]
        labels = labels.expand(outputs.shape[0], -1)
        val_loss = nn.functional.mse_loss(outputs, labels)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        # type: () -> torch.optim.Optimizer
        """Configure the optimizer for training the model.
        Adam optimizer is used with the specified learning rate.

        Returns:
            torch.optim.Optimizer: The optimizer for training the model
        """
        return torch.optim.Adam(self.parameters(), lr=self.lr)
