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

    stm_prev_window: int = 4
    stm_next_window: int = 3
    ltm_prev_window: int = 90
    ltm_next_window: int = 60

    # ShortTermTemporalModule
    stm_heads: int = 8

    # LongTermTemporalModule
    ltm_heads: int = 8
    ltm_layers: int = 8

    # BiasedConditionalSelfAttention
    attn_heads: int = 8
    attn_layers: int = 8  
    attn_bias_factor: float = 0.1

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


class BiasedConditionalSelfAttention(nn.Module):
    """Biased Conditional Self-Attention with Transformer"""

    def __init__(self, embed_dim, num_heads, num_layers, bias_factor=0.1):
        super(BiasedConditionalSelfAttention, self).__init__()
        self.bias_factor = bias_factor

        # Transformerエンコーダーの定義
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, input_features, speaker_info=None):
        # input_features: (seq_len, batch_size, embed_dim)

        # Transformerエンコーダーの適用
        attn_output = self.transformer_encoder(input_features)

        # スピーカーバイアスの適用
        if speaker_info is not None:
            speaker_embed = speaker_info.unsqueeze(0)  # (1, batch_size, embed_dim)
            bias = speaker_embed * self.bias_factor
            attn_output = attn_output + bias  # seq_len方向にブロードキャスト

        output = self.layer_norm(attn_output)
        return output


class DiffusionModel(nn.Module):
    """Diffusion Model"""

    def __init__(self, embed_dim, output_dim, num_steps, beta_start=0.0001, beta_end=0.02):
        # type: (int, int, int, float, float) -> None

        super(DiffusionModel, self).__init__()
        self.num_steps = num_steps
        self.output_dim = output_dim

        # Define betas for forward diffusion process (linear schedule)
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, num_steps))

        # Precompute alphas and alpha bars
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

        # Model to predict noise ε_θ(x_t, t)
        self.network = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.output_dim)
        )

    def forward(self, x, t):
        # type: (Float[Array, "batch", "seq_len", "embed_dim"], Float[Array, "batch"]) -> Float[Array, "batch", "seq_len", "embed_dim"]
        """Forward pass for training.

        Args:
            x: Clean data sample
            t: Time step tensor
        """

        # Add noise to the data
        noise = torch.randn_like(x)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(1).unsqueeze(1)
        x_noisy = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise

        # Predict the noise
        predicted_noise = self.network(x_noisy)

        return predicted_noise, noise

    def sample(self, x, num_samples, device):
        # type: (torch.Tensor, int, torch.device) -> Float[Array, "batch", "seq_len", "embed_dim"]
        """Generate samples using the reverse diffusion process."""

        # x = torch.randn(num_samples, self.output_dim, device=device)
        for t in reversed(range(self.num_steps)):

            t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)

            predicted_noise = self.network(x, t_tensor)
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alpha_bars[t]

            # 次のステップのノイズを追加（t > 0の場合）
            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            # 逆拡散プロセスの更新式
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise) + torch.sqrt(beta_t) * noise
        return x


class EmotionConstraintLayer(torch.nn.Module):
    """Emotion Constraint Layer

    The sumulation of the emotion columns should be 1.
    """

    def __init__(self):
        super(EmotionConstraintLayer, self).__init__()

    def forward(self, x):
        # type: (Float[Array, "batch", "seq_len", "embed_dim"]) -> Float[Array, "batch", "seq_len", "embed_dim"]

        # col21から27 (facial expressions) の値をsoftmaxで正規化
        emotion_outputs = x[:, 21:28]
        emotion_outputs = torch.nn.functional.softmax(emotion_outputs, dim=1)

        # avoid in-place operation
        x = x.clone()  # type: ignore
        x[:, 21:28] = emotion_outputs
        return x


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
        self.biased_attention = BiasedConditionalSelfAttention(hparams.embed_dim, hparams.attn_heads, hparams.attn_layers)
        self.diffusion = DiffusionModel(hparams.embed_dim, hparams.output_dim, hparams.diff_steps, hparams.diff_beta_start, hparams.diff_beta_end)
        # self.emotion_constraint_layer = EmotionConstraintLayer()
        # self.emotion_constraint_penalty = 0.001
        self.diff_steps = hparams.diff_steps
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
        seq_len, batch_size, embed_dim = frame_features.size()

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

        attention_output = self.biased_attention(combined_features, speaker_labels)

        # Permute the output features back to the original shape
        attention_output = attention_output.permute(1, 0, 2)  # type: Float[Array, "batch_size", seq_len, embed_dim]

        if inference:
            # Perform reverse diffusion sampling during inference
            num_samples = frame_features.shape[1]  # batch_size
            device = frame_features.device
            # Prepare the conditioning information
            condition = attention_output.permute(1, 0, 2)  # Shape: (batch_size, seq_len, embed_dim)
            # Use the diffusion model's sampling method
            generated_output = self.diffusion.sample(
                condition,
                num_samples=num_samples,
                # condition=condition,
                device=device
            )
            # final_output = self.emotion_constraint_layer(generated_output)
            return generated_output

        else:
            # Training mode (existing code)
            batch_size = attention_output.shape[0]
            t = torch.randint(0, self.diffusion.num_steps, (batch_size,), device=attention_output.device)
            denoised_output, _ = self.diffusion(attention_output, t)
            # constrained_output = self.emotion_constraint_layer(denoised_output)
            final_output = denoised_output[:, -1, :]  # Only return the final frame output
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

        # Introduce the constraint of sum of the emotions should be 1
        # the col 21 to 27 are the emotion columns
        # emotion_sum = torch.sum(outputs[:, 21:28], dim=1)
        # emo_loss = nn.functional.mse_loss(emotion_sum, torch.ones_like(emotion_sum)) * self.emotion_constraint_penalty
        # self.log("train_emo_loss", emo_loss)

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
