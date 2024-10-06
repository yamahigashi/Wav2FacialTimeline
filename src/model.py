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
    )


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
        self.attention = nn.MultiheadAttention(embed_dim, num_heads * 3)
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

    def __init__(self, embed_dim, max_len=5000):
        # type: (int, int) -> None

        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x):
        # type: (Float[Array, "seq_len", "batch", "embed_dim"]) -> Float[Array, "seq_len", "batch", "embed_dim"]

        pos = self.pe.expand(x.shape[1], -1, -1).transpose(0, 1)
        x = x + pos[:x.shape[0], :, :]
        return x


class BiasedConditionalSelfAttention(nn.Module):
    """Biased Conditional Self-Attention"""

    def __init__(self, embed_dim, num_heads, num_speakers=None, bias_factor=0.1):
        # type: (int, int, int | None, float) -> None
        """Initialize the BiasedConditionalSelfAttention

        Args:
            embed_dim (int): The input feature dimension
            num_heads (int): The number of heads in the multi-head attention
            bias_factor (float): The bias factor to be applied to the attention output
        """
        super(BiasedConditionalSelfAttention, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.bias_factor = bias_factor
        self.layer_norm = nn.LayerNorm(embed_dim)

        if num_speakers is not None:
            self.speaker_embeddings = nn.Embedding(num_speakers, embed_dim)
        else:
            self.speaker_embeddings = None

    def forward(self, input_features, speaker_info=None):
        # type: (Float[Array, "seq_len", "batch", "embed_dim"], Float[Array, "batch", "embed_dim"]|None) -> Float[Array, "seq_len", "batch", "embed_dim"]
        """Forward pass of the BiasedConditionalSelfAttention

        Args:
            x (torch.Tensor): The input features
            condition (torch.Tensor): The conditional input features

        Returns:
            torch.Tensor: The output features
        """

        attn_output, _ = self.attention(input_features, input_features, input_features)

        if speaker_info is not None:

            if self.speaker_embeddings is not None and speaker_info.dim() == 1:
                speaker_embed = self.speaker_embeddings(speaker_info).unsqueeze(0)
            else:
                speaker_embed = speaker_info.unsqueeze(0)

            bias = speaker_embed * self.bias_factor
            attn_output += bias

        output = self.layer_norm(input_features + attn_output)

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

    def sample(self, num_samples, device):
        # type: (int, torch.device) -> Float[Array, "batch", "seq_len", "embed_dim"]
        """Generate samples using the reverse diffusion process."""

        x = torch.randn(num_samples, self.output_dim, device=device)
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

        # カラム21から27にソフトマックスを適用して和が1になるようにする
        emotion_outputs = x[:, 21:28]
        emotion_outputs = torch.nn.functional.softmax(emotion_outputs, dim=1)

        # avoid in-place operation
        x = x.clone()  # type: ignore
        x[:, 21:28] = emotion_outputs
        return x


class SpeechToExpressionModel(pl.LightningModule):
    """Temporal Diffusion Speaker Model"""

    def __init__(self, embed_dim, output_dim, num_heads, num_steps, num_layers, num_speakers=1, lr=1e-3):
        # type: (int, int, int, int, int, int, float) -> None
        """Initialize the SpeechToExpressionModel

        Args:
            embed_dim (int): The input feature dimension
            num_heads (int): The number of heads in the multi-head attention
            num_steps (int): The number of diffusion steps
            num_layers (int): The number of transformer encoder layers
            num_speakers (int): The number of speakers
            lr (float): The learning rate for training the model
        """
        super(SpeechToExpressionModel, self).__init__()
        
        self.short_term_module = ShortTermTemporalModule(embed_dim, num_heads)
        self.long_term_module = LongTermTemporalModule(embed_dim, num_heads, num_layers)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.biased_attention = BiasedConditionalSelfAttention(embed_dim, num_heads, num_speakers)
        self.diffusion = DiffusionModel(embed_dim, output_dim, num_steps)
        self.emotion_constraint_layer = EmotionConstraintLayer()
        self.num_steps = num_steps
        self.lr = lr
        self.emotion_constraint_penalty = 0.001

        self.memory_weights = nn.Parameter(torch.ones(2))

        self.save_hyperparameters()

    def forward(
            self,
            frame_features,
            global_frame_features,
            frame_masks=None,
            global_frame_masks=None,
            speaker_labels=None
    ):
        # type: (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor|None) -> torch.Tensor
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
        combined_features = torch.cat((weighted_short_term, weighted_long_term), dim=0)
        combined_features = self.positional_encoding(combined_features)

        attention_output = self.biased_attention(combined_features, speaker_labels)

        # Permute the output features back to the original shape
        attention_output = attention_output.permute(1, 0, 2)  # type: Float[Array, "batch_size", seq_len, embed_dim]

        # Generate random timesteps for diffusion
        batch_size = attention_output.shape[0]
        t = torch.randint(0, self.num_steps, (batch_size,), device=attention_output.device)
        denoised_output, _ = self.diffusion(attention_output, t)
        constrained_output = self.emotion_constraint_layer(denoised_output)

        final_output = constrained_output [:, -1, :]  # Only return the final frame output
        # final_output = final_output.permute(1, 0, 2)  # permute back to (batch_size, seq_len, embed_dim)
        
        return final_output

    def training_step(self, batch, batch_idx):
        # type: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], int) -> torch.Tensor
        """Training step of the SpeechToExpressionModel

        Lightning calls this method during the training loop.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A batch of input data
            batch_idx (int): The index of the batch

        Returns:
            torch.Tensor: The loss value for the training step
        """

        frame_features, global_frame_features, frame_masks, global_frame_masks, labels = batch
        outputs = self(frame_features, global_frame_features, frame_masks, global_frame_masks)  # type: Float[Array, "batch_size", "embed_dim"]
        labels = labels.expand(outputs.shape[0], -1)  # extend to batch dimension

        # Calculate the mean squared error loss
        loss = nn.functional.mse_loss(outputs, labels)

        # Introduce the constraint of sum of the emotions should be 1
        # the col 21 to 27 are the emotion columns
        emotion_sum = torch.sum(outputs[:, 21:28], dim=1)
        emo_loss = nn.functional.mse_loss(emotion_sum, torch.ones_like(emotion_sum)) * self.emotion_constraint_penalty

        weights = torch.softmax(self.memory_weights, dim=0)

        loss += emo_loss
        self.log("train_emo_loss", emo_loss)
        self.log("train_loss", loss)
        self.log("short_term_weight", weights[0])
        self.log("long_term_weight", weights[1])

        return loss

    def validation_step(self, batch, batch_idx):
        # type: (Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], int) -> None
        """Validation step of the SpeechToExpressionModel
        Lightning calls this method during the validation loop.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A batch of input data
            batch_idx (int): The index of the batch
        """

        frame_features, global_frame_features, frame_masks, global_frame_masks, labels = batch
        outputs = self(frame_features, global_frame_features, frame_masks, global_frame_masks) # type: Float[Array, "batch_size", "embed_dim"]
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
