import dataclasses

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl

import config as conf

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

    Label = Float[Array, "batch_size", "output_dim"]
    Seq1st = Float[Array, "seq_len", "batch", "embed_dim"]
    Batch1st = Float[Array, "batch", "seq_len", "embed_dim"]


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

    def forward(self, frame_features, key_padding_mask, timestep_emb):
        # type: (Float[Array, "seq_len", "batch", "embed_dim"], Float[Array, "batch", "seq_len"], Float[Array, "batch"]) -> Float[Array, "seq_len", "batch", "embed_dim"]
        """Forward pass of the ShortTermTemporalModule

        Args:
            frame_features (torch.Tensor): The input frame features
        """

        # TODO: this should be extracted from the input
        seq_len, batch_size, _ = frame_features.size()
        timestep_emb_expanded = timestep_emb.unsqueeze(0).expand(seq_len, batch_size, -1)
        frame_features = torch.cat([frame_features, timestep_emb_expanded], dim=-1)

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

    def forward(self, global_frame_features, src_key_padding_mask, timestep_emb):
        # type: (Float[Array, "seq_len", "batch", "embed_dim"], Float[Array, "batch", "seq_len"], Float[Array, "batch"]) -> Float[Array, "seq_len", "batch", "embed_dim"]
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

        # TODO: this should be extracted from the input
        seq_len, batch_size, _ = global_frame_features.size()
        timestep_emb_expanded = timestep_emb.unsqueeze(0).expand(seq_len, batch_size, -1)
        global_frame_features = torch.cat([global_frame_features, timestep_emb_expanded], dim=-1)

        attn_output = self.transformer_encoder(
            global_frame_features,
            src_key_padding_mask=src_key_padding_mask
        )
        
        output = self.layer_norm(attn_output)
        
        return output


class PositionalEncoding(pl.LightningModule):
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


class SiLU(nn.Module):

    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)


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
        short_term_module: ShortTermTemporalModule,
        long_term_module: LongTermTemporalModule,
        st_window: int,
        lt_window: int
    ):

        super().__init__()

        self.time_step_num = config.time_step_num
        self.time_embed_dim = time_embed_dim
        self.estimate_mode = config.estimate_mode

        self.timestep_embedding = nn.Embedding(self.time_step_num + 1, self.time_embed_dim)
        self.short_term_module = short_term_module
        self.long_term_module = long_term_module

        self.model = NoiseDecoder(
            config.noise_decorder_config,
            output_dim,
            st_window + lt_window,
            embed_dim + time_embed_dim,
        )

        betas = self._generate_diffusion_schedule()
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas)

        self.register_buffer("betas", torch.tensor(betas))
        self.register_buffer("alphas", torch.tensor(alphas))
        self.register_buffer("alphas_cumprod", torch.tensor(alphas_cumprod))

        self.register_buffer("sqrt_alphas_cumprod", torch.tensor(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.tensor(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer("reciprocal_sqrt_alphas", torch.tensor(np.sqrt(1. / alphas)))
        # self.register_buffer("reciprocal_sqrt_alphas_cumprod", torch.tensor(np.sqrt(1. / alphas_cumprod)))
        # self.register_buffer("reciprocal_sqrt_alphas_cumprod_m1", torch.tensor(np.sqrt(1. / alphas_cumprod -1)))
        # self.register_buffer("remove_noise_coeff", torch.tensor(betas / np.sqrt(1. - alphas_cumprod)))
        # self.register_buffer("sigma", torch.tensor(np.sqrt(betas)))

    def _generate_diffusion_schedule(self, s=0.008):
        # type: (float) -> np.ndarray

        def f(t, T):
            # type: (int, int) -> float
            return (np.cos((t / T + s) / (1 + s) * np.pi / 2)) ** 2

        # Cosine schedule for beta
        # from https://arxiv.org/abs/2102.09672  
        alphas = []
        f0 = f(0, self.time_step_num)

        for t in range(self.time_step_num + 1):
            alphas.append(f(t, self.time_step_num) / f0)
        
        betas = []

        for t in range(1, self.time_step_num + 1):
            betas.append(min(1 - alphas[t] / alphas[t - 1], 0.999))
        return np.array(betas)

    @torch.no_grad()
    def extract(self, a, ts, x_shape):
        b, *_ = ts.shape
        out = a.gather(-1, ts)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def remove_noise(self, xt, pred_noise, ts):
        coef1 = self.extract(self.reciprocal_sqrt_alphas, ts, xt.shape)
        coef2 = self.extract(self.betas / self.sqrt_one_minus_alphas_cumprod, ts, xt.shape)
        x_prev = coef1 * (xt - coef2 * pred_noise)
        return x_prev

    def add_noise(self, x, ts, noise):
        return (
            self.extract(self.sqrt_alphas_cumprod, ts, x.shape) * x +
            self.extract(self.sqrt_one_minus_alphas_cumprod, ts, x.shape) * noise
        )   

    def forward(self, cur_x, next_x, ts, st_features, lt_features, key_padding_mask, global_key_padding_mask):

        timestep_emb = self.timestep_embedding(ts)
        st_latent = self.short_term_module(st_features, key_padding_mask, timestep_emb)
        lt_latent = self.long_term_module(lt_features, global_key_padding_mask, timestep_emb)

        estimated = self.model(cur_x, next_x, st_latent, lt_latent)

        return estimated


class NoiseDecoder(nn.Module):

    def __init__(self, config, output_dim, window_size, embed_dim):
        # type: (conf.NoiseDecoderConfig, int, int, int) -> None

        super().__init__()

        self.hidden_dim = config.hidden_dim
        self.query_proj = nn.Linear(output_dim * 2, config.hidden_dim)
        self.kv_proj = nn.Linear(embed_dim, config.hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            config.hidden_dim,
            num_heads=config.head_num
        )

        self.fin = nn.Linear(config.hidden_dim, output_dim)
  
    def forward(self, xcur, xnext, st_latent, lt_latent):
        # type: (Label, Label, Seq1st, Seq1st) -> Label

        temporal_features = torch.cat([st_latent, lt_latent], dim=0)  # type: Seq1st

        queries = torch.cat([xcur, xnext], dim=-1).unsqueeze(0).float()  # (1, batch_size, 2*output_dim)
        proj_queries = self.query_proj(queries)  # (1, batch_size, hidden_dim)
        proj_kv = self.kv_proj(temporal_features)  # (seq_len, batch_size, hidden_dim)

        attn_output, _ = self.cross_attn(proj_queries, proj_kv, proj_kv)

        x = attn_output.squeeze(0)
        x = self.fin(x)

        return x 


class SpeechToExpressionModel(pl.LightningModule):
    """Temporal Diffusion Speaker Model"""

    def __init__(self, config):
        # type: (conf.SpeechToExpressionConfig) -> None
        """Initialize the SpeechToExpressionModel

        Args:
            embed_dim (int): The input feature dimension
            num_heads (int): The number of heads in the multi-head attention
            num_steps (int): The number of diffusion steps
            num_layers (int): The number of transformer encoder layers
            lr (float): The learning rate for training the model
        """
        super(SpeechToExpressionModel, self).__init__()

        short_term_window = config.st.prev_window + config.st.next_window + 1
        long_term_window = config.lt.prev_window + config.lt.next_window + 1
        
        self.short_positional_encoding = PositionalEncoding(config.embed_dim, short_term_window)
        self.long_positional_encoding = PositionalEncoding(config.embed_dim, long_term_window)

        embed_dim = config.embed_dim + config.time_embed_dim
        self.short_term_module = ShortTermTemporalModule(embed_dim, config.st.head_num)
        self.long_term_module = LongTermTemporalModule(embed_dim, config.lt.head_num, config.lt.layer_num)

        self.diffusion = GaussianDiffusion(
            config.diffusion,
            config.embed_dim,
            config.output_dim,
            config.time_embed_dim,
            self.short_term_module,
            self.long_term_module,
            short_term_window,
            long_term_window
        )

        self.lr = config.lr

        self.memory_weights = nn.Parameter(torch.ones(2))

        self.save_hyperparameters()

    def forward(
        self,
        input_last_x,
        frame_features,
        global_frame_features,
        frame_masks,
        global_frame_masks,
        current_short_frame,
        current_long_frame,
        input_ts=None,
        inference=False
    ):
        # type: (Label, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int|None, bool) -> torch.Tensor|Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
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

        Returns:
            torch.Tensor: The output features
        """
        batch_size, _, embed_dim = frame_features.size()

        # Permute the input features to the correct shape
        # The nn.Transformer expects the input to be sequence-first data. Then,
        # (batch_size, seq_len, embed_dim) -> （(sequence_length, batch_size, embed_dim)）
        frame_features = frame_features.permute(1, 0, 2)
        global_frame_features = global_frame_features.permute(1, 0, 2)

        st_encoded = self.short_positional_encoding(frame_features, current_short_frame)
        lt_encoded = self.long_positional_encoding(global_frame_features, current_long_frame)

        if inference:

            x = torch.randn_like(input_last_x)  # Start from pure noise
            for t in reversed(range(self.diffusion.time_step_num)):
                ts = torch.full((batch_size,), t, device=x.device, dtype=torch.long)

                pred_noise = self.diffusion(
                    input_last_x,
                    x,
                    ts,
                    st_encoded,
                    lt_encoded,
                    frame_masks,
                    global_frame_masks
                )

                x = self.diffusion.remove_noise(x, pred_noise, ts)

                # if t > 0:
                #     # Optionally add noise based on the sampling strategy
                #     noise = torch.randn_like(x)
                #     x = x + self.diffusion.get_noise_level(ts, x.shape) * noise

            return x

        else:
            if input_ts is None:
                input_ts = torch.randint(
                    0,
                    self.diffusion.time_step_num,
                    (batch_size,),
                    device=frame_features.device
                )

            noise = torch.randn_like(input_last_x)
            perturbed_x = self.diffusion.add_noise(input_last_x, input_ts, noise)

            # Model predicts the noise
            pred_noise = self.diffusion(
                input_last_x,
                perturbed_x,
                input_ts,
                st_encoded,
                lt_encoded,
                frame_masks,
                global_frame_masks
            )

            return pred_noise, noise, input_ts

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
            input_last_x,
            frame_features,
            global_frame_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
            labels
        ) = batch

        pred_noise, noise, input_ts = self(
            input_last_x,
            frame_features,
            global_frame_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame
        )

        loss = nn.functional.mse_loss(pred_noise, noise)
        self.log("train_loss", loss)
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
            input_last_x,
            frame_features,
            global_frame_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
            labels
        ) = batch

        outputs, noise, input_ts = self(
            input_last_x,
            frame_features,
            global_frame_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame
        )

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
