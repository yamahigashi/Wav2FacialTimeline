import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import diffusers.schedulers

import config as conf

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
    FeatSeq1st = Float[Array, "seq_len", "batch", "embed_dim"]
    FeatBatch1st = Float[Array, "batch", "seq_len", "embed_dim"]
    MaskSeq1st = Float[Array, "seq_len", "batch"]
    MaskBatch1st = Float[Array, "batch", "seq_len"]


def find_closest_divisible_num_heads(embed_dim, target_num_heads, max_heads=16):
    # type: (int, int, int) -> int
    """
    embed_dimを割り切れる、target_num_headsに最も近いnum_headsを見つける関数。
    
    Args:
        embed_dim (int): 埋め込みの次元数
        target_num_heads (int): 目標のヘッド数
        max_heads (int): 最大のヘッド数の制限（デフォルトは16）

    Returns:
        int: target_num_headsに最も近い割り切れるnum_headsの値
    """
    # 上限を制限して、1 から max_heads の範囲で割り切れる num_heads を探す
    possible_heads = []
    for num_heads in range(1, max_heads + 1):
        if embed_dim % num_heads == 0:
            possible_heads.append(num_heads)

    # 最も target_num_heads に近い値を選択
    closest_num_heads = min(possible_heads, key=lambda x: abs(x - target_num_heads))
    
    return closest_num_heads


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
        num_heads = find_closest_divisible_num_heads(embed_dim, num_heads)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, st_features, st_key_padding_mask, timestep_emb):
        # type: (FeatSeq1st, MaskBatch1st, Float[Array, "batch"]) -> FeatSeq1st
        """Forward pass of the ShortTermTemporalModule

        Args:
            st_features (torch.Tensor): The input frame features
        """

        seq_len, batch_size, _ = st_features.size()
        timestep_emb_expanded = timestep_emb.unsqueeze(0).expand(seq_len, batch_size, -1)
        st_features = torch.cat([st_features, timestep_emb_expanded], dim=-1)

        attn_output, _ = self.attention(
            st_features, st_features, st_features,
            key_padding_mask=st_key_padding_mask
        )
        output = self.layer_norm(st_features + attn_output)
        return output


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

        num_heads = find_closest_divisible_num_heads(embed_dim, num_heads)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, lt_features, src_key_padding_mask, timestep_emb):
        # type: (FeatSeq1st, MaskBatch1st, Float[Array, "batch"]) -> FeatSeq1st
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

        seq_len, batch_size, _ = lt_features.size()
        timestep_emb_expanded = timestep_emb.unsqueeze(0).expand(seq_len, batch_size, -1)
        lt_features = torch.cat([lt_features, timestep_emb_expanded], dim=-1)

        attn_output = self.transformer_encoder(
            lt_features,
            src_key_padding_mask=src_key_padding_mask
        )
        
        output = self.layer_norm(attn_output)
        
        return output


class PositionalEncoding(pl.LightningModule):
    """Positional Encoding Module"""

    def __init__(self, embed_dim, max_len):
        # type: (int, int) -> None

        super(PositionalEncoding, self).__init__()

        # Absolute positional encoding using sine and cosine functions
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(1)  # (max_len, 1, embed_dim)
        self.register_buffer("pe", pe)

        # Temporal encoding (1) for past, (2) for present, and (3) for future
        self.temporal_embeddings = nn.Embedding(3, embed_dim)

    def forward(self, x, current_frame_idx):
        # type: (FeatSeq1st, Int[Array, "batch"]) -> FeatSeq1st
        """
        Args:
            x (torch.Tensor): 入力テンソル（形状： (seq_len, batch_size, embed_dim)）
            current_frame_idx (torch.Tensor): 各バッチの現在フレームのインデックス（形状： (batch_size,)）

        Returns:
            torch.Tensor: 位置エンコーディングと時間的エンコーディングが付加された入力テンソル
        """
        seq_len, batch_size, embed_dim = x.size()
        device = x.device

        # Get the positional encoding
        pe = self.pe[:seq_len, :].to(device)  # (seq_len, 1, embed_dim)
        pe = pe.expand(-1, batch_size, -1)    # (seq_len, batch_size, embed_dim)

        # Calculate the temporal indices
        frame_indices = torch.arange(seq_len, device=device).unsqueeze(1)  # type: Int[Array, "seq_len", 1, "batch"]
        frame_indices = frame_indices.expand(-1, batch_size)  # type: Int[Array, "seq_len", "batch", "embed_dim"]
        current_frame_idx_expanded = current_frame_idx.unsqueeze(0).expand(seq_len, batch_size)

        temporal_indices = torch.zeros((seq_len, batch_size), dtype=torch.long, device=device)
        temporal_indices[frame_indices < current_frame_idx_expanded] = 0  # past
        temporal_indices[frame_indices == current_frame_idx_expanded] = 1  # present
        temporal_indices[frame_indices > current_frame_idx_expanded] = 2  # future

        # Get the temporal embeddings
        temporal_encodings = self.temporal_embeddings(temporal_indices)  # (seq_len, batch_size, embed_dim)

        # Add the positional and temporal encodings
        x = x + pe + temporal_encodings

        return x


class NoiseDecoder(nn.Module):

    def __init__(self, config, output_dim, embed_dim):
        # type: (conf.NoiseDecoderConfig, int, int) -> None

        super().__init__()
        self.hidden_dim = config.hidden_dim
        num_heads = find_closest_divisible_num_heads(self.hidden_dim, config.head_num)

        # Project input features to hidden dimensions
        self.query_proj = nn.Linear(output_dim * 2, self.hidden_dim)
        self.key_proj = nn.Linear(embed_dim, self.hidden_dim)
        self.value_proj = nn.Linear(embed_dim, self.hidden_dim)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="relu"
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.layer_num
        )

        # Output layer
        self.output_layer = nn.Linear(self.hidden_dim, output_dim)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, xcur, xnext, st_latent, lt_latent):
        """
        Args:
            xcur (Tensor): Current noisy input (batch_size, output_dim)
            xnext (Tensor): Next noisy input (batch_size, output_dim)
            st_latent (Tensor): Short-term features (seq_len_s, batch_size, embed_dim)
            lt_latent (Tensor): Long-term features (seq_len_l, batch_size, embed_dim)
        
        Returns:
            Tensor: Estimated noise (batch_size, output_dim)
        """
        # Prepare queries
        queries = torch.cat([xcur, xnext], dim=-1)  # (batch_size, output_dim * 2)
        queries = self.query_proj(queries)          # (batch_size, hidden_dim)
        queries = queries.unsqueeze(0)              # (1, batch_size, hidden_dim)

        # Prepare keys and values
        # Concatenate short-term and long-term features along the sequence dimension
        kv_features = torch.cat([st_latent, lt_latent], dim=0)  # (seq_len, batch_size, embed_dim)
        keys = self.key_proj(kv_features)    # (seq_len, batch_size, hidden_dim)
        # values = self.value_proj(kv_features)  # (seq_len, batch_size, hidden_dim)

        # Transformer decoder expects target sequence first
        # Since we have only one query, the target sequence length is 1
        # The memory (keys and values) is the concatenated temporal features

        # Pass through the Transformer decoder
        decoder_output = self.transformer_decoder(
            tgt=queries,
            memory=keys,
            tgt_mask=None,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None  # You can provide masks if needed
        )

        decoder_output = self.layer_norm(decoder_output)  # (1, batch_size, hidden_dim)
        decoder_output = decoder_output.squeeze(0)        # (batch_size, hidden_dim)

        # Output layer
        output = self.output_layer(decoder_output)  # (batch_size, output_dim)

        return output


class GaussianDiffusion(pl.LightningModule):
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
        noise_decoder_module: NoiseDecoder,
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
        cur_x,
        next_x,
        ts,
        st_features,
        lt_features,
        st_key_padding_mask,
        lt_key_padding_mask
    ):
        # type: (Label, Label, Int[Array, "batch"], FeatSeq1st, FeatSeq1st, MaskBatch1st, MaskBatch1st) -> Label
     
        # Ensure ts is within the valid range
        assert ts.max() < self.train_timesteps_num, "ts contains indices out of bounds for the embedding layer."
        timestep_emb = self.timestep_embedding(ts)
        st_latent = self.short_term_module(st_features, st_key_padding_mask, timestep_emb)
        lt_latent = self.long_term_module(lt_features, lt_key_padding_mask, timestep_emb)

        estimated = self.model(cur_x, next_x, st_latent, lt_latent)

        return estimated

    def generate(
        self,
        input_last_x,
        st_encoded,
        lt_encoded,
        frame_masks,
        global_frame_masks,
        current_short_frame,
        current_long_frame,
        num_inference_steps=30
    ):
        # type: (Label, FeatBatch1st, FeatBatch1st, MaskBatch1st, MaskBatch1st, Int[Array, "batch"], Int[Array, "batch"], int) -> Label

        batch_size = input_last_x.size(0)
        device = input_last_x.device

        # 推論用のスケジューラを設定
        self.scheduler.set_timesteps(
            num_inference_steps,
            device=device
        )

        # 初期入力をノイズから開始
        x = torch.randn_like(input_last_x).to(device)

        for t in self.scheduler.timesteps:
            ts = t.expand(batch_size).to(device)

            timestep_emb = self.timestep_embedding(ts)
            st_latent = self.short_term_module(st_encoded, frame_masks, timestep_emb)
            lt_latent = self.long_term_module(lt_encoded, global_frame_masks, timestep_emb)

            # ノイズの予測
            model_output = self.model(input_last_x, x, st_latent, lt_latent)

            # 前のサンプルを計算
            x = self.scheduler.step(
                model_output,
                t,
                x
            ).prev_sample

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
        self.noise_decoder_module = NoiseDecoder(
            config.diffusion.noise_decoder_config,
            config.output_dim,
            embed_dim,
        )

        self.diffusion = GaussianDiffusion(
            config.diffusion,
            config.embed_dim,
            config.output_dim,
            config.time_embed_dim,
            self.short_term_module,
            self.long_term_module,
            self.noise_decoder_module,
        )

        self.lr = config.lr

        self.save_hyperparameters()

    def forward(
        self,
        input_last_x,
        current_x,
        ts,
        st_features,
        lt_features,
        frame_masks,
        global_frame_masks,
        current_short_frame,
        current_long_frame,
    ):
        # type: (Label, Label, int, FeatBatch1st, FeatBatch1st, MaskBatch1st, MaskBatch1st, Int[Array, "batch"], Int[Array, "batch"]) -> Label
        """Forward pass of the SpeechToExpressionModel.

        First, the short-term temporal attention module is applied to the frame features.
        Then, the long-term temporal attention module is applied to the global frame features.
        The outputs of the short-term and long-term modules are concatenated and passed to the
        biased conditional self-attention module. Finally, the output is passed through the diffusion
        model to denoise and refine the output.

        The bias factor in the biased conditional self-attention module is conditioned on the input condition.

        Args:
            st_features (torch.Tensor): The input frame features
            lt_features (torch.Tensor): The input global frame features

        Returns:
            torch.Tensor: The output features
        """
        batch_size, _, embed_dim = st_features.size()

        # Permute the input features to the correct shape
        # The nn.Transformer expects the input to be sequence-first data. Then,
        # (batch_size, seq_len, embed_dim) -> （(sequence_length, batch_size, embed_dim)）
        st_features = st_features.permute(1, 0, 2)
        lt_features = lt_features.permute(1, 0, 2)

        st_encoded = self.short_positional_encoding(st_features, current_short_frame)
        lt_encoded = self.long_positional_encoding(lt_features, current_long_frame)

        estimated = self.diffusion(
            input_last_x,
            current_x,
            ts,
            st_encoded,
            lt_encoded,
            frame_masks,
            global_frame_masks
        )

        return estimated

    def generate(
        self,
        input_last_x,
        st_features,
        lt_features,
        frame_masks,
        global_frame_masks,
        current_short_frame,
        current_long_frame,
        num_inference_steps=20,
    ):
        # type: (Label, FeatBatch1st, FeatBatch1st, MaskBatch1st, MaskBatch1st, Int[Array, "batch"], Int[Array, "batch"], int) -> Label

        batch_size, _, embed_dim = st_features.size()

        # Permute the input features to the correct shape
        # The nn.Transformer expects the input to be sequence-first data. Then,
        # (batch_size, seq_len, embed_dim) -> （(sequence_length, batch_size, embed_dim)）
        st_features = st_features.permute(1, 0, 2)
        lt_features = lt_features.permute(1, 0, 2)

        st_encoded = self.short_positional_encoding(st_features, current_short_frame)
        lt_encoded = self.long_positional_encoding(lt_features, current_long_frame)

        generated_output = self.diffusion.generate(
            input_last_x,
            st_encoded,
            lt_encoded,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
            num_inference_steps
        )
        return generated_output

    def custom_loss(self, pred_noise, noisy_x, noise, steps):
        # type: (Label, Label, Label, Int[Array, "batch"]) -> torch.Tensor
        """Custom loss function for the SpeechToExpressionModel

        We already know that the labels have some constraints. We can use this information to
        create a custom loss function.

        The constraints are:
            Range of [0:28] is 0 - 1.0
            Sum of [21:28] is 1.0
        """
        loss_noisy = torch.nn.functional.mse_loss(pred_noise, noise)

        # Calculate x0_pred
        alpha_t = self.diffusion.scheduler.alphas_cumprod[steps].to(noisy_x.device).view(-1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        x0_pred = (noisy_x - sqrt_one_minus_alpha_t * pred_noise) / sqrt_alpha_t

        # Condition 1: Range of [0:28] is 0 - 1.0
        range_penalty = torch.relu(x0_pred[:, :28] - 1).mean() + torch.relu(-x0_pred[:, :28]).mean()

        # Condition 2: Sum of [21:28] is 1.0
        sum_constraint_penalty = torch.abs(x0_pred[:, 21:28].sum(dim=1) - 1).mean()

        total_loss = (
            loss_noisy +
            0.2 * range_penalty +
            0.2 * sum_constraint_penalty
        )
     
        self.log("range_penalty", range_penalty)
        self.log("sum_constraint_penalty", sum_constraint_penalty)

        return total_loss

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
            st_features,
            lt_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
            labels
        ) = batch

        device = st_features.device
        noise = torch.randn_like(labels).to(device)
        steps = torch.randint(
            0,
            self.diffusion.scheduler.config.num_train_timesteps,
            (input_last_x.size(0),),
            device=device
        )

        # ノイズを加えたデータの生成
        noisy_x = self.diffusion.scheduler.add_noise(labels, noise, steps)

        # モデルの順伝播
        residual = self(
            input_last_x,
            noisy_x,
            steps,
            st_features,
            lt_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
        )

        # 損失の計算
        loss = self.custom_loss(residual, noisy_x, noise, steps)

        self.log("train_loss", loss, prog_bar=True)
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
            st_features,
            lt_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
            labels
        ) = batch

        device = st_features.device
        noise = torch.randn_like(labels).to(device)
        steps = torch.randint(
            0,
            self.diffusion.scheduler.config.num_train_timesteps,
            (input_last_x.size(0),),
            device=device
        )

        noisy_x = self.diffusion.scheduler.add_noise(labels, noise, steps)

        # モデルの順伝播
        residual = self(
            input_last_x,
            noisy_x,
            steps,
            st_features,
            lt_features,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
        )

        val_loss = self.custom_loss(residual, noisy_x, noise, steps)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        # type: () -> Dict[str, torch.optim.Optimizer|torch.optim.lr_scheduler._LRScheduler]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
