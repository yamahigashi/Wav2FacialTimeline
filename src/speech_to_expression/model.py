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
        # type: (FeatBatch1st, MaskBatch1st, Float[Array, "batch"]) -> FeatBatch1st
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


class NoiseDecoder(nn.Module):
    def __init__(self, config, output_dim, embed_dim):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        num_heads = find_closest_divisible_num_heads(self.hidden_dim, config.head_num)

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
        past_frames,
        current_x,
        ts,
        st_features,
        lt_features,
        past_mask,
        frame_masks,
        global_frame_masks,
        current_short_frame,
        current_long_frame,
    ):
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

        st_encoded = self.short_positional_encoding(st_features, current_short_frame)
        lt_encoded = self.long_positional_encoding(lt_features, current_long_frame)

        estimated = self.diffusion(
            past_frames,
            current_x,
            ts,
            st_encoded,
            lt_encoded,
            past_mask,
            frame_masks,
            global_frame_masks
        )

        return estimated

    def generate(
        self,
        past_frames,
        st_features,
        lt_features,
        past_frame_masks,
        frame_masks,
        global_frame_masks,
        current_short_frame,
        current_long_frame,
        num_inference_steps=20,
    ):

        batch_size, _, embed_dim = st_features.size()

        st_encoded = self.short_positional_encoding(st_features, current_short_frame)
        lt_encoded = self.long_positional_encoding(lt_features, current_long_frame)

        generated_output = self.diffusion.generate(
            past_frames,
            st_encoded,
            lt_encoded,
            past_frame_masks,
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
        loss_noisy = torch.nn.functional.mse_loss(pred_noise, noisy_x[:, -1, :])
        return loss_noisy

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
            input_frames,
            st_features,
            lt_features,
            input_frames_mask,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
        ) = batch

        batch_size = input_frames.size(0)
        labels = input_frames[:, -1:, :]
        past_frames = input_frames[:, :-1, :]  # 最後のフレームは現在フレームなので除外
        past_frame_masks = input_frames_mask[:, :-1]  # 最後のフレームは現在フレームなので除外

        device = st_features.device
        noise = torch.randn_like(labels).to(device)
        steps = torch.randint(
            0,
            self.diffusion.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device
        )

        # ノイズを加えたデータの生成
        noisy_x = self.diffusion.scheduler.add_noise(labels, noise, steps)

        # モデルの順伝播
        residual = self(
            past_frames,
            noisy_x,
            steps,
            st_features,
            lt_features,
            past_frame_masks,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
        )
        mean = torch.mean(residual)
        std = torch.std(residual)
        self.log("residual mean", mean)
        self.log("residual std", std)

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
            input_frames,
            st_features,
            lt_features,
            input_frames_mask,
            frame_masks,
            global_frame_masks,
            current_short_frame,
            current_long_frame,
        ) = batch

        batch_size = input_frames.size(0)
        labels = input_frames[:, -1:, :]
        past_frames = input_frames[:, :-1, :]  # 最後のフレームは現在フレームなので除外
        past_frame_masks = input_frames_mask[:, :-1]  # 最後のフレームは現在フレームなので除外

        device = st_features.device
        noise = torch.randn_like(labels).to(device)
        steps = torch.randint(
            0,
            self.diffusion.scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device
        )

        noisy_x = self.diffusion.scheduler.add_noise(labels, noise, steps)

        # モデルの順伝播
        residual = self(
            past_frames,
            noisy_x,
            steps,
            st_features,
            lt_features,
            past_frame_masks,
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
