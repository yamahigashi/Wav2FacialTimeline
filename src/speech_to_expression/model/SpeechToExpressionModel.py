import torch
import torch.nn as nn
import pytorch_lightning as pl

from model import (
    ShortTermTemporalModule,
    LongTermTemporalModule,
    NoiseDecoder,
    GaussianDiffusion,
    PositionalEncoding,
)


import typing
if typing.TYPE_CHECKING:
    from .. import config as conf  # noqa: F401
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
        
        self.short_positional_encoding = PositionalEncoding.PositionalEncoding(config.embed_dim, short_term_window)
        self.long_positional_encoding = PositionalEncoding.PositionalEncoding(config.embed_dim, long_term_window)

        embed_dim = config.embed_dim + config.time_embed_dim
        self.short_term_module = ShortTermTemporalModule.ShortTermTemporalModule(embed_dim, config.st.head_num)
        self.long_term_module = LongTermTemporalModule.LongTermTemporalModule(embed_dim, config.lt.head_num, config.lt.layer_num)
        self.noise_decoder_module = NoiseDecoder.NoiseDecoder(
            config.diffusion.noise_decoder_config,
            config.output_dim,
            embed_dim,
        )

        self.diffusion = GaussianDiffusion.GaussianDiffusion(
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
        pred_noise = pred_noise.unsqueeze(1)
        loss_noisy = torch.nn.functional.mse_loss(pred_noise, noise)
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
        pred_noise = self(
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
        mean = torch.mean(pred_noise)
        std = torch.std(pred_noise)
        self.log("pred_noise mean", mean)
        self.log("pred_noise std", std)

        # 損失の計算
        loss = self.custom_loss(pred_noise, noisy_x, noise, steps)

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
        pred_noise = self(
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

        val_loss = self.custom_loss(pred_noise, noisy_x, noise, steps)
        self.log("val_loss", val_loss)

    def configure_optimizers(self):
        # type: () -> Dict[str, torch.optim.Optimizer|torch.optim.lr_scheduler._LRScheduler]
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.99)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
