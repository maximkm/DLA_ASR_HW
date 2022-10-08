import torch
import torch.nn as nn

from hw_asr.base import BaseModel


def get_op_output_dim(some_op: torch.nn.Module, input_dim: int, in_channels: int) -> (int, int):
    """
    Returns dimensions after 2D operation
    """
    x = torch.randn(10, in_channels, 200, input_dim)
    x = some_op(x)

    output_dim = x.size(3)
    return output_dim


def get_encoder_key_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    convert lengths int tensor to 2-D binary mask tensor
    """
    max_lengths = torch.max(lengths).item()
    bsz = lengths.size(0)

    arange_tensor = torch.arange(max_lengths, device=lengths.device).view(1, max_lengths).expand(bsz, -1)
    encoder_padding_mask = arange_tensor >= lengths.view(bsz, 1).expand(-1, max_lengths)
    return encoder_padding_mask


class VGGBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            conv_kernel_size,
            num_convs,
            input_dim,
            pool_kernel_size,
            conv_stride=1,
            layer_norm=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layers = nn.ModuleList()
        padding = conv_kernel_size // 2
        for k in range(num_convs):
            in_channels = self.in_channels if k == 0 else self.out_channels
            conv_op = nn.Conv2d(
                in_channels,
                out_channels,
                conv_kernel_size,
                conv_stride,
                padding,
            )
            nn.init.xavier_normal_(conv_op.weight.data)
            nn.init.zeros_(conv_op.bias.data)
            self.layers.append(conv_op)

            if layer_norm:
                output_dim = get_op_output_dim(conv_op, input_dim, in_channels)
                self.layers.append(nn.LayerNorm(output_dim))
                input_dim = output_dim

            self.layers.append(nn.ReLU())

        if pool_kernel_size is not None:
            pool_op = nn.MaxPool2d(kernel_size=pool_kernel_size, ceil_mode=True)
            self.layers.append(pool_op)
            self.output_dim = get_op_output_dim(pool_op, input_dim, out_channels)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CTCTransformerEncoder(BaseModel):
    def __init__(
            self,
            n_feats,
            n_class,
            vggblock_config,
            transformer_config,
            in_channels=1,
            encoder_output_dim=512,
            **batch
    ):
        super().__init__(n_feats, n_class, **batch)
        self.n_feats = n_feats
        self.n_class = n_class
        self.in_channels = in_channels
        self.pool_kernel_sizes = []
        input_dim = n_feats

        self.conv_layers = nn.ModuleList()
        for config in eval(vggblock_config):
            out_channels, conv_kernel_size, num_convs, pool_kernel_size = config
            self.conv_layers.append(
                VGGBlock(in_channels, out_channels, conv_kernel_size, num_convs, input_dim, pool_kernel_size)
            )
            in_channels = out_channels
            input_dim = self.conv_layers[-1].output_dim
            self.pool_kernel_sizes.append(pool_kernel_size)

        transformer_input_dim = input_dim * in_channels
        transformer_config = eval(transformer_config)
        self.transformer_layers = nn.ModuleList()

        if input_dim != transformer_config[0][0]:
            self.transformer_layers.append(nn.Linear(transformer_input_dim, transformer_config[0][0]))
        self.transformer_layers.append(nn.TransformerEncoderLayer(*transformer_config[0], norm_first=True))

        for i in range(1, len(transformer_config)):
            if transformer_config[i - 1][0] != transformer_config[i][0]:
                self.transformer_layers.append(nn.Linear(transformer_config[i - 1][0], transformer_config[i][0]))
            self.transformer_layers.append(nn.TransformerEncoderLayer(*transformer_config[i], norm_first=True))

        self.transformer_layers.extend([
                nn.Linear(transformer_config[-1][0], encoder_output_dim),
                nn.LayerNorm(encoder_output_dim),
        ])

        self.classifier = nn.Linear(encoder_output_dim, n_class)

    def forward(self, spectrogram, spectrogram_length, **batch):
        spectrogram_length = spectrogram_length.to(spectrogram.device)
        spectrogram = spectrogram.transpose(1, 2).contiguous()  # (B, T, C * feat)

        bsz, max_spec_len, _ = spectrogram.size()
        spectrogram = spectrogram.view(bsz, max_spec_len, self.in_channels, self.n_feats)
        spectrogram = spectrogram.transpose(1, 2).contiguous()  # (B, C, T, feat)

        for layer in self.conv_layers:
            spectrogram = layer(spectrogram)

        bsz, _, output_spec_len, _ = spectrogram.size()
        spectrogram = spectrogram.transpose(1, 2).transpose(0, 1)
        spectrogram = spectrogram.contiguous().view(output_spec_len, bsz, -1)  # (T, B, C * feat)

        lengths = spectrogram_length.clone()
        for s in self.pool_kernel_sizes:
            lengths = (lengths.float() / s).ceil().int()

        key_padding_mask = get_encoder_key_padding_mask(lengths)
        if not key_padding_mask.any():
            key_padding_mask = None

        for transformer_layer in self.transformer_layers:
            if isinstance(transformer_layer, nn.TransformerEncoderLayer):
                spectrogram = transformer_layer(spectrogram, None, key_padding_mask)
            else:
                spectrogram = transformer_layer(spectrogram)

        spectrogram = spectrogram.transpose(0, 1)
        spectrogram = self.classifier(spectrogram)

        return {
            "logits": spectrogram,  # (B, T, C)
        }

    def transform_input_lengths(self, input_lengths):
        for s in self.pool_kernel_sizes:
            input_lengths = (input_lengths.float() / s).ceil().int()
        return input_lengths
