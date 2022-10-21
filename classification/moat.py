# --------------------------------------------------------
# MOAT
# Written by ZeChen Wu
# --------------------------------------------------------
""" MOAT
A PyTorch implementation of the paper:'MOAT: Alternating Mobile Convolution and Attention Brings Strong Vision Models'
"""
from typing import Type, Callable, Tuple, Optional, Set, List, Union

import torch
import torch.nn as nn

from timm.models.efficientnet_blocks import SqueezeExcite, DepthwiseSeparableConv
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath


def window_partition(
        input: torch.Tensor,
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Window partition function.

    Args:
        input (torch.Tensor): Input tensor of the shape [B, C, H, W].
        window_size (Tuple[int, int], optional): Window size to be applied. Default (7, 7)

    Returns:
        windows (torch.Tensor): Unfolded input tensor of the shape [B * windows, window_size[0], window_size[1], C].
    """
    # Get size of input
    B, C, H, W = input.shape
    # Unfold input
    windows = input.view(B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1])
    # Permute and reshape to [B * windows, window_size[0], window_size[1], channels]
    windows = windows.permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(
        windows: torch.Tensor,
        original_size: Tuple[int, int],
        window_size: Tuple[int, int] = (7, 7)
) -> torch.Tensor:
    """ Reverses the window partition.

    Args:
        windows (torch.Tensor): Window tensor of the shape [B * windows, window_size[0], window_size[1], C].
        original_size (Tuple[int, int]): Original shape.
        window_size (Tuple[int, int], optional): Window size which have been applied. Default (7, 7)

    Returns:
        output (torch.Tensor): Folded output tensor of the shape [B, C, original_size[0], original_size[1]].
    """
    # Get height and width
    H, W = original_size
    # Compute original batch size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # Fold grid tensor
    output = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    output = output.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    return output


def _gelu_ignore_parameters(
        *args,
        **kwargs
) -> nn.Module:
    """ Bad trick to ignore the inplace=True argument in the DepthwiseSeparableConv of Timm.

    Args:
        *args: Ignored.
        **kwargs: Ignored.

    Returns:
        activation (nn.Module): GELU activation function.
    """
    activation = nn.GELU()
    return activation


class MBConvBlock(nn.Module):
    """
        Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        downscale (bool, optional): If true downscale by a factor of two is performed. Default: False
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        drop_path (float, optional): Dropout rate to be applied during training. Default 0.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            downscale: bool = False,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            drop_path: float = 0.,
            expand_ratio: int = 4.,
            use_se=False,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(MBConvBlock, self).__init__()
        # Save parameter
        self.drop_path_rate: float = drop_path
        if not downscale:
            assert in_channels == out_channels, "If downscaling is utilized input and output channels must be equal."
        if act_layer == nn.GELU:
            act_layer = _gelu_ignore_parameters
        # Make main path
        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            DepthwiseSeparableConv(in_chs=in_channels,
                                   out_chs=int(out_channels * expand_ratio // 2) if downscale else int(
                                       out_channels * expand_ratio),
                                   stride=2 if downscale else 1,
                                   act_layer=act_layer, norm_layer=norm_layer, drop_path_rate=drop_path),
            SqueezeExcite(
                in_chs=int(out_channels * expand_ratio // 2) if downscale else int(out_channels * expand_ratio),
                rd_ratio=0.25) if use_se else nn.Identity(),
            nn.Conv2d(
                in_channels=int(out_channels * expand_ratio // 2) if downscale else int(out_channels * expand_ratio),
                out_channels=out_channels, kernel_size=(1, 1))
        )
        # Make skip path
        self.skip_path = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))
        ) if downscale else nn.Identity()

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H (// 2), W (// 2)] (downscaling is optional).
        """
        output = self.main_path(input)
        if self.drop_path_rate > 0.:
            output = drop_path(output, self.drop_path_rate, self.training)
        output = output + self.skip_path(input)
        return output


def get_relative_position_index(
        win_h: int,
        win_w: int
) -> torch.Tensor:
    """ Function to generate pair-wise relative position index for each token inside the window.
        Taken from Timms Swin V1 implementation.

    Args:
        win_h (int): Window/Grid height.
        win_w (int): Window/Grid width.

    Returns:
        relative_coords (torch.Tensor): Pair-wise relative position indexes [height * width, height * width].
    """
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)]))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)


class RelativeSelfAttention(nn.Module):
    """ Relative Self-Attention similar to Swin V1. Implementation inspired by Timms Swin V1 implementation.
    Args:
        in_channels (int): Number of input channels.
        num_heads (int, optional): Number of attention heads. Default 32
        window_size (Tuple[int, int], optional): Grid/Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
            self,
            in_channels: int,
            num_heads: int = 32,
            window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop: float = 0.
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(RelativeSelfAttention, self).__init__()
        # Save parameters
        self.in_channels: int = in_channels
        self.num_heads: int = in_channels // num_heads
        self.window_size: Tuple[int, int] = window_size
        self.scale: float = num_heads ** -0.5
        self.attn_area: int = window_size[0] * window_size[1]
        # Init layers
        self.qkv_mapping = nn.Linear(in_features=in_channels, out_features=3 * in_channels, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)
        self.softmax = nn.Softmax(dim=-1)
        # Define a parameter table of relative position bias, shape: 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads))

        # Get pair-wise relative position index for each token inside the window
        self.register_buffer("relative_position_index", get_relative_position_index(window_size[0],
                                                                                    window_size[1]))
        # Init relative positional bias
        trunc_normal_(self.relative_position_bias_table, std=.02)

    def _get_relative_positional_bias(
            self
    ) -> torch.Tensor:
        """ Returns the relative positional bias.

        Returns:
            relative_position_bias (torch.Tensor): Relative positional bias.
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(self.attn_area, self.attn_area, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias.unsqueeze(0)

    def forward(
            self,
            input: torch.Tensor
    ) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B_, N, C].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B_, N, C].
        """
        # Get shape of input
        B_, N, C = input.shape
        # Perform query key value mapping
        qkv = self.qkv_mapping(input).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # Scale query
        q = q * self.scale
        # Compute attention maps
        attn = self.softmax(q @ k.transpose(-2, -1) + self._get_relative_positional_bias())
        # Map value with attention maps
        output = (attn @ v).transpose(1, 2).reshape(B_, N, -1)
        # Perform final projection and dropout
        output = self.proj(output)
        output = self.proj_drop(output)
        return output


class MOATAttnetion(nn.Module):
    def __init__(
            self,
            in_channels: int,
            partition_function: Callable,
            reverse_function: Callable,
            img_size: Tuple[int, int] = (224, 224),
            num_heads: int = 32,
            window_size: Tuple[int, int] = (7, 7),
            use_window: bool = False,
            attn_drop: float = 0.,
            drop: float = 0.,
            drop_path: float = 0.,
            norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        """ Constructor method """
        super(MOATAttnetion, self).__init__()
        # Save parameters
        self.use_window = use_window
        self.partition_function: Callable = partition_function
        self.reverse_function: Callable = reverse_function
        if self.use_window:
            self.window_size: Tuple[int, int] = window_size
        else:
            self.window_size: Tuple[int, int] = img_size
        # Init layers
        self.norm_1 = norm_layer(in_channels)
        self.attention = RelativeSelfAttention(
            in_channels=in_channels,
            num_heads=num_heads,
            window_size=self.window_size,
            attn_drop=attn_drop,
            drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W].

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H, W].
        """
        # Save original shape
        B, C, H, W = input.shape
        if self.use_window:
            # Perform partition
            input_partitioned = self.partition_function(input, self.window_size)
            input_partitioned = input_partitioned.view(-1, self.window_size[0] * self.window_size[1], C)
            # Perform normalization, attention, and dropout
            output = input_partitioned + self.drop_path(self.attention(self.norm_1(input_partitioned)))
            # Reverse partition
            output = self.reverse_function(output, (H, W), self.window_size)
        else:
            # flatten: [B, C, H, W] -> [B, C, HW]
            # transpose: [B, C, HW] -> [B, HW, C]
            input_partitioned = input.flatten(2).transpose(1, 2).contiguous()
            output = input_partitioned + self.drop_path(self.attention(self.norm_1(input_partitioned)))
            output = output.transpose(1, 2).contiguous().view(B, C, H, W)
        return output


class MOATBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 partition_function: Callable,
                 reverse_function: Callable,
                 img_size: Tuple[int, int] = (224, 224),
                 num_heads: int = 32,
                 window_size: Tuple[int, int] = (7, 7),
                 use_window: bool = False,
                 downscale: bool = False,
                 attn_drop: float = 0.,
                 drop: float = 0.,
                 drop_path: float = 0.,
                 expand_ratio: float = 4.,
                 act_layer: Type[nn.Module] = nn.GELU,
                 norm_layer: Type[nn.Module] = nn.BatchNorm2d,
                 norm_layer_transformer: Type[nn.Module] = nn.LayerNorm,
                 use_se=False,
                 ):
        super(MOATBlock, self).__init__()
        # Init MBConv
        self.mb_conv = MBConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            downscale=downscale,
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_path=drop_path,
            expand_ratio=expand_ratio,
            use_se=use_se,
        )
        # Init Attention
        self.moat_attention = MOATAttnetion(
            in_channels=out_channels,
            img_size=img_size,
            partition_function=partition_function,
            reverse_function=reverse_function,
            num_heads=num_heads,
            window_size=window_size,
            use_window=use_window,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            norm_layer=norm_layer_transformer
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            input (torch.Tensor): Input tensor of the shape [B, C_in, H, W]

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C_out, H // 2, W // 2] (downscaling is optional)
            :param x:
        """
        output = self.mb_conv(input)
        output = self.moat_attention(output)
        return output


class MOAT(nn.Module):
    """ Implementation of the MOAT proposed in:
            https://arxiv.org/pdf/2210.01820.pdf
    Args:
        in_channels (int, optional): Number of input channels to the convolutional stem. Default 3
        depths (Tuple[int, ...], optional): Depth of each network stage. Default (2, 3, 7, 2)
        channels (Tuple[int, ...], optional): Number of channels in each network stage. Default (96, 192, 384, 768)
        num_classes (int, optional): Number of classes to be predicted. Default 1000
        embed_dim (int, optional): Embedding dimension of the convolutional stem. Default 64
        num_heads (int, optional): Number of attention heads. Default 32
        window_size (Tuple[int, int], optional): Window size to be utilized. Default (7, 7)
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        drop (float, optional): Dropout ratio of output. Default: 0.0
        drop_path (float, optional): Dropout ratio of path. Default: 0.0
        mlp_ratio (float, optional): Ratio of mlp hidden dim to embedding dim. Default: 4.0
        act_layer (Type[nn.Module], optional): Type of activation layer to be utilized. Default: nn.GELU
        norm_layer (Type[nn.Module], optional): Type of normalization layer to be utilized. Default: nn.BatchNorm2d
        norm_layer_transformer (Type[nn.Module], optional): Normalization layer in Transformer. Default: nn.LayerNorm
        global_pool (str, optional): Global polling type to be utilized. Default "avg"
    """

    def __init__(
            self,
            in_channels: int = 3,
            depths: Tuple[int, ...] = (2, 3, 7, 2),
            channels: Tuple[int, ...] = (96, 192, 384, 768),
            img_size: Tuple[int, int] = (224, 224),
            num_classes: int = 1000,
            embed_dim: int = 64,
            num_heads: int = 32,
            use_window: bool = False,
            window_size: Tuple[int, int] = (7, 7),
            attn_drop: float = 0.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.BatchNorm2d,
            norm_layer_transformer=nn.LayerNorm,
            global_pool: str = "avg",
    ) -> None:
        super(MOAT, self).__init__()
        assert len(depths) == len(channels), "For each stage a channel dimension must be given."
        assert global_pool in ["avg", "max"], f"Only avg and max is supported but {global_pool} is given"
        self.num_classes: int = num_classes
        # image size
        self.H = img_size[0]
        self.W = img_size[1]
        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=(3, 3), stride=(2, 2),
                      padding=(1, 1)),
            act_layer(),
            nn.Conv2d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(3, 3), stride=(1, 1),
                      padding=(1, 1)),
            act_layer(),
        )
        # h w 64
        # stochastic depth rate
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        # h w 64 --> h w 96
        # h w 96 --> h w 96
        self.MBConv1 = nn.ModuleList([
            MBConvBlock(in_channels=embed_dim if i == 0 else channels[0],
                        out_channels=channels[0],
                        downscale=i == 0,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        drop_path=dpr[i],
                        expand_ratio=4.,
                        use_se=True)
            for i in range(depths[0])])

        self.MBConv2 = nn.ModuleList([
            MBConvBlock(in_channels=channels[0] if i == 0 else channels[1],
                        out_channels=channels[1],
                        downscale=i == 0,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        drop_path=dpr[i + depths[0]],
                        expand_ratio=4.,
                        use_se=True)
            for i in range(depths[1])])
        self.MOAT1 = nn.ModuleList([
            MOATBlock(in_channels=channels[1] if i == 0 else channels[2],
                      out_channels=channels[2],
                      partition_function=window_partition,
                      reverse_function=window_reverse,
                      img_size=(self.H // 16, self.W // 16),
                      num_heads=num_heads,
                      use_window=use_window,
                      window_size=window_size,
                      downscale=i == 0,
                      attn_drop=attn_drop,
                      drop=drop,
                      drop_path=dpr[i + depths[0] + depths[1]],
                      act_layer=act_layer,
                      norm_layer=norm_layer,
                      norm_layer_transformer=norm_layer_transformer,
                      use_se=False)
            for i in range(depths[2])])
        self.MOAT2 = nn.ModuleList([
            MOATBlock(in_channels=channels[2] if i == 0 else channels[3],
                      out_channels=channels[3],
                      partition_function=window_partition,
                      reverse_function=window_reverse,
                      img_size=(self.H // 32, self.W // 32),
                      num_heads=num_heads,
                      use_window=use_window,
                      window_size=window_size,
                      downscale=i == 0,
                      attn_drop=attn_drop,
                      drop=drop,
                      drop_path=dpr[i + depths[0] + depths[1] + depths[2]],
                      act_layer=act_layer,
                      norm_layer=norm_layer,
                      norm_layer_transformer=norm_layer_transformer,
                      use_se=False)
            for i in range(depths[3])])
        self.global_pool: str = global_pool
        self.head = nn.Linear(channels[-1], num_classes)

    def forward_features(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass of feature extraction.

        Args:
            input (torch.Tensor): Input images of the shape [B, C, H, W].

        Returns:
            output (torch.Tensor): Image features of the backbone.
        """
        # MBConv
        output = input
        for mbconv in self.MBConv1:
            output = mbconv(output)
        for mbconv in self.MBConv2:
            output = mbconv(output)
        # MOAT
        for moat in self.MOAT1:
            output = moat(output)
        for moat in self.MOAT2:
            output = moat(output)
        return output

    def forward_head(self, input: torch.Tensor, pre_logits: bool = False):
        """ Forward pass of classification head.

        Args:
            input (torch.Tensor): Input features
            pre_logits (bool, optional): If true pre-logits are returned

        Returns:
            output (torch.Tensor): Classification output of the shape [B, num_classes].
        """
        if self.global_pool == "avg":
            input = input.mean(dim=(2, 3))
        elif self.global_pool == "max":
            input = torch.amax(input, dim=(2, 3))
        return input if pre_logits else self.head(input)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """ Forward pass

        Args:
            input (torch.Tensor): Input images of the shape [B, C, H, W].

        Returns:
            output (torch.Tensor): Classification output of the shape [B, num_classes].
        """
        output = self.forward_features(self.stem(input))
        output = self.forward_head(output)
        return output


def moat_0(**kwargs) -> MOAT:
    """ MOAT_0 for a resolution of 224 X 224"""
    return MOAT(
        depths=(2, 3, 7, 2),
        channels=(96, 192, 384, 768),
        embed_dim=64,
        **kwargs
    )


def moat_1(**kwargs) -> MOAT:
    """ MOAT_1 for a resolution of 224 X 224"""
    return MOAT(
        depths=(2, 6, 14, 2),
        channels=(96, 192, 384, 768),
        embed_dim=64,
        **kwargs
    )


def moat_2(**kwargs) -> MOAT:
    """ MOAT_2 for a resolution of 224 X 224"""
    return MOAT(
        depths=(2, 6, 14, 2),
        channels=(128, 256, 512, 1024),
        embed_dim=128,
        **kwargs
    )


def moat_3(**kwargs) -> MOAT:
    """ MOAT_3 for a resolution of 224 X 224"""
    return MOAT(
        depths=(2, 12, 28, 2),
        channels=(160, 320, 640, 1280),
        embed_dim=160,
        **kwargs
    )


def moat_4(**kwargs) -> MOAT:
    """ MOAT_0 for a resolution of 224 X 224"""
    return MOAT(
        depths=(2, 12, 28, 2),
        channels=(256, 512, 1024, 2048),
        embed_dim=256,
        **kwargs
    )


def tiny_moat_0(**kwargs) -> MOAT:
    """ tiny_MOAT_0 for a resolution of 224 X 224"""
    return MOAT(
        depths=(2, 3, 7, 2),
        channels=(32, 64, 128, 256),
        embed_dim=32,
        **kwargs
    )


def tiny_moat_1(**kwargs) -> MOAT:
    """ tiny_MOAT_1 for a resolution of 224 X 224"""
    return MOAT(
        depths=(2, 3, 7, 2),
        channels=(40, 80, 160, 320),
        embed_dim=40,
        **kwargs
    )


def tiny_moat_2(**kwargs) -> MOAT:
    """ tiny_MOAT_2 for a resolution of 224 X 224"""
    return MOAT(
        depths=(2, 3, 7, 2),
        channels=(56, 112, 224, 448),
        embed_dim=56,
        **kwargs
    )


def tiny_moat_3(**kwargs) -> MOAT:
    """ tiny_MOAT_3 for a resolution of 224 X 224"""
    return MOAT(
        depths=(2, 3, 7, 2),
        channels=(80, 160, 320, 640),
        embed_dim=56,
        **kwargs
    )


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def test():
    print("*" * 100)
    print("MOAT:")
    print("use_window: False")
    print(f'MAOT_0:{get_n_params(moat_0(num_classes=10))}')
    print(f'MAOT_1:{get_n_params(moat_1(num_classes=10))}')
    print(f'MAOT_2:{get_n_params(moat_2(num_classes=10))}')
    print(f'MAOT_3:{get_n_params(moat_3(num_classes=10))}')
    print(f'MAOT_4:{get_n_params(moat_4(num_classes=10))}')

    print("use_window: True")
    print(f'MAOT_0:{get_n_params(moat_0(use_window=True, num_classes=10))}')
    print(f'MAOT_1:{get_n_params(moat_1(use_window=True, num_classes=10))}')
    print(f'MAOT_2:{get_n_params(moat_2(use_window=True, num_classes=10))}')
    print(f'MAOT_3:{get_n_params(moat_3(use_window=True, num_classes=10))}')
    print(f'MAOT_4:{get_n_params(moat_4(use_window=True, num_classes=10))}')
    print("*" * 100)
    print("-" * 100)
    print("tiny_MOAT:")
    print(f'tiny_MOAT_0:{get_n_params(tiny_moat_0(num_classes=10))}')
    print(f'tiny_MOAT_1:{get_n_params(tiny_moat_1(num_classes=10))}')
    print(f'tiny_MOAT_2:{get_n_params(tiny_moat_2(num_classes=10))}')
    print(f'tiny_MOAT_3:{get_n_params(tiny_moat_3(num_classes=10))}')
    print("-" * 100)


if __name__ == '__main__':
    network = moat_0(use_window=True, num_classes=10)
    input = torch.rand(1, 3, 224, 224)
    output = network(input)
    print(output.shape)
    test()
