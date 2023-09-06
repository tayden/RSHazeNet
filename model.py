import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=to_2tuple(3), padding=1, padding_mode='reflect', bias=False),
                                  nn.PixelUnshuffle(2))
            
    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=to_2tuple(3), padding=1, padding_mode='reflect', bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


# Cross-stage Multi-scale Interaction (CMIM) module
class CMFI(nn.Module):
    def __init__(self, dim, bias=False):
        super(CMFI, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))

        self.norm_1 = LayerNorm2d(dim)
        self.norm_2 = LayerNorm2d(dim * 2)

        self.q_1 = nn.Sequential(
            Downsample(dim),
            nn.Conv2d(dim * 2, dim * 2, kernel_size=to_2tuple(1), bias=bias)
        )

        self.v_1 = nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias)
        self.k_2 = nn.Conv2d(dim * 2, dim, kernel_size=to_2tuple(1), bias=bias)
        self.v_2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=to_2tuple(1), bias=bias)

        self.proj_1 = nn.Conv2d(dim * 2, dim, kernel_size=to_2tuple(1), bias=bias)
        self.proj_2 = nn.Conv2d(dim, dim * 2, kernel_size=to_2tuple(1), bias=bias)

    def forward(self, x_1, x_2):
        input_1, input_2 = x_1, x_2
        x_1 = self.norm_1(x_1)
        x_2 = self.norm_2(x_2)

        b, c, h, w = x_2.shape
        q_1 = self.q_1(x_1).reshape(b, c, h * w)
        k_2 = self.k_2(x_2).reshape(b, c // 2, h * w)

        q_1 = F.normalize(q_1, dim=-1)
        k_2 = F.normalize(k_2, dim=-1)

        v_1 = self.v_1(x_1).reshape(b, c // 2, (h * 2) * (w * 2))
        v_2 = self.v_2(x_2).reshape(b, c, h * w)

        attn = (q_1 @ k_2.transpose(-2, -1)) * self.alpha

        attn_1 = attn.softmax(dim=-1)
        attn_2 = attn.transpose(-1, -2).softmax(dim=-1)

        x_1 = (attn_1 @ v_1).reshape(b, c, h * 2, w * 2)
        x_2 = (attn_2 @ v_2).reshape(b, c // 2, h, w)

        x_1 = self.proj_1(x_1) + input_1
        x_2 = self.proj_2(x_2) + input_2

        return x_1, x_2


# Intra-stage Transposed Fusion (ITFM) module
class IFTE(nn.Module):
    def __init__(self, dim, bias=False):
        super(IFTE, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1))

        self.norm_dec = LayerNorm2d(dim)
        self.norm_skip = LayerNorm2d(dim)

        self.qk_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.qk = nn.Conv2d(dim * 2, dim * 2, kernel_size=to_2tuple(1), bias=False)

        self.v = nn.Conv2d(dim * 2, dim, kernel_size=to_2tuple(1), bias=bias)

        self.proj_out = nn.Conv2d(dim, dim, kernel_size=to_2tuple(1), bias=bias)

    def forward(self, x):
        x_dec, x_skip = x
        x_dec = self.norm_dec(x_dec)
        x_skip = self.norm_skip(x_skip)

        b, c, h, w = x[0].shape
        x = torch.cat((x_dec, x_skip), dim=1)

        q, k = self.qk(self.qk_avg_pool(x)).chunk(2, dim=1)
        v = self.v(x)
        q = q.reshape(b, c, 1)
        k = k.reshape(b, c, 1)
        v = v.reshape(b, c, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.alpha
        attn = attn.softmax(dim=-1)
        x = (attn @ v).reshape(b, c, h, w)

        x = self.proj_out(x)

        return x


class BasicBlock(nn.Module):
    def __init__(self, dim, division_ratio=4):
        super(BasicBlock, self).__init__()
        self.dim = dim
        self.dim_partial = int(dim // division_ratio)
        hidden_features = int(dim * 4)

        self.conv_1 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(1))
        self.conv_3 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(3), padding=3, dilation=3, groups=self.dim_partial)
        self.conv_5 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(5), padding=6, dilation=3, groups=self.dim_partial)
        self.conv_7 = nn.Conv2d(self.dim_partial, self.dim_partial, kernel_size=to_2tuple(7), padding=9, dilation=3, groups=self.dim_partial)

        self.mlp = self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden_features, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_features, dim, 1, bias=False)
        )

        layer_scale_init_value = 0.
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        input_ = x
        x_1, x_2, x_3, x_4 = torch.split(x, [self.dim_partial, self.dim_partial, self.dim_partial, self.dim_partial], dim=1)
        x_1 = self.conv_1(x_1)
        x_2 = self.conv_3(x_2)
        x_3 = self.conv_5(x_3)
        x_4 = self.conv_7(x_4)

        x = torch.cat((x_1, x_2, x_3, x_4), 1)
        x = self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x) + input_

        return x


class RSHazeNet(nn.Module):
    def __init__(self, in_chans=3, out_chans=4, dim=32, depths=(2, 3, 4)):
        super(RSHazeNet, self).__init__()

        self.patch_embed_level_1 = OverlapPatchEmbed(in_c=in_chans, embed_dim=dim, bias=False)
        self.skip_connection_level_1_pre = nn.Sequential(*[BasicBlock(dim) for _ in range(depths[0] // 2)])

        self.skip_connection_level_1_post = nn.Sequential(*[BasicBlock(dim) for _ in range(depths[0] // 2)])

        self.down_level_2 = Downsample(dim)
        self.skip_connection_level_2_pre = nn.Sequential(*[BasicBlock(int(dim * 2 ** 1)) for _ in range(depths[1] // 3)])

        self.skip_connection_level_2_mid = nn.Sequential(*[BasicBlock(int(dim * 2 ** 1)) for _ in range(depths[1] // 3)])

        self.skip_connection_level_2_post = nn.Sequential(*[BasicBlock(int(dim * 2 ** 1)) for _ in range(depths[1] // 3)])

        self.down_level_3 = Downsample(int(dim * 2 ** 1))
        self.skip_connection_level_3_pre = nn.Sequential(*[BasicBlock(int(dim * 2 ** 2)) for _ in range(depths[2] // 2)])

        self.skip_connection_level_3_post = nn.Sequential(*[BasicBlock(int(dim * 2 ** 2)) for _ in range(depths[2] // 2)])

        self.up_level_3 = Upsample(int(dim * 2 ** 2))
        self.up_level_2 = Upsample(int(dim * 2 ** 1))

        self.cmfi_level_1_2 = CMFI(dim)
        self.cmfi_level_2_3 = CMFI(int(dim * 2 ** 1))
        self.ifte_level_2 = IFTE(int(dim * 2 ** 1))
        self.ifte_level_1 = IFTE(dim)

        self.output_level_1 = nn.Conv2d(dim, out_chans, kernel_size=to_2tuple(3), padding=1, padding_mode='reflect', bias=False)

    def forward_features(self, x):
        x = self.patch_embed_level_1(x)
        skip_level_1_pre = self.skip_connection_level_1_pre(x)

        x = self.down_level_2(x)
        skip_level_2_pre = self.skip_connection_level_2_pre(x)

        x = self.down_level_3(x)
        latent_pre = self.skip_connection_level_3_pre(x)

        skip_level_2_pre, latent_pre = self.cmfi_level_2_3(skip_level_2_pre, latent_pre)

        skip_level_2_mid = self.skip_connection_level_2_mid(skip_level_2_pre)

        skip_level_1_pre, skip_level_2_mid = self.cmfi_level_1_2(skip_level_1_pre, skip_level_2_mid)

        skip_level_2 = self.skip_connection_level_2_post(skip_level_2_mid)
        skip_level_1 = self.skip_connection_level_1_post(skip_level_1_pre)

        latent_post = self.skip_connection_level_3_post(latent_pre)
        x = self.up_level_3(latent_post)

        x = self.ifte_level_2([x, skip_level_2]) + x
        x = self.up_level_2(x)

        x = self.ifte_level_1([x, skip_level_1]) + x
        x = self.output_level_1(x)
        return x

    def forward(self, x):
        input_ = x
        _, _, h, w = input_.shape

        x = self.forward_features(x)
        K, B = torch.split(x, [1, 3], dim=1)

        x = K * input_ - B + input_
        x = x[:, :, :h, :w]

        return x


if __name__ == '__main__':
    x = torch.randn((1, 3, 512, 512)).cuda()
    net = RSHazeNet().cuda()

    from thop import profile, clever_format

    flops, params = profile(net, (x,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)