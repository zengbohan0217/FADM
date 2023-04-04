import imp
import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from .condition_deal import condition_net
from .util import TimeEmbedding, Swish, ResnetBlocWithAttn, Downsample, Upsample, Block, default, exists


class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        # inner_channel=32,
        inner_channel=64,
        norm_groups=32,
        # channel_mults=(1, 2, 4, 8, 8),
        channel_mults=(1, 2, 4, 4, 8, 8),
        # attn_res=(8,),
        attn_res=(16,),
        res_blocks=3,
        dropout=0,
        with_time_emb=True,
        image_size=256,
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None

        self.condition_conv = condition_net(time_dim=inner_channel, condition_dim=56)

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, time_emb_dim=time_dim, dropout=dropout, norm_groups=norm_groups, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time, condition_input):
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        warp_s = condition_input['warping']
        g = condition_input['generated']
        attr = condition_input['attribute']
        c, color_loss = self.condition_conv(warp_source=warp_s, color_cond=g, attribute=attr, t=t)
        x = torch.cat((c, x), dim=1)

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x), color_loss
