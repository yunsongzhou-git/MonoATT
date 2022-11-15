import torch
import torch.nn as nn
from functools import partial
import math
from .tcformer_layers import Block, TCBlock, OverlapPatchEmbed, CTM
from .tcformer_utils import (load_checkpoint, get_root_logger, token2map, vis_tokens)
from .transformer_utils import trunc_normal_
from lib.models.monodetr.heatmap.utils import (
    nms_hm,
    select_topk,
    select_point_of_interest,
)

class TCFormer(nn.Module):
    def __init__(
            self, img_size=224, in_chans=3, embed_dims=[64, 128, 256, 512],
            num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
            qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            num_stages=4, pretrained=None,
            k=5, sample_ratios=[0.25, 0.25, 0.25, 0.25],
            return_map=False,
            **kwargs
    ):
        super().__init__()

        self.depths = depths
        self.num_stages = num_stages
        self.grid_stride = sr_ratios[0]
        self.embed_dims = embed_dims
        self.sr_ratios = sr_ratios
        self.mlp_ratios = mlp_ratios
        self.sample_ratios = sample_ratios
        self.return_map = return_map

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0


        # In stage 2~4, use TCBlock for dynamic tokens
        for i in range(0, num_stages):
            ctm = CTM(sample_ratios[i], embed_dims[i], embed_dims[i+1], k)
            norm = norm_layer(embed_dims[i+1])
            cur += depths[i]

            setattr(self, f"ctm{i}", ctm)
            setattr(self, f"norm{i + 1}", norm)
        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    def forward_features(self, features, pos, img, heatmap, img_id):
        outs = []
        outs_fmap = []
        x = features.tensors
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1).permute(0,2,1)

        # init token dict
        B, N, _ = x.shape
        device = x.device
        idx_token = torch.arange(N)[None, :].repeat(B, 1).to(device)
        agg_weight = x.new_ones(B, N, 1)
        token_dict = {'x': x,
                      'token_num': N,
                      'map_size': [H, W],
                      'init_grid_size': [H, W],
                      'idx_token': idx_token,
                      'agg_weight': agg_weight}
        init_k = H*W//4
        width = W//2
        heatmap = nms_hm(heatmap)

        # stage 2~4
        for i in range(0, self.num_stages-1):

            _, _, clses, ys, xs = select_topk(heatmap, K=init_k)

            ys, xs = ys//(2**(i+1)), xs//(2**(i+1))
            indexs = ys*width+xs
            token_dict['indexs'] = indexs.long()

            ctm = getattr(self, f"ctm{i}")
            norm = getattr(self, f"norm{i + 1}")

            token_dict,_ = ctm(token_dict)  # down sample
            token_dict['x'] = norm(token_dict['x'])
            init_k = init_k//4
            width = width//2
        i = i+1

        _, _, clses, ys, xs = select_topk(heatmap, K=init_k)
        ys, xs = ys//(2**(i+1)), xs//(2**(i+1))
        indexs = ys*width+xs
        token_dict['indexs'] = indexs.long()


        ctm = getattr(self, f"ctm{i}")
        norm = getattr(self, f"norm{i + 1}")

        token_dict,_ = ctm(token_dict)  # down sample
        token_dict['x'] = norm(token_dict['x'])
        idx_tokens = token_dict['idx_token']


        def pick_center(idx_tokens, H, W, ratio):
            from random import sample
            import random
            output = []
            B, N = idx_tokens.shape
            for i in range(B):
                output_temp = []
                for j in range(1,N-1,3):
                    temp = idx_tokens[i,j]
                    if temp!=idx_tokens[i,j-1] and temp!=idx_tokens[i,j+1]:
                        h, w = j//H*ratio, j%H*ratio
                        output_temp.append([h, w])
                output.append([random.choice(output_temp) for _ in range(50)])

            output = torch.Tensor(output).reshape(B, 50, 2)
            return output

        
        ratio = 1280//H

        return token2map(token_dict)

    def forward(self, x, y, img, heatmap, img_id):
        x = self.forward_features(x, y, img, heatmap, img_id)
        return x


class tcformer_light(TCFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            **kwargs)


class tcformer(TCFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            nh_list=[1, 1, 1], nw_list=[1, 1, 1],
            **kwargs)


class tcformer_large(TCFormer):
    def __init__(self, **kwargs):
        super().__init__(
            embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            **kwargs)


