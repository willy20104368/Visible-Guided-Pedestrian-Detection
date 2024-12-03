import torch.nn as nn
from models.ConvMLP.utils.tokenizer import ConvTokenizer
from models.ConvMLP.utils.modules import ConvStage, BasicStage
import torch

__all__ = ['ConvMLP', 'convmlp_s', 'convmlp_m', 'convmlp_l']


weight_path = {
    'convmlp_s': "/home/willy20104368/yolov7/runs/backbone/convmlp_s_imagenet.pth",
    'convmlp_m': "/home/willy20104368/yolov7/runs/backbone/convmlp_m_imagenet.pth",
    'convmlp_l': 'https://shi-labs.com/projects/convmlp/checkpoints/convmlp_l_imagenet.pth',
}


class ConvMLP(nn.Module):
    def __init__(self,
                 blocks,
                 dims,
                 mlp_ratios,
                 channels=64,
                 n_conv_blocks=3,
                 num_classes=1000,
                 *args, **kwargs):
        super(ConvMLP, self).__init__()
        assert len(blocks) == len(dims) == len(mlp_ratios), \
            f"blocks, dims and mlp_ratios must agree in size, {len(blocks)}, {len(dims)} and {len(mlp_ratios)} passed."

        self.tokenizer = ConvTokenizer(embedding_dim=channels)
        self.conv_stages = ConvStage(n_conv_blocks,
                                     embedding_dim_in=channels,
                                     hidden_dim=dims[0],
                                     embedding_dim_out=dims[0])

        self.stages = nn.ModuleList()
        for i in range(0, len(blocks)):
            stage = BasicStage(num_blocks=blocks[i],
                               embedding_dims=dims[i:i + 2],
                               mlp_ratio=mlp_ratios[i],
                               stochastic_depth_rate=0.1,
                               downsample=(i + 1 < len(blocks)))
            self.stages.append(stage)
        # if classifier_head:
        #     self.norm = nn.LayerNorm(dims[-1])
        #     self.head = nn.Linear(dims[-1], num_classes)
        # else:
        #     self.head = None
        self.apply(self.init_weight)

    def forward(self, x):
        x = self.tokenizer(x)
        x = self.conv_stages(x)
        p3 = x
        x = x.permute(0, 2, 3, 1).contiguous()
        
        for i,stage in enumerate(self.stages):
            x = stage(x)
            if i == 0:
                p4 = x.permute(0,3,1,2).contiguous()
            elif i == 2:
                p5 = x.permute(0,3,1,2).contiguous()
        return p3, p4, p5


    @staticmethod
    def init_weight(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, (nn.Linear, nn.Conv1d)) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)


def _convmlp(arch, pretrained, progress, classifier_head, blocks, dims, mlp_ratios, *args, **kwargs):
    model = ConvMLP(blocks=blocks, dims=dims, mlp_ratios=mlp_ratios, *args, **kwargs)
    # if pretrained and arch in weight_path:
    #     state_dict = torch.load(weight_path[arch])
    #     if not classifier_head:
    #         for k in list(state_dict.keys()):    
    #             if k.startswith("norm") or k.startswith("head"):
    #                 print("delete layer frmo weights", k)
    #                 del state_dict[k]
    #     model.load_state_dict(state_dict)
    #     print("loading pretrained weigths for ConvMLP")
    return model


def convmlp_s(pretrained=True, progress=False, classifier_head=False, *args, **kwargs):
    return _convmlp('convmlp_s', pretrained=pretrained, progress=progress,
                    blocks=[2, 4, 2], mlp_ratios=[2, 2, 2], dims=[128, 256, 512],
                    channels=64, n_conv_blocks=2, classifier_head=classifier_head,
                    *args, **kwargs)


def convmlp_m(pretrained=False, progress=False, classifier_head=True, *args, **kwargs):
    return _convmlp('convmlp_m', pretrained=pretrained, progress=progress,
                    blocks=[3, 6, 3], mlp_ratios=[3, 3, 3], dims=[128, 256, 512],
                    channels=64, n_conv_blocks=3, classifier_head=classifier_head,
                    *args, **kwargs)


def convmlp_l(pretrained=False, progress=False, classifier_head=True, *args, **kwargs):
    return _convmlp('convmlp_l', pretrained=pretrained, progress=progress,
                    blocks=[4, 8, 3], mlp_ratios=[3, 3, 3], dims=[192, 384, 768],
                    channels=96, n_conv_blocks=3, classifier_head=classifier_head,
                    *args, **kwargs)
