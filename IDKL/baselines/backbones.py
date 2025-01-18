import torch.nn as nn
from torch.nn import functional as F
# from torchvision.models.utils import load_state_dict_from_url   #原来
from torch.hub import load_state_dict_from_url
import torch
from baselines.resnet import resnet50
from baselines.efficientnet import EfficientNetV2


class Shared_module_fr(nn.Module):
    def __init__(self, model_name='resnet', version='s', pretrained=True, drop_last_stride=False):
        super(Shared_module_fr, self).__init__()
        
        if model_name == 'resnet50':
            self.model = resnet50(pretrained=pretrained, drop_last_stride=drop_last_stride, modality_attention=0)
        elif model_name == 'efficientnet':
            self.model = EfficientNetV2(version=version, pretrained=pretrained, drop_last_stride=drop_last_stride)
        else:
            raise ValueError("Unsupported backbone type. Choose 'resnet' or 'efficientnet'.")

    def forward(self, x):
        if hasattr(self.model, 'conv1'):
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
        else:
            print('Shared_module_fr')
            print(x.size())
            x = self.model(x)  # EfficientNet processes the input directly
        return x

class Special_module(nn.Module):
    def __init__(self, model_name='resnet', version='s', pretrained=True, drop_last_stride=False):
        super(Special_module, self).__init__()

        if model_name == 'resnet50':
            self.model = resnet50(pretrained=pretrained, drop_last_stride=drop_last_stride)
        elif model_name == 'efficientnet':
            self.model = EfficientNetV2(version=version, pretrained=pretrained, drop_last_stride=drop_last_stride)
        else:
            raise ValueError("Unsupported backbone type. Choose 'resnet' or 'efficientnet'.")

    def forward(self, x):
        if hasattr(self.model, 'layer3'):
            x = self.model.layer3(x)
            x = self.model.layer4(x)
        else:
            x = self.model(x)  # EfficientNet processes the input directly
        return x

class Shared_module_bh(nn.Module):
    def __init__(self, model_name='resnet', version='s', pretrained=True, drop_last_stride=False):
        super(Shared_module_bh, self).__init__()

        if model_name == 'resnet50':
            self.model = resnet50(pretrained=pretrained, drop_last_stride=drop_last_stride)
        elif model_name == 'efficientnet':
            self.model = EfficientNetV2(version=version, pretrained=pretrained, drop_last_stride=drop_last_stride)
        else:
            raise ValueError("Unsupported backbone type. Choose 'resnet' or 'efficientnet'.")

    def forward(self, x):
        if hasattr(self.model, 'layer3'):
            x_sh3 = self.model.layer3(x)
            x_sh4 = self.model.layer4(x_sh3)
        else:
            x_sh3 = x
            x_sh4 = self.model(x)  # EfficientNet processes the input directly
        return x_sh3, x_sh4






























class Mask(nn.Module):
    def __init__(self, dim, r=16):
        super(Mask, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.channel_attention(x)
        return mask


class special_att(nn.Module):
    def __init__(self, dim, r=16):
        super(special_att, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Conv2d(dim, dim // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // r, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.IN = nn.InstanceNorm2d(dim, track_running_stats=False) #self.IN = nn.InstanceNorm2d(dim, track_running_stats=True, affine=True)

    def forward(self, x):
        x_IN = self.IN(x)
        x_R = x - x_IN
        pooled = gem(x_R)
        mask = self.channel_attention(pooled)
        x_sp = x_R * mask + x_IN  # x

        return x_sp, x_IN


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)


        
class embed_net(nn.Module):
    def __init__(self, drop_last_stride,  decompose=False, model_name='efficientnet'):
        super(embed_net, self).__init__()

        self.shared_module_fr = Shared_module_fr(drop_last_stride=drop_last_stride,
                                                model_name=model_name)
        self.shared_module_bh = Shared_module_bh(drop_last_stride=drop_last_stride,
                                                model_name=model_name)

        self.special = Special_module(drop_last_stride=drop_last_stride, 
                                      model_name=model_name)

        self.decompose = decompose
        self.IN = nn.InstanceNorm2d(2048, track_running_stats=True, affine=True)
        if decompose:
            self.special_att = special_att(2048)
            self.mask1 = Mask(2048)
            self.mask2 = Mask(2048)
        self.model_name = model_name

    def forward(self, x):
        x2 = self.shared_module_fr(x)
        x3, x_sh = self.shared_module_bh(x2)  # bchw

        sh_pl = gem(x_sh).squeeze()  # Gem池化
        sh_pl = sh_pl.view(sh_pl.size(0), -1)  # Gem池化

        if self.decompose:
            ######special structure
            x_sp_f = self.special(x2)
            sp_IN = self.IN(x_sp_f)
            m_IN = self.mask1(sp_IN)
            m_F = self.mask2(x_sp_f)
            sp_IN_p = m_IN * sp_IN
            x_sp_f_p = m_F * x_sp_f
            x_sp = m_IN * x_sp_f_p + m_F * sp_IN_p

            sp_pl = gem(x_sp).squeeze()  # Gem池化
            sp_pl = sp_pl.view(sp_pl.size(0), -1)  # Gem池化


        return x_sh,  sh_pl, sp_pl,sp_IN,sp_IN_p,x_sp_f,x_sp_f_p     
    