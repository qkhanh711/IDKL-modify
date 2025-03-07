import torch.nn as nn
from torch.nn import functional as F
# from torchvision.models.utils import load_state_dict_from_url   #原来
from torch.hub import load_state_dict_from_url
import torch
from baselines.resnet import resnet50, resnet18, resnet34, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d
from baselines.efficientnet import efficientnet_b0
from baselines.shufflenet import shufflenet_v2_x0_5
from baselines.mnasnet import mnasnet1_0



class FeatureExtractor(torch.nn.Module):
    def __init__(self, original_model, model_name='efficientnet'):
        super(FeatureExtractor, self).__init__()
        if model_name == "efficientnet":
            self.stem = original_model.features[0]  # Stem (initial conv layer)
            self.stage1 = original_model.features[1]  # MBConv 1   
            self.stage2 = torch.nn.Sequential(*original_model.features[2:4])  # MBConv 2-3
            self.stage3 = torch.nn.Sequential(*original_model.features[4:6])  # MBConv 4-5
            self.stage4 = torch.nn.Sequential(*original_model.features[6:8])  # MBConv 6-7
        else:
            self.stem = original_model.layers[0]
            self.stage1 = original_model.layers[1]
            self.stage2 = original_model.layers[2]
            self.stage3 = original_model.layers[3]
            self.stage4 = original_model.layers[4]
    

    def forward(self, x):
        x1 = self.stem(x)   # Layer 1
        x2 = self.stage1(x1)  # Layer 2
        x3 = self.stage2(x2)  # Layer 3
        x4 = self.stage3(x3)  # Layer 4
        return x1, x2, x3, x4  # Return all intermediate outputs

class Shared_module_fr(nn.Module):
    def __init__(self, model_name='resnet', version='s', pretrained=True, drop_last_stride=False):
        super(Shared_module_fr, self).__init__()
        self.model_name = model_name        
        if model_name == 'resnet18':
            self.model = resnet18(pretrained=pretrained, drop_last_stride=drop_last_stride, modality_attention=0)
        elif model_name == 'resnet50':
            self.model = resnet50(pretrained=pretrained, drop_last_stride=drop_last_stride, modality_attention=0)
        elif model_name == 'efficientnet':
            self.model = efficientnet_b0(pretrained=pretrained)
        elif model_name == "shufflenetv2":
            self.model = shufflenet_v2_x0_5(progress = True)  
        elif model_name == 'mnasnet':
            self.model = mnasnet1_0(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone type. Choose 'resnet' or 'efficientnet'.")
            
    def forward(self, x):
        if self.model_name == "resnet50":
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
            x = self.model.layer1(x)
            x = self.model.layer2(x)
        elif self.model_name == "shufflenetv2":
            x = self.model.conv1(x)
            x = self.model.maxpool(x)
            x = self.model.stage2(x)
        else:
            x = FeatureExtractor(self.model, self.model_name).stem(x)
            x = FeatureExtractor(self.model, self.model_name).stage1(x)
            x = FeatureExtractor(self.model, self.model_name).stage2(x)
        return x

class Special_module(nn.Module):
    def __init__(self, model_name='resnet', version='s', pretrained=True, drop_last_stride=False):
        super(Special_module, self).__init__()
        self.model_name = model_name
        if model_name == 'resnet18':
            self.model = resnet18(pretrained=pretrained, drop_last_stride=drop_last_stride, modality_attention=0)
        elif model_name == 'resnet50':       
            self.model = resnet50(pretrained=pretrained, drop_last_stride=drop_last_stride)
        elif model_name == 'efficientnet':
            self.model = efficientnet_b0(pretrained=pretrained)
        elif model_name == "shufflenetv2":
            self.model = shufflenet_v2_x0_5(progress = True)
        elif model_name == 'mnasnet':
            self.model = mnasnet1_0(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone type. Choose 'resnet50' or 'efficientnet'.")
    def forward(self, x):
        if self.model_name == 'resnet50':
            x = self.model.layer3(x)
            x = self.model.layer4(x)
        elif self.model_name == "shufflenetv2":
            x = self.model.stage3(x)
            x = self.model.stage4(x)
        else: 
            x = FeatureExtractor(self.model, self.model_name).stage3(x)
            x = FeatureExtractor(self.model, self.model_name).stage4(x)
        return x


class Shared_module_bh(nn.Module):
    def __init__(self, model_name='resnet50', version='s', pretrained=True, drop_last_stride=False):
        super(Shared_module_bh, self).__init__()
        self.model_name = model_name
        if model_name == 'resnet18':
            self.model = resnet18(pretrained=pretrained, drop_last_stride=drop_last_stride, modality_attention=0)
        elif model_name == 'resnet50':
            self.model = resnet50(pretrained=pretrained, drop_last_stride=drop_last_stride)
        elif model_name == 'efficientnet':
            self.model = efficientnet_b0(pretrained=pretrained)
        elif model_name == "shufflenetv2":
            self.model = shufflenet_v2_x0_5(progress = True)
        elif model_name == 'mnasnet':
            self.model = mnasnet1_0(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone type. Choose 'resnet50' or 'efficientnet'.")

    def forward(self, x):
        if hasattr(self.model, 'layer3'):
            x_sh3 = self.model.layer3(x)
            x_sh4 = self.model.layer4(x_sh3)
        elif self.model_name == "shufflenetv2":
            x_sh3 = self.model.stage3(x)
            x_sh4 = self.model.stage4(x_sh3)
        else:
            x_sh3 = FeatureExtractor(self.model, self.model_name).stage3(x)
            x_sh4 = FeatureExtractor(self.model, self.model_name).stage4(x_sh3)
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
    def __init__(self, drop_last_stride,  decompose=False, base_dim= 2048, model_name='efficientnet'):
        super(embed_net, self).__init__()
        self.base_dim = base_dim

        self.shared_module_fr = Shared_module_fr(drop_last_stride=drop_last_stride,
                                                model_name=model_name)
        self.shared_module_bh = Shared_module_bh(drop_last_stride=drop_last_stride,
                                                model_name=model_name)

        self.special = Special_module(drop_last_stride=drop_last_stride, 
                                      model_name=model_name)

        self.decompose = decompose
        self.IN = nn.InstanceNorm2d(self.base_dim, track_running_stats=True, affine=True)
        if decompose:
            self.special_att = special_att(self.base_dim)
            self.mask1 = Mask(self.base_dim)
            self.mask2 = Mask(self.base_dim)
        self.model_name = model_name

    def forward(self, x):
        # print("Passing through Shared_module_fr")
        x2 = self.shared_module_fr(x)
        # print("Passing through Shared_module_bh")
        x3, x_sh = self.shared_module_bh(x2)  # bchw

        sh_pl = gem(x_sh).squeeze()  # Gem池化
        sh_pl = sh_pl.view(sh_pl.size(0), -1)  # Gem池化

        if self.decompose:
            ######special structure
            # print("Passing through Special_module")
            x_sp_f = self.special(x2)
            # if self.model_name == "shufflenetv2":
            #     conv = nn.Conv2d(in_channels=192, out_channels=2048, kernel_size=1, stride=1, padding=1, bias=False).to(device="cuda")
            #     x_sp_f = conv(x_sp_f)  # Giờ x_sp_f có shape (60, 2048, 8, 4)
            # print(x_sp_f.size())
            sp_IN = self.IN(x_sp_f)
            # print(sp_IN.size())
            m_IN = self.mask1(sp_IN)
            # print(m_IN.size())
            m_F = self.mask2(x_sp_f)
            sp_IN_p = m_IN * sp_IN
            x_sp_f_p = m_F * x_sp_f
            x_sp = m_IN * x_sp_f_p + m_F * sp_IN_p

            sp_pl = gem(x_sp).squeeze()  # Gem池化
            sp_pl = sp_pl.view(sp_pl.size(0), -1)  # Gem池化


        return x_sh,  sh_pl, sp_pl,sp_IN,sp_IN_p,x_sp_f,x_sp_f_p     
    