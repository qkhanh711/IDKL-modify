import torch.nn as nn
from torch.nn import functional as F
# from torchvision.models.utils import load_state_dict_from_url   #原来
from torch.hub import load_state_dict_from_url
import torch
from models.resnet import resnet50, special_att, Mask, gem
from models.EfficientNet import  efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l
from models.shufflenet import shufflenet_v2_x2_0
    

class Shared_module_bh(nn.Module):
    def __init__(self, drop_last_stride, modality_attention=0, model_name='efficientnetv2_l'):
        super(Shared_module_bh, self).__init__()

        # model_sh_bh = resnet50(pretrained=True, drop_last_stride=drop_last_stride,)  # model_sh_fr  model_sh_bh
        if model_name == 'resnet50':
            model_sh_bh = resnet50(pretrained=True, drop_last_stride=drop_last_stride,
                                   modality_attention=modality_attention)
        elif model_name == 'efficientnetv2_s':
            model_sh_bh = efficientnet_v2_s(pretrained=True)
        elif model_name == 'efficientnetv2_m':
            model_sh_bh = efficientnet_v2_m(pretrained=True)
        elif model_name == 'efficientnetv2_l':
            model_sh_bh = efficientnet_v2_l(pretrained=True)
        elif model_name == 'shufflenetv2_x2.0':
            model_sh_bh = shufflenet_v2_x2_0(pretrained=True)
        else:
            raise ValueError('model_name not supported')
        
        self.model_sh_bh = model_sh_bh  # self.model_sh_bh = model_sh_bh  #self.model_sh_fr = model_sh_fr
        self.model_name = model_name    
            
    def forward(self, x):
        if self.model_name == "resnet50": 
            # x = self.model_sh_bh.layer2(x)
            x_sh3 = self.model_sh_bh.layer3(x)  # self.model_sh_fr  self.model_sh_bh
            x_sh4 = self.model_sh_bh.layer4(x_sh3)  # self.model_sh_fr  self.model_sh_bh
            return x_sh3, x_sh4
        else:
            x_sh3 = self.model_sh_bh(x)
            return x_sh3



class Shared_module_fr(nn.Module):
    def __init__(self, drop_last_stride, modality_attention = 0, model_name='efficientnetv2_l'):
        super(Shared_module_fr, self).__init__()

        # model_sh_fr = resnet50(pretrained=True, drop_last_stride=drop_last_stride,
        #                        modality_attention=modality_attention)
        if model_name == 'resnet50':
            model_sh_fr = resnet50(pretrained=True, drop_last_stride=drop_last_stride,
                                   modality_attention=modality_attention)
        elif model_name == 'efficientnetv2_s':
            model_sh_fr = efficientnet_v2_s(pretrained=True)
        elif model_name == 'efficientnetv2_m':
            model_sh_fr = efficientnet_v2_m(pretrained=True)
        elif model_name == 'efficientnetv2_l':
            model_sh_fr = efficientnet_v2_l(pretrained=True)
        elif model_name == 'shufflenetv2_x2.0':
            model_sh_fr = shufflenet_v2_x2_0(pretrained=True)
        else:
            raise ValueError('model_name not supported')
        # avg pooling to global pooling
        self.model_sh_fr = model_sh_fr
        self.model_name = model_name

    def forward(self, x):
        if self.model_name == "resnet50":
            x = self.model_sh_fr.conv1(x)
            x = self.model_sh_fr.bn1(x)
            x = self.model_sh_fr.relu(x)
            x = self.model_sh_fr.maxpool(x)
            x = self.model_sh_fr.layer1(x)
            x = self.model_sh_fr.layer2(x)
        # x = self.model_sh_fr.layer3(x)
        # x = self.model_sh_fr.layer4(x)
        else:
            x = self.model_sh_fr(x)
        return x

    
class Special_module(nn.Module):
    def __init__(self, drop_last_stride, modality_attention, model_name):
        super(Special_module, self).__init__()
        self.model_name = model_name
        if model_name == 'resnet50':
            special_module = resnet50(pretrained=True, drop_last_stride=drop_last_stride,)
        elif model_name == 'efficientnetv2_s':
            special_module = efficientnet_v2_s(pretrained=True)
        elif model_name == 'efficientnetv2_m':
            special_module = efficientnet_v2_m(pretrained=True)
        elif model_name == 'efficientnetv2_l':
            special_module = efficientnet_v2_l(pretrained=True)
        elif model_name == 'shufflenetv2_x2.0':
            special_module = shufflenet_v2_x2_0(pretrained=True)
        else:
            raise ValueError('model_name not supported')
        
        self.special_module = special_module
        
    def forward(self, x):
        if self.model_name == 'resnet50':
            x = self.special_module.layer3(x)
            x = self.special_module.layer4(x)
            return x 
        else:
            print("Training on special module", self.model_name)
            print("Input shape", x.shape)
            return self.special_module(x)
        
class embed_net(nn.Module):
    def __init__(self, drop_last_stride,  decompose=False, model_name='efficientnetv2_l'):
        super(embed_net, self).__init__()

        self.shared_module_fr = Shared_module_fr(drop_last_stride=drop_last_stride,
                                                model_name=model_name)
        self.shared_module_bh = Shared_module_bh(drop_last_stride=drop_last_stride,
                                                model_name=model_name)

        self.special = Special_module(drop_last_stride=drop_last_stride, modality_attention=0, model_name=model_name)

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
    