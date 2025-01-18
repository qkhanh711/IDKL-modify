import torch
import torch.nn as nn
from torchvision.models.efficientnet import efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l

class EfficientNetV2(nn.Module):
    def __init__(self, version='s', pretrained=True, drop_last_stride=False):
        super(EfficientNetV2, self).__init__()
        
        if version == 's':
            self.backbone = efficientnet_v2_s(pretrained=pretrained)
        elif version == 'm':
            self.backbone = efficientnet_v2_m(pretrained=pretrained)
        elif version == 'l':
            self.backbone = efficientnet_v2_l(pretrained=pretrained)
        else:
            raise ValueError("Invalid EfficientNet version. Choose from 's', 'm', or 'l'.")

        self.drop_last_stride = drop_last_stride

        # Remove the fully connected layer to retain feature extractor only
        self.features = self.backbone.features
        # self.out_channels = self.features[-1][-1].out_channels

        if self.drop_last_stride:
            self._modify_last_stride()

    def _modify_last_stride(self):
        """Modify the last convolutional layer to have stride 1."""
        for idx in range(len(self.features) - 1, -1, -1):
            for layer in self.features[idx]:
                if isinstance(layer, nn.Conv2d) and layer.stride == (2, 2):
                    layer.stride = (1, 1)
                    return

    def forward(self, x):
        print("EfficientNetV2 forward")
        print(x.size())
        x = self.features(x)
        return x

# Example usage
if __name__ == "__main__":
    model = EfficientNetV2(version='s', pretrained=True, drop_last_stride=True)
    x = torch.randn(1, 3, 224, 224)
    features = model(x)
    print("Feature shape:", features.shape)
