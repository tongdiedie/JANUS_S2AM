import os
import warnings

import torch
import torch.nn as nn
import torchvision


class Res101Encoder(nn.Module):
    """
    ResNet-101 backbone from DeepLabV3/vanilla ResNet.

    ``pretrained_weights`` can be:
      - ``'COCO'``: read ``DEEPLABV3_RESNET101_COCO`` env or
        ``./checkpoints/deeplabv3_resnet101_coco-586e9e4e.pth``;
      - ``'resnet101'``: read ``RESNET101_IMAGENET`` env or
        ``./checkpoints/resnet101-63fe2227.pth``;
      - a concrete checkpoint path;
      - a state dict;
      - ``None``/``'none'`` to train without loading encoder weights.

    Missing checkpoint paths now produce a warning instead of a hard crash, so
    scripts fail only when the actual model/data dependency is unavailable.
    """

    def __init__(self, replace_stride_with_dilation=None, pretrained_weights='resnet101'):
        super().__init__()
        self.pretrained_weights = self._resolve_pretrained(pretrained_weights)

        try:
            _model = torchvision.models.resnet101(
                weights=None,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )
        except TypeError:  # torchvision<=0.12
            _model = torchvision.models.resnet.resnet101(
                pretrained=False,
                replace_stride_with_dilation=replace_stride_with_dilation,
            )

        self.backbone = nn.ModuleDict()
        for dic, m in _model.named_children():
            self.backbone[dic] = m

        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.reduce1d = nn.Linear(in_features=1000, out_features=1, bias=True)

        self._init_weights()

    def _resolve_pretrained(self, pretrained_weights):
        if pretrained_weights is None:
            return None
        if isinstance(pretrained_weights, dict):
            return pretrained_weights
        if not isinstance(pretrained_weights, str):
            return pretrained_weights

        key = pretrained_weights.strip()
        if key.lower() in {"", "none", "false", "no"}:
            return None
        if key == 'COCO':
            path = os.environ.get(
                'DEEPLABV3_RESNET101_COCO',
                os.path.join('checkpoints', 'deeplabv3_resnet101_coco-586e9e4e.pth'),
            )
        elif key == 'resnet101':
            path = os.environ.get(
                'RESNET101_IMAGENET',
                os.path.join('checkpoints', 'resnet101-63fe2227.pth'),
            )
        else:
            path = key

        if not os.path.exists(path):
            warnings.warn(
                f"Encoder checkpoint not found: {path}. The encoder will be randomly initialized. "
                "Set encoder_pretrained_weights to a valid path or export the expected environment variable.",
                RuntimeWarning,
            )
            return None
        return torch.load(path, map_location='cpu')

    def forward(self, x):
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["relu"](x)

        x = self.backbone["maxpool"](x)
        x = self.backbone["layer1"](x)
        x = self.backbone["layer2"](x)
        x = self.backbone["layer3"](x)

        feature = self.reduce1(x)
        x = self.backbone["layer4"](x)
        t = self.backbone["avgpool"](x)
        t = torch.flatten(t, 1)
        t = self.backbone["fc"](t)
        t = self.reduce1d(t)
        return feature, t

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.pretrained_weights is not None:
            keys = list(self.pretrained_weights.keys())
            new_dic = self.state_dict()
            new_keys = set(new_dic.keys())

            for key in keys:
                if key in new_keys and new_dic[key].shape == self.pretrained_weights[key].shape:
                    new_dic[key] = self.pretrained_weights[key]

            self.load_state_dict(new_dic)
