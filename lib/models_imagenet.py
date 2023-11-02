
import torch
import torch.nn.functional as F
import torch.nn as nn

''' ConvNet '''
class ConvNet_200(torch.nn.Module):
    def __init__(self, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', channel=3, num_classes=200, im_size = (64,64)):
        super(ConvNet_200, self).__init__()

        self.features, shape_feat = self._make_layers(channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size)
        num_feat = shape_feat[0]*shape_feat[1]*shape_feat[2]
        self.classifier = torch.nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return torch.nn.Sigmoid()
        elif net_act == 'relu':
            return torch.nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return torch.nn.LeakyReLU(negative_slope=0.01)


    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return torch.nn.MaxPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return torch.nn.AvgPool2d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*h*w)
        if net_norm == 'batchnorm':
            return torch.nn.BatchNorm2d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return torch.nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return torch.nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return torch.nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, channel, net_width, net_depth, net_norm, net_act, net_pooling, im_size):
        layers = []
        in_channels = channel
        if im_size[0] == 28:
            im_size = (32, 32)
        shape_feat = [in_channels, im_size[0], im_size[1]]
        for d in range(net_depth):
            layers += [torch.nn.Conv2d(in_channels, net_width, kernel_size=3, padding=3 if channel == 1 and d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2
                shape_feat[2] //= 2

        return torch.nn.Sequential(*layers), shape_feat
    
