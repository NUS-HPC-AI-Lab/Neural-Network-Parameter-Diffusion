
import torch
import torch.nn.functional as F

class MNIST_5066(torch.nn.Module):
    
    def __init__(self):
        super(MNIST_5066, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,2,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(2,4,kernel_size=3,stride=1,padding=0),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(6*6*4,32),
                                         torch.nn.ReLU(),
                                        #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                         torch.nn.Linear(32, 10)
                                        )
        # import pdb; pdb.set_trace()
    def forward(self, x):
        x = self.conv1(x)
        #x = self.conv2(x)
        x = x.view(-1, 6*6*4)
        x = self.dense(x)
        return x

class MNIST_466(torch.nn.Module): #90acc
    
    def __init__(self):
        super(MNIST_466, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,2,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(2,4,kernel_size=3,stride=2,padding=0),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(3*3*4,10),
                                        #  torch.nn.ReLU(),
                                        # #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                        #  torch.nn.Linear(32, 10)
                                        )
        # import pdb; pdb.set_trace()
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 3*3*4)
        x = self.dense(x)
        return x

class MNIST_1066(torch.nn.Module): #94acc
    
    def __init__(self):
        super(MNIST_1066, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,4,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(4,8,kernel_size=3,stride=2,padding=0),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(3*3*8,10),
                                        #  torch.nn.ReLU(),
                                        # #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                        #  torch.nn.Linear(32, 10)
                                        )
        # import pdb; pdb.set_trace()
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 3*3*8)
        x = self.dense(x)
        return x

class MNIST_9914(torch.nn.Module): #97acc
    
    def __init__(self):
        super(MNIST_9914, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,4,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(4,8,kernel_size=3,stride=1,padding=0),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(6*6*8,32),
                                         torch.nn.ReLU(),
                                        #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                         torch.nn.Linear(32, 10)
                                        )
        # import pdb; pdb.set_trace()
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 6*6*8)
        x = self.dense(x)
        return x

class MNIST_13354(torch.nn.Module): #96acc
    
    def __init__(self):
        super(MNIST_13354, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,2,kernel_size=3,stride=1,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(2,4,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(7*7*4,64),
                                         torch.nn.ReLU(),
                                        #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                         torch.nn.Linear(64, 10)
                                        )
        # import pdb; pdb.set_trace()
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 7*7*4)
        x = self.dense(x)
        return x

class MNIST_19664(torch.nn.Module): #96acc
    
    def __init__(self):
        super(MNIST_19664, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,2,kernel_size=3,stride=1,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(2,6,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(7*7*6,64),
                                         torch.nn.ReLU(),
                                        #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                         torch.nn.Linear(64, 10)
                                        )
        # import pdb; pdb.set_trace()
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 7*7*6)
        x = self.dense(x)
        return x

class MNIST_25974(torch.nn.Module): #98acc
    
    def __init__(self):
        super(MNIST_25974, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,2,kernel_size=3,stride=1,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(2,8,kernel_size=3,stride=2,padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2,kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(7*7*8,64),
                                         torch.nn.ReLU(),
                                        #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                         torch.nn.Linear(64, 10)
                                        )
        # import pdb; pdb.set_trace()
    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 7*7*8)
        x = self.dense(x)
        return x

class MNIST_277(torch.nn.Module):#88acc
    
    def __init__(self):
        super(MNIST_277, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1,1,kernel_size=3,stride=1,padding=0,bias=False),
                                         torch.nn.LeakyReLU())
        self.conv2 = torch.nn.Sequential(
                                         torch.nn.Conv2d(1,1,kernel_size=3,stride=2,padding=0,bias=False),
                                         torch.nn.LeakyReLU(),
                                         torch.nn.Conv2d(1,1,kernel_size=3,stride=2,padding=0,bias=False),
                                        #  torch.nn.Conv2d(1,1,kernel_size=3,stride=1,padding=0,bias=False),
                                         )
        self.dense = torch.nn.Sequential(
                                        torch.nn.Linear(25,10,bias=False),
                                        #  torch.nn.ReLU(),
                                        #  torch.nn.Dropout(p=0.5),  #drop out 在训练时作用 在测试时自动关闭 所以会导致出现train acc > test acc的现象出现
                                        #  torch.nn.Linear(32, 10,bias=False)
                                        )
    def forward(self, x):
        bs = x.shape[0]
        x = self.conv1(x)
        # pdb.set_trace()
        x = self.conv2(x)
        # pdb.set_trace()
        x = x.view(bs, -1)
        x = self.dense(x)
        return x

class Lenet(torch.nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 =torch.nn.Conv2d(1, 6, 5)
        self.relu1 =torch.nn.ReLU()
        self.pool1 =torch.nn.MaxPool2d(2)
        self.conv2 =torch.nn.Conv2d(6, 16, 5)
        self.relu2 =torch.nn.ReLU()
        self.pool2 =torch.nn.MaxPool2d(2)
        self.fc1 =torch.nn.Linear(256, 120)
        self.relu3 =torch.nn.ReLU()
        self.fc2 =torch.nn.Linear(120, 84)
        self.relu4 =torch.nn.ReLU()
        self.fc3 =torch.nn.Linear(84, 10)
        self.relu5 =torch.nn.ReLU()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pool1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pool2(y)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        # y = self.relu5(y)
        return y

''' ConvNet '''
class ConvNet(torch.nn.Module):
    def __init__(self, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', channel=1, num_classes=10, im_size = (28,28)):
        super(ConvNet, self).__init__()

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
    


class ConvNet_1stride(torch.nn.Module):
    def __init__(self, net_width=128, net_depth=3, net_act='relu', net_norm='instancenorm', net_pooling='avgpooling', channel=1, num_classes=10, im_size = (28,28)):
        super(ConvNet_1stride, self).__init__()

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
            return torch.nn.MaxPool2d(kernel_size=2, stride=1)
        elif net_pooling == 'avgpooling':
            return torch.nn.AvgPool2d(kernel_size=2, stride=1)
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
                shape_feat[1] -= 1
                shape_feat[2] -= 1

        return torch.nn.Sequential(*layers), shape_feat


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(torch.nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv3 = torch.nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(torch.nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = torch.nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out






def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet10():
    return ResNet(BasicBlock, [1, 1, 1, 1])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])