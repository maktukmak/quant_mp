import torch.nn as nn
import torch.nn.functional as F

from quant_mp.config import QuantModuleConfig
from quant_mp.QModules import QConv2d, QLinear
from quant_mp.utils import patch_model

# TODO: All models not validated working under new paradigm yet


class LinNet(nn.Module):
    def __init__(self, rconfig: QuantModuleConfig):
        super(LinNet, self).__init__()
        self.fci = QLinear(784, 1024, qlinear_config=rconfig)
        self.fcs = QLinear(1024, 1024, qlinear_config=rconfig)
        self.fco = QLinear(1024, 10, qlinear_config=rconfig)

    def forward(self, x):
        x = F.relu(self.fci(x))
        x = F.relu(self.fcs(x))
        x = self.fco(x)
        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    def __init__(self, rconfig: QuantModuleConfig):
        super(ConvNet, self).__init__()
        self.ci = QConv2d(rconfig, 3, 50, (3, 3), stride=(1, 1), padding=1)
        self.cs = nn.ModuleList(
            [
                QConv2d(rconfig, 50, 64, (3, 3), stride=(1, 1), padding=1),
                QConv2d(rconfig, 64, 128, (3, 3), stride=(1, 1), padding=1),
                QConv2d(rconfig, 128, 256, (3, 3), stride=(1, 1), padding=1),
                QConv2d(rconfig, 256, 256, (3, 3), stride=(1, 1), padding=1),
            ]
        )

        self.fcs = nn.ModuleList(
            [QLinear(1024, 500, rconfig), QLinear(500, 250, rconfig)]
        )

        self.fco = QLinear(250, 10, rconfig)

    def forward(self, x):
        x = F.relu(self.ci(x))

        for c in self.cs:
            x = F.relu(c(x))
            x = F.max_pool2d(x, 2)
            x = F.dropout(x, 0.2)

        x = x.flatten(1)

        for fc in self.fcs:
            x = F.relu(fc(x))
            x = F.dropout(x, 0.2)

        x = self.fco(x)
        output = F.log_softmax(x, dim=1)
        return output


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

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


def ResNet18(rconfig: QuantModuleConfig):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    patch_model(model, rconfig)
    return model


def ResNet34(rconfig: QuantModuleConfig):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    patch_model(model, rconfig)
    return model


def ResNet50(rconfig: QuantModuleConfig):
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    patch_model(model, rconfig)
    return model


def ResNet101(rconfig: QuantModuleConfig):
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    patch_model(model, rconfig)
    return model


def ResNet152(rconfig: QuantModuleConfig):
    model = ResNet(Bottleneck, [3, 8, 36, 3])
    patch_model(model, rconfig)
    return model
