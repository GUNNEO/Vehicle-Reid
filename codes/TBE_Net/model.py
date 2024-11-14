import torch
import torch.nn as nn
from resnetBackbone import resnet50, conv1x1


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class sim(nn.Module):
    def __init__(
        self,
        in_planes: int,
        num_id: int
    ):
        super().__init__()
        out_planes = in_planes // 8
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = conv1x1(in_planes=in_planes, out_planes=out_planes)
        self.bn1 = nn.BatchNorm2d(num_features=out_planes)
        self.conv2 = conv1x1(in_planes=in_planes, out_planes=out_planes)
        self.bn2 = nn.BatchNorm2d(num_features=out_planes)
        self.conv3 = conv1x1(in_planes=out_planes * 2, out_planes=out_planes)
        self.bn3 = nn.BatchNorm2d(num_features=out_planes)
        self.gelu = QuickGELU()
        self.fc = nn.Linear(out_planes, num_id)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ):
        x = self.gelu(self.bn1(self.conv1(x)))
        y = self.avg(y)
        y = self.gelu(self.bn2(self.conv2(y)))
        y = torch.cat((x, y), dim=1)
        y = self.gelu(self.bn3(self.conv3(y)))
        # residual elment wise sum
        y = x + y
        y = torch.flatten(y, start_dim=1)
        y = self.fc(y)
        return y


class headBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        kernel_size: int,
        num_id: int
    ):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_features=in_planes)
        self.leakyRelu = nn.LeakyReLU()
        out_planes = in_planes // 8
        if kernel_size == 2:
            self.conv1 = nn.Conv2d(
                in_channels=in_planes, out_channels=out_planes, kernel_size=2, stride=2, padding=0)
        else:
            self.conv1 = conv1x1(in_planes=in_planes, out_planes=out_planes)
        self.bn2 = nn.BatchNorm2d(num_features=out_planes)
        self.fc = nn.Linear(out_planes, num_id)

    def forward(
        self,
        x: torch.Tensor
    ):
        x = self.bn1(x)
        x = self.leakyRelu(x)
        x = self.bn2(self.conv1(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class complementaryBranch(nn.Module):
    def __init__(
        self,
        in_planes: int,
        num_id: int
    ):
        super().__init__()
        # define the first subbranch
        self.gmp1 = nn.AdaptiveMaxPool2d((4, 1))
        # define sim part
        self.sim1 = sim(in_planes=in_planes, num_id=num_id)
        self.sim2 = sim(in_planes=in_planes, num_id=num_id)
        self.sim3 = sim(in_planes=in_planes, num_id=num_id)
        self.sim4 = sim(in_planes=in_planes, num_id=num_id)
        self.sim = [self.sim1, self.sim2, self.sim3, self.sim4]

        # define the rest branches
        self.gmp2 = nn.AdaptiveMaxPool2d((1, 1))
        self.bn2 = nn.BatchNorm2d(num_features=in_planes)
        self.avg2 = nn.AdaptiveAvgPool2d((1, 1))
        self.bn3 = nn.BatchNorm2d(num_features=in_planes)
        self.gmp3 = nn.AdaptiveMaxPool2d((2, 2))
        self.bn4 = nn.BatchNorm2d(num_features=in_planes)
        self.avg3 = nn.AdaptiveAvgPool2d((2, 2))
        self.bn5 = nn.BatchNorm2d(num_features=in_planes)

        # define headblock
        self.hb1 = headBlock(in_planes=in_planes, kernel_size=1, num_id=num_id)
        self.hb2 = headBlock(in_planes=in_planes, kernel_size=2, num_id=num_id)

    def forward(
        self,
        x: torch.Tensor
    ):
        x1, x2, x3, x4, x5 = x, x, x, x, x
        x1 = self.gmp1(x1)

        # sim part
        y1 = []
        for i in range(x1.shape[2]):
            x11 = x1[:, :, i, :].unsqueeze(2)
            x121 = x1[:, :, 0:i, :]
            x122 = x1[:, :, i + 1:, :]
            x12 = torch.cat((x121, x122), dim=2)
            y1.append(self.sim[i](x11, x12))

        # the rest branch with different pooling manners
        # store the values for triplet loss to optimize embedding space
        x2 = self.gmp2(x2)
        x3 = self.avg2(x3)
        x4 = self.gmp3(x4)
        x5 = self.avg3(x5)
        triplet_loss_tensor = [x2, x3, x4, x5]

        # apply BNNeck
        y2 = self.bn2(x2) + self.bn3(x3)
        y3 = self.bn4(x4) + self.bn5(x5)
        y2 = self.hb1(y2)
        y3 = self.hb2(y3)
        return y1, y2, y3, triplet_loss_tensor


class localBranch(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        pass


class globalBranch(nn.Module):
    def __init__(
        self,
        in_planes: int,
        num_id: int
    ):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        out_planes = in_planes // 8
        self.conv1 = conv1x1(in_planes=in_planes, out_planes=out_planes)
        self.bn = nn.BatchNorm2d(num_features=out_planes)
        self.gelu = QuickGELU()
        self.fc = nn.Linear(out_planes, num_id)

    def forward(
        self,
        x: torch.Tensor
    ):
        x = self.avg(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.gelu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class TBE(nn.Module):
    def __init__(
        self,
        num_id: int
    ):
        super().__init__()
        # define resnet50 backbone
        self.backbone = resnet50(pretrained=True, progress=True)

        # define the global branch
        self.globalBranch = globalBranch(in_planes=2048, num_id=num_id)

        # define the local branch
        self.localBranch = localBranch()

        # define the complementary branch
        self.complementaryBranch = complementaryBranch(
            in_planes=2048, num_id=num_id)

    def forward(
        self,
        x: torch.Tensor
    ):
        x = self.backbone(x)
        g, c = x, x
        g = self.globalBranch(g)
        c1, c2, c3, triplet_loss_tensor = self.complementaryBranch(c)
        return c2


if __name__ == "__main__":
    model = TBE(num_id=1000)
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    print(out.shape)
