"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=scale_factor, mode="bilinear", align_corners=True
        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        # D = 41
        # C = 64
        # downsample = 16
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        # x.shape = torch.Size([24, 3, 128, 352])
        x = self.get_eff_depth(x)  # 24, 512, 8, 22    长宽各下采样 16 倍
        # Depth
        x = self.depthnet(x)  # 24, 105, 8, 22    channel 变成 41 + 64

        depth = self.get_depth_dist(x[:, : self.D])  # 24, 41, 8, 22
        # [24, 1, 41, 8, 22] * [24, 64, 1, 8, 22] = [24, 64, 41, 8, 22]
        new_x = depth.unsqueeze(1) * x[:, self.D : (self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(
                    self.trunk._blocks
                )  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints["reduction_{}".format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints["reduction_{}".format(len(endpoints) + 1)] = x
        x = self.up1(endpoints["reduction_5"], endpoints["reduction_4"])
        return x

    def forward(self, x):
        # x.shape = [24, 3, 128, 352]
        depth, x = self.get_depth_feat(x)  # [24, 64, 41, 8, 22]

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):

        # grid_conf = {
        #     "xbound": [-50.0, 50.0, 0.5], # 一共 200 种可能
        #     "ybound": [-50.0, 50.0, 0.5], # 一共 200 种可能
        #     "zbound": [-10.0, 10.0, 20.0],
        #     "dbound": [4.0, 45.0, 1.0],   # 一共 41 种可能
        # }

        # data_aug_conf = {
        #     "resize_lim": (0.193, 0.225),
        #     "final_dim": (128, 352),
        #     "rot_lim": (-5.4, 5.4),
        #     "H": 900,
        #     "W": 1600,
        #     "rand_flip": True,
        #     "bot_pct_lim": (0.0, 0.22),
        #     "cams": [
        #         "CAM_FRONT_LEFT",
        #         "CAM_FRONT",
        #         "CAM_FRONT_RIGHT",
        #         "CAM_BACK_LEFT",
        #         "CAM_BACK",
        #         "CAM_BACK_RIGHT",
        #     ],
        #     "Ncams": 5,
        # }

        # outC = 1
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(
            self.grid_conf["xbound"],
            self.grid_conf["ybound"],
            self.grid_conf["zbound"],
        )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16  # 下采样 16 倍
        self.camC = 64  # 与模型大小/精度相关，没有实际意义
        # frustum = 锥台
        self.frustum = self.create_frustum()  # shape = torch.Size([41, 8, 22, 3])
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        # 创建一个视锥映射表，作用是：
        #   1. H 和 W 上从小图（8, 22）向大图（128, 352）映射
        #   2. depth 上从 [0, 40] 向真实深度 [4, 45] 映射表
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf["final_dim"]
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = (
            torch.arange(*self.grid_conf["dbound"], dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )  # 41, 8, 22
        # ds = [4, 5, 6, ..., 44].expand_to(41, 8, 22)
        D, _, _ = ds.shape
        xs = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )  # 41, 8, 22
        # xs = [0, 16.7, 33.4, ..., 351].expand_to(41, 8, 22)
        ys = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )  # 41, 8, 22
        # ys = [0, 18.1, 36.2, ..., 127].expand_to(41, 8, 22)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # 像素坐标系内部转换
        # 由于相机的内外参（内参：intrins ; 外参：rots, trans）是指在原图（900, 1600）像素坐标上的内外参数
        # 所以需要将 frustum 从小图（128, 352）映射回原图（900, 1600）上
        # 之前图像预处理时：（900, 1600）-> matmul(post_rots) -> add(post_trans) -> (128, 352)
        # 所以：(128, 352) -> sub(post_trans) -> matmul(inverse(post_rots)) -> (900, 1600)
        # 所以叫 undo post-transformation

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)  # 求矩阵的逆
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )

        # cam_to_ego
        # 深度 * 像素坐标 = 内参 @ 相机坐标
        # 所以：inverse(内参) @ 深度 * 像素坐标 = 相机坐标
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )

        # combine = rots.matmul(torch.inverse(intrins))
        # points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        # points += trans.view(B, N, 1, 1, 1, 3)

        # 这里不太理解，理论上：外参 @ 世界坐标 = 相机坐标
        # 所以理论上应该是：inverse(外参) @ 相机坐标 = 世界坐标
        # 但实际上，这里给出的不是常规的外参，这里：外参 @ 相机坐标 = 世界坐标
        # 所以：外参 @ inverse(内参) @ 深度 * 像素坐标 = 世界坐标
        points = rots.view(B, N, 1, 1, 1, 3, 3).matmul(
            torch.inverse(intrins).view(B, N, 1, 1, 1, 3, 3).matmul(points)
        ).squeeze(-1) + trans.view(B, N, 1, 1, 1, 3)

        # [4, 6, 41, 8, 22, 3]：根据 batch_number, camera_no, depth, h, w 索引世界坐标系下的 X，Y，Z
        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C"""
        B, N, C, imH, imW = x.shape  # 4, 6, 3, 128, 352

        x = x.view(B * N, C, imH, imW)  # 24, 3, 128, 352
        x = self.camencode(x)  # 24, 64, 41, 8, 22
        x = x.view(
            B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample
        )  # 4, 6, 64, 41, 8, 22
        x = x.permute(0, 1, 3, 4, 5, 2)  # 4, 6, 41, 8, 22, 64

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape  # 4, 6, 41, 8, 22, 64
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)  # -1, 64

        # self.bx = tensor([-49.7500, -49.7500,   0.0000])
        # self.dx = tensor([ 0.5000,  0.5000, 20.0000])
        # self.nx = tensor([200, 200,   1])

        # flatten indices
        # 将世界坐标 X, Y, Z 中的负数和小数变成正整数
        # 负数则减去负向偏置，小数则向下取整
        # 保证所有索引结果都是正整数组成的三维向量
        # "xbound": [-50.0, 50.0, 0.5]
        #       [-50, -49.5, ... ,49.5] -> [0, 1, 2, ..., 199]
        # "ybound": [-50.0, 50.0, 0.5]
        #       [-50, -49.5, ... ,49.5] -> [0, 1, 2, ..., 199]
        # "zbound": [-10.0, 10.0, 20.0]
        #       [-10] -> [0]
        geom_feats = (
            (geom_feats - (self.bx - self.dx / 2.0)) / self.dx
        ).long()  # 4, 6, 41, 8, 22, 3
        geom_feats = geom_feats.view(Nprime, 3)  # -1, 3

        # batch_ix.shape = [Nprime, 1]
        # torch.unique(batch_ix) = [0, 1, 2, 3]
        # 仅用于表示 Nprime 中那些数据属于同一个 batch（不同 batch 之间的数据是没有关系的）
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        # 将 out of xbound/ybound/zbound 的点云过滤掉
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # geom_feats 中所有世界坐标系下坐标点相同的点（同一个体素内的点），rank 相同
        # rank sort + cumsum（累加）目的是将同一个体素内的点合并，本质上只是一个加速操作
        # get tensors from the same voxel next to each other
        ranks = (
            geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)
            + geom_feats[:, 1] * (self.nx[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[
            geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]
        ] = x

        # collapse Z
        # final.squeeze()
        # [4, 64, 1, 200, 200] -> [4, 64, 200, 200]
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        # 获得一个映射表，shape = [4, 6, 41, 8, 22, 3]
        # 在（8, 22）feature map 下根据 batch_number, camera_no, depth, h, w 索引世界坐标系下的 X，Y，Z
        # 世界坐标系的原点是汽车后轴中点，车前进方向为 x 轴正方向，面向 x 轴正方向向左为 y 轴正方向，z 轴正方向垂直地面向上
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)

        # [4, 6, 3, 128, 352] -> [4, 6, 41, 8, 22, 64]
        # 原图长宽各下采样 16 倍，得到一个 [8, 22, 64] 的 feature map
        # 对其每个点进行深度估计（每个点 41 分类任务），每个 hw 预测 64 次，类似集成学习
        x = self.get_cam_feats(x)

        # 体素化
        x = self.voxel_pooling(geom, x)  # 4, 64, 200, 200

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        # x.shape = torch.Size([4, 6, 3, 128, 352])
        # rots.shape = torch.Size([4, 6, 3, 3])
        # trans.shape = torch.Size([4, 6, 3])
        # intrins.shape = torch.Size([4, 6, 3, 3])
        # post_rots.shape = torch.Size([4, 6, 3, 3])
        # post_trans.shape = torch.Size([4, 6, 3])
        x = self.get_voxels(
            x, rots, trans, intrins, post_rots, post_trans
        )  # 4, 64, 200, 200

        # 降维，[4, 64, 200, 200] -> [4, 1, 200, 200]
        x = self.bevencode(x)  # 4, 1, 200, 200
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
