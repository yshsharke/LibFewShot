# -*- coding: utf-8 -*-
"""
@inproceedings{Liu_2022_CVPR,
    author    = {Liu, Yang and Zhang, Weifeng and Xiang, Chao and Zheng, Tu and Cai, Deng and He, Xiaofei},
    title     = {Learning To Affiliate: Mutual Centralized Learning for Few-Shot Classification},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {14411-14420}
}
https://arxiv.org/abs/2106.05517

Adapted from https://github.com/LouieYang/MCL.
"""
import torch
from torch import nn
from torch.nn import functional as F

from core.utils import accuracy
from .metric_model import MetricModel


class MCLLayer(nn.Module):
    def __init__(self, encoding, gamma, gamma2, katz_factor):
        super(MCLLayer, self).__init__()
        self.encoding = encoding
        self.gamma = gamma
        self.gamma2 = gamma2
        self.katz_factor = katz_factor

    def _parse_encoding_params(self):
        idx = self.encoding.find('-')
        if idx < 0:
            return []
        blocks = self.encoding[idx + 1:].split(',')
        blocks = [int(s) for s in blocks]
        return blocks

    def _encoding_FCN(self, x):
        return x

    def _encoding_Grid(self, x):
        b, n, _, _, _ = x.shape
        grids = 1

        out = F.adaptive_avg_pool2d(x, 1)
        out = out.view(b, n, grids, -1).permute(0, 1, 3, 2).unsqueeze(-1)

        return out

    def _encoding_PyramidFCN(self, x):
        b, n, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        feature_list = []
        for size_ in self._parse_encoding_params():
            feature_list.append(F.adaptive_avg_pool2d(x, size_).view(b, n, c, 1, -1))

        if not feature_list:
            out = x.view(b, n, c, 1, -1)
        else:
            out = torch.cat(feature_list, dim=-1)

        return out

    def forward(self, query_feat, support_feat, way_num, shot_num, query_num):
        """

        :param query_feat: b * q * c * h * w
        :param support_feat: b * s * c * h * w
        :return:
        """
        if self.encoding == "FCN":
            support_feat, query_feat = self._encoding_FCN(support_feat), self._encoding_FCN(query_feat)
        elif self.encoding == "Grid":
            support_feat, query_feat = self._encoding_Grid(support_feat), self._encoding_Grid(query_feat)
        elif self.encoding == "PyramidFCN":
            support_feat, query_feat = self._encoding_PyramidFCN(support_feat), self._encoding_PyramidFCN(query_feat)

        b, q, c, h, w = query_feat.size()
        _, s, _, _, _ = support_feat.size()

        # b, q, c, h, w -> b, q, h*w, c -> b, q, 1, h*w, c
        query_feat = query_feat.view(b, way_num * query_num, c, h * w).permute(
            0, 1, 3, 2
        )
        query_feat = F.normalize(query_feat, p=2, dim=-1).unsqueeze(2)
        query_feat = _l2norm(query_feat, dim=-1)

        # b, s, c, h, w -> b, way, shot, c, h*w -> b, way, c, shot*h*w -> b, 1, way, c, shot*h*w
        support_feat = (
            support_feat.view(b, way_num, shot_num, c, h * w)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .view(b, way_num, c, shot_num * h * w)
        )
        support_feat = F.normalize(support_feat, p=2, dim=2).unsqueeze(1)
        support_feat = _l2norm(support_feat, dim=-2)

        # b, q, way, h*w, shot*h*w
        S = torch.matmul(query_feat, support_feat)

        M_q = S.shape[-2]
        M_s = S.shape[2] * S.shape[-1]
        S = S.permute(0, 1, 3, 2, 4).contiguous().view(b * q, M_q, M_s)

        N_examples, M_q, M_s = S.shape
        St = S.transpose(-2, -1)
        device = S.device

        T_sq = torch.exp(self.gamma * (S - S.max(-1, keepdim=True)[0]))
        T_sq = T_sq / T_sq.sum(-1, keepdim=True)  # row-wise stochastic
        T_qs = torch.exp(self.gamma2 * (St - St.max(-1, keepdim=True)[0])) # [b * q, M_s, M_q]
        T_qs = T_qs / T_qs.sum(-1, keepdim=True)  # row-wise stochastic

        T = torch.cat([
            torch.cat([torch.zeros((N_examples, M_s, M_s), device=device), T_sq.transpose(-2, -1)], dim=-1),
            torch.cat([T_qs.transpose(-2, -1), torch.zeros((N_examples, M_q, M_q), device=device)], dim=-1),
        ], dim=-2)
        katz = (torch.inverse(torch.eye(M_s + M_q, device=device)[None].repeat(N_examples, 1, 1) - self.katz_factor * T) - \
                torch.eye(M_s + M_q, device=S.device)[None].repeat(N_examples, 1, 1)) @ torch.ones(
            (N_examples, M_s + M_q, 1), device=device)
        partial_katz = katz.squeeze(-1)[:, :M_s] / katz.squeeze(-1)[:, :M_s].sum(-1, keepdim=True)
        predicts = partial_katz.view(N_examples, way_num, -1).sum(-1)

        return predicts

def _l2norm(x, dim=1, keepdim=True):
    return x / (1e-16 + torch.norm(x, 2, dim, keepdim))

class MCL(MetricModel):
    def __init__(self, encoding="FCN", gamma=20.0, gamma2=10.0, katz_factor=0.5, **kwargs):
        super(MCL, self).__init__(**kwargs)
        self.mcl_layer = MCLLayer(encoding, gamma, gamma2, katz_factor)
        self.loss_func = nn.NLLLoss()

    def set_forward(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2
        )

        output = self.mcl_layer(
            query_feat, support_feat, self.way_num, self.shot_num, self.query_num
        )
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc

    def set_forward_loss(self, batch):
        """

        :param batch:
        :return:
        """
        image, global_target = batch
        image = image.to(self.device)
        episode_size = image.size(0) // (
            self.way_num * (self.shot_num + self.query_num)
        )
        feat = self.emb_func(image)
        support_feat, query_feat, support_target, query_target = self.split_by_episode(
            feat, mode=2
        )

        output = self.mcl_layer(
            query_feat,
            support_feat,
            self.way_num,
            self.shot_num,
            self.query_num,
        )

        loss = self.loss_func(output, query_target.reshape(-1))
        acc = accuracy(output, query_target.reshape(-1))

        return output, acc, loss
