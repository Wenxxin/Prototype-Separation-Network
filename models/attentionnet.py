import torch
import torch.nn.functional as F
from torch import autograd, nn
import numpy as np

# from utils.distributed import tensor_gather

def compute(part1, part2):
    projected = part1.mm(part1.t())
    projected1 = part1.mm(part2.t())

    diag = torch.diag(projected)
    diag = torch.diag_embed(diag)
    projected = projected - diag

    diag = torch.diag(projected1)
    diag = torch.diag_embed(diag)
    projected = projected + diag
    projected *= 0.25

    projected = projected.float()
    soft_out = F.softmax(projected, dim=1)  # 给每个样本的pred向量做指数归一化---softmax
    key = soft_out.diag()+0.001
    a = key.shape
    loss1 = -torch.log(key).sum() / a[0]
    return loss1

#转为余弦相似度,但个没法做很多运算

def computeloss(part,lut,target):
    # lut = lut[:, :256]
    pro = part.mm(lut.t())
    pro *= 10.0#duide
    part_oim = F.cross_entropy(pro, target, ignore_index=5554)
    # part_oim = F.cross_entropy(part, target, ignore_index=5554)
    return part_oim

def compare(inputs1,inputs2,lut,target):
    out=0
    for m, n, y in zip(inputs1, inputs2, target):
        if y < len(lut):
            m = m.unsqueeze(dim=0)
            n = n.unsqueeze(dim=0)
            # keym = m / torch.norm(m, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模
            # keyn = n / torch.norm(n, dim=-1, keepdim=True)  # 方差归一化，即除以各自的模
            # key= keym.mm(lut[y].t())+ keyn.mm(lut[y].t())
            keym = torch.cosine_similarity(m, lut[y])
            keyn = torch.cosine_similarity(n, lut[y])
            key = torch.abs(keym)+torch.abs(keyn)

    return out

#单个像素可以用compare计算
#可能有同身份的框同时出现
#用查询表里的值做相关运算

#不是标记行人的类型没有得分

def correlation(inputs,map,lut,target):
    sum = 0
    n =0
    zero = torch.zeros(256)
    zero = zero.cuda(0)
    for m, y, z in zip(inputs, target,map):
        if y < len(lut):
            if(torch.equal(lut[y],zero)):
                q=0
            else:
                t = lut[y].unsqueeze(dim=1).unsqueeze(dim=1)
                keym = torch.cosine_similarity(m, t, dim=0)
                submap = torch.abs(keym - z)
                sum = submap.sum()/324 + sum
                n = n + 1

            # sum = torch.abs(tmap[i][j] - map[i][j])
    # if(n!=0):
    #     sum = sum/n
    return sum

class attention(nn.Module):
    def __init__(self):
        super(attention, self).__init__()
        self.atten = SpatialAttentionModule()
        # self.atten = CBAM()
    def forward(self, input):#,lut,label
        # targets = torch.cat(label)
        # label = targets - 1  # background label = -1
        # inds = label >= 0
        # label = label[inds]
        #
        # map = self.atten(input)
        # feature = input * map

        net = CBAM().cuda(0)
        out,map = net(input)
        feature = out * map

        # map = ChannelAttentionModule(input)
        # feature = input * map

        return feature,map


class mapaa(nn.Module):
    def __init__(self):
        super(mapaa, self).__init__()
        self.atten = SpatialAttentionModule()
        self.conv111 = nn.Conv2d(2048, 256, kernel_size=1)
    def forward(self,feature,input,lut,label):#
        targets = torch.cat(label)
        label = targets - 1  # background label = -1
        inds = label >= 0
        label = label[inds]
        input = input.squeeze(dim = 1)
        input = input[inds.unsqueeze(1).unsqueeze(1).expand_as(input)].view(-1, 7, 7)
        map = input

        feature = self.conv111(feature)
        feature = feature[inds.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand_as(feature)].view(-1,256, 7, 7)
        # print(feature.shape)
        loss = correlation(feature,map,lut,label)+0.01

        return loss



class apool(nn.Module):
    def __init__(self):
        super(apool, self).__init__()

    def forward(self, input,dim):
        feature = input.reshape(-1,2048,49)
        a, idx = torch.sort(feature,descending=True,dim = 2)  # descending为alse，升序，为True，降序
        a = a[:,:,:4]
        feature = a.sum(dim = 2)/4
        feature = feature.unsqueeze(dim = 2).unsqueeze(dim = 2)
        # feature = F.adaptive_max_pool2d(input, 1)

        return feature

# 通道注意力模块
class ChannelAttentionModule(nn.Module):
    def __init__(self, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        channel = 2048
        mid_channel = channel // reduction
        # mid_channel = 1024
        # print(mid_channel)
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2).unsqueeze(3)
        return self.sigmoid(avgout + maxout)


# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


# CBAM模块
class CBAM(nn.Module):
    def __init__(self):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule()
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        map = self.spatial_attention(out)
        return out,map





