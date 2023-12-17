import torch
import torch.nn.functional as F
from torch import autograd, nn
import numpy as np
from torch.nn import Module, Parameter
import math
# from utils.distributed import tensor_gather


class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets,box,boxnum, lut, cq, header, momentum, detectionscore):
        ctx.save_for_backward(inputs, targets,box,boxnum, lut, cq, header, momentum, detectionscore)
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)#jia

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets,box,boxnum, lut, cq, header, momentum, detectionscore = ctx.saved_tensors
        # inputs, targets = tensor_gather((inputs, targets))

        grad_inputs = None
        grad_box = None
        if ctx.needs_input_grad[0]:
            grad_outputs =grad_outputs.to(torch.half)
            lutt = lut.to(torch.half)
            cqq = cq.to(torch.half)
            grad_box = box.to(torch.half)
            grad_boxnum = boxnum.to(torch.half)
            grad_inputs = grad_outputs.mm(torch.cat([lutt, cqq], dim=0))
            if grad_inputs.dtype == torch.float16:
                grad_inputs = grad_inputs.to(torch.float32)
            if grad_box.dtype == torch.float16:
                grad_box = grad_box.to(torch.float32)
            if grad_boxnum.dtype == torch.float16:
                grad_boxnum = grad_box.to(torch.float32)

        h=0.05
        wmin=9999
        indx=-1
        k=0


        for x, y, dete in zip(inputs, targets, detectionscore):#x输入特征，y指身份标签
            allput = torch.cat([lut, cq])
            getscore = inputs.mm(lut.t())
            indx += 1
            if y < len(lut):
                if box[y]!= 0:
                    momentum =dete/(box[y]+dete)
                else:
                    momentum = 0.5
                momentum = torch.clip(dete, 0.4, 0.6)


                # if momentum>0.7:
                #     momentum = 0.7
                # if momentum<0.3:
                #     momentum = 0.3
                boxnum[y] +=1
                box[y] = (dete+box[y]*(boxnum[y]-1))/boxnum[y]
                lut[y] = (1 - momentum) * lut[y] + momentum * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None, None,grad_box,grad_boxnum


def oim(inputs, targets, box,boxnum, lut, cq, header, momentum=0.5,detectionscore=0.5):
    return OIM.apply(inputs, targets, box,boxnum,lut, cq, torch.tensor(header), torch.tensor(momentum), detectionscore.detach())#detectionscore.clone().detach()


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar,detectionscore):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.detectionscore = detectionscore
        self.m = 0.4
        self.t = 0.1
        self.eps = 1e-3
        #两个缓冲区
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
        self.register_buffer("box", torch.zeros(self.num_pids, 1))
        self.register_buffer("boxnum", torch.zeros(self.num_pids, 1))

        self.header_cq = 0

    def forward(self, inputs, roi_label,detectionscore):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1
        inds = label >= 0

        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)
        # detectionscore = detectionscore[inds.unsqueeze(1).expand_as(detectionscore)].view(-1, 1)#jiancedefencaijia
        projected = oim(inputs, label, self.box,self.boxnum,self.lut, self.cq, self.header_cq, momentum=self.momentum, detectionscore=detectionscore)#jia


        lianhe = self.lut
        # #
        # #
        # #
        projected *= self.oim_scalar
        self.header_cq = (
            self.header_cq + (label >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled

        B, K = getmaxlut.shape
        loss1 = F.mse_loss(getmaxlut - torch.eye(K).unsqueeze(0).cuda(1), torch.zeros(B, K).cuda(1), size_average=False)/(K*K)

        loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
        loss_oim = loss_oim + loss1

        return loss_oim, lianhe#jia








