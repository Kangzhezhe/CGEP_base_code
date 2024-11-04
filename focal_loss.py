import torch
from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, preds, labels):
        eps = 1e-7
        # 计算 log-softmax
        preds_logsoft = F.log_softmax(preds, dim=1)
        # 取 softmax 概率值
        preds_softmax = torch.exp(preds_logsoft)
        # 选择真实类别对应的概率
        preds_softmax = preds_softmax.gather(1, labels.unsqueeze(1)).squeeze(1)
        preds_logsoft = preds_logsoft.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        # 处理 alpha 参数
        if self.alpha is not None:
            if isinstance(self.alpha, (list, torch.Tensor)):
                # 若 alpha 为列表或张量，则为类别权重
                alpha = self.alpha[labels]
            else:
                alpha = self.alpha
        else:
            alpha = 1.0

        # 计算 Focal Loss
        loss = -alpha * torch.pow(1 - preds_softmax, self.gamma) * preds_logsoft

        if self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()
