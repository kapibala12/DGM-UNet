import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary


__all__ = ['BCEDiceLoss']



class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

# 深监督损失
class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=False):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        # assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        # assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        # if self.weight_factors is None:
        #     weights = [1] * len(x)
        # else:
        #     weights = self.weight_factors

        if self.weight_factors:
            weights = nn.Parameter(torch.tensor([0.6, 0.2, 0.1, 0.1, 0.05]), requires_grad=True)
            weights=F.softmax(weights, dim=0)
        else:
            weights = [1] * len(x)

        # l = weights[0] * self.loss(x[0], y[0])
        l = torch.mul(weights[0], self.loss(x[0], y))
        for i in range(1, len(x)):
            if weights[i] != 0:
                l += weights[i] * self.loss(x[i], y)
        return l
