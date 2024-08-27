from asteroid.losses.sdr import SingleSrcNegSDR
import torch
import torch.nn as nn
import math

class CompositeLoss(nn.Module):
    def __init__(self, r = 0.95, loss_type = "sisdr") -> None:
        super().__init__()
        self.l1_loss = nn.L1Loss(reduce=False)
        self.loss_type = loss_type
        if loss_type == "sdsdr":
            self.snrloss = SingleSrcNegSDR("snr")
            self.sdsdrloss = SingleSrcNegSDR("sdsdr")
            self.sisdrloss = SingleSrcNegSDR("sisdr")
        else:
            self.sisdrloss = SingleSrcNegSDR(loss_type)
        self.r = r
    
    def forward(self, est: torch.Tensor, gt: torch.Tensor, decompose=False, **kwargs):
        """
        input: (N, 1, t) (N, 1, t)
        """
        output = est
        assert gt.shape[1] == 1
        assert output.shape[1] == 1

        gt = gt[:, 0]
        output = output[:, 0]
        
        mask = (torch.absolute(gt).max(dim=1)[0] == 0)

        l1loss = self.l1_loss(output, gt)
        if self.loss_type == "sdsdr":
            snrloss = self.snrloss(output, gt)
            sdsdrloss = self.sdsdrloss(output, gt)
            sisdrloss = self.sisdrloss(output, gt)
            sisdrloss = 0.75 *sisdrloss + 0.25*torch.maximum(snrloss, sdsdrloss)
        else:
            sisdrloss = self.sisdrloss(output, gt)
        
        comp_loss = 0
        neg_loss = None
        pos_loss  = None

        # print(mask)
        # print((~mask))

        # If there's at least one negative sample
        if any(mask):
            comp_loss += 30*torch.mean(l1loss[mask])
            neg_loss = 30*torch.mean(l1loss[mask])
            
        # If there's at least one positive sample
        if any((~ mask)):
            comp_loss += torch.mean(l1loss[~ mask]) * self.r + torch.mean(sisdrloss[~ mask]) * (1 - self.r)
            pos_loss = torch.mean(l1loss[~ mask]) * self.r + torch.mean(sisdrloss[~ mask]) * (1 - self.r)
        if decompose:
            return comp_loss, pos_loss, neg_loss
        else:
            return comp_loss