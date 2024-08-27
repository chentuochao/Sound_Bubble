import torch
import torch.nn as nn
from asteroid.losses.sdr import SingleSrcNegSDR


class SNRLosses(nn.Module):
    def __init__(self, name, **kwargs) -> None:
        super().__init__()
        self.name = name
        if name == 'sisdr':
            self.loss_fn = SingleSrcNegSDR('sisdr')
        elif name == 'snr':
            self.loss_fn = SingleSrcNegSDR('snr')
        # elif name == 'sdsdr':
        #     self.loss_fn = SingleSrcNegSDR('sdsdr')
        elif name == 'fused':
            self.loss1 = SingleSrcNegSDR('sisdr')
            self.loss2 = SingleSrcNegSDR('snr')
        elif name == "max_fused":
            self.loss1 = SingleSrcNegSDR('sisdr')
            self.loss2 = SingleSrcNegSDR('snr')
        elif name == "sdsdr":
            self.loss1 = SingleSrcNegSDR("snr")
            self.loss2 = SingleSrcNegSDR("sdsdr")
        elif name == "full":
            self.loss1 = SingleSrcNegSDR("snr")
            self.loss2 = SingleSrcNegSDR("sdsdr")
            self.loss3 = SingleSrcNegSDR('sisdr')
        else:
            assert 0, f"Invalid loss function used: Loss {name} not found"

    def forward(self, est: torch.Tensor, gt: torch.Tensor, **kwargs):
        """
        est: (B, C, T)
        gt: (B, C, T)
        """
        B, C, T = est.shape

        est = est.reshape(B*C, T)
        gt = gt.reshape(B*C, T)
        
        if self.name == "fused":
            return 0.5*self.loss1(est_target=est, target=gt) + 0.5*self.loss2(est_target=est, target=gt)
        elif self.name == "max_fused" or self.name == "sdsdr":
            return torch.maximum(self.loss1(est_target=est, target=gt), self.loss2(est_target=est, target=gt)  )
        elif self.name == "full":
            l1 = self.loss1(est_target=est, target=gt)
            l2 = self.loss2(est_target=est, target=gt)
            l3 = self.loss3(est_target=est, target=gt)

            return  0.5*l3 + 0.5* torch.maximum(l1, l2)
        else:
            return self.loss_fn(est_target=est, target=gt)
