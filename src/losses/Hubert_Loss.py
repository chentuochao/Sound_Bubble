import os
from torchaudio.functional import resample

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, layer_norm
from transformers import AutoProcessor, HubertModel
from asteroid.losses.sdr import SingleSrcNegSDR


class HubertLoss(nn.Module):
    def __init__(self, device = 'cuda', fs = 24000, norm=False, output_feat = False, distance_function = "MSE", **kwargs) -> None:
        super().__init__()

        model =  HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.output_feat = output_feat
        self.fs = fs
        self.fs_new = 16000
        self.norm =norm

        self.model = model.to(device)


        ### freeze model params
        self.model.train()

        #for param in self.model.parameters():
        #    param.requires_grad = False
            

        self.model.feature_extractor._freeze_parameters()
        for param in self.model.parameters():
            param.requires_grad = False

        self.snr_loss = SingleSrcNegSDR('snr') 

        ###for
        #for name, param in self.model.named_parameters():
        #    print(name, param.requires_grad)

        if distance_function == "MSE":
            self.dis = nn.MSELoss()
        elif distance_function == "L1":
            self.dis = nn.L1Loss()
        elif distance_function == "Cos":
            self.dis = nn.CosineEmbeddingLoss()
        else:
            raise ValueError("Invalid distance function")

    def forward(self, est: torch.Tensor, gt: torch.Tensor): 
        #print(est.shape, gt.shape)
        gt = gt.squeeze(1)
        est = est.squeeze(1)
        


        if self.norm:
            gt = layer_norm(gt, gt.shape)
            est = layer_norm(est, gt.shape)
        if self.fs != self.fs_new:
            gt = resample(gt, self.fs, self.fs_new)
            est = resample(est, self.fs, self.fs_new)

        hidden_gt = self.model.feature_extractor(gt)
        hidden_est = self.model.feature_extractor(est)
        if self.output_feat:
            feat_gt = hidden_gt
            feat_est =  hidden_est
        else:
            feat_gt = self.model.feature_projection(hidden_gt.transpose(1, 2))
            feat_est = self.model.feature_projection(hidden_est.transpose(1, 2))

        #print(feat_est.shape, feat_gt.shape)
        loss =  self.dis(feat_est, feat_gt)
        #loss =  self.dis(feat_est, feat_gt)

        return loss


