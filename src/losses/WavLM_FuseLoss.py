import os
from torchaudio.functional import resample

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, layer_norm
from transformers import AutoProcessor, WavLMModel
from asteroid.losses.sdr import SingleSrcNegSDR


class WavLMFuseLoss(nn.Module):
    def __init__(self, device = 'cuda', fs = 24000, norm=False, output_feat = False, scale = 0.5, neg_weight = 25, distance_function = "MSE", **kwargs) -> None:
        super().__init__()

        model =  WavLMModel.from_pretrained(pretrained_model_name_or_path = "microsoft/wavlm-base", cache_dir = "/gscratch/intelligentsystems/tuochao/Large_Model")
        self.output_feat = output_feat
        self.fs = fs
        self.fs_new = 16000
        self.norm =norm
        self.scale = scale
        self.neg_weight = neg_weight
        self.model = model.to(device)


        ### freeze model params
        self.model.train()

        #for param in self.model.parameters():
        #    param.requires_grad = False
            

        self.model.freeze_feature_extractor()
        for param in self.model.parameters():
            param.requires_grad = False

        self.snr_loss = SingleSrcNegSDR('snr') 
        self.lp_loss = nn.L1Loss()#LogPowerLoss()
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
        # print(est.shape, gt.shape)
        gt = gt.squeeze(1)
        est = est.squeeze(1)


        comp_loss = torch.zeros((est.shape[0]), device=est.device)
        mask = (torch.max(torch.abs(gt), dim=1)[0] == 0)
        #print(mask)
        # loss1 = self.snr_loss(est_target=est, target=gt)
        # If there's at least one negative sample
        if any(mask):
            est_neg, gt_neg = est[mask], gt[mask]
            neg_loss = self.lp_loss(est_neg, gt_neg)
            #print(neg_loss, self.neg_weight)
            comp_loss[mask] = neg_loss * self.neg_weight
            
        # If there's at least one positive sample
        if any((~ mask)):
            est_pos, gt_pos = est[~mask], gt[~mask]
            pos_loss = self.snr_loss(est_pos, gt_pos)
            #print(pos_loss,self.scale)
            # Compute_joint_loss
            comp_loss[~mask] = self.scale * pos_loss

        if self.norm:
            gt = layer_norm(gt, gt.shape)
            est = layer_norm(est, gt.shape)
        if self.fs != self.fs_new:
            gt = resample(gt, self.fs, self.fs_new)
            est = resample(est, self.fs, self.fs_new)
        hidden_gt = self.model.feature_extractor(gt)
        hidden_est = self.model.feature_extractor(est)
        #print(hidden_gt.shape)
        #print(hidden_est.shape)
        if self.output_feat:
            feat_gt = hidden_gt
            feat_est =  hidden_est
        else:
            feat_gt, norm_gt = self.model.feature_projection(hidden_gt.transpose(1, 2))
            feat_est, norm_est = self.model.feature_projection(hidden_est.transpose(1, 2))

        #print(feat_est.shape, feat_gt.shape)
        # print(comp_loss.mean(), 10*self.dis(feat_est, feat_gt))
        loss =  comp_loss.mean() + 10*self.dis(feat_est, feat_gt)
        #loss =  self.dis(feat_est, feat_gt)

        return loss


