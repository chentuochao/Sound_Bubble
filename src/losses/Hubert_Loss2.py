import fairseq
import os
from torchaudio.functional import resample

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss, layer_norm



class HubertLoss(nn.Module):
    def __init__(self, use_gpu = True, fs = 24000, norm=False, output_feat = True, model_path = "../Large_Model/hubert_base_ls960.pt",  distance_function = "MSE", **kwargs) -> None:
        super().__init__()

        models = fairseq.checkpoint_utils.load_model_ensemble([model_path])
        self.model = models[0][0]
        self.output_feat = output_feat
        self.fs = fs
        self.fs_new = 16000
        self.norm =norm

        if use_gpu:
            self.model = self.model.cuda()

        if distance_function == "MSE":
            self.dis = nn.MSELoss()
        elif distance_function == "L1":
            self.dis = nn.L1Loss()
        elif distance_function == "Cos":
            self.dis = nn.CosineEmbeddingLoss()
        else:
            raise ValueError("Invalid distance function")

    def forward(self, est: torch.Tensor, gt: torch.Tensor): 

        if self.norm:
            gt = layer_norm(gt, gt.shape)
            est = layer_norm(est, gt.shape)

        features_gt, _ = self.model.extract_features(gt, ret_conv = self.output_feat)
        features_est, _ = self.model.extract_features(est, ret_conv = self.output_feat)


        loss = self.dis(features_est, features_gt)

        return loss


