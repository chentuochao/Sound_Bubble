"""
The main training script for training on synthetic data
"""
import argparse
import multiprocessing
import os
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm  # pylint: disable=unused-import

from asteroid.metrics import get_metrics
from .network import Network



def compute_metrics(orig: torch.Tensor,
                    est: torch.Tensor,
                    gt: torch.Tensor,
                    sr: torch.Tensor):
    """
    input: (N, 1, t) (N, 1, t)
    """
    if gt.shape[1] != 1:
        N, C, t = gt.shape
        gt.reshape(N * C, 1, t)
        est.reshape(N * C, 1, t)

    gt = gt[:, 0].detach().cpu().numpy()
    est = est[:, 0].detach().cpu().numpy()
    orig = orig[:, 0].detach().cpu().numpy() # Take first channel of original input
    
    mask = (np.absolute(gt).max(axis=1) > 0)

    metrics = []

    # Only consider positive samples because of complications with computing SI-SNR
    # If there's at least one positive sample
    if np.sum(mask) > 0:
        gt = gt[mask]
        est = est[mask]
        orig = orig[mask]
    
        for i in range(gt.shape[0]):
            metrics_dict = get_metrics(orig[i], gt[i], est[i], sample_rate=sr, metrics_list=['si_sdr'])
            metrics.append(metrics_dict)

    return metrics

def train_epoch(model: nn.Module, device: torch.device,
                optimizer: optim.Optimizer,
                train_loader: torch.utils.data.dataloader.DataLoader,
                training_params: dict,
                epoch: int = 0, log_interval: int = 20) -> float:

    """
    Train a single epoch.
    """
    # Set the model to training.
    model.train()

    # Training loop
    losses = []
    interval_losses = []
    t1 = time.time()

    for batch_idx, (data, gt_inside, gt_outside) in enumerate(train_loader):
        data = data.to(device)
        gt_inside = gt_inside.to(device)

        # Reset grad
        optimizer.zero_grad()
        # data, means, stds = normalize_input(data)
        # Run through the model n_mics
        output_signal = model(data)
        # Un-normalize
        # output_signal = unnormalize_input(output_signal, means, stds)

        loss = model.module.loss(output_signal, gt_inside)

        interval_losses.append(loss.item())

        # Backpropagation
        loss.backward()

        # # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_params['gradient_clip'])

        # Update the weights
        optimizer.step()

        # Print the loss
        if batch_idx % log_interval == 0:
            t2 = time.time()
            
            print("Train Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f} \t Time taken: {:.4f}s ({} examples)".format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                np.mean(interval_losses),
                t2 - t1,
                log_interval * output_signal[0].shape[0] * (batch_idx > 0) + output_signal[0].shape[0] * (batch_idx == 0)))

            losses.extend(interval_losses)
            interval_losses = []
            t1 = time.time()


    return np.mean(losses)


def test_epoch(model: nn.Module, device: torch.device,
               test_loader: torch.utils.data.dataloader.DataLoader,
               sr: int,
               log_interval: int = 20) -> float:
    """
    Evaluate the network.
    """
    model.eval()
    test_loss = 0
    metrics = []
    with torch.no_grad():
        losses = []
        pos_losses = []
        neg_losses = []

        for batch_idx, (data, gt_inside, gt_outside) in enumerate(test_loader):
            data = data.to(device)
            gt_inside = gt_inside.to(device)
            gt_outside = gt_outside.to(device)

            # Normalize input, each batch item separately
            # data, means, stds = normalize_input(data)

            # Run through the model
            output_signal = model(data)
            # Un-normalize
            # output_signal = unnormalize_input(output_signal, means, stds)


            loss, pos_loss, neg_loss = model.module.loss(output_signal, gt_inside, True)
            test_loss = loss.item()
            losses.append(test_loss)
            if pos_loss is not None:
                pos_losses.append(pos_loss.item())
            if neg_loss is not None:
                neg_losses.append(neg_loss.item())
            # Compute metrics
            m = compute_metrics(data, output_signal, gt_inside, sr)
            metrics.extend(m)
        

            if batch_idx % log_interval == 0:
                print("Loss: {:.4f}".format(test_loss))

        average_loss = np.mean(losses)
        average_loss_pos = np.mean(pos_losses)
        average_loss_neg = np.mean(neg_losses)
        print("\nTest set: Average Loss: {:.4f}, pos={:.4f}, neg={:.4f}\n".format(average_loss, average_loss_pos, average_loss_neg) )

        return average_loss, metrics

