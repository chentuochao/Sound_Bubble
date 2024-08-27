import os, glob
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import StochasticWeightAveraging
import math
import shutil

import src.utils as utils
import torchaudio

def main(args, hparams):
    # Setting all the random seeds to the same value.
    # This is important in a distributed training setting.
    # Each rank will get its own set of initial weights.
    # If they don't match up, the gradients will not match either,
    # leading to training that may not converge.
    pl.seed_everything(167)
    os.makedirs(args.run_dir, exist_ok = args.resume)

    # Initialize the model
    pl_module = utils.import_attr(hparams.pl_module)(**hparams.pl_module_args)
    if hasattr(hparams, "project_name"):
        project_name = hparams.project_name
    else:
        project_name = "AcousticBubble"
    # Init trainer
    wandb_run_name = os.path.basename(args.run_dir.rstrip('/'))
    wandb_logger = WandbLogger(
        project=project_name, save_dir=args.run_dir, name=wandb_run_name)

    # Init callbacks
    callbacks = [utils.import_attr(hparams.pl_logger)()]
    
    callbacks += [
        ModelCheckpoint( # Save last model
            dirpath=args.run_dir, monitor=None, save_last=True,
            auto_insert_metric_name=False,),
        ModelCheckpoint( # Save top 5 models
            dirpath=os.path.join(args.run_dir, 'best'),
            filename="best_epoch={epoch}-step={step}",
            monitor=pl_module.monitor,
            mode=pl_module.monitor_mode,
            auto_insert_metric_name=False,
            save_top_k=1)
    ]

    # If a scheduler is provided, log learning rate
    if 'scheduler' in hparams.pl_module_args and hparams.pl_module_args['scheduler'] is not None:
        print("Logging learning rate")
        lr_logger = LearningRateMonitor(logging_interval='epoch')
        callbacks += [lr_logger]
        
    # Gradient clipping
    grad_clip = 0
    if hasattr(hparams, 'grad_clip'):
        print("USING GRADIENT CLIPPING", hparams.grad_clip)
        grad_clip = hparams.grad_clip

    if hasattr(hparams, 'swa') and hparams.swa == True:
        print("USING STOCHASTIC WEIGHT AVERAGING")
        callbacks += [StochasticWeightAveraging(swa_lrs=1e-3, swa_epoch_start=0.5)]

    # Use the largest amoung of divices that divides batch size
    devices = math.gcd(torch.cuda.device_count(), hparams.batch_size)

    print(f"**** USING {devices} DEVICES ****")
    
    # Init trainer
    trainer = pl.Trainer(
        accelerator="gpu", devices=devices, strategy='ddp_find_unused_parameters_true', max_epochs=hparams.epochs,
        logger=wandb_logger, callbacks=callbacks, limit_train_batches=args.frac, gradient_clip_val=grad_clip,
        limit_val_batches=args.frac, limit_test_batches=args.frac)

    # Maximum 4 worker per GPU
    num_workers = min(len(os.sched_getaffinity(0)), 4)

    # Choose checkpoint to use
    # If provided, use checkpoint given
    ckpt_path = None
    if os.path.exists(os.path.join(args.run_dir, 'last.ckpt')):
        ckpt_path = os.path.join(args.run_dir, 'last.ckpt')

    # Initialize the train dataset
    train_ds = utils.import_attr(hparams.train_dataset)(**hparams.train_data_args, split='train')
    train_batch_size = hparams.batch_size // devices # Batch size per GPU
    print("train", len(train_ds))
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )


    # Initialize the validation dataset
    val_ds = utils.import_attr(hparams.val_dataset)(**hparams.val_data_args, split='val')
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=hparams.eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    print("val", len(val_ds))

    # from asteroid.losses.sdr import SingleSrcNegSDR
    # sisdrloss = SingleSrcNegSDR('sisdr')
    # for i in range(0, 10):
    #     inputs, targets = val_ds[i]
    #     mix = inputs['mixture']
    #     gt = targets['target']
    #     n_speakers = targets['num_target_speakers']
    #     print(i, targets["num_target_speakers"],-sisdrloss(mix[0:1], gt))
    #     torchaudio.save("./debug/gt{:02d}.wav".format(i), gt, 24000)
    #     torchaudio.save("./debug/mixture{:02d}.wav".format(i), mix, 24000)
    # raise KeyboardInterrupt

    # Create run dir and copy config file
    shutil.copyfile(args.config, os.path.join(args.run_dir, 'config.json'))

    # Train
    trainer.fit(pl_module, train_dl, val_dl, ckpt_path=ckpt_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training with an existing run_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Fraction of the dataset to use for train, val and test.')
    args = parser.parse_args()

    if os.environ.get('LOCAL_RANK', None) is None:
        if not args.resume:
            assert not os.path.exists(args.run_dir), \
                f"run_dir {args.run_dir} already exists. " \
                "Use --resume if you intend to resume an existing run."

    # Load hyperparameters
    hparams = utils.Params(args.config)

    # Run
    main(args, hparams)
