import os
import argparse
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.outputlib import WriteConfusionSeaborn
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--max_epochs', type=int, default=15)
parser.add_argument('--maxseqlen', type=float, default=10)
parser.add_argument('--nworkers', type=int, default=4)
parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
parser.add_argument('--saving_path', type=str, default='downstream/checkpoints/custom')

parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--labeldir', type=str, required=True)
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--model_type', type=str, choices=['wav2vec', 'wav2vec2'], default='wav2vec2')

parser.add_argument('--save_top_k', type=int, default=1)
parser.add_argument('--num_exps', type=int, default=1)

parser.add_argument('--outputfile', type=str, default=None)

args = parser.parse_args()
hparams = args

from downstream.Custom.trainer import DownstreamGeneral

if not os.path.exists(hparams.saving_path):
    os.makedirs(hparams.saving_path)

nfolds = len(os.listdir(hparams.labeldir))
for foldlabel in os.listdir(hparams.labeldir):
    assert foldlabel[-5:] == '.json'
metrics, confusion = np.zeros((4, args.num_exps, nfolds)), 0.
for exp in range(args.num_exps):
    for ifold, foldlabel in enumerate(os.listdir(hparams.labeldir)):
        print (f"Running experiment {exp+1} / {args.num_exps}, fold {ifold+1} / {nfolds}...")
        hparams.labelpath = os.path.join(hparams.labeldir, foldlabel)
        model = DownstreamGeneral(hparams)
        checkpoint_callback = ModelCheckpoint(
            dirpath=hparams.saving_path,
            filename='{epoch:02d}-{valid_loss:.3f}-{valid_UAR:.5f}' if hasattr(model, 'valid_met') else None,
            save_top_k=args.save_top_k if hasattr(model, 'valid_met') else 0,
            verbose=True,
            save_weights_only=True,
            monitor='valid_UAR' if hasattr(model, 'valid_met') else None,
            mode='max'
        )

        trainer = Trainer(
            precision=args.precision,
            amp_backend='native',
            callbacks=[checkpoint_callback] if hasattr(model, 'valid_met') else None,
            checkpoint_callback=hasattr(model, 'valid_met'),
            resume_from_checkpoint=None,
            check_val_every_n_epoch=1,
            max_epochs=hparams.max_epochs,
            num_sanity_val_steps=2 if hasattr(model, 'valid_met') else 0,
            gpus=1,
            logger=False
        )
        trainer.fit(model)

        if hasattr(model, 'valid_met'):
            trainer.test()
        else:
            trainer.test(model)
        met = model.test_met
        metrics[:, exp, ifold] = np.array([met.uar*100, met.war*100, met.macroF1*100, met.microF1*100])
        confusion += met.m

outputstr = "+++ SUMMARY +++\n"
for nm, metric in zip(('UAR [%]', 'WAR [%]', 'macroF1 [%]', 'microF1 [%]'), metrics):
    outputstr += f"Mean {nm}: {np.mean(metric):.2f}\n"
    outputstr += f"Fold Std. {nm}: {np.mean(np.std(metric, 1)):.2f}\n"
    outputstr += f"Fold Median {nm}: {np.mean(np.median(metric, 1)):.2f}\n"
    outputstr += f"Run Std. {nm}: {np.std(np.mean(metric, 1)):.2f}\n"
    outputstr += f"Run Median {nm}: {np.median(np.mean(metric, 1)):.2f}\n"
if args.outputfile:
    with open(args.outputfile, 'w') as f:
        f.write(outputstr)
else:
    print (outputstr)

#This may cause trouble if emotion categories are not consistent across folds?
WriteConfusionSeaborn(
    confusion,
    model.dataset.emoset,
    os.path.join(args.saving_path, 'confmat.png')
)
