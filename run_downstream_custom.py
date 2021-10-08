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

parser.add_argument('--output_path', type=str, required=True)
parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--labelpath', type=str, required=True)
parser.add_argument('--pretrained_path', type=str, default=None)
parser.add_argument('--model_type', type=str, choices=['wav2vec', 'wav2vec2'], default='wav2vec2')

parser.add_argument('--save_top_k', type=int, default=1)
parser.add_argument('--cpu_torchscript', action='store_true')

args = parser.parse_args()
hparams = args

from downstream.Custom.trainer import DownstreamGeneral

if not os.path.exists(hparams.saving_path):
    os.makedirs(hparams.saving_path)
if not os.path.exists(hparams.output_path):
    os.makedirs(hparams.output_path)

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
    callbacks=[checkpoint_callback],
    resume_from_checkpoint=None,
    check_val_every_n_epoch=1,
    max_epochs=hparams.max_epochs,
    gpus=1
)
trainer.fit(model)

if hasattr(model, 'test_met'):
    trainer.test(model)
    met = model.test_met
    print("+++ SUMMARY +++")
    for nm, metric in zip(('UAR [%]', 'WAR [%]', 'macroF1 [%]', 'microF1 [%]'),
                          (met.uar*100, met.war*100, met.macroF1*100, met.microF1*100)):
        print(f"Mean {nm}: {np.mean(metric):.2f}")
        print(f"Std. {nm}: {np.std(metric):.2f}")
    WriteConfusionSeaborn(
        met.m,
        model.dataset.emoset,
        os.path.join(args.saving_path, 'confmat.png')
    )

if hasattr(model, 'valid_met'):
    trainer.checkpoint_connector.restore_model_weights(checkpoint_callback.best_model_path)
model.eval()
model.freeze()
if not args.cpu_torchscript:
    model.cuda()
example_inputs = torch.randn(10, 16000*8, device=model.device)
example_lengths = torch.randint(low=16000*4, high=16000*8, size=(10,), device=model.device)
m = torch.jit.trace(model.model.forward, (example_inputs, example_lengths))
torch.jit.save(m, os.path.join(args.output_path, "script.pt"))
torch.save(model.model.state_dict(), os.path.join(args.output_path, "eager.pt"))
