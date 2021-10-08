from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pretrain.trainer import PretrainedEmoClassifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--saving_path', type=str, default='pretrain/checkpoints')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
parser.add_argument('--training_step', type=int, default=300000)
parser.add_argument('--save_top_k', type=int, default=1)
parser.add_argument('--resume_checkpoint', type=str, default=None)
parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
parser.add_argument('--valid_split', type=float, default=0.9)
parser.add_argument('--labeling_method', type=str, default='soft', choices=['hard', 'soft'])

parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--labelpath', type=str, required=True)
parser.add_argument('--wav2vecpath', type=str, required=True)
args = parser.parse_args()


checkpoint_callback = ModelCheckpoint(
    dirpath=args.saving_path,
    filename='w2v-{epoch:02d}-{valid_loss:.2f}-{valid_acc:.2f}',
    save_top_k=args.save_top_k,
    verbose=True,
    monitor='valid_loss',
    mode='min'
)
wrapper = Trainer(
    precision=args.precision,
    amp_backend='native',
    callbacks=[checkpoint_callback],
    resume_from_checkpoint=args.resume_checkpoint,
    check_val_every_n_epoch=args.check_val_every_n_epoch,
    max_steps=args.training_step,
    gpus=1,
    logger=False
)
model = PretrainedEmoClassifier(maxstep=args.training_step,
                                batch_size=args.batch_size,
                                lr=args.lr,
                                datadir=args.datadir,
                                labeldir=args.labelpath,
                                modelpath=args.wav2vecpath,
                                labeling_method=args.labeling_method,
                                valid_split=args.valid_split)
wrapper.fit(model)
