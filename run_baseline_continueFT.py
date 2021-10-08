from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pretrain.trainer import ContinueFinetuningBaseline
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--saving_path', type=str, default='pretrain/checkpoints_baseline')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--training_step', type=int, default=120000)
parser.add_argument('--warmup_step', type=int, default=4000)
parser.add_argument('--maxseqlen', type=float, default=10.0)
parser.add_argument('--resume_checkpoint', type=str, default=None)
parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--accelerator', type=str, default='ddp')
parser.add_argument('--train_bucket_size', type=int, default=50)
parser.add_argument('--val_bucket_size', type=int, default=20)
parser.add_argument('--use_additional_obj', action='store_true')
parser.add_argument('--use_bucket_sampler', action='store_true')
parser.add_argument('--unsupdatadir', type=str, default=None)
parser.add_argument('--num_clusters', type=str, default=None)
#parser.add_argument('--val_check_interval', type=int, default=100)
parser.add_argument('--save_top_k', type=int, default=2)

parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--labelpath', type=str, default=None)
args = parser.parse_args()
nclusters = None
if args.num_clusters:
    nclusters = [int(x) for x in args.num_clusters.split(',')]

checkpoint_callback = ModelCheckpoint(
    dirpath=args.saving_path,
    filename='w2v2-{epoch:02d}-{valid_loss:.2f}',
#    save_top_k=args.save_top_k,
    verbose=True,
#    monitor='valid_loss',
#    mode='min',
    save_last=True
)
wrapper = Trainer(
    precision=args.precision,
    amp_backend='native',
    callbacks=[checkpoint_callback],
    resume_from_checkpoint=args.resume_checkpoint,
#    val_check_interval=args.val_check_interval,
    max_steps=args.training_step,
    gpus=(-1 if args.distributed else 1),
    accelerator=(args.accelerator if args.distributed else None),
    replace_sampler_ddp=False,
    logger=False
)
model = ContinueFinetuningBaseline(maxstep=args.training_step,
                                   batch_size=args.batch_size,
                                   lr=args.lr,
                                   warmup_step=args.warmup_step,
                                   maxseqlen=int(16000*args.maxseqlen),
                                   nclusters=nclusters,
                                   datadir=args.datadir,
                                   labelpath=args.labelpath,
                                   distributed=args.distributed,
                                   use_bucket_sampler=args.use_bucket_sampler,
                                   train_bucket_size=args.train_bucket_size,
                                   val_bucket_size=args.val_bucket_size,
                                   use_additional_obj=args.use_additional_obj)
wrapper.fit(model)
