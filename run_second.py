from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pretrain.trainer import SecondPassEmoClassifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--saving_path', type=str, default='pretrain/checkpoints_second')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--dynamic_batch', action='store_true')
parser.add_argument('--training_step', type=int, default=120000)
parser.add_argument('--warmup_step', type=int, default=4000)
parser.add_argument('--maxseqlen', type=float, default=10.0)
parser.add_argument('--resume_checkpoint', type=str, default=None)
parser.add_argument('--precision', type=int, choices=[16, 32], default=32)
parser.add_argument('--num_clusters', type=str, default='8,64,512,4096')
parser.add_argument('--distributed', action='store_true')
parser.add_argument('--accelerator', type=str, default='ddp')
parser.add_argument('--use_bucket_sampler', action='store_true')
parser.add_argument('--train_bucket_size', type=int, default=50)
parser.add_argument('--val_bucket_size', type=int, default=20)
parser.add_argument('--unsupdatadir', type=str, default=None)
parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
parser.add_argument('--save_top_k', type=int, default=2)
parser.add_argument('--valid_split', type=float, default=1.0)

parser.add_argument('--w2v2_pretrain_path', type=str, default=None)

parser.add_argument('--datadir', type=str, required=True)
parser.add_argument('--labelpath', type=str, required=True)
args = parser.parse_args()
nclusters = [int(x) for x in args.num_clusters.split(',')]


checkpoint_callback = ModelCheckpoint(
    dirpath=args.saving_path,
    filename='w2v2-{epoch:02d}-{valid_loss:.2f}-{valid_acc:.2f}',
#    save_top_k=args.save_top_k,
    verbose=True,
#    monitor='valid_acc',
#    mode='max',
    save_last=True
)
wrapper = Trainer(
    precision=args.precision,
    amp_backend='native',
    callbacks=[checkpoint_callback],
    resume_from_checkpoint=args.resume_checkpoint,
    check_val_every_n_epoch=args.check_val_every_n_epoch,
    max_steps=args.training_step,
    gpus=(-1 if args.distributed else 1),
    accelerator=(args.accelerator if args.distributed else None),
    replace_sampler_ddp=False,
    logger=False
)
if args.w2v2_pretrain_path is None:
    model = SecondPassEmoClassifier(maxstep=args.training_step,
                                    batch_size=args.batch_size,
                                    lr=args.lr,
                                    warmup_step=args.warmup_step,
                                    nclusters=nclusters,
                                    maxseqlen=int(16000*args.maxseqlen),
                                    datadir=args.datadir,
                                    unsupdatadir=args.unsupdatadir,
                                    labeldir=args.labelpath,
                                    distributed=args.distributed,
                                    use_bucket_sampler=args.use_bucket_sampler,
                                    train_bucket_size=args.train_bucket_size,
                                    val_bucket_size=args.val_bucket_size,
                                    dynamic_batch=args.dynamic_batch,
                                    valid_split=args.valid_split)
else:
    model = SecondPassEmoClassifier.load_from_checkpoint(args.w2v2_pretrain_path, strict=False,
                                                         maxstep=args.training_step,
                                                         batch_size=args.batch_size,
                                                         lr=args.lr,
                                                         warmup_step=args.warmup_step,
                                                         nclusters=nclusters,
                                                         maxseqlen=int(16000*args.maxseqlen),
                                                         datadir=args.datadir,
                                                         unsupdatadir=args.unsupdatadir,
                                                         labeldir=args.labelpath,
                                                         distributed=args.distributed,
                                                         use_bucket_sampler=args.use_bucket_sampler,
                                                         train_bucket_size=args.train_bucket_size,
                                                         val_bucket_size=args.val_bucket_size,
                                                         dynamic_batch=args.dynamic_batch,
                                                         valid_split=args.valid_split)
    for linear in model.linearheads:
        linear.reset_parameters()
wrapper.fit(model)
