from utils.helper_funcs import multilabel2vec
import os
from torch.utils import data
import torch
import json
import numpy as np
from collections import Counter
import soundfile as sf
import random
from pathlib import Path
from torch.utils.data.dataloader import default_collate
import math
from tqdm import tqdm

class PretrainEmoDataset(data.Dataset):
    def __init__(self, datadir, labeldir, returnname=False, maxseqlen=12*16000, labeling_method='hard'):
        with open(labeldir, 'r') as f:
            self.label = json.load(f) #{wavname: {emotion: number}} or {wavname: emotion}
        if list(self.label.keys()) == ['Train', 'Val', 'Test']: #Format of fine-tuning, only use training set
            self.label = self.label['Train']
        self.datasetbase = [x for x in os.listdir(datadir) if x in self.label]
        self.dataset = [os.path.join(datadir, x) for x in self.datasetbase]
        self.returnname = returnname
        self.maxseqlen = maxseqlen
        if type(list(self.label.values())[0]) == str:
            print (list(self.label.values())[0])
            self.emos = Counter([emo for emo in self.label.values()])
        else:
            self.emos = Counter([k for sparse_emo in self.label.values() for k in sparse_emo.keys()])
        self.emoset = list(self.emos.keys())
        self.nemos = len(self.emoset)
        self.labeling_method = labeling_method
        self.labeldict = {k: i for i, k in enumerate(self.emoset)}

        #Print statistics:
        print ('----Involved Emotions----')
        for k, v in self.emos.items():
            print (f'{k}: {v} examples')
        l = len(self.dataset)
        print (f'Total {l} examples')
        print ('----Examples Involved----')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        dataname = self.dataset[i]
        wav, _sr = sf.read(dataname)
        _label = self.label[self.datasetbase[i]]
        if type(_label) == str:
            _label = { _label: 1.0 }
        #turn into multi-class vector
        p = multilabel2vec(_label, self.labeldict)
        #random sample a label according to the label distribution
        if self.labeling_method == 'hard':
            label = np.argmax(p)
        elif self.labeling_method == 'soft':
            label = np.random.choice(len(self.emoset), p=p)
        if not self.returnname:
            return wav.astype(np.float32), label
        return wav.astype(np.float32), label, self.datasetbase[i]

    def seqCollate(self, batch):
        getlen = lambda x: x[0].shape[0]
        max_seqlen = max(map(getlen, batch))
        target_seqlen = min(self.maxseqlen, max_seqlen)
        def trunc(x):
            x = list(x)
            if x[0].shape[0] >= target_seqlen:
                x[0] = x[0][:target_seqlen]
                output_length = target_seqlen
            else:
                output_length = x[0].shape[0]
                over = target_seqlen - x[0].shape[0]
                x[0] = np.pad(x[0], [0, over])
            ret = (x[0], x[1], output_length)
            return ret
        batch = list(map(trunc, batch))
        return default_collate(batch)

    def truncCollate(self, batch):
        getlen = lambda x: x[0].shape[0]
        min_seqlen = min(map(getlen, batch))
        def trunc(x):
            x = list(x)
            if x[0].shape[0] == min_seqlen:
                return x
            over = (x[0].shape[0] - min_seqlen)
            start = np.random.randint(over)
            x[0] = x[0][start: start + min_seqlen]
            return tuple(x)
        batch = list(map(trunc, batch))
        return default_collate(batch)

class UnlabeledDataset(data.Dataset):
    def __init__(self, datadir, returnname=False):
        self.datasetbase = [str(x)[len(datadir)+1:] for x in Path(datadir).rglob('*.wav')]
        self.dataset = [os.path.join(datadir, x) for x in self.datasetbase]
        self.returnname = returnname

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        dataname = self.dataset[i]
        wav, _sr = sf.read(dataname)
        if not self.returnname:
            return wav.astype(np.float32)
        return wav.astype(np.float32), self.datasetbase[i]

class MixedDataset(data.Dataset):
    def __init__(self, datadir, unsupdatadir, labelpath=None):
        if not labelpath:
            self.datasetbase = [x for x in os.listdir(datadir) if x[-4:] == '.wav']
        else:
            with open(labelpath, 'r') as f:
                label = json.load(f)
            self.datasetbase = list(label['Train'].keys())
        self.dataset = [os.path.join(datadir, x) for x in self.datasetbase]
        if unsupdatadir:
            unsupdatadir = unsupdatadir.rstrip('/')
            self.unsupdatasetbase = [str(x)[len(unsupdatadir)+1:] for x in Path(unsupdatadir).rglob('*.wav')]
            self.unsupdataset = [os.path.join(unsupdatadir, x) for x in self.unsupdatasetbase]
            self.datasetbase = self.datasetbase + self.unsupdatasetbase
            self.dataset = self.dataset + self.unsupdataset
        #Print statistics:
        l = len(self.dataset)
        print (f'Total {l} examples')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        dataname = self.dataset[i]
        wav, _sr = sf.read(dataname)
        return wav.astype(np.float32), self.datasetbase[i]

class SecondPhaseEmoDataset(data.Dataset):
    def __init__(self, datadir, unsupdatadir,
                 labeldir, maxseqlen, returnname=False, final_length_fn=None):
        with open(labeldir, 'r') as f:
            self.label = json.load(f) #{wavname: [cluster labels]}
        self.datasetbase = [x for x in os.listdir(datadir) if x[-4:] == '.wav']
        self.datasetbase = [x for x in self.datasetbase if x in self.label.keys()]
        self.dataset = [os.path.join(datadir, x) for x in self.datasetbase]
        self.returnname = returnname
        self.maxseqlen = maxseqlen
        self.final_length_fn = final_length_fn
        if unsupdatadir:
            unsupdatadir = unsupdatadir.rstrip('/')
            self.unsupdatasetbase = [str(x)[len(unsupdatadir)+1:] for x in Path(unsupdatadir).rglob('*.wav')]
            self.unsupdatasetbase = [x for x in self.unsupdatasetbase if x in self.label.keys()]
            self.unsupdataset = [os.path.join(unsupdatadir, x) for x in self.unsupdatasetbase]
            self.dataset = self.dataset + self.unsupdataset
            self.datasetbase = self.datasetbase + self.unsupdatasetbase

        #Print statistics:
        l = len(self.dataset)
        print (f'Total {l} examples')

        self.lengths = []
        print ("Loading over the dataset once...")
        for dataname in tqdm(self.dataset):
            wav, _sr = sf.read(dataname)
            self.lengths.append(len(wav))
        avglen = float(sum(self.lengths)) / len(self.lengths) / 16000
        print (f"Average duration of audio: {avglen} sec")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        dataname = self.dataset[i]
        wav, _sr = sf.read(dataname)
        label = np.array(self.label[self.datasetbase[i]], dtype=np.int_)
        if not self.returnname:
            return wav.astype(np.float32), label
        return wav.astype(np.float32), label, self.datasetbase[i]

    def seqCollate(self, batch):
        getlen = lambda x: x[0].shape[0]
        max_seqlen = max(map(getlen, batch))
        target_seqlen = min(self.maxseqlen, max_seqlen)
        nlabel = self.final_length_fn(target_seqlen)
        def trunc(x):
            x = list(x)
            if x[0].shape[0] >= target_seqlen:
                #RandomCrop
                start_point = random.randint(0, x[0].shape[0] - target_seqlen)
                label_start_point = self.final_length_fn(start_point) #Approximate label start
                x[0] = x[0][start_point: start_point+target_seqlen]
                output_length = target_seqlen
                x[1] = x[1][:, label_start_point: label_start_point+nlabel]
            else:
                output_length = x[0].shape[0]
                over = target_seqlen - x[0].shape[0]
                x[0] = np.pad(x[0], [0, over])
            #Double check labels
            if x[1].shape[1] >= nlabel:
                x[1] = x[1][:, :nlabel]
            else:
                over = nlabel - x[1].shape[1]
                x[1] = np.pad(x[1], [[0, 0], [0, over]], constant_values=-100)
            if self.returnname:
                ret = (x[0], x[1], output_length, x[2])
            else:
                ret = (x[0], x[1], output_length)
            return ret
        batch = list(map(trunc, batch))
        return default_collate(batch)

class BaselineDataset(data.Dataset):
    def __init__(self, datadir, labelpath, maxseqlen):
        if not labelpath:
            self.datasetbase = [x for x in os.listdir(datadir) if x[-4:] == '.wav']
        else:
            with open(labelpath, 'r') as f:
                label = json.load(f)
            print (labelpath)
            self.datasetbase = list(label['Train'].keys())
        self.dataset = [os.path.join(datadir, x) for x in self.datasetbase]
        self.maxseqlen = maxseqlen

        #Print statistics:
        l = len(self.dataset)
        print (f'Total {l} examples')

        self.lengths = []
        print ("Loading over the dataset once...")
        for dataname in tqdm(self.dataset):
            wav, _sr = sf.read(dataname)
            self.lengths.append(len(wav))
        avglen = float(sum(self.lengths)) / len(self.lengths) / 16000
        print (f"Average duration of audio: {avglen} sec")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        dataname = self.dataset[i]
        wav, _sr = sf.read(dataname)
        return wav.astype(np.float32)

    def seqCollate(self, batch):
        getlen = lambda x: x.shape[0]
        max_seqlen = max(map(getlen, batch))
        target_seqlen = min(self.maxseqlen, max_seqlen)
        def trunc(x):
            if x.shape[0] >= target_seqlen:
                #RandomCrop
                start_point = random.randint(0, x.shape[0] - target_seqlen)
                x = x[start_point: start_point+target_seqlen]
                output_length = target_seqlen
            else:
                output_length = x.shape[0]
                over = target_seqlen - x.shape[0]
                x = np.pad(x, [0, over])
            ret = (x, output_length)
            return ret
        batch = list(map(trunc, batch))
        return default_collate(batch)

#Samplers
def StandardSampler(dataset, shuffle, distributed=False,
                    world_size=None, rank=None, dynamic_batch=False):
    assert not dynamic_batch, "We currently only support dynamic batching for bucket sampler, please unflag the option \"--dynamic_batch\" or use bucket sampler \"--use_bucket_sampler\"..."
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle,
                                                   num_replicas=world_size, rank=rank)
    if shuffle:
        return data.RandomSampler(dataset)
    return data.SequentialSampler(dataset)

def RandomBucketSampler(nbuckets, length, batch_size, drop_last, distributed=False,
                        world_size=None, rank=None, dynamic_batch=True):
    if dynamic_batch:
        assert not distributed, "We currently don't support dynamic batching in Multi-GPU distributed training, please unflag the option \"--dynamic_batch\"..."
        batch_size = int(16000 * float(batch_size) / 64. * 250)
        print (f"Using dynamic batching, the batch duration is {float(batch_size) / 16000} seconds")
    if distributed:
        return DistributedRandomBucketSampler(nbuckets, length, batch_size, drop_last, world_size, rank)
    return SingleRandomBucketSampler(nbuckets, length, batch_size, drop_last, dynamic_batch)

class SingleRandomBucketSampler(data.Sampler):
    def __init__(self, nbuckets, length, batch_size, drop_last, dynamic_batch):
        self.length = length
        self.batch_size = batch_size
        self.dynamic_batch = dynamic_batch
        self.drop_last = drop_last
        indices = np.argsort(length)
        split = len(indices) // nbuckets
        self.indices = []
        for i in range(nbuckets):
            self.indices.append(indices[i*split:(i+1)*split])
        if nbuckets * split < len(length):
            self.indices.append(indices[nbuckets*split:])

    def __iter__(self):
        if not self.dynamic_batch:
            random.shuffle(self.indices)
        for x in self.indices:
            random.shuffle(x)
        idxs = [i for x in self.indices for i in x]
        batches, batch, sum_len = [], [], 0
        for idx in idxs:
            batch.append(idx)
            sum_len += (self.length[idx] if self.dynamic_batch else 1)
            if sum_len >= self.batch_size:
                batches.append(batch)
                batch, sum_len = [], 0
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        if self.dynamic_batch:
            return sum(self.length) // self.batch_size
        if self.drop_last:
            return len(self.length) // self.batch_size  # type: ignore
        else:
            return (len(self.length) + self.batch_size - 1) // self.batch_size

class DistributedRandomBucketSampler(data.Sampler):
    def __init__(self, nbuckets, length, batch_size, drop_last, num_replicas, rank, seed=787):
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        indices = np.argsort(length)
        split = len(indices) // nbuckets
        self.length = length
        self.batch_size = batch_size
        self.indices = []
        for i in range(nbuckets):
            self.indices.append(indices[i*split:(i+1)*split])
        if nbuckets * split < len(length):
            self.indices.append(indices[nbuckets*split:])
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed
        self.drop_last = drop_last
        if self.drop_last and len(length) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(length) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(length) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        #Deterministic shuffling
        random.Random(self.epoch + self.seed).shuffle(self.indices)
        for i, x in enumerate(self.indices):
            seed = self.epoch + self.seed + i * 5
            random.Random(seed).shuffle(x)
        indices = [i for x in self.indices for i in x]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank*self.num_samples: (self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples

        #Batching
        batches, batch = [], []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batches.append(batch)
        #Stochastic suffling
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        if self.drop_last:
            return self.num_samples // self.batch_size  # type: ignore
        else:
            return (self.num_samples + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch
