import os
from torch.utils import data
import torch
import json
import numpy as np
from collections import Counter
import soundfile as sf
from torch.utils.data.dataloader import default_collate

class CustomEmoDataset:
    def __init__(self, datadir, labeldir, maxseqlen):
        super().__init__()
        self.maxseqlen = maxseqlen * 16000 #Assume sample rate of 16000
        with open(labeldir, 'r') as f:
            self.label = json.load(f) #{split: {wavname: emotion_label}}
        self.emoset = list(set([emo for split in self.label.values() for emo in split.values()]))
        self.emoset = list(sorted(self.emoset))
        self.nemos = len(self.emoset)
        self.train_dataset = _CustomEmoDataset(datadir, self.label['Train'], self.emoset, 'training')
        if self.label['Val']:
            self.val_dataset = _CustomEmoDataset(datadir, self.label['Val'], self.emoset, 'validation')
        if self.label['Test']:
            self.test_dataset = _CustomEmoDataset(datadir, self.label['Test'], self.emoset, 'testing')

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
            ret = (x[0], output_length, x[1])
            return ret
        batch = list(map(trunc, batch))
        return default_collate(batch)

class _CustomEmoDataset(data.Dataset):
    def __init__(self, datadir, label, emoset,
                 split, maxseqlen=12):
        super().__init__()
        self.maxseqlen = maxseqlen * 16000 #Assume sample rate of 16000
        self.split = split
        self.label = label #{wavname: emotion_label}
        self.emos = Counter([self.label[n] for n in self.label.keys()])
        self.emoset = emoset
        self.labeldict = {k: i for i, k in enumerate(self.emoset)}
        self.datasetbase = list(self.label.keys())
        self.dataset = [os.path.join(datadir, x) for x in self.datasetbase]

        #Print statistics:
        print (f'Statistics of {self.split} splits:')
        print ('----Involved Emotions----')
        for k, v in self.emos.items():
            print (f'{k}: {v} examples')
        l = len(self.dataset)
        print (f'Total {l} examples')
        print ('----Examples Involved----\n')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        dataname = self.dataset[i]
        wav, _sr = sf.read(dataname)
        _label = self.label[self.datasetbase[i]]
        label = self.labeldict[_label]
        return wav.astype(np.float32), label
