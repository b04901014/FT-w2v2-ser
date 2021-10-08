import numpy as np
import matplotlib.pyplot as plt
import torch
import fairseq

def tonumpy(x):
    return x.detach().cpu().numpy()

def loadwav2vec(path):
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([path])
    model = model[0]
    model.cuda()
    return model, cfg

def multilabel2vec(labels, labeldict):
    # Transform a list of labels for one utterance to the proability vector
    numlabels = len(labeldict.keys())
    ret = np.zeros(numlabels, dtype=np.float32)
    totalvote = 0
    for k, v in labels.items():
        if k in labeldict:
            ret[labeldict[k]] = v
        totalvote += v
    ret = ret / totalvote# May not sum up to 1# / (ret.sum() + 1e-12)
    return ret

#Modified from hugging face https://github.com/huggingface/transformers/blob/7eee950ac3135476c811daf23eca8cedbbaa3879/src/transformers/models/wav2vec2/modeling_wav2vec2.py
#to cope with no masking
def _compute_mask_indices(
    shape,
    mask_prob,
    mask_length,
    attention_mask=None,
    min_masks=0,
):
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        # add a random number for probabilistic rounding
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    padding_mask = attention_mask.ne(1) if attention_mask is not None else None
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if num_mask == 0: #no masking
            mask_idcs.append([])
            continue

        lengths = np.full(num_mask, mask_length)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
        mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    for i, mask_idc in enumerate(mask_idcs):
        mask[i, mask_idc] = True

    return mask
