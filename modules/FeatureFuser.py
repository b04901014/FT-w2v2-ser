import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.helper_funcs import loadwav2vec
from transformers import Wav2Vec2ForPreTraining, Wav2Vec2Config
import argparse
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

class Wav2vecWrapper(nn.Module):
    def __init__(self, modelpath):
        super().__init__()
        self.wav2vec, self.wav2vec_cfg = loadwav2vec(modelpath)
        self.conv = nn.Conv1d(512, 512, 4, 2) #Match w2v to w2v2

        #SpecAug
        self.mask_time_length = 15
        self.mask_time_prob = 0.08
        self.observe_time_prob = 0.0

        self.mask_feature_length = 64
        self.mask_feature_prob = 0.05

    def trainable_params(self):
        ret = list(self.wav2vec.feature_aggregator.parameters()) + list(self.conv.parameters())
        return ret

    def forward(self, x, length=None):
        #INPUT: (N, L)
        #OUTPUT: (N, L, C)
        all_outs = []
        with torch.no_grad():
            wav2vec_z = self.wav2vec.feature_extractor(x)
            wav2vec_z = wav2vec_z.transpose(1, 2)
            if self.training:
                # apply SpecAugment along time axis
                batch_size, sequence_length, hidden_size = wav2vec_z.size()
                mask_time_indices = _compute_mask_indices(
                    (batch_size, sequence_length),
                    self.mask_time_prob,
                    self.mask_time_length,
                    min_masks=2,
                    device=x.device
                )
                masked_indicies = mask_time_indices
                flip_mask = torch.rand((batch_size, sequence_length), device=masked_indicies.device) > self.observe_time_prob
                wav2vec_z[masked_indicies & flip_mask] = 0.

                # apply SpecAugment along feature axis
                mask_feature_indices = _compute_mask_indices(
                    (batch_size, hidden_size),
                    self.mask_feature_prob,
                    self.mask_feature_length,
                    device=x.device,
                    min_masks=1
                )
                wav2vec_z[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
            wav2vec_z = wav2vec_z.transpose(1, 2)
        wav2vec_c = self.wav2vec.feature_aggregator(wav2vec_z)
        x = F.relu(self.conv(wav2vec_c))
        return x.transpose(1, 2)

    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for _, kernel_size, stride in eval(self.wav2vec_cfg['model']['conv_feature_layers']):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        for _, kernel_size, stride in eval(self.wav2vec_cfg['model']['conv_aggregator_layers']):
            ka = kernel_size // 2
            kb = ka - 1 if kernel_size % 2 == 0 else ka
            pad = ka + kb
            input_length += pad
            input_length = _conv_out_length(input_length, kernel_size, stride)
        input_length = (input_length - 4) // 2 + 1 #Last Convolution
        return input_length


def prepare_mask(length, shape, dtype, device):
    #Modified from huggingface
    mask = torch.zeros(
        shape, dtype=dtype, device=device
    )
    # these two operations makes sure that all values
    # before the output lengths indices are attended to
    mask[
        (torch.arange(mask.shape[0], device=device), length - 1)
    ] = 1
    mask = mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    return mask


class Wav2vec2Wrapper(nn.Module):
    def __init__(self, pretrain=True):
        super().__init__()
        self.wav2vec2 = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base", revision='2dcc7b7f9b11f0ef271067e62599a27317a03114').wav2vec2
        #Disable gradient checkpointing for ddp
        self.wav2vec2.encoder.config.gradient_checkpointing = False
        self.pretrain = pretrain
        if pretrain:
            self.mask_time_length = 15
            self.mask_time_prob = 0.06 #Probability of each time step is masked!
            self.observe_time_prob = 0.0 #Percentage of tokens that are perserved
            self.mask_feature_prob = 0
        else:
            #SpecAug
            self.mask_time_length = 15
            self.mask_time_prob = 0.08
            self.observe_time_prob = 0.0

            self.mask_feature_length = 64
            self.mask_feature_prob = 0.05


    def trainable_params(self):
        ret = list(self.wav2vec2.encoder.parameters())
        return ret

    def forward(self, x, length=None):
        with torch.no_grad():
            x = self.wav2vec2.feature_extractor(x)
            x = x.transpose(1, 2) #New version of huggingface
            x, _ = self.wav2vec2.feature_projection(x) #New version of huggingface
            mask = None
            if length is not None:
                length = self.get_feat_extract_output_lengths(length)
                mask = prepare_mask(length, x.shape[:2], x.dtype, x.device)
            if self.pretrain or self.training:
                batch_size, sequence_length, hidden_size = x.size()

                # apply SpecAugment along time axis
                if self.mask_time_prob > 0:
                    mask_time_indices = _compute_mask_indices(
                        (batch_size, sequence_length),
                        self.mask_time_prob,
                        self.mask_time_length,
                        min_masks=2,
                        device=x.device
                    )
                    masked_indicies = mask_time_indices & mask
                    flip_mask = torch.rand((batch_size, sequence_length), device=masked_indicies.device) > self.observe_time_prob
                    x[masked_indicies & flip_mask] = self.wav2vec2.masked_spec_embed.to(x.dtype)

                # apply SpecAugment along feature axis
                if self.mask_feature_prob > 0:
                    mask_feature_indices = _compute_mask_indices(
                        (batch_size, hidden_size),
                        self.mask_feature_prob,
                        self.mask_feature_length,
                        device=x.device,
                        min_masks=1
                    )
                    x[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
        x = self.wav2vec2.encoder(x, attention_mask=mask)[0]
        reps = F.relu(x)
        if self.pretrain:
            return reps, masked_indicies
        return reps

    #From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.wav2vec2.config.conv_kernel, self.wav2vec2.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length

class Wav2vec2PretrainWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.wav2vec2PT = Wav2Vec2ForPreTraining.from_pretrained("facebook/wav2vec2-base", revision='2dcc7b7f9b11f0ef271067e62599a27317a03114')
        self.wav2vec2 = self.wav2vec2PT.wav2vec2
#        self.wav2vec2PT.freeze_feature_extractor()

    def trainable_params(self):
        ret = list(self.wav2vec2PT.parameters())
        return ret

    def forward(self, x, length=None):
        self.wav2vec2PT.train()
        with torch.no_grad():
            batch_size, sequence_length = x.size()
            sequence_length = self.get_feat_extract_output_lengths(sequence_length)
            feat_shape = (batch_size, sequence_length)
            length = self.get_feat_extract_output_lengths(length)
            attn_mask = prepare_mask(length, feat_shape, x.dtype, x.device)
            mask_time_indices = _compute_mask_indices(
                feat_shape,
                self.wav2vec2PT.config.mask_time_prob,
                self.wav2vec2PT.config.mask_time_length,
                min_masks=2,
                device=x.device,
                attention_mask=attn_mask
            )
        x = self.wav2vec2PT(x, mask_time_indices=mask_time_indices)#, attention_mask=attn_mask)
        return x

    #From huggingface
    def get_feat_extract_output_lengths(self, input_length):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return (input_length - kernel_size) // stride + 1
        for kernel_size, stride in zip(self.wav2vec2.config.conv_kernel, self.wav2vec2.config.conv_stride):
            input_length = _conv_out_length(input_length, kernel_size, stride)
        return input_length
