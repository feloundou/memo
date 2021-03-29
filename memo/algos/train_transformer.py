import pytorch_lightning
import os
from pathlib import Path
import numpy as np
import math, copy, time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torch.autograd import Variable

# from torchtext import data, datasets

import torchtext
from torchtext.data import Field, BucketIterator
# from torchtext.datasets import IMDB
from torchtext.datasets import IWSLT
# from torchtext.datasets
# import WikiText2 #vocab size of 33,278
import spacy
import pytorch_lightning as pl
pl.seed_everything(hash("set random seeds") % 2**32 - 1)

import wandb
from pytorch_lightning.loggers import WandbLogger

wandb.login()

nlp = spacy.load("en")
doc = nlp("This is an English sentence.")
print([(w.text, w.pos_) for w in doc])

nlpf = spacy.load('fr')
docu = nlpf("Voici une phrase en francais.")
print([(w.text, w.pos_) for w in docu])


PATH = './transformer_fr_en.pth'
DATA_PATH=Path('./data/')
DATA_PATH.mkdir(exist_ok=True)


BOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
PAD_TOKEN = "<pad>"
BLANK_TOKEN = "<blank>"
MAX_LEN = 100  # filter out examples that have more than MAX_LEN tokens
MIN_FREQ = 2


args = {
    "full_data_dir": DATA_PATH,
    "model_dimension" : 512,
    "num_layers" : 6,
    "num_heads" : 8,
    "batch_size" : 8,
    "dropout" : 0.1,
    "label_smoothing" : 0.1
}


def greedy_decode_sequence(model, X, start_idx_Y, pad_idx_X, pad_idx_Y, max_len):
    '''
    :param model:
    :param X:
    :param start_idx_Y:
    :param pad_idx_X:
    :param pad_idx_Y:
    :param max_len:
    :return:
    '''
    b, inp_len = X.shape
    inp_pads = (X == pad_idx_X).int()

    # encode inputs
    # shape (b, inp_len, d_model)
    encoded_memory = model.encode(X, inp_pads)

    # shape (b, max_len) e.g. (b, 10)
    Y = torch.ones(b, max_len).type_as(X) * pad_idx_Y
    Y[:, 0] = start_idx_Y
    # shape (b, max_len)
    out_pads = (Y == pad_idx_Y).int()

    # generate one token at a time
    for t in range(1, max_len):

        # shape (b, t, d_model)
        decoder_output = model.decode(encoded_memory, Y[:, :t], inp_pads, out_pads[:, :t])

        # shape (b, t, V)
        decoded_logits = model.classifier(decoder_output)
        # decoded_probs = torch.softmax(decoded_logits, dim=-1)

        # shape (b,), (b,)
        max_prob, max_idx = torch.max(decoded_logits[:, -1, :], dim=-1)

        # update Y, out_pads for next timestep
        Y[:, t] = max_idx
        out_pads[:, t] = (max_idx == pad_idx_Y).int()

    return Y[:, 1:]


spacy_en = spacy.load('en')
spacy_fr = spacy.load('fr')

def tokenize_french(text):
    return [tok.text for tok in spacy_fr.tokenizer(text)]

def tokenize_english(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize=tokenize_french, pad_token=BLANK_TOKEN)
TGT = Field(tokenize=tokenize_english, init_token=BOS_TOKEN, eos_token = EOS_TOKEN, pad_token=BLANK_TOKEN)

train, val, test = IWSLT.splits(
    exts=('.fr', '.en'), fields=(SRC, TGT),
    filter_pred=lambda x: len(vars(x)['src']) <= MAX_LEN and len(vars(x)['trg']) <= MAX_LEN)

SRC.build_vocab(train.src, min_freq=MIN_FREQ)
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)


print([tok.text for tok in spacy_fr.tokenizer("Je ne suis pas une malade.")])

# Let's look at a batch of 4 sentences
train_iter = BucketIterator(train, batch_size=4, sort_key=lambda x: len(x.trg), shuffle=True)

batch = next(iter(train_iter))
'''In each batch, the sentences have been transposed so they are descending vertically 
(important: we will need to transpose these again to work with the transformer). Each index represents a token (word), 
and each column represents a sentence. We have 10 columns, as 10 was the batch_size we specified.'''

print(batch.src) # source

def save_cache(cache_path, dataset):
    with open(cache_path, 'w', encoding='utf-8') as cache_file:
        # Interleave source and target tokenized examples, source is on even lines, target is on odd lines
        for ex in dataset.examples:
            cache_file.write(' '.join(ex.src) + '\n')
            cache_file.write(' '.join(ex.trg) + '\n')