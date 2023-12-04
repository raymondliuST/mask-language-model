from dataset import *

import argparse
import sys
sys.path.extend(["../","./"])
import os
from torch.utils.data import DataLoader
from dataset import WordVocab
from model.bert import BERT
from dataset import BERTDataset,collate_mlm
from driver import BERTTrainer
from module import Paths
import torch
import numpy as np
import config.hparams as hp
import random

# d = BERTDataset("./data/category.txt")
print("Loading Vocab")
vocab = WordVocab.load_vocab("./data/category.vocab")
print("Vocab Size: ", vocab.vocab_size)

d = BERTDataset("./data/category.txt", vocab = vocab)

item = d.__getitem__(0)

train_data_loader = DataLoader(d, batch_size=20, collate_fn=lambda batch: collate_mlm(batch),num_workers=20, shuffle=False)

for batch in train_data_loader:

    import pdb

    pdb.set_trace()