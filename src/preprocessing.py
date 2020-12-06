#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import itertools
import numpy as np
from Bio import SeqIO
from sampler import *
from tqdm import tqdm
import pytorch_lightning as pl
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split


# In[2]:


class KMerList():
    
    def __init__(self, fasta, ks):
        self.fasta = itertools.groupby(fasta.seq, key=lambda x: int(x != "N"))
        self.ks = ks
        
    def __iter__(self):
        return(self)
    
    def __next__(self):
        is_not_n, non_overlapping = self.fasta.__next__()
        if is_not_n:
            overlapping = KMerList.overlap(
                "".join(non_overlapping), self.ks
            )
        else:
            overlapping = self.__next__()
        return(overlapping)
    
    @staticmethod
    def overlap(s, ks):
        kmers = []
        for a in range(len(s)):
            kmers.append(s[a:a+np.random.choice(ks)])
        return(np.array(kmers[0:len(kmers)-min(ks)+1]))


# In[3]:


class DNA2vec_set(Dataset):
    
    def __init__(self, fastas, kmers, train=False, cuda=True, transform=None):
        self.fastas = [list(KMerList(a, kmers)) for a in fastas]
        self.train = train
        if train:
            self.Y = np.array([int(a.id) for a in fastas])
        self.cuda = cuda
        self.T = transform
        
    def __len__(self):
        return(len(self.fastas))
    
    def __getitem__(self, ind):
        x = torch.from_numpy(self.T(self.fastas[ind])).type(torch.FloatTensor)
        if self.cuda:
            x = x.cuda()
        if self.train:
            y = self.Y[ind]
            return(x,y)
        else:
            return(x, 0)


# In[4]:


class Vectorizer():
    
    def __init__(self, vector_path):
        self.v = KeyedVectors.load_word2vec_format(vector_path, binary=False)
        
    def __call__(self, kmers):
        return(np.mean(np.mean([self.v[a] for a in kmers], 0), 0))


# In[5]:


class SiameseSet(Dataset):
    
    def __init__(self, ds):
        self.ds = ds
        self.train = ds.train
        
    def __len__(self):
        return(len(self.ds)**2)
    
    def __getitem__(self, ind):
        left = int(ind//len(self.ds))
        right = int(ind%len(self.ds))
        left = self.ds[left]
        right = self.ds[right]
        x = torch.cat((left[0], right[0]))
        y = int(left[1] == right[1]) if not self.train else 0
        return(x,y)


# In[8]:


class dna2vecDataModule(pl.LightningDataModule):
    
    def __init__(
        self, vector_path, fasta_path, kmers, test_size=0.25, batch_size=128, cv=False, val_size=0.1
    ):
        super().__init__()
        self.vp = vector_path
        self.fp = fasta_path
        self.kmers = kmers
        self.test_size = test_size
        self.batch_size = batch_size
        self.cv = cv
        self.val_size = val_size
    
    def prepare_data(self):
        self.v = Vectorizer(self.vp)
        f_gen = SeqIO.parse(self.fp, "fasta")
        self.fasta = [a for a in f_gen]
        
    def setup(self, stage=None):
        train_X, test_X = train_test_split(
            np.arange(len(self.fasta)), test_size=self.test_size
        )
        train_fasta = []
        val_fasta = []
        if not self.cv:
            train_X, val_X = train_test_split(
                np.arange(train_X.shape[0]), test_size=self.val_size
            )
            for a in val_X:
                val_fasta.append(self.fasta[a])
            self.val = SiameseSet(
                DNA2vec_set(val_fasta, self.kmers, transform=self.v, train=True)
            )
        for a in train_X:
            train_fasta.append(self.fasta[a])
        test_fasta = []
        for a in test_X:
            test_fasta.append(self.fasta[a])
        self.train = SiameseSet(
            DNA2vec_set(train_fasta, self.kmers, transform=self.v, train=True)
        )
        self.test = SiameseSet(
            DNA2vec_set(test_fasta, self.kmers, transform=self.v, train=True)
        )
        
    def train_dataloader(self):
        y = []
        tl = DataLoader(
            self.train, batch_size=self.batch_size,
            shuffle=False
        )
        for a in tqdm(tl):
            y.extend(a[1])
        y = torch.Tensor(y).type(torch.LongTensor)
        self.train_loader = DataLoader(
            self.train, batch_size=self.batch_size,
            sampler=BalancedBatchSampler(self.train, labels=y)
        )
        return(self.train_loader)
    
    def test_dataloader(self):
        self.test_loader = DataLoader(
            self.test, shuffle=False, batch_size=self.batch_size,
        )
        return(self.test_loader)
    
    def val_dataloader(self):
        if self.cv:
            return(self.test_dataloader())
        else:
            self.val_loader = DataLoader(
                self.val, shuffle=False, batch_size=self.batch_size
            )
            return(self.val_loader)
