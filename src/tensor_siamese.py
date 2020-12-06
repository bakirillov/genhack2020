#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pytorch_ranger
from contr_loss import *
from tensorlayer import *
from preprocessing import *
import pytorch_lightning as pl
import torch.nn.functional as F


# In[2]:



# In[3]:


class TensorSiamese(pl.LightningModule):
    
    def __init__(self, order=2, input_dim=100, output_dim=50, rank_tucker=5):
        super(TensorSiamese, self).__init__()
        self.tensor = NeuralTensorLayer(
            order, input_dim, output_dim, rank_tucker=rank_tucker
        )
        self.bn = nn.BatchNorm1d(output_dim)
        self.s = nn.Sigmoid()
        self.loss = ContrastiveLoss()
        
    def forward(self, x):
        return(self.s(self.bn(self.tensor(x))))
    
    def configure_optimizers(self):
        optimizer = pytorch_ranger.RangerQH(self.parameters())
        return(optimizer)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x1 = x[:,0:100]
        x2 = x[:,100:]
        left = self.forward(x1)
        right = self.forward(x2)
        loss = self.loss(left, right, y)+0.5*self.tensor.get_orthogonality_loss()
        self.log("train_loss", loss)
        return(loss)
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x1 = x[:,0:100]
        x2 = x[:,100:]
        left = self.forward(x1)
        right = self.forward(x2)
        loss = self.loss(left, right, y)+0.5*self.tensor.get_orthogonality_loss()
        self.log("val_loss", loss)
        return(loss)
    
    def predict(self, val_batch):
        x, _ = val_batch
        x1 = x[:,0:100]
        x2 = x[:,100:]
        left = self.forward(x1)
        right = self.forward(x2)
        return(F.pairwise_distance(left,right))


# In[4]:
if __name__ == "__main__":
    dm = dna2vecDataModule(
        vector_path="../dna2vec/pretrained/dna2vec-20161219-0153-k3to8-100d-10c-29320Mbp-sliding-Xat.w2v", 
        fasta_path="training_set.fasta",
        kmers=[3,4,5,6,7,8], batch_size=128
    )
    model = TensorSiamese(2, 100, 256).cuda()
    trainer = pl.Trainer(
        gpus="0", max_epochs=3,
        default_root_dir="../checkpoints"
    )
    trainer.fit(model, dm)
    torch.save(model, "tsm.ptl")

