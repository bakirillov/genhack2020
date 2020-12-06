import numpy as np
import torch
from tensorlayer import NeuralTensorLayer
from contr_loss import ContrastiveLoss

if __name__ == '__main__':
    data1 = torch.from_numpy(np.random.randn(500, 30)).type(torch.FloatTensor)
    data2 = torch.from_numpy(np.random.randn(500, 30)).type(torch.FloatTensor)
    labels = torch.from_numpy(np.random.randint(0, 1, 500))

    layer = NeuralTensorLayer(3, 30, 10, 15)

    loss = ContrastiveLoss()

    out1 = layer(data1)
    out2 = layer(data2)
    loss_val = loss(out1, out2, labels)
    print(loss_val.detach().numpy())