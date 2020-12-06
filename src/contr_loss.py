import torch
import torch.nn.functional as F
from torch.nn import CosineSimilarity

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.CS = CosineSimilarity(1, 10**-5)

    def forward(self, output1, output2, label):
        cosine_distance = 1-self.CS(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(cosine_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))


        return loss_contrastive