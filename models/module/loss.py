import torch
import torch.nn as nn

class CornerNet_Loss(nn.Module):
    '''
    input : (List) [heat_tl, embed_tl, off_tl, heat_br, embed_br, off_br]
    output : Loss
    '''
    def __init__(self):
        super(CornerNet_Loss, self).__init__()
        self.pull_weight = 0.1
        self.push_weight = 0.1
        self.off_weight = 1

    def focal_loss(self,alpha=2,beta=4):

    def offset_loss(self):

    def triplet_loss(self):

    def forward(self, true, pred):
