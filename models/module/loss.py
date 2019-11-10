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

    def focal_loss(self, true, pred, alpha=2, beta=4):
        pos_true = true.eq(1) # = 1
        neg_true = true.lt(1) # < 1

        loss = 0

        for i, p in enumerate(pred):
            pos_pred = pred[pos_true[i] == 1.]
            neg_pred = pred[neg_true[i] == 1.]

            pos_loss = torch.pow(1 - pos_true, alpha) * torch.log(pos_pred)
            neg_loss = torch.pow(1 - true[i][neg_true[i]], beta) * torch.pow(neg_pred, alpha) * torch.log(1 - neg_pred)

            n = pos_true[i].float().sum()

            if n == 0:
                loss = loss - neg_loss.sum()
            else:
                loss = loss - (pos_loss + neg_loss) / n

        return loss

    def offset_loss(self, true, pred):

    def triplet_loss(self, true, pred):

    def forward(self, true, pred):
        '''
        [top left heatmap,
        bottom right heatmap,
        top left offset,
        bottom right offset,
        top left embedding,
        bottom right embedding]
        '''
        mask = pred[-1]

        true_tl_heatmap = true[0]
        pred_tl_heatmap = pred[0].sigmoid()
        true_br_heatmap = true[1]
        pred_br_heatmap = pred[1].sigmoid()

        true_tl_offset = true[2]
        pred_tl_offset = pred[2]
        true_br_offset = true[3]
        pred_br_offset = pred[3]

        true_tl_embedding = true[4]
        pred_tl_embedding = pred[4]
        true_br_embedding = true[5]
        pred_br_embedding = pred[5]

        # heatmap
        det_loss = self.focal_loss(true_tl_heatmap, pred_tl_heatmap) + self.focal_loss(true_br_heatmap, pred_br_heatmap)
        det_loss = det_loss * 0.5














