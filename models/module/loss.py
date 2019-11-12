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

    def trans_output(self, embedding, pred):
        # contiguous : 메모리에 값을 저장하는 방식
        # output shape --> embedding shape
        pred = pred.permute(0, 2, 3, 1).contiguous()
        pred = pred.view(pred.size(0), -1, pred.size(3))

        embedding = embedding.unsqueeze(2).expand(embedding.size(0), embedding.size(1), pred.size(2))
        # embedding에 해당하는 index값(100개)으로 뽑는다.
        pred = pred.gather(1, embedding)

        return pred

    def focal_loss(self, true, pred, alpha=2, beta=4):
        pos_true = true.eq(1) # = 1
        neg_true = true.lt(1) # < 1

        loss = 0

        for i, p in enumerate(pred):
            pos_pred = p[pos_true[i] == 1.]
            neg_pred = p[neg_true[i] == 1.]

            pos_loss = torch.pow(1 - pos_pred, alpha) * torch.log(pos_pred)
            neg_loss = torch.pow(1 - true[i][neg_true[i]], beta) * torch.pow(neg_pred, alpha) * torch.log(1 - neg_pred)

            n = pos_true[i].float().sum()

            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()

            if n == 0:
                loss = loss - neg_loss.sum()
            else:
                loss = loss - (pos_loss + neg_loss) / (n)

        return loss

    def offset_loss(self, true, pred, mask):
        num = mask.float().sum() * 2

        mask = mask.unsqueeze(2).expand_as(true)

        pred = pred[mask==1]
        true = true[mask==1]

        loss = nn.functional.smooth_l1_loss(pred, true, size_average=False)
        loss = loss / (num + 1e-4)

        return loss

    def triplet_loss(self, tl, br, mask):
        num = mask.sum(dim=1, keepdim=True).unsqueeze(1).expand_as(tl)

        mask = mask.unsqueeze(2) # 1,100,1

        ek = (tl+br) / 2

        tl = torch.pow(tl - ek, 2) / (num + 1e-4)
        tl = (tl*mask).sum()

        br = torch.pow(br - ek, 2) / (num + 1e-4)
        br = (br * mask).sum()

        pull = tl + br

        mask = mask.unsqueeze(1) + mask.unsqueeze(2)
        mask = mask.eq(2)
        num = num.unsqueeze(2).expand_as(mask)

        num2 = (num - 1) * num
        m = 2

        dist = ek.unsqueeze(1) - ek.unsqueeze(2)
        dist = m - torch.abs(dist)
        dist = nn.functional.relu(dist, inplace=True)
        dist = dist - m / (num + 1e-4)
        dist = dist / (num2 + 1e-4)
        dist = dist[mask]
        push = dist.sum()
        return pull, push

    def forward(self, true, pred):
        '''
        [top left heatmap,
        bottom right heatmap,
        top left offset,
        bottom right offset,
        top left embedding,
        bottom right embedding]
        '''
        mask = true[-1]

        true_tl_heatmap = true[0]
        true_br_heatmap = true[1]
        pred_tl_heatmap = pred[0].sigmoid()
        pred_br_heatmap = pred[1].sigmoid()

        true_tl_offset = true[2]
        true_br_offset = true[3]
        pred_tl_offset = pred[2]
        pred_br_offset = pred[3]

        true_tl_embedding = true[4].long()
        true_br_embedding = true[5].long()
        pred_tl_embedding = pred[4]
        pred_br_embedding = pred[5]

        # heatmap
        det_loss = self.focal_loss(true_tl_heatmap, pred_tl_heatmap) + self.focal_loss(true_br_heatmap, pred_br_heatmap)
        det_loss = det_loss * 0.5

        # offset
        pred_tl_offset = self.trans_output(true_tl_embedding, pred_tl_offset)
        pred_br_offset = self.trans_output(true_br_embedding, pred_br_offset)

        offset_loss = self.offset_loss(true_tl_offset, pred_tl_offset, mask)*self.off_weight + \
                    self.offset_loss(true_br_offset, pred_br_offset, mask)*self.off_weight

        # embedding
        tl_embedding = self.trans_output(true_tl_embedding, pred_tl_embedding)
        br_embedding = self.trans_output(true_br_embedding, pred_br_embedding)

        pull_loss, push_loss = self.triplet_loss(tl_embedding, br_embedding, mask)

        loss = (det_loss + pull_loss + push_loss + offset_loss) / len(true_tl_heatmap)

        return loss, [det_loss.item(), offset_loss.item(), pull_loss.item(), push_loss.item()]




