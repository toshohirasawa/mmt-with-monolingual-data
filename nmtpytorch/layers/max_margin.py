# -*- coding: utf-8 -*-
import torch
from torch import nn

# Layer contributed by @elliottd


class MaxMargin(nn.Module):
    """A max-margin layer for ranking-based loss functions."""

    def __init__(self, margin, max_violation=False):
        super().__init__()

        assert margin > 0., "margin must be > 0."

        # Other arguments
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, enc1, enc2):
        """Computes the max-margin loss given a pair of rank-2
           annotation matrices. The matrices must have the same number of
           batches and the same number of feats.

        Arguments:
            enc1(Tensor): A tensor of `B*feats` representing the
                annotation vectors of the first encoder.
            enc2(Tensor): A tensor of `B*feats` representation the
                annotation vectors of the second encoder.
        """

        assert enc1.shape == enc2.shape, \
            "shapes must match: enc1 {} enc2 {}".format(enc1.shape, enc2.shape)

        enc1 = enc1 / enc1.norm(p=2, dim=1).unsqueeze(1)
        enc2 = enc2 / enc2.norm(p=2, dim=1).unsqueeze(1)
        loss = self.constrastive_loss(enc1, enc2)

        return {'loss': loss}

    def constrastive_loss(self, enc1, enc2):
        if enc1.shape[0] == 1:
            # There is no error when we have a single-instance batch.
            # Return a dummy error of 1e-5 as a regularizer
            return torch.tensor([1e-3], device=enc1.device)

        # compute enc1-enc2 score matrix
        scores = self.cosine_sim(enc1, enc2)
        diagonal = scores.diag().view(enc1.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        cost_enc1 = (self.margin + scores - d2).clamp(min=0)
        cost_enc2 = (self.margin + scores - d1).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0), device=enc1.device) > .5
        cost_enc2 = cost_enc2.masked_fill_(mask, 0)
        cost_enc1 = cost_enc1.masked_fill_(mask, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_enc2 = cost_enc2.max(1)[0]
            cost_enc1 = cost_enc1.max(0)[0]
        denom = cost_enc1.shape[0]**2 - cost_enc1.shape[0]
        return (cost_enc2 + cost_enc1).sum() / denom

    def cosine_sim(self, one, two):
        '''Cosine similarity between all the first and second encoder pairs'''
        return one.mm(two.t())

class MaxMarginLoss(nn.Module):
    """Alternative implementation of max-margin layer for ranking-based loss function"""

    def __init__(self, margin):
        super().__init__()

        assert margin > 0., "margin must be > 0."

        self.margin = margin

    def forward(self, pred, gold):
        B = pred.shape[0]

        # for simplifying cosine distance calculation below
        U_norm = pred / pred.norm(dim=-1, keepdim=True)

        # extract and normalize feats
        Y = gold
        Y_norm = Y / Y.norm(dim=-1, keepdim=True)

        # Implementation from original paper (Max-Margin)
        errors = U_norm.matmul(Y_norm.t())
        diag = errors.diag()

        # all contrastive images for each sentence
        loss_s = self.margin - errors + diag.unsqueeze(-1)
        loss_s = torch.max(loss_s, torch.zeros_like(loss_s))

        # all contrastive sentences for each image
        loss_i = self.margin - errors + diag.unsqueeze(0)
        loss_i = torch.max(loss_i, torch.zeros_like(loss_i))

        # total loss
        loss_tot = loss_s + loss_i
        loss_tot[range(B), range(B)] = 0.0

        # one for positive sample
        return loss_tot.mean()

class MaxMarginLoss3D(nn.Module):
    """Alternative implementation of max-margin layer for ranking-based loss function"""

    def __init__(self, margin):
        super().__init__()

        assert margin > 0., "margin must be > 0."

        self.margin = margin

    def forward(self, pred, gold):
        n_feat = pred.shape[0]
        n_batch = pred.shape[1]

        # for simplifying cosine distance calculation below
        U_norm = pred / pred.norm(dim=-1, keepdim=True)

        # extract and normalize feats
        Y = gold
        Y_norm = Y / Y.norm(dim=-1, keepdim=True)

        # Implementation from original paper (Max-Margin)
        errors = torch.bmm(U_norm, Y_norm.transpose(-2,-1))
        preds = errors.diagonal(dim1=-2, dim2=-1)

        # all contrastive images for each sentence
        loss_s = self.margin - errors + preds.unsqueeze(-1)
        loss_s = torch.max(loss_s, torch.zeros_like(loss_s))

        # all contrastive sentences for each image
        loss_i = self.margin - errors + preds.unsqueeze(-2)
        loss_i = torch.max(loss_i, torch.zeros_like(loss_i))

        # total loss
        loss_tot = loss_s + loss_i
        loss_tot[:, range(n_batch), range(n_batch)] = 0.0

        # mean over non-zero loss if having losses
        zero_loss_mask = loss_tot != 0
        n_zero_loss = zero_loss_mask.sum()

        final_loss = (loss_tot * zero_loss_mask).sum()
        if n_zero_loss > 0:
            final_loss = final_loss / n_zero_loss
        # otherwise, final_loss should be zero

        # one for positive sample
        return final_loss

class MaxMarginForEmbeddingPrediction(nn.Module):
    def __init__(self, margin=0.5, constrastive_type='intruder'):
        super().__init__()

        self.constrastive_type = constrastive_type.lower()

        assert margin > 0., "margin must be positive"
        assert self.constrastive_type in ('all', 'intruder'), \
            f"Unknown constrastive_type '{self.constrastive_type}'"

        self.margin = margin
        self.forward = getattr(self, f'_forward_{self.constrastive_type}')

    def _forward_all(self, O, Y):
        B = O.shape[0]
        corrects = O[range(B), Y].unsqueeze(-1)    # Bx1
        
        # calculate loss for all canditees over vocabulary
        loss = self.margin + O - corrects   # BxV
        # mask corrects
        loss[range(B), Y] = 0.0
        # mean loss except correct one
        loss = loss.sum(dim=-1) / (loss.shape[-1] - 1)

        # suppress loss　negative values and padding
        loss[loss < 0] = 0.0
        loss[Y == 0] = 0.0

        return loss.sum()

    def _forward_intruder(self, O, Y):
        B = O.shape[0]
        corrects = O[range(B), Y].unsqueeze(-1)    # Bx1
        
        # Lazaridou, Dinu, Baroni - ACL 2015 - Hubness and Pollution Delving into Cross-Space Mapping for Zero-Shot Learning
        # max() returns both exact values AND indicators, only first one is needed
        loss = self.margin + (O - corrects).max(dim=-1)[0]

        # suppress loss　negative values and padding
        loss[loss < 0] = 0.0
        loss[Y == 0] = 0.0

        return loss.sum()

    def __repr__(self):
        return self.__class__.__name__ + \
            f"(margin={self.margin}, constrastive_type={self.constrastive_type})" 