import torch
from torch import nn
from torch.optim import Adam

class TransformerAdam:
    def __init__(self, params, lr=512**(-0.5), betas=(0.9, 0.98), eps=1e-9, 
        weight_decay=0, amsgrad=False, warmup_steps=4000):
        self.adam = Adam(params, lr, betas, eps, weight_decay, amsgrad)

        self.init_lr = lr
        self.current_steps = 0
        self.warmup_steps = warmup_steps
        self.defaults = self.adam.defaults

        if self.warmup_steps == 0:
            self.get_lr_multiplier = self.get_lr_multiplier_no_warmup
        else:
            self.get_lr_multiplier = self.get_lr_multiplier_warmup

    def get_lr_multiplier_warmup(self):
        return min([
            self.current_steps ** -0.5,
            self.current_steps * (self.warmup_steps ** -1.5)
        ])

    def get_lr_multiplier_no_warmup(self):
        return self.current_steps ** -0.5

    def zero_grad(self):
        self.adam.zero_grad()

    def step(self, closure=None):
        # update learning rate
        self.current_steps += 1
        new_lr = self.init_lr * self.get_lr_multiplier()
        for pg in self.adam.param_groups:
            pg['lr'] = new_lr

        self.adam.step(closure)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, eps, n_vocab, padding_index=0):
        super(LabelSmoothingLoss, self).__init__()

        self.eps = eps
        self.n_vocab = n_vocab
        self.padding_index = padding_index

        unk_eps = eps / (n_vocab - 1)
        one_hot = torch.full((n_vocab,), unk_eps)
        one_hot[padding_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - eps
        
    def forward(self, pred, y):
        # pred: batch_size x len x n_vocab
        # y: batch_size x len
        pred = pred.view(-1, self.n_vocab)
        y = y.view(-1)

        pad_mask = (y == self.padding_index)

        # smoothed probability over whole vocabulary
        gold = self.one_hot.repeat(y.size(0), 1)
        gold.scatter_(1, y.unsqueeze(-1), self.confidence)
        gold.masked_fill_(pad_mask.unsqueeze(-1).bool(), 0)

        return nn.functional.kl_div(pred, gold, reduction='sum')
