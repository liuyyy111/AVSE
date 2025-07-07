import torch
import torch.nn as nn
import torch.nn.init
import lib.utils as utils
import logging

from timm.models.vision_transformer import Block

from lib.encoders import GPO, EncoderSimilarity, get_image_encoder, get_sim_encoder, get_text_encoder
from lib.loss import ContrastiveLoss, InfoNCE, loss_select

from lib.cross_net import CrossSparseAggrNet_v2

logger = logging.getLogger(__name__)

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X



class VSEModel(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        self.img_enc = get_image_encoder(opt)
        self.txt_enc = get_text_encoder(opt)
        self.sim_enc = EncoderSimilarity(opt)
        
        self.criterion = loss_select(opt, loss_type=opt.loss)
        # self.criterion = InfoNCE(tau=0.05)
        # self.triplet = False
        self.triplet = True
        # iteration
        self.Eiters = 0

    def img_token_compress_on(self):
        self.img_enc.token_compress_on()
    
    def freeze_backbone(self):
        self.img_enc.freeze_backbone()
        self.txt_enc.freeze_backbone()

    def unfreeze_backbone(self):
        self.img_enc.unfreeze_backbone()
        self.txt_enc.unfreeze_backbone()

    def set_max_violation(self, max_violation=True):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()
    
    def set_triplet_loss(self):
        self.triplet = True
        self.criterion = ContrastiveLoss(opt=self.opt, margin=self.opt.margin, max_violation=True)

    # Compute the image and caption embeddings
    def forward_emb(self, images, captions, lengths):
        img_emb = self.img_enc(images)

        cap_emb, word_emb = self.txt_enc(captions, lengths)

        return img_emb, cap_emb, lengths
    
    # compute the similarity on cross-attention interaction
    def forward_sim(self, img_embs, cap_embs, cap_lens=None):
        img_embs = l2norm(img_embs, -1)
        cap_embs = l2norm(cap_embs, -1)

        sims = self.sim_enc(img_embs, cap_embs)

        return sims

    def forward_loss(self, img_emb, cap_emb):
        """Compute the loss given pairs of image and caption embeddings
        """
        sims = self.forward_sim(img_emb, cap_emb)

        loss0 = self.criterion(sims)

        embs = torch.chunk(img_emb, 2, -1)
        embs1 = l2norm(embs[0], 0)
        embs2 = l2norm(embs[1], 0)
        c = embs1.T @ embs2
        # # # sum the cross-correlation matrix between all gpus
        c.div_(embs[0].size(0))

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss1 = on_diag + 0.0051 * off_diag

        loss = loss0 + loss1

        return loss 

    # One training step given images and captions
    def forward(self, images, captions, lengths, long_captions=None, long_lengths=None, img_ids=None, warmup_alpha=1.,):

        self.Eiters += 1
      
        img_emb = self.img_enc(images)
        cap_emb, word_emb = self.txt_enc(captions, lengths)

        # get all samples for compute loss function
        # if self.opt.multi_gpu and (not self.opt.cross_attention):
        if self.opt.multi_gpu:
            lengths = utils.concat_all_gather(lengths, keep_grad=False)
            img_ids = utils.concat_all_gather(img_ids, keep_grad=False)

            img_emb = utils.all_gather_with_grad(img_emb)
            cap_emb = utils.all_gather_with_grad(cap_emb) 

            if self.opt.distill:
                long_lengths = utils.concat_all_gather(long_lengths, keep_grad=False)
                prompt = utils.all_gather_with_grad(prompt)           
                long_cap_emb = utils.all_gather_with_grad(long_cap_emb)

        loss = self.forward_loss(img_emb, cap_emb)

        # loss = vicreg_loss(img_emb, cap_emb)
        return loss
    

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


# optimizer init
def create_optimizer(opt, model):

    # Set up the lr for different parts of the VSE model
    decay_factor = 1e-4  
    cross_lr_rate = 1.0
        
    # bert params
    all_text_params = list(model.txt_enc.parameters())
    bert_params = list(model.txt_enc.bert.parameters())
    bert_params_ptr = [p.data_ptr() for p in bert_params]
    text_params_no_bert = list()

    for p in all_text_params:
        if p.data_ptr() not in bert_params_ptr:
            text_params_no_bert.append(p)

    # bert   
    params_list = [
        {'params': text_params_no_bert, 'lr': opt.learning_rate},
        {'params': bert_params, 'lr': opt.learning_rate * 0.1},
    ]

    # vit
    params_list += [
        {'params': model.img_enc.visual_encoder.parameters(), 'lr': opt.learning_rate * 0.1},
        {'params': model.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
    ]

    params_list += [
        {'params': model.sim_enc.parameters(), 'lr': opt.learning_rate}
    ]

    optimizer = torch.optim.AdamW(params_list, lr=opt.learning_rate, weight_decay=decay_factor)
    
    return optimizer


if __name__ == '__main__':

    pass

