import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertTokenizer, SwinModel, ViTModel
# from lib.module.modeling_vit import ViTModel
import logging
import timm
from transformers.modeling_outputs import BaseModelOutputWithPooling
from timm.models.vision_transformer import VisionTransformer
from timm.layers import resample_abs_pos_embed
from timm.layers import PatchEmbed, Mlp

from lib.module.vision_transformers import Block, Transformer


logger = logging.getLogger(__name__)


# 手动插值位置编码到 384x384
def resize_vit_pos_embed(model: VisionTransformer, img_size: int):
    # 原始 pos_embed
    posemb = model.pos_embed  # [1, 197, C]
    patch_size = model.patch_embed.patch_size[0]  # 通常为 16
    new_grid = [img_size // patch_size, img_size // patch_size]  # [24, 24]
    old_grid = None  # 可以省略，timm 会自动推
    posemb_resized = resample_abs_pos_embed(
        posemb=posemb,
        new_size=new_grid,       # ✅ 这里是 patch grid size，不是 tensor shape
        num_prefix_tokens=1,
        interpolation='bicubic',
        antialias=True,
        verbose=True
    )

    # 替换模型中的位置编码
    model.pos_embed = torch.nn.Parameter(posemb_resized)


def token_merge(x, merge_ratio=0.2):
    B, N, D = x.shape
    cls_token, tokens = x[:, :1], x[:, 1:]  # [B, 1, D], [B, N-1, D]
    N_patch = tokens.shape[1]
    N_merge = int(N_patch * merge_ratio)
    if N_merge == 0:
        return x

    # Step 1: cosine similarity
    tokens_norm = F.normalize(tokens, dim=-1)
    sim = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # [B, N-1, N-1]
    sim -= torch.eye(N_patch, device=x.device).unsqueeze(0) * 1e6

    sim_flat = sim.view(B, -1)
    topk_vals, topk_idx = torch.topk(sim_flat, k=N_merge, dim=1)

    row = topk_idx // N_patch
    col = topk_idx % N_patch

    # Step 2: scatter + count
    merged_tokens = torch.zeros_like(tokens)
    count = torch.zeros(B, N_patch, device=x.device)

    row_idx = row.unsqueeze(-1).expand(-1, -1, D)
    col_idx = col.unsqueeze(-1).expand(-1, -1, D)

    merged_tokens.scatter_add_(1, row_idx, tokens.gather(1, row_idx))
    merged_tokens.scatter_add_(1, row_idx, tokens.gather(1, col_idx))

    count.scatter_add_(1, row, torch.ones_like(row, dtype=tokens.dtype))
    count.scatter_add_(1, col, torch.ones_like(col, dtype=tokens.dtype))

    count = count.clamp(min=1).unsqueeze(-1)
    merged_tokens = merged_tokens / count

    # Step 3: mask and gather kept
    used_mask = torch.zeros(B, N_patch, dtype=torch.bool, device=x.device)
    used_mask.scatter_(1, row, True)
    used_mask.scatter_(1, col, True)
    keep_mask = ~used_mask  # [B, N-1]

    # 新方式：用 gather 高效提取
    idxs = torch.arange(N_patch, device=x.device).unsqueeze(0).expand(B, -1)
    kept_idxs = idxs[keep_mask].split((keep_mask.sum(1).tolist()))
    padded_kept = torch.zeros(B, N_patch - N_merge, D, device=x.device)

    for i in range(B):
        padded_kept[i, :kept_idxs[i].numel()] = tokens[i, kept_idxs[i]]

    merged_final = merged_tokens[used_mask].view(B, -1, D)

    new_tokens = torch.cat([padded_kept, merged_final], dim=1)  # [B, N-1, D]
    x_out = torch.cat([cls_token, new_tokens], dim=1)

    return x_out



def token_compress(x, s, query_proj, key_proj, value_proj, num_heads, prune_ratio=0.05):
    b, n, d = x.shape
    N = n - 1  # exclude CLS token
    h = num_heads
    d_h = d // h

    cls_token = x[:, :1, :]
    patch_tokens = x[:, 1:, :]
    s_patch = s[:, 1:]

    # Step 1: QKV projection
    with torch.no_grad():
        q = query_proj(patch_tokens).reshape(b,N,h,d_h).transpose(1, 2)
        k = key_proj(patch_tokens).reshape(b,N,h,d_h).transpose(1, 2)
        v = value_proj(patch_tokens).reshape(b,N,h,d_h).transpose(1, 2)  
    # q = q.view(b, N, h, d_h).transpose(1, 2)
    # k = k.view(b, N, h, d_h).transpose(1, 2)
    # v = v.view(b, N, h, d_h).transpose(1, 2)

    # Step 2: Token voting
    k_vote = k.mean(1)  # (b, N, d/h)
    k_vote = F.normalize(k_vote, dim=-1)
    sim = F.cosine_similarity(k_vote.unsqueeze(2), k_vote.unsqueeze(1), dim=-1)  # (b, N, N)
    diag_mask = torch.eye(N, device=x.device).bool().unsqueeze(0)
    sim.masked_fill_(diag_mask, float('-inf'))

    vote_weight, vote_index = sim.max(dim=2)

    vote_weight = vote_weight.masked_fill(vote_weight == float('-inf'), 0)

    score = torch.zeros(b, N, device=x.device)
    score = score.scatter_add(1, vote_index, vote_weight)

    # Step 3: Sort and prune
    num_retained = int(N * (1 - prune_ratio))
    sorted_idx = score.argsort(dim=1, descending=True)  # (b, N)
    retained_idx = sorted_idx[:, :num_retained] 
    pruned_idx = sorted_idx[:, num_retained:] 

    # Step 4: Token mixing

    sim_gather = sim.gather(1, pruned_idx.unsqueeze(-1).expand(-1, -1, N))  # (b, N_p, N)
    W = sim_gather.gather(2, retained_idx.unsqueeze(1).expand(-1, pruned_idx.shape[1], num_retained))  # (b, N_p, N_r)
    W = F.softmax(W, dim=-1)  # (b, N_p, N_r)

    # A = sim
    # W = F.softmax(
    #     A.gather(1, pid.unsqueeze(-1).expand(-1, -1, num_retained)).gather(
    #         2, rid.unsqueeze(1).expand(-1, pid.shape[1], -1)
    #     ), dim=-1
    # )

    q_ = q.permute(0, 2, 1, 3).reshape(b, N, d)
    q_weighted = q_ * s_patch.unsqueeze(-1)
    mixed = torch.zeros_like(q_weighted)
    for i in range(b):
        mixed[i, retained_idx[i]] += W[i].T @ q_weighted[i, pruned_idx[i]]

    s_pruned = torch.gather(s_patch, 1, pruned_idx)  # (b, N_p)
    s_retained = torch.gather(s_patch, 1, retained_idx)  # (b, N_r)
    s_new_patch = s_retained + torch.bmm(W.transpose(1, 2), s_pruned.unsqueeze(-1)).squeeze(-1)  # (b, N_r)

    q_new_patch = mixed.gather(1, retained_idx.unsqueeze(-1).expand(-1, -1, d)) / s_new_patch.unsqueeze(-1)  # (b, N_r, d)
    q_new_patch = q_new_patch.view(b, num_retained, h, d_h).transpose(1, 2)  # (b, h, N_r, d_h)

    # Step 5: Attention
    # Gather k, v
    k_new = k.gather(2, retained_idx.unsqueeze(1).unsqueeze(-1).expand(-1, h, -1, d_h))  # (b, h, N_r, d_h)
    v_new = v.gather(2, retained_idx.unsqueeze(1).unsqueeze(-1).expand(-1, h, -1, d_h))  # (b, h, N_r, d_h)

    attn = (q_new_patch) @ k_new.transpose(-2, -1)  # (b, h, N_r, N_r)
    attn = attn / math.sqrt(d_h)
    attn = attn.softmax(dim=-1)
    out_patch = (attn @ v_new).transpose(1, 2).reshape(b, num_retained, d)  # (b, N_r, d)

    # Step 6: CLS + output
    x_new = torch.cat([cls_token, out_patch], dim=1)  # (b, N_r + 1, d)
    s_new = torch.cat([s[:, :1], s_new_patch], dim=1)  # (b, N_r + 1)
    x_retained = torch.cat([cls_token, patch_tokens.gather(1, retained_idx.unsqueeze(-1).expand(-1, -1, d))], dim=1)

    return x_new, s_new, x_retained


class ViTWithTokenCompress(nn.Module):
    def __init__(self, prune_ratio=0.5, compress_at_layer=6, pretrained_name="google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(pretrained_name)
        self.compress = TokenCompressor(prune_ratio)
        self.compress_at_layer = compress_at_layer

    def forward(
        self, 
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:

        
        x = self.vit.embeddings(pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding)  # (b, n, d)
        s = torch.ones(x.shape[:2], device=x.device)

        for i, layer in enumerate(self.vit.encoder.layer):
            
            if i == self.compress_at_layer:
                qkv_proj = layer.attention.attention.qkv
                x, s, _ = self.compress(x, s, qkv_proj)
            x = layer(x).last_hidden_state

        return x
    
    
def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):
        B, N, D = x.size()
        x = x.reshape(B * N, D)
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            x = F.relu(bn(layer(x))) if i < self.num_layers - 1 else layer(x)
        x = x.view(B, N, self.output_dim)
        return x


def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)

        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features, sort = sorted_features.sort(dim=1, descending=True)

        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features, pool_weights

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.)


def get_text_encoder(opt):
    txt_enc = EncoderText_BERT(opt)   
    return txt_enc


def get_image_encoder(opt):
    img_enc = VisionTransEncoder(opt)
    return img_enc

def get_sim_encoder(opt):
    return EncoderSimilarity(opt)


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.fc = nn.Linear(img_dim, embed_size)
        self.gpool = GPO(32, 32)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)

        features, pool_weights = self.gpool(features, image_lengths)

        # if not self.no_imgnorm:
        features = l2norm(features, dim=-1)

        return features

# ViT encoder
class VisionTransEncoder(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt

        # Swin model
        if 'swin' in opt.vit_type:                           
            # img_res 224 * 224, 7*7 patch
            # self.visual_encoder = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
            # self.visual_encoder = SwinModel.from_pretrained("/home/sculiuyang/.cache/huggingface/hub/swin", local_files_only=True)
            # self.visual_encoder = SwinModel.from_pretrained("/home/sculiuyang/.cache/huggingface/hub/swin-base-patch4-window7-224-in22k", local_files_only=True)
            self.visual_encoder = SwinModel.from_pretrained(opt.vit_type)

            opt.num_patches = 49
            print('swin model')
        #  ViT model
        elif 'vit' in opt.vit_type: 
            # img_res 224 * 224, 14*14 patch
            # self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
            # self.visual_encoder = ViTModel.from_pretrained("/home/sculiuyang/.cache/huggingface/hub/vit-base-patch16-224-in21k")
            self.visual_encoder = ViTModel.from_pretrained(opt.vit_type)
            # self.visual_encoder = ViTModel.from_pretrained("../weights_models/google--vit-base-patch16-224-in21k")
            opt.num_patches = 196
            print('vit model')
        else:
            raise ValueError(f"unsupport type: {opt.vit_type}")

        self.dropout = nn.Dropout(0.2)  
        if 'swin' in opt.vit_type:
            self.image_encoder = EncoderImageAggr(1024, opt.embed_size)

        else:
            self.image_encoder = EncoderImageAggr(768, opt.embed_size)
            
    def radial_bias_sampling_groups(self, image_patches, n_group=2, new_n=64, alpha=4.0):
        """
        Radial Bias Sampling: generate n_group sampled patch sets per image.

        Args:
            image_patches (Tensor): (bs, n, d) — input patches
            n_group (int): number of views (groups) to sample
            new_n (int): number of patches to sample per view
            alpha (float): exponential decay rate (controls bias sharpness)

        Returns:
            List[Tensor]: list of n_group tensors, each of shape (bs, new_n, d)
        """
        cls_token = image_patches[:,0,:]
        image_patches = image_patches[:,1:,:]
        bs, n, d = image_patches.shape
        side_len = int(n ** 0.5)
        assert side_len ** 2 == n, "n must be a perfect square (e.g., 196=14x14)"
        device = image_patches.device

        # === Step 1: 2D coordinates of each patch (n, 2)
        grid_y, grid_x = torch.meshgrid(
            torch.arange(side_len, device=device),
            torch.arange(side_len, device=device),
            indexing='ij'
        )
        coords = torch.stack([grid_y, grid_x], dim=-1).reshape(n, 2).float()  # (n, 2)

        # === Step 2: Random center indices for each view & batch: (bs, n_group)
        center_indices = torch.randint(0, n, (bs, n_group), device=device)  # (bs, n_group)
        center_coords = coords[center_indices]  # (bs, n_group, 2)

        # === Step 3: Expand coords for broadcasting
        coords_exp = coords.unsqueeze(0).unsqueeze(0)  # (1, 1, n, 2)
        center_coords_exp = center_coords.unsqueeze(2)  # (bs, n_group, 1, 2)

        # === Step 4: Compute distances: (bs, n_group, n)
        dists = torch.norm(coords_exp - center_coords_exp, dim=-1)  # (bs, n_group, n)

        # === Step 5: Compute sampling probabilities
        weights = torch.exp(-alpha * dists)  # (bs, n_group, n)
        probs = weights / weights.sum(dim=-1, keepdim=True)  # (bs, n_group, n)

        # === Step 6: Sample indices: (bs, n_group, new_n)
        probs_2d = probs.reshape(bs * n_group, n)  # flatten batch × group
        # sampled_indices_2d = torch.multinomial(probs_2d, new_n, replacement=False)  # (bs, n_group, new_n)
        _, sampled_indices_2d = torch.topk(probs_2d, new_n, dim=-1, largest=True, sorted=False)
        sampled_indices = sampled_indices_2d.view(bs, n_group, new_n)  # (bs, n_group, new_n)

        # === Step 7: Gather patches
        # Prepare gather index: (bs, n_group, new_n, d)
        idx_expanded = sampled_indices.unsqueeze(-1).expand(-1, -1, -1, d)  # (bs, n_group, new_n, d)
        image_patches_exp = image_patches.unsqueeze(1).expand(-1, n_group, -1, -1)  # (bs, n_group, n, d)

        sampled = torch.gather(image_patches_exp, dim=2, index=idx_expanded)  # (bs, n_group, new_n, d)
        # === Step 8: Return as list of (bs, new_n, d)
        return [torch.cat([cls_token.unsqueeze(1), sampled[:, i]], dim=1) for i in range(n_group)]  # list of (bs, new_n, d)    

    def forward_vit(self, images, n_group=2):
        embedding_output = self.visual_encoder.embeddings(images, interpolate_pos_encoding=True)  # (b, n, d)
        new_n = int(embedding_output.shape[1] * 0.8) if self.training else int(embedding_output.shape[1] * 0.9)

        # print(embedding_output.shape)
        embeddings = self.radial_bias_sampling_groups(embedding_output, n_group=n_group, new_n=new_n, alpha=4.0)
        # print(embeddings[0].shape)
        image_features = []

        for embedding in embeddings:
            encoder_outputs = self.visual_encoder.encoder(embedding)
            # sequence_output = encoder_outputs[0]
            # image_feature = self.layernorm(sequence_output)
            image_features.append(encoder_outputs[0])
            # print(encoder_outputs[0].shape)
        return image_features
    
    def forward_img_encoder(self, group_features):
        image_features = []
        lengths = []
        for base_features in group_features:
            # print(base_features.shape)
            if self.training:
                # Size Augmentation during training, randomly drop grids
                base_length = base_features.size(1)
                features = []
                feat_lengths = []
                rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
                rand_list_2 = np.random.rand(base_features.size(0))
                for i in range(base_features.size(0)):
                    if rand_list_2[i] > 0.2:
                        feat_i = base_features[i][np.where(rand_list_1[i] > 0.20 * rand_list_2[i])]
                        len_i = len(feat_i)
                        pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                        feat_i = torch.cat([feat_i, pads_i], dim=0)
                    else:
                        feat_i = base_features[i]
                        len_i = base_length
                    feat_lengths.append(len_i)
                    features.append(feat_i)
                base_features = torch.stack(features, dim=0)
                base_features = base_features[:, :max(feat_lengths), :]
                feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
            else:
                feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
                feat_lengths[:] = base_features.size(1)

            image_features.append(base_features)
            lengths.append(feat_lengths)

        image_features = torch.cat([self.image_encoder(image_features[0], lengths[0]), self.image_encoder(image_features[1], lengths[1])], dim=1)
        return image_features

    def forward(self, images, return_atttention=False):
        # (B, L_v, C_hidden)
        
        base_features = self.forward_vit(images)
        
        features = self.forward_img_encoder(base_features)

        # features = base_features.mean(dim=1)
        return features
        
    def freeze_backbone(self):
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.visual_encoder.parameters():  
            param.requires_grad = True     

# Language Model with BERT backbone
class EncoderText_BERT(nn.Module):
    def __init__(self, opt):
        super(EncoderText_BERT, self).__init__()

        self.opt = opt
        self.embed_size = opt.embed_size
        
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # from transformers import CLIPModel
        # clip = CLIPModel.from_pretrained("/home/sculiuyang/.cache/huggingface/hub/clip-vit-base-patch32", local_files_only=True)
        # # self.visual_encoder = 
        # self.bert = clip.text_model
        
        # self.tokenizer = BertTokenizer.from_pretrained(opt.bert_path)
        # self.bert = BertModel.from_pretrained(opt.bert_path)
        
        self.fc = nn.Linear(self.bert.config.hidden_size, opt.embed_size)
        self.dropout = nn.Dropout(0.2)
        self.ln = nn.LayerNorm(self.embed_size)

        self.mlp = MLP(768, self.embed_size // 2, self.embed_size, 2)
        self.gpool = GPO(32, 32)


    def forward(self, x, lengths):

        # Embed word ids to vectors
        # pad 0 for redundant tokens in previous process
        bert_attention_mask = (x != 0).float()

        # all hidden features, D=768 in bert-base model
        # attention_mask： Mask to avoid performing attention on padding token indices.
        # bert_output[0] is the last/final hidden states of all tokens
        # bert_output[1] is the hidden state of [CLS] + one fc layer + Tanh, can be used for classification tasks.

        # N = max_cap_lengths, D = 768
        bert_emb = self.bert(input_ids=x, attention_mask=bert_attention_mask)[0]  # B x N x D

        bert_emb = self.dropout(bert_emb)

        cap_len = lengths
        cap_emb = self.fc(bert_emb)

        features = self.mlp(bert_emb) + cap_emb
        pooled_features, pool_weights = self.gpool(features, cap_len.to(features.device))

        pooled_features = self.ln(pooled_features)
        
        return pooled_features, features

    def freeze_backbone(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.bert.parameters():  
            param.requires_grad = True  


def log_sinkhorn_iterations(Z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m * one).to(scores), (n * one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


class EncoderSimilarity(nn.Module):
    def __init__(self, opt):
        super(EncoderSimilarity, self).__init__()
        self.opt = opt
        self.block_dim = [256]
        bin_score = torch.nn.Parameter(torch.tensor(0.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, img_emb, cap_emb):
        cap_emb = l2norm(cap_emb, -1)
        n_cap, cap_dim = cap_emb.size(0), cap_emb.size(1)
        n_img, img_dim = img_emb.size(0), img_emb.size(1)
        sims = []
        for i, block_dim in enumerate(self.block_dim):
            img_blk_num, cap_blk_num = img_emb.size(1) // block_dim, cap_emb.size(1) // block_dim
            img_emb_blocks = torch.chunk(img_emb, img_blk_num, -1)  # (bs, 2*n, block_dim)
            cap_emb_blocks = torch.chunk(cap_emb, cap_blk_num, -1)  # (bs, n, block_dim)

            img_emb_blocks = torch.stack(img_emb_blocks, dim=1)  # (bs, 2*n, block_dim)
            cap_emb_blocks = torch.stack(cap_emb_blocks, dim=1)  # (bs, n, block_dim)

            img_emb_blocks = l2norm(img_emb_blocks, -1)  # (bs, 2*n, block_dim)
            cap_emb_blocks = l2norm(cap_emb_blocks, -1)

            logits = torch.einsum("avc,btc->abvt", [img_emb_blocks, cap_emb_blocks])  # (bs, bs, 2*n, n)

            # logits = log_optimal_transport(logits.reshape(-1, img_blk_num, cap_blk_num), self.bin_score, 20)[:, :-1,
            #          :-1].reshape(n_img, n_cap, img_blk_num, cap_blk_num)
            t2i_logits = logits.max(dim=-2)[0]
            sims.append(t2i_logits.sum(dim=-1))

        sims = torch.stack(sims, -1).sum(-1)

        return sims


def create_efficient_transformer(vit, img_size=224):
    # vit = timm.create_model("timm/vit_base_patch16_224.orig_in21k", pretrained=True, img_size=224)
    # img_size = 224
    patch_size = vit.patch_embed.proj.weight.shape[-1]
    in_chans = vit.patch_embed.proj.weight.shape[1]
    embed_dim = vit.embed_dim
    depth = len(vit.blocks)
    num_heads = vit.blocks[0].attn.num_heads
    mlp_ratio = vit.blocks[0].mlp.fc1.out_features // vit.blocks[0].mlp.fc1.in_features
    qkv_bias = vit.blocks[0].attn.qkv.bias is not None
    qk_norm = not isinstance(vit.blocks[0].attn.k_norm, torch.nn.Identity)
    # init_val = None if isinstance(vit.blocks[0].ls1, nn.Identity) else vit.blocks[0].ls1.gamma.data[0].item()
    class_token = vit.has_class_token
    no_embed_class = vit.no_embed_class
    reg_tokens = vit.num_reg_tokens
    pre_norm = not isinstance(vit.norm_pre, nn.Identity)
    fc_norm = not isinstance(vit.fc_norm, nn.Identity)
    dynamic_img_size = vit.dynamic_img_size
    dynamic_img_pad = vit.patch_embed.dynamic_img_pad
    # distilled = 
    drop_rate = vit.head_drop.p
    pos_drop_rate = vit.pos_drop.p
    patch_drop_rate = 0 if isinstance(vit.patch_drop, torch.nn.Identity) else vit.patch_drop.prob
    proj_drop_rate = vit.blocks[0].mlp.drop1.p
    attn_drop_rate = vit.blocks[0].attn.attn_drop.p
    drop_path_rate = 0 if isinstance(vit.blocks[-1].drop_path1, torch.nn.Identity) else vit.blocks[-1].drop_path1.drop_prob
    embed_layer = PatchEmbed
    norm_layer = type(vit.norm)
    act_layer = type(vit.blocks[0].mlp.act)
    block_fn = Block
    mlp_layer = Mlp

    new_vit = Transformer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm,
                class_token=class_token, no_embed_class=no_embed_class, reg_tokens=reg_tokens, pre_norm=pre_norm, fc_norm=fc_norm, dynamic_img_size=dynamic_img_size, dynamic_img_pad=dynamic_img_pad,
                drop_rate=drop_rate, pos_drop_rate=pos_drop_rate, patch_drop_rate=patch_drop_rate, proj_drop_rate=proj_drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=embed_layer,
                norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn, mlp_layer=mlp_layer, keep_rate=(1,), fuse_token=False)
    
    new_vit.load_state_dict(vit.state_dict(), strict=False)
    
    return new_vit


if __name__ == '__main__':

    pass
