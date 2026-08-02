
import torch.nn as nn
import numpy as np
from math import sqrt
#from flash_attn import flash_attn_func
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange
from .rotary import apply_rotary_emb
# Code implementation from https://github.com/thuml/Flowformer
class FlowAttention(nn.Module):
    def __init__(self, attention_dropout=0.1):
        super(FlowAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def kernel_method(self, x):
        return torch.sigmoid(x)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        # kernel
        queries = self.kernel_method(queries)
        keys = self.kernel_method(keys)
        # incoming and outgoing
        normalizer_row = 1.0 / (torch.einsum("nhld,nhd->nhl", queries + 1e-6, keys.sum(dim=2) + 1e-6))
        normalizer_col = 1.0 / (torch.einsum("nhsd,nhd->nhs", keys + 1e-6, queries.sum(dim=2) + 1e-6))
        # reweighting
        normalizer_row_refine = (
            torch.einsum("nhld,nhd->nhl", queries + 1e-6, (keys * normalizer_col[:, :, :, None]).sum(dim=2) + 1e-6))
        normalizer_col_refine = (
            torch.einsum("nhsd,nhd->nhs", keys + 1e-6, (queries * normalizer_row[:, :, :, None]).sum(dim=2) + 1e-6))
        # competition and allocation
        normalizer_row_refine = torch.sigmoid(
            normalizer_row_refine * (float(queries.shape[2]) / float(keys.shape[2])))
        normalizer_col_refine = torch.softmax(normalizer_col_refine, dim=-1) * keys.shape[2]  # B h L vis
        # multiply
        kv = keys.transpose(-2, -1) @ (values * normalizer_col_refine[:, :, :, None])
        x = (((queries @ kv) * normalizer_row[:, :, :, None]) * normalizer_row_refine[:, :, :, None]).transpose(1,
                                                                                                                2).contiguous()
        return x, None


# Code implementation from https://github.com/shreyansh26/FlashAttention-PyTorch
class FlashAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FlashAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def flash_attention_forward(self, Q, K, V, mask=None):
        BLOCK_SIZE = 32
        NEG_INF = -1e10  # -infinity
        EPSILON = 1e-10
        # mask = torch.randint(0, 2, (128, 8)).to(device='cuda')
        O = torch.zeros_like(Q, requires_grad=True)
        l = torch.zeros(Q.shape[:-1])[..., None]
        m = torch.ones(Q.shape[:-1])[..., None] * NEG_INF

        O = O.to(device='cuda')
        l = l.to(device='cuda')
        m = m.to(device='cuda')

        Q_BLOCK_SIZE = min(BLOCK_SIZE, Q.shape[-1])
        KV_BLOCK_SIZE = BLOCK_SIZE

        Q_BLOCKS = torch.split(Q, Q_BLOCK_SIZE, dim=2)
        K_BLOCKS = torch.split(K, KV_BLOCK_SIZE, dim=2)
        V_BLOCKS = torch.split(V, KV_BLOCK_SIZE, dim=2)
        if mask is not None:
            mask_BLOCKS = list(torch.split(mask, KV_BLOCK_SIZE, dim=1))

        Tr = len(Q_BLOCKS)
        Tc = len(K_BLOCKS)

        O_BLOCKS = list(torch.split(O, Q_BLOCK_SIZE, dim=2))
        l_BLOCKS = list(torch.split(l, Q_BLOCK_SIZE, dim=2))
        m_BLOCKS = list(torch.split(m, Q_BLOCK_SIZE, dim=2))

        for j in range(Tc):
            Kj = K_BLOCKS[j]
            Vj = V_BLOCKS[j]
            if mask is not None:
                maskj = mask_BLOCKS[j]

            for i in range(Tr):
                Qi = Q_BLOCKS[i]
                Oi = O_BLOCKS[i]
                li = l_BLOCKS[i]
                mi = m_BLOCKS[i]

                scale = 1 / np.sqrt(Q.shape[-1])
                Qi_scaled = Qi * scale

                S_ij = torch.einsum('... i d, ... j d -> ... i j', Qi_scaled, Kj)
                if mask is not None:
                    # Masking
                    maskj_temp = rearrange(maskj, 'b j -> b 1 1 j')
                    S_ij = torch.where(maskj_temp > 0, S_ij, NEG_INF)

                m_block_ij, _ = torch.max(S_ij, dim=-1, keepdims=True)
                P_ij = torch.exp(S_ij - m_block_ij)
                if mask is not None:
                    # Masking
                    P_ij = torch.where(maskj_temp > 0, P_ij, 0.)

                l_block_ij = torch.sum(P_ij, dim=-1, keepdims=True) + EPSILON

                P_ij_Vj = torch.einsum('... i j, ... j d -> ... i d', P_ij, Vj)

                mi_new = torch.maximum(m_block_ij, mi)
                li_new = torch.exp(mi - mi_new) * li + torch.exp(m_block_ij - mi_new) * l_block_ij

                O_BLOCKS[i] = (li / li_new) * torch.exp(mi - mi_new) * Oi + (
                        torch.exp(m_block_ij - mi_new) / li_new) * P_ij_Vj
                l_BLOCKS[i] = li_new
                m_BLOCKS[i] = mi_new

        O = torch.cat(O_BLOCKS, dim=2)
        l = torch.cat(l_BLOCKS, dim=2)
        m = torch.cat(m_BLOCKS, dim=2)
        return O, l, m

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        res = \
        self.flash_attention_forward(queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), values.permute(0, 2, 1, 3),
                                     attn_mask)[0]
        return res.permute(0, 2, 1, 3).contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


# Code implementation from https://github.com/zhouhaoyi/Informer2020
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn

class DiffAttention_Claude(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DiffAttention_Claude, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        # Lambda parameters for differential attention
        self.lambda_q1 = nn.Parameter(torch.randn(factor) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(factor) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(factor) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(factor) * 0.1)
        self.lambda_init = 0.8

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # 将queries和keys分成两部分
        queries1, queries2 = torch.chunk(queries, 2, dim=-1)
        keys1, keys2 = torch.chunk(keys, 2, dim=-1)

        # 计算两组不同的注意力分数
        scores_q1 = torch.einsum("blhe,bshe->bhls", queries1, keys1)
        scores_q2 = torch.einsum("blhe,bshe->bhls", queries2, keys2)

        # 应用掩码（如果需要）
        if self.mask_flag and attn_mask is not None:
            scores_q1.masked_fill_(attn_mask.mask, -float('inf'))
            scores_q2.masked_fill_(attn_mask.mask, -float('inf'))

        # 计算softmax注意力
        attn_q1 = torch.softmax(scale * scores_q1, dim=-1)
        attn_q2 = torch.softmax(scale * scores_q2, dim=-1)

        # 应用dropout
        attn_q1 = self.dropout(attn_q1)
        attn_q2 = self.dropout(attn_q2)

        # 计算lambda因子
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 计算最终的差分注意力权重
        attn_weights = attn_q1 - lambda_full * attn_q2

        # 计算最终的注意力输出
        V = torch.einsum("bhls,bshd->blhd", attn_weights, values)

        if self.output_attention:
            return (V.contiguous(), attn_weights)
        else:
            return (V.contiguous(), None)

# Usage
# diff_attention = DiffAttention(scale=1.0, output_attention=True)
# output, attention_weights = diff_attention(queries, keys, values, attn_mask)

import torch
import torch.nn as nn
import numpy as np
from math import sqrt

class DiffAttention_GPT(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False, lambda_init=1.0):
        """
        差分注意力机制类
        :param mask_flag: 是否使用掩码，默认为True
        :param scale: 注意力分数的缩放因子，如果为None，则使用1/sqrt(E)
        :param attention_dropout: 注意力的dropout概率
        :param output_attention: 是否输出注意力权重，默认为False
        :param lambda_init: 差分计算中的初始lambda值，用于调整两个注意力图之间的权重差异
        """
        super(DiffAttention_GPT, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))

    def forward(self, queries, keys, values, attn_mask=None):
        """
        前向传播过程
        :param queries: 查询向量，形状为 (B, L, H, E)
        :param keys: 键向量，形状为 (B, S, H, E)
        :param values: 值向量，形状为 (B, S, H, D)
        :param attn_mask: 注意力掩码，默认None
        :return: 如果output_attention为True，返回值 (V, A)，否则返回 (V, None)
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # 将查询和键向量分别拆分为两个部分，用于计算差分注意力图
        Q1, Q2 = torch.chunk(queries, 2, dim=-1)  # 分成两个查询向量，形状为 (B, L, H, E/2)
        K1, K2 = torch.chunk(keys, 2, dim=-1)    # 分成两个键向量，形状为 (B, S, H, E/2)

        # 计算两个独立的注意力图
        scores1 = torch.einsum("blhe,bshe->bhls", Q1, K1)  # (B, H, L, S)
        scores2 = torch.einsum("blhe,bshe->bhls", Q2, K2)  # (B, H, L, S)

        # 如果需要应用掩码，则对分数进行掩码处理
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores1.masked_fill_(attn_mask.mask, -np.inf)
            scores2.masked_fill_(attn_mask.mask, -np.inf)

        # 对两个注意力图分别进行softmax和缩放，然后计算差分
        A1 = torch.softmax(scale * scores1, dim=-1)  # 第一组注意力分数
        A2 = torch.softmax(scale * scores2, dim=-1)  # 第二组注意力分数

        # 差分注意力图，通过参数lambda_param来调整两个注意力图的权重
        A_diff = A1 - self.lambda_param * A2

        # 应用dropout并计算加权的值向量
        A_diff = self.dropout(A_diff)
        V = torch.einsum("bhls,bshd->blhd", A_diff, values)  # 计算注意力加权后的输出

        # 如果设置为输出注意力权重，则返回 (V, A_diff)
        if self.output_attention:
            return (V.contiguous(), A_diff)
        else:
            return (V.contiguous(), None)

#原始注意力层        
class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None
class MultiheadFlashDiff1(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, d_model=None, n_heads=None, depth=None):
        super().__init__()
        # 基础参数设置
        self.embed_dim = d_model
        self.num_heads = n_heads
        self.num_kv_heads = n_heads  # 如果使用GQA，这里应该是n_heads的一半
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = d_model // n_heads // 2
        self.scaling = self.head_dim ** -0.5

        # 投影层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model // self.n_rep, bias=False)
        self.v_proj = nn.Linear(d_model, d_model // self.n_rep, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # lambda参数初始化
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        # RMSNorm层
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
        
        # 保存其他参数
        self.mask_flag = mask_flag
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        bsz, tgt_len, _ = queries.shape
        src_len = keys.shape[1]

        # 重塑维度
        q = queries.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = keys.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = values.view(bsz, src_len, self.num_kv_heads, 2*self.head_dim)

        # 应用旋转位置编码
        if hasattr(self, 'rotary_emb_generator'):
            cos, sin = self.rotary_emb_generator.get_rotary_embedding(tgt_len, queries.device)
            q = apply_rotary_emb(q, cos, sin, interleaved=True)
            k = apply_rotary_emb(k, cos, sin, interleaved=True)

        # 分离q和k用于差分注意力
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]

        # 使用Flash Attention计算
        attn1 = flash_attn_func(q1, k1, v, causal=self.mask_flag)
        attn2 = flash_attn_func(q2, k2, v, causal=self.mask_flag)

        # 计算lambda因子
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 差分注意力计算
        attn = attn1 - lambda_full * attn2

        # 应用RMSNorm和缩放
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)

        # 重塑并投影输出
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        attn = self.out_proj(attn)

        if self.output_attention:
            return attn, None
        return attn, None
class MultiheadFlashDiff2(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, d_model=None, n_heads=None, depth=None):
        super().__init__()
        # 基础参数设置
        self.embed_dim = d_model
        self.num_heads = n_heads
        self.num_kv_heads = n_heads  # 如果使用GQA，这里应该是n_heads的一半
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = d_model // n_heads // 2
        self.scaling = self.head_dim ** -0.5

        # 投影层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model // self.n_rep, bias=False)
        self.v_proj = nn.Linear(d_model, d_model // self.n_rep, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # lambda参数初始化
        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        # RMSNorm层
        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)
        
        # 保存其他参数
        self.mask_flag = mask_flag
        self.output_attention = output_attention

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        bsz, tgt_len, _ = queries.shape
        src_len = keys.shape[1]

        # 重塑维度
        q = queries.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = keys.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim)
        v = values.view(bsz, src_len, self.num_kv_heads, 2*self.head_dim)

        # 应用旋转位置编码
        if hasattr(self, 'rotary_emb_generator'):
            cos, sin = self.rotary_emb_generator.get_rotary_embedding(tgt_len, queries.device)
            q = apply_rotary_emb(q, cos, sin, interleaved=True)
            k = apply_rotary_emb(k, cos, sin, interleaved=True)

        # 分离q、k和v用于差分注意力
        q = q.reshape(bsz, tgt_len, self.num_heads, 2, self.head_dim)
        k = k.reshape(bsz, src_len, self.num_kv_heads, 2, self.head_dim)
        q1, q2 = q[:, :, :, 0], q[:, :, :, 1]
        k1, k2 = k[:, :, :, 0], k[:, :, :, 1]
        v1, v2 = v[:, :, :, 0], v[:, :, :, 1]

        # 计算四组注意力
        attn11 = flash_attn_func(q1, k1, v1, causal=self.mask_flag)
        attn12 = flash_attn_func(q1, k1, v2, causal=self.mask_flag)
        attn1 = torch.cat([attn11, attn12], dim=-1)
        
        attn21 = flash_attn_func(q2, k2, v1, causal=self.mask_flag)
        attn22 = flash_attn_func(q2, k2, v2, causal=self.mask_flag)
        attn2 = torch.cat([attn21, attn22], dim=-1)

        # 计算lambda因子
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        # 差分注意力计算
        attn = attn1 - lambda_full * attn2

        # 应用RMSNorm和缩放
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)

        # 重塑并投影输出
        attn = attn.reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)
        attn = self.out_proj(attn)

        if self.output_attention:
            return attn, None
        return attn, None
import math
import torch
import torch.nn.functional as F
from torch import nn
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True, memory_efficient=False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'
    
#错误的差分注意力计算，不能使用chunk,除非是物理切分两个携带相同信息的变量
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )
def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class DiffAttention_Cursor(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,d_model=None,n_heads=None,depth=None):
        super(DiffAttention_Cursor, self).__init__()
        if d_model is None or n_heads is None or depth is None:
            raise ValueError("DiffAttention_Cursor requires d_model, n_heads, and depth.")
        if d_model % (2 * n_heads) != 0:
            raise ValueError("DiffAttention_Cursor requires d_model to be divisible by 2 * n_heads.")
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention

        self.embed_dim = d_model
        self.num_heads = n_heads
        self.num_kv_heads = n_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = d_model // n_heads // 2
        self.scaling = self.scale or self.head_dim ** -0.5

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        bsz, tgt_len, _, query_dim = queries.shape
        _, src_len, _, key_dim = keys.shape
        _, _, _, value_dim = values.shape
        expected_qk_dim = 2 * self.head_dim
        expected_v_dim = 2 * self.head_dim
        if query_dim != expected_qk_dim or key_dim != expected_qk_dim or value_dim != expected_v_dim:
            raise ValueError("DiffAttention_Cursor expects q/k/v head dim to equal d_model // n_heads.")

        q = queries.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        k = keys.view(bsz, src_len, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        v = values.view(bsz, src_len, self.num_heads, 2 * self.head_dim).transpose(1, 2)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q = q * self.scaling
        attn_weights = torch.matmul(q, k.transpose(-1, -2))

        if self.mask_flag and attn_mask is None:
            offset = src_len - tgt_len
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_weights),
                1 + offset,
            )

        if attn_mask is not None:
            mask = attn_mask.mask if hasattr(attn_mask, 'mask') else attn_mask
            if mask.dtype == torch.bool:
                attn_weights = attn_weights.masked_fill(mask, -float('inf'))
            else:
                attn_weights = attn_weights + mask.type_as(attn_weights)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2)

        if self.output_attention:
            return attn.contiguous(), attn_weights
        else:
            return attn.contiguous(), None
#完全兼容AttentionLayer的输入输出
class DiffAttention_Perfect(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,d_model=None,n_heads=None,depth=None):
        super(DiffAttention_Perfect, self).__init__()
        if d_model is None or n_heads is None or depth is None:
            raise ValueError("DiffAttention_Perfect requires d_model, n_heads, and depth.")
        if d_model % (2 * n_heads) != 0:
            raise ValueError("DiffAttention_Perfect requires d_model to be divisible by 2 * n_heads.")
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention

        self.embed_dim = d_model
        self.num_heads = n_heads
        self.num_kv_heads = n_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = d_model // n_heads // 2
        self.scaling = self.scale or self.head_dim ** -0.5

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0,std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        bsz, tgt_len, _, query_dim = queries.shape
        _, src_len, _, key_dim = keys.shape
        _, _, _, value_dim = values.shape

        if query_dim != 2 * self.head_dim or key_dim != 2 * self.head_dim or value_dim != 2 * self.head_dim:
            raise ValueError("DiffAttention_Perfect expects q/k/v head dim to equal d_model // n_heads.")

        q = queries.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim).transpose(1, 2)
        k = keys.view(bsz, src_len, 2 * self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = values.view(bsz, src_len, self.num_kv_heads, 2 * self.head_dim).transpose(1, 2)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        q = q * self.scaling
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = torch.nan_to_num(attn_scores)

        if self.mask_flag and attn_mask is None:
            offset = src_len - tgt_len
            attn_mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                .float()
                .fill_(float("-inf"))
                .type_as(attn_scores),
                1 + offset,
            )

        if attn_mask is not None:
            mask = attn_mask.mask if hasattr(attn_mask, 'mask') else attn_mask
            if mask.dtype == torch.bool:
                attn_scores = attn_scores.masked_fill(mask, -float('inf'))
            else:
                attn_scores = attn_scores + mask.type_as(attn_scores)

        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).type_as(attn_scores)

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]

        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2)

        if self.output_attention:
            return attn.contiguous(), attn_weights
        else:
            return attn.contiguous(), None
def build_rel_pos(self, x, start_pos):
        if self._precomputed_freqs_cis is None:
            angle = 1.0 / (self.args.rope_theta ** torch.linspace(0, 1, self.head_dim // 2, dtype=torch.float, device=x.device))
            index = torch.arange(self.args.max_seq_len).to(angle)
            self._precomputed_freqs_cis = index[:, None] * angle

        cos = torch.cos(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        sin = torch.sin(self._precomputed_freqs_cis[start_pos:start_pos+x.size(1)])
        rel_pos = (cos.to(x.dtype), sin.to(x.dtype))
        return rel_pos
class RotaryPositionalEmbeddingGenerator(nn.Module):
    def __init__(self, dim: int, base: int = 10_000, rotary_dim: int = None):
        super().__init__()
        self.dim = dim
        self.base = base
        self.rotary_dim = rotary_dim if rotary_dim is not None else dim // 2

    def get_rotary_embedding(self, seq_len: int, device: torch.device):
        theta = 1.0 / (self.base ** (torch.arange(0, self.rotary_dim, 2).float().to(device) / self.rotary_dim))
        seq_idx = torch.arange(seq_len, device=device).float()
        idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
        cos = idx_theta.cos()
        sin = idx_theta.sin()
        return cos, sin
