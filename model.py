import torch
import torch.nn as nn
import math


# a pointer network layer for policy output
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.key_dim = embedding_dim
        self.norm_factor = self.key_dim ** -0.5
        self.tanh_clipping = 10.0

        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def forward(self, q, k, mask=None):
        B, T_q, _ = q.size()
        T_k = k.size(1)

        Q = self.W_q(q)
        K = self.W_k(k)

        scores = torch.bmm(Q, K.transpose(1, 2)) * self.norm_factor
        scores = self.tanh_clipping * scores.tanh()

        if mask is not None:
            scores = scores.masked_fill(mask == 1, -6e4)

        attn = torch.log_softmax(scores, dim=-1)  # B, T_q, T_k

        return attn


# standard multi head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8, bias=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads
        self.norm_factor = self.head_dim ** -0.5

        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=bias)
        self.W_o = nn.Linear(embedding_dim, embedding_dim, bias=bias)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        k = q if k is None else k
        v = q if v is None else v

        B, T_q, _ = q.size()
        T_k = k.size(1)

        Q = self.W_q(q).view(B, T_q, self.n_heads, self.head_dim).transpose(1, 2)  # B, n_heads, T, head_dim
        K = self.W_k(k).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(v).view(B, T_k, self.n_heads, self.head_dim).transpose(1, 2)

        scores = (Q @ K.transpose(-2, -1)) * self.norm_factor  # B, n_heads, T_q, T_k

        if attn_mask is not None:  # attn_mask: B, T_q, T_k
            attn_mask = attn_mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(attn_mask > 0, -6e4)
        if key_padding_mask is not None:  # key_padding_mask: B, 1, T_k
            key_padding_mask = key_padding_mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(key_padding_mask > 0, -6e4)

        attn = torch.softmax(scores, dim=-1)

        context = attn @ V  # B, n_heads, T_q, head_dim
        context = context.transpose(1, 2).contiguous().view(B, T_q, self.embedding_dim)
        out = self.W_o(context)

        return out, attn  # out: B, T_q, embedding_dim, attn: B, n_heads, T_q, T_k


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        return self.normalizer(x)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head, ff_dim=512, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(embedding_dim, n_head, bias=False)
        self.norm1 = Normalization(embedding_dim)
        self.norm2 = Normalization(embedding_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(embedding_dim, ff_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(ff_dim, embedding_dim))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h = self.norm1(src)
        attn_out, _ = self.mha(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        attn_out = self.drop1(attn_out)
        h = src + attn_out

        h2 = self.norm2(h)
        ffn_out = self.ffn(h2)
        ffn_out = self.drop2(ffn_out)
        h2 = h + ffn_out
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head, ff_dim=512, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.mha = MultiHeadAttention(embedding_dim, n_head, bias=False)
        self.norm1 = Normalization(embedding_dim)
        self.norm2 = Normalization(embedding_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(embedding_dim, ff_dim),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(ff_dim, embedding_dim))

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.norm1(tgt)
        attn_out, w = self.mha(q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        attn_out = self.drop1(attn_out)
        h = h0 + attn_out

        h1 = h
        h = self.norm2(h)
        ffn_out = self.ffn(h)
        ffn_out = self.drop2(ffn_out)
        h2 = ffn_out + h1
        return h2, w


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for _ in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for _ in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w


class PolicyNet(nn.Module):
    def __init__(self, node_dim, embedding_dim):
        super(PolicyNet, self).__init__()

        # graph encoder
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)

        # decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        # pointer
        self.pointer = SingleHeadAttention(embedding_dim)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature,
                                                         key_padding_mask=node_padding_mask,
                                                         attn_mask=edge_mask)

        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1,
                                                  current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(current_node_feature,
                                                                    enhanced_node_feature,
                                                                    node_padding_mask)

        return current_node_feature, enhanced_current_node_feature

    def output_policy(self, current_node_feature, enhanced_current_node_feature,
                      enhanced_node_feature, current_edge, edge_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_state_feature = self.current_embedding(torch.cat((enhanced_current_node_feature,
                                                                current_node_feature), dim=-1))

        neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                           current_edge.repeat(1, 1, embedding_dim))

        logp = self.pointer(current_state_feature, neighboring_feature, edge_padding_mask)
        logp = logp.squeeze(1)

        return logp

    # @torch.compile
    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        current_node_feature, enhanced_current_node_feature = self.decode_state(
            enhanced_node_feature, current_index, node_padding_mask)
        logp = self.output_policy(current_node_feature, enhanced_current_node_feature,
                                  enhanced_node_feature, current_edge, edge_padding_mask)

        return logp


class QNet(nn.Module):
    def __init__(self, node_dim, embedding_dim):
        super(QNet, self).__init__()

        # graph encoder
        self.initial_embedding = nn.Linear(node_dim, embedding_dim)
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)

        # decoder
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        self.q_values_layer = nn.Linear(embedding_dim * 3, 1)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature,
                                                         key_padding_mask=node_padding_mask,
                                                         attn_mask=edge_mask)

        return enhanced_node_feature

    def decode_state(self, enhanced_node_feature, current_index, node_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        current_node_feature = torch.gather(enhanced_node_feature, 1,
                                                  current_index.repeat(1, 1, embedding_dim))
        enhanced_current_node_feature, _ = self.decoder(current_node_feature,
                                                                    enhanced_node_feature,
                                                                    node_padding_mask)

        return current_node_feature, enhanced_current_node_feature

    def output_q(self, current_node_feature, enhanced_current_node_feature, enhanced_node_feature,
                 current_edge, edge_padding_mask):
        embedding_dim = enhanced_node_feature.size()[2]
        k_size = current_edge.size()[1]
        current_state_feature = torch.cat((enhanced_current_node_feature, current_node_feature), dim=-1)

        neighboring_feature = torch.gather(enhanced_node_feature, 1,
                                           current_edge.repeat(1, 1, embedding_dim))

        action_features = torch.cat((current_state_feature.repeat(1, k_size, 1), neighboring_feature), dim=-1)
        q_values = self.q_values_layer(action_features)
        return q_values

    def forward(self, node_inputs, node_padding_mask, edge_mask, current_index,
                current_edge, edge_padding_mask):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        current_node_feature, enhanced_current_node_feature = self.decode_state(enhanced_node_feature, current_index, node_padding_mask)
        q_values = self.output_q(current_node_feature, enhanced_current_node_feature,
                                 enhanced_node_feature, current_edge, edge_padding_mask)

        return q_values
