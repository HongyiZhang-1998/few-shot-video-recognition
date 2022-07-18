import torch
from torch import nn
import math
import gl

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class CrossAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, hidden_dropout_prob):
        super(CrossAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        attention_probs_dropout_prob = 0.2
        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x, y, debug):
        mixed_query_layer = self.query(x)
        mixed_key_layer = self.key(y)
        mixed_value_layer = self.value(x)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        # attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # import os
        # import numpy as np
        # if debug == True and gl.epoch % 5 == 0 and gl.iter % 100 == 0:
        #     path = os.path.join(gl.experiment_root, 'save_attention_probs', 'epoch_{}_iter_{}'.format(gl.epoch, gl.iter))
        #     if not os.path.exists(path):
        #         os.makedirs(path)
        #     np.save(os.path.join(path, 'attention_y_probs.npy'), attention_probs.cpu().detach().numpy())


        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + x)  # residual

        return hidden_states

# 位置编码层
def PositionEncoding(inputs, model_dim):
    '''
    :param model_dim:词嵌入维度
    inputs:(batch, seq_length, model_dim)
    '''
    batch, seq_length = inputs.shape[0], inputs.shape[1]
    position_encodings = torch.zeros((seq_length, model_dim)).to(gl.device)

    for pos in range(seq_length):
        for i in range(model_dim):
            position_encodings[pos, i] = pos / torch.pow(torch.tensor(10000.0), (i - i % 2) / model_dim)

    position_encodings = position_encodings.repeat(batch, 1, 1)

    position_encodings[:, :, 0::2] = torch.sin(position_encodings[:, :, 0::2])  # 2i
    position_encodings[:, :, 1::2] = torch.cos(position_encodings[:, :, 1::2])  # 2i+1

    return position_encodings

