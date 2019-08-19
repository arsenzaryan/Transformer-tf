import numpy as np
import tensorflow as tf
import logging

Dense = tf.keras.layers.Dense
initializers = tf.keras.initializers

from utils import logger, TensorLoggingLevels, Dim, log_size


class ScaledDotProductAttention():
    level = TensorLoggingLevels.attention
    batch_dim, heads_dim, seq_dim, feat_dim = 0,1,2,3

    def __init__(self, dropout = 0.1):
        self.dropout_rate = dropout
        self.attn_weights = None


    def __call__(self, q, k, v, mask = None):
        '''q,k,v are tensors of shape batch x heads x sequence x features'''
        d_k = k.shape[-1].value
        assert d_k == q.shape[-1].value

        attn = tf.matmul(q, tf.transpose(k, perm=[self.batch_dim, self.heads_dim, self.feat_dim, self.seq_dim]))
        attn = attn / np.sqrt(d_k)

        attn = tf.exp(attn)
        log_size(self.level, attn, 'ScaledDotProductAttention, attention weights')
        if mask is not None:
            attn = tf.multiply(attn, tf.cast(mask, tf.float32))

        attn = attn / tf.reduce_sum(attn, axis=-1, keepdims=True)
        self.attn_weights = attn
        attn = tf.nn.dropout(attn, keep_prob=1-self.dropout_rate)

        output = tf.matmul(attn, v)
        output = tf.transpose(output, perm=[self.batch_dim, self.seq_dim, self.heads_dim, self.feat_dim])
        log_size(self.level, output, 'ScaledDotProductAttention, attention output')

        return output, self.attn_weights




class AttentionHead():
    '''A single attention head
    Just for demonstration purposes, is not used in the further code'''
    level = TensorLoggingLevels.attention_head

    def __init__(self, d_feature, dropout = 0.1, name='attn_head'):
        self.attn = ScaledDotProductAttention(dropout=dropout)
        self.q_transform = Dense(d_feature, kernel_initializer=initializers.he_normal(),
                                 name = name + '_Qtnsfm', dtype=tf.float32)
        self.k_transform = Dense(d_feature, kernel_initializer=initializers.he_normal(),
                                 name=name + '_Ktnsfm', dtype=tf.float32)
        self.v_transform = Dense(d_feature, kernel_initializer=initializers.he_normal(),
                                 name=name + '_Vtnsfm', dtype=tf.float32)


    def __call__(self, queries, keys, values, mask = None):
        Q = self.q_transform(queries)
        K = self.k_transform(keys)
        V = self.v_transform(values)
        log_size(self.level, Q, 'AttentionHead, transformed Q, K, V')

        attn_head_output, _ = self.attn(Q, K, V)
        return attn_head_output



class MultiHeadAttnetion():
    level = TensorLoggingLevels.multihead_attention_block

    def __init__(self, d_model, n_heads, dropout = 0.1, name = 'Multihead_attn'):
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.dropout = dropout
        self.attn_weights = None
        self.attn = ScaledDotProductAttention(dropout=dropout)

        self.q_transform = Dense(d_model, kernel_initializer=initializers.he_normal(),
                                 name=name + '_Qtnsfm', dtype=tf.float32)
        self.k_transform = Dense(d_model, kernel_initializer=initializers.he_normal(),
                                 name=name + '_Ktnsfm', dtype=tf.float32)
        self.v_transform = Dense(d_model, kernel_initializer=initializers.he_normal(),
                                 name=name + '_Vtnsfm', dtype=tf.float32)


    def __call__(self, queries, keys, values, mask = None):
        if mask is not None:
            mask = None #to be implemented
        n_batch = queries.get_shape().as_list()[0]

        Q = self.transform_vec(queries, self.q_transform, n_batch)
        K = self.transform_vec(keys, self.k_transform, n_batch)
        V = self.transform_vec(values, self.v_transform, n_batch)

        attn_head_output, self.attn_weights = self.attn(Q, K, V, mask=mask)
        attn_head_output = tf.reshape(attn_head_output, shape=(n_batch, -1, self.n_heads*self.d_k))
        log_size(self.level, attn_head_output, 'Multihead Attention, output')
        return attn_head_output


    def transform_vec(self, vec, dense_transform, n_batch):
        linear_proj =  tf.reshape(dense_transform(vec), shape=[n_batch, -1, self.n_heads, self.d_k])
        return tf.transpose(linear_proj, perm=[0,2,1,3]) #bringing to shape (batch x heads x seq x feat)



class Position_Wise_ff():
    def __init__(self, d_model, d_ff, name='pos_wise_ff'):
        self.dense_feed_forward_1 = Dense(d_ff, activation=tf.nn.relu,
                                          kernel_initializer=initializers.he_normal(), name=name + '_ff1')
        self.dense_feed_forward_2 = Dense(d_model,
                                          kernel_initializer=initializers.he_normal(), name=name + '_ff2')

    def __call__(self, x):
        return self.dense_feed_forward_2(self.dense_feed_forward_1(x))



class EncoderBlock():
    level = TensorLoggingLevels.enc_dec_block

    def __init__(self, d_model, d_feature, d_ff=2048, n_heads = 8, dropout = 0.1, name='Enc_block'):
        assert d_model == d_feature*n_heads
        self.dropout = dropout
        self.multihead = MultiHeadAttnetion(d_model, n_heads, dropout=dropout, name = name)
        self.pos_wise_ff = Position_Wise_ff(d_model, d_ff, name=name)


    def __call__(self, enc_input, mask = None):
        log_size(self.level, enc_input, 'Encoder, input')
        attn = self.multihead(enc_input, enc_input, enc_input, mask=mask)
        log_size(self.level, attn, 'Encoder, attention output')

        enc_state = enc_input + tf.nn.dropout(tf.contrib.layers.layer_norm(attn), keep_prob=1-self.dropout)

        pos = self.pos_wise_ff(enc_state)
        enc_state = enc_state + tf.nn.dropout(tf.contrib.layers.layer_norm(pos), keep_prob=1-self.dropout)
        log_size(self.level, enc_state, 'Encoder, output')

        return enc_state



class TransformerEncoder():
    level = TensorLoggingLevels.enc_dec

    def __init__(self, n_blocks=6, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        self.encoders = [EncoderBlock(d_model, d_model//n_heads, d_ff, n_heads,
                                        dropout, name='Enc_block_'+str(i)) for i in range(n_blocks)]

    def __call__(self, x, mask=None):
        for encoder in self.encoders:
            x = encoder(x, mask=mask)
        return x


class DecoderBlock():
    level = TensorLoggingLevels.enc_dec_block

    def __init__(self, d_model, d_feature, d_ff=2048, n_heads = 8, dropout = 0.1, name='Dec_block'):
        assert d_model == d_feature*n_heads
        self.dropout = dropout
        self.masked_attn = MultiHeadAttnetion(d_model, n_heads, dropout=dropout, name=name+'_masked')
        self.attn = MultiHeadAttnetion(d_model, n_heads, dropout=dropout, name = name)
        self.pos_wise_ff = Position_Wise_ff(d_model, d_ff, name=name)


    def __call__(self, x, enc_out, src_mask=None, tgt_mask=None):
        #apply decoder self attention
        att = self.masked_attn(x, x, x, mask=tgt_mask)
        x = x + tf.nn.dropout(tf.contrib.layers.layer_norm(att), keep_prob=1 - self.dropout)

        #apply encoder-decoder attention
        att = self.attn(queries=x, keys=enc_out, values=enc_out)
        x = x + tf.nn.dropout(tf.contrib.layers.layer_norm(att), keep_prob=1 - self.dropout)

        #position wise feed forward
        pos = self.pos_wise_ff(x)
        x = x + tf.nn.dropout(tf.contrib.layers.layer_norm(pos), keep_prob=1 - self.dropout)
        return x


class TransformerDecoder():
    level = TensorLoggingLevels.enc_dec

    def __init__(self, n_blocks=6, d_model=512, n_heads=8, d_ff=2048, dropout=0.1):
        self.decoders = [DecoderBlock(d_model, d_model//n_heads, d_ff, n_heads,
                                        dropout, name='Dec_block_'+str(i)) for i in range(n_blocks)]


    def __call__(self, x, enc_out, src_mask = None, tgt_mask=None):
        for decoder in self.decoders:
            x = decoder(x, enc_out, src_mask=src_mask, tgt_mask=tgt_mask)
        return x


class Generator():
    def __init__(self, d_output_vocab):
        self.proj = Dense(d_output_vocab, kernel_initializer=initializers.he_normal(), name='generator')

    def __call__(self, x):
        return tf.nn.softmax(self.proj(x), axis = -1)



class PositionalEmbedding():
    def __init__(self, max_len, d_model):
        self.pe = np.zeros([max_len, d_model]).astype(np.float32)
        position = np.arange(0, max_len)[:, None]
        div_term = np.power(10000, np.arange(0, d_model, 2) / d_model)

        self.pe[:, 0::2] = np.sin(position * div_term)
        self.pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(self.pe[None, :, :], dtype = tf.float32, name = 'Positional_embs') #adding batch dimension

    def __call__(self, seq_len):
        return tf.gather(self.pe, np.arange(seq_len), axis=1)




class WordEmbeddings():
    def __init__(self, d_model, d_vocab, max_len = 30):
        self.embeddings = tf.Variable(initial_value=np.random.randn(d_vocab, d_model), dtype=tf.float32)
        self.pos_embeddings = PositionalEmbedding(max_len, d_model)

    def __call__(self, indices):
        seq_len = indices.shape[1] if len(indices.shape)>1 else len(indices)
        return tf.nn.embedding_lookup(self.embeddings, indices) + self.pos_embeddings(seq_len)























