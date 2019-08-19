import numpy as np
import tensorflow as tf
import logging
import inspect
from enum import IntEnum



logger = logging.getLogger("tensor_shapes")
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
# if you want the model to continuously print tensor shapes, set to DEBUG!
logger.setLevel(1)




def log_size(level: int, tnsr: tf.Tensor, name: str):
    logger.log(level=level, msg="{} size={}".format(name, tnsr.get_shape()))


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return (subsequent_mask) == 0


def gen_batch(data, batch_size=10):
    inp_seq, out_seq = data[0], data[1]
    n_data = len(inp_seq)

    n_batches = n_data // batch_size + 1
    perm = np.random.permutation((n_data))
    for iter_i in range(n_batches):
        next_batch_ixs = perm[iter_i*batch_size : (iter_i+1)*batch_size]
        yield inp_seq[next_batch_ixs], out_seq[next_batch_ixs]










# Control how much debugging output we want
class TensorLoggingLevels(IntEnum):
    attention = 1
    attention_head = 2
    multihead_attention_block = 3
    enc_dec_block = 4
    enc_dec = 5

class Dim(IntEnum):
    batch = 0
    seq = 1
    head = 2
    feature = 3