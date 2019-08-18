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