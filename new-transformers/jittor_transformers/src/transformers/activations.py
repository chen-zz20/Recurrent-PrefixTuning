import math

import jittor as jt


from .utils import logging


logger = logging.get_logger(__name__)


def swish(x):
    return x * jt.sigmoid(x)


def _gelu_python(x):
    """Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    This is now written in C in torch.nn.functional
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + jt.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + jt.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * jt.pow(x, 3.0))))



gelu = jt.nn.gelu


def gelu_fast(x):
    return 0.5 * x * (1.0 + jt.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def mish(x):
    return x * jt.tanh(jt.nn.softplus(x))


ACT2FN = {
    "relu": jt.nn.relu,
    "swish": swish,
    "gelu": gelu,
    "tanh": jt.tanh,
    "gelu_new": gelu_new,
    "gelu_fast": gelu_fast,
    "mish": mish,
}


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError("function {} not found in ACT2FN mapping {}".format(activation_string, list(ACT2FN.keys())))


def mish(x):
    return x * jt.tanh(jt.nn.softplus(x))
