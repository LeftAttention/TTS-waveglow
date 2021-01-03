from tacotron2.model import Tacotron2
from waveglow.model import WaveGlow
import torch

def model_parser(model_name, parser, add_help=False):
    if model_name == 'Tacotron2':
        from tacotron2.arg_parser import tacotron2_parser
        return tacotron2_parser(parser, add_help)
    if model_name == 'WaveGlow':
        from waveglow.arg_parser import waveglow_parser
        return waveglow_parser(parser, add_help)
    else:
        raise NotImplementedError(model_name)

def batchnorm_to_float(module):
    """Converts batch norm to FP32"""
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.float()
    for child in module.children():
        batchnorm_to_float(child)
    return module

def init_bn(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        if module.affine:
            module.weight.data.uniform_()
    for child in module.children():
        init_bn(child)
