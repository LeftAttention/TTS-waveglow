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
