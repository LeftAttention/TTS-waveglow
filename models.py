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
        
def get_model(model_name, model_config, cpu_run,
              uniform_initialize_bn_weight=False, forward_is_infer=False):
    """ Code chooses a model based on name"""
    model = None
    if model_name == 'Tacotron2':
        if forward_is_infer:
            class Tacotron2__forward_is_infer(Tacotron2):
                def forward(self, inputs, input_lengths):
                    return self.infer(inputs, input_lengths)
            model = Tacotron2__forward_is_infer(**model_config)
        else:
            model = Tacotron2(**model_config)
    elif model_name == 'WaveGlow':
        if forward_is_infer:
            class WaveGlow__forward_is_infer(WaveGlow):
                def forward(self, spect, sigma=1.0):
                    return self.infer(spect, sigma)
            model = WaveGlow__forward_is_infer(**model_config)
        else:
            model = WaveGlow(**model_config)
    else:
        raise NotImplementedError(model_name)

    if uniform_initialize_bn_weight:
        init_bn(model)

    if not cpu_run:
        model = model.cuda()
    return model
