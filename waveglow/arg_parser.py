import argparse

def waveglow_parser(parent, add_help=False):
    """
    Parse commandline arguments.
    """
    parser = argparse.ArgumentParser(parents=[parent], add_help=add_help)
    
    # misc parameters
    parser.add_argument('--n-mel-channels', default=80, type=int,
                        help='Number of bins in mel-spectrograms')

    # glow parameters
    parser.add_argument('--flows', default=12, type=int,
                        help='Number of steps of flow')
    parser.add_argument('--groups', default=8, type=int,
                        help='Number of samples in a group processed by the steps of flow')
    parser.add_argument('--early-every', default=4, type=int,
                        help='Determines how often (i.e., after how many coupling layers) \
                        a number of channels (defined by --early-size parameter) are output\
                        to the loss function')
    parser.add_argument('--early-size', default=2, type=int,
                        help='Number of channels output to the loss function')
    parser.add_argument('--sigma', default=1.0, type=float,
                        help='Standard deviation used for sampling from Gaussian')
    parser.add_argument('--segment-length', default=4000, type=int,
                        help='Segment length (audio samples) processed per iteration')

    # wavenet parameters
    wavenet = parser.add_argument_group('WaveNet parameters')
    wavenet.add_argument('--wn-kernel-size', default=3, type=int,
                        help='Kernel size for dialted convolution in the affine coupling layer (WN)')
    wavenet.add_argument('--wn-channels', default=512, type=int,
                        help='Number of channels in WN')
    wavenet.add_argument('--wn-layers', default=8, type=int,
                        help='Number of layers in WN')

    return parser