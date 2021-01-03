import torch
import random
import common.layers as layers
from common.utils import load_wav_to_torch, load_filepaths_and_text, to_gpu


class MelAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) computes mel-spectrograms from audio files.
    """

    def __init__(self, dataset_path, audiopaths_and_text, args):
        self.audiopaths_and_text = load_filepaths_and_text(dataset_path, audiopaths_and_text)
        self.max_wav_value = args.max_wav_value
        self.sampling_rate = args.sampling_rate
        self.stft = layers.TacotronSTFT(
            args.filter_length, args.hop_length, args.win_length,
            args.n_mel_channels, args.sampling_rate, args.mel_fmin,
            args.mel_fmax)
        self.segment_length = args.segment_length
        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

    def get_mel_audio_pair(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)

        if sampling_rate != self.stft.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.stft.sampling_rate))

        # Take segment
        if audio.size(0) >= self.segment_length:
            max_audio_start = audio.size(0) - self.segment_length
            audio_start = random.randint(0, max_audio_start)
            audio = audio[audio_start:audio_start+self.segment_length]
        else:
            audio = torch.nn.functional.pad(
                audio, (0, self.segment_length - audio.size(0)), 'constant').data

        audio = audio / self.max_wav_value
        audio_norm = audio.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        melspec = self.stft.mel_spectrogram(audio_norm)
        melspec = melspec.squeeze(0)

        return (melspec, audio, len(audio))

    def __getitem__(self, index):
        return self.get_mel_audio_pair(self.audiopaths_and_text[index][0])

    def __len__(self):
        return len(self.audiopaths_and_text)


def batch_to_gpu(batch):
    x, y, len_y = batch
    x = to_gpu(x).float()
    y = to_gpu(y).float()
    len_y = to_gpu(torch.sum(len_y))
    return ((x, y), y, len_y)
