import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline.model import register_model
from baseline.pytorch.tagger import TaggerModelBase
from baseline.pytorch.classify import ClassifierModelBase
from baseline.pytorch.torchy import pytorch_activation


class SlidingWindow(nn.Module):
    def __init__(self, insz, outsz, windowsz):
        super(SlidingWindow, self).__init__()
        self.linear = nn.Linear(insz * windowsz, outsz)
        self.pad = (
            0, 0,  # Padding on the C dim
            # Padding on the T dim
            (windowsz - 1) // 2,  # Padding on the top
            (windowsz - 1) - (windowsz - 1) // 2  # Padding on the bottom
        )
        self.window_size = windowsz

    def forward(self, x):
        B = x.size(0)
        T = x.size(1)
        x = F.pad(x, self.pad, 'constant', 0)
        x = x.unfold(1, self.window_size, 1)
        x = x.transpose(-1, -2)
        x = x.contiguous().view(B, T, -1)
        return self.linear(x)


class ParallelSlidingWindows(nn.Module):
    def __init__(self, insz, outsz, windowsz, activation_type, pdrop):
        super(ParallelSlidingWindows, self).__init__()
        windows = []
        if isinstance(outsz, int):
            outsz = [outsz] * len(windowsz)

        self.outsz = sum(outsz)
        for osz, wsz in zip(outsz, windowsz):
            window = nn.Sequential(
                SlidingWindow(insz, osz, wsz),
                pytorch_activation(activation_type)
            )
            windows.append(window)
        self.windows = nn.ModuleList(windows)
        self.drop = nn.Dropout(pdrop)

    def forward(self, btc):
        mots = []
        for win in self.windows:
            win_out = win(btc)
            mot, _ = win_out.max(1)
            mots.append(mot)
        mots = torch.cat(mots, 1)
        return self.drop(mots)


@register_model(task='classify', name='window')
class WindowClassifier(ClassifierModelBase):
    def init_pool(self, dsz, **kwargs):
        windowsz = kwargs['filtsz']
        outsz = kwargs['cmotsz']
        self.parallel_windows = ParallelSlidingWindows(dsz, outsz, windowsz, 'relu', self.pdrop)
        return self.parallel_windows.outsz

    def pool(self, btc, lengths):
        return self.parallel_windows(btc)



@register_model(task='tagger', name='window')
class WindowTaggerModel(TaggerModelBase):
    def init_encoder(self, input_sz, **kwargs):
        self.drop = nn.Dropout(float(kwargs.get('dropout')))
        hsz = int(kwargs['hsz'])
        window_sz = int(kwargs['window_size'])
        self.trans = SlidingWindow(input_sz, hsz, window_sz)
        self.act = pytorch_activation(kwargs.get('act', 'relu'))
        return hsz

    def encode(self, words_over_time, lengths):
        x = words_over_time.transpose(0, 1).contiguous()
        x = self.drop(self.act(self.trans(x)))
        return x.transpose(0, 1).contiguous()
