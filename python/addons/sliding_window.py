import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline.model import register_model
from baseline.pytorch.tagger import TaggerModelBase
from baseline.pytorch.classify import ClassifierModelBase
from baseline.pytorch.torchy import pytorch_activation


class SlidingWindow(nn.Module):
    def __init__(self, insz, outsz, window_size):
        assert window_size % 2 == 1
        super(SlidingWindow, self).__init__()
        self.linear = nn.Linear(insz * window_size, outsz)
        self.pad = window_size // 2
        self.window_size = window_size

    def forward(self, x):
        B = x.size(0)
        T = x.size(1)
        x = F.pad(x, (0, 0, self.pad, self.pad), 'constant', 0)
        x = x.unfold(1, self.window_size, 1)
        x = x.transpose(-1, -2).view(B, T, -1)
        return self.linear(x)


@register_model(task='classify', name='window')
class WindowClassifier(ClassifierModelBase):
    def init_pool(self, dsz, **kwargs):
        windowsz = kwargs['filtsz']
        outsz = kwargs['cmotsz']
        act = kwargs.get('act', 'relu')
        windows = []
        for w in windowsz:
            wind = nn.Sequential(
                SlidingWindow(dsz, outsz, w),
                pytorch_activation(act)
            )
            windows.append(wind)
        self.windows = nn.ModuleList(windows)
        self.drop = nn.Dropout(self.pdrop)
        return outsz * len(windowsz)

    def pool(self, btc, lengths):
        mots = []
        for win in self.windows:
            wind_out = win(btc)
            mot, _ = wind_out.max(1)
            mots.append(mot)
        mots = torch.cat(mots, 1)
        return self.drop(mots)



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
