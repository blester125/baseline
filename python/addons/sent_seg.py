import re
import codecs
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from baseline.model import register_model
from baseline.reader import register_reader, CONLLSeqReader
from baseline.vectorizers import register_vectorizer, Dict1DVectorizer
from baseline.pytorch.torchy import *
from baseline.pytorch.tagger import TaggerModelBase
from baseline.pytorch.embeddings import PyTorchEmbeddings, register_embeddings


@register_embeddings(name='continuous')
class PassthroughEmbeddings(PyTorchEmbeddings):
    def __init__(self, _, **kwargs):
        super(PassthroughEmbeddings, self).__init__()
        self.dsz = kwargs.get('dsz')

    def get_dsz(self):
        return self.dsz

    def forward(self, x):
        return x


@register_reader(task='tagger', name='doc-conll')
class DocCONLLReader(CONLLSeqReader):
    """Read a conll file into documents rather than sentences."""
    def __init__(self, vectorizers, trim=False, truncate=False, mxlen=-1, **kwargs):
        super(DocCONLLReader, self).__init__(vectorizers, trim, truncate, mxlen, **kwargs)
        self.label2index = {}

    def build_vocab(self, files, **kwargs):
        vocabs = {k: Counter() for k in self.vectorizers.keys()}
        labels = Counter()

        for file_name in files:
            if file_name is None: continue
            docs = self.read_examples(file_name)
            for doc in docs:
                for sentence in doc:
                    labels.update(self.label_vectorizer.count(sentence))
                    for k, vectorizer in self.vectorizers.items():
                        vocab_example = vectorizer.count(sentence)
                        vocabs[k].update(vocab_example)

        self.label2index = {k: i for i, k in enumerate(labels.keys())}
        return vocabs

    def read_examples(self, tsfile):
        docs = []
        doc = []
        sentence = []
        with codecs.open(tsfile, encoding='utf-8', mode='r') as f:
            for line in f:
                if line.startswith('#'):
                    if line.startswith('# newdoc'):
                        if doc:
                            docs.append(doc)
                            doc = []
                    continue
                states = re.split("\t", line.rstrip())
                token = {}
                if len(states) > 1:
                    for j in range(len(states)):
                        noff = j - len(states)
                        if noff >= 0:
                            noff = j
                        field_name = self.named_fields.get(str(j), self.named_fields.get(str(noff), str(j)))
                        token[field_name] = states[j]
                    sentence.append(token)
                else:
                    doc.append(sentence)
                    sentence = []
            if sentence:
                doc.append(sentence)
            if doc:
                docs.append(doc)
        return docs


@register_vectorizer(name='features')
class FeatureDictVectorizer(Dict1DVectorizer):
    """Read features from a conll file into a vector."""
    def __init__(self, **kwargs):
        kwargs['fields'] = kwargs.get('fields', ['start_space', 'start_cap', 'all_caps', 'numeric'])
        super(FeatureDictVectorizer, self).__init__(**kwargs)
        self.featsz = len(self.fields)

    def _next_element(self, tokens, _):
        for atom in self.iterable(tokens):
            yield np.array(atom.split(self.delim), dtype=np.float32)

    def count(self, tokens):
        return {}

    def run(self, tokens, _):
        if self.mxlen < 0:
            self.mxlen = self.max_seen

        vec1d = np.zeros((self.mxlen, self.featsz))
        for i, atom in enumerate(self._next_element(tokens, _)):
            if i == self.mxlen:
                i -= 1
                break
            vec1d[i] = atom
        valid_length = i + 1

        if self.time_reverse:
            vec1d = vec1d[::-1]
            valid_length = None

        vec1d = np.stack(vec1d)
        return vec1d, valid_length


NAMED_FIELDS = {
    '0': 'text',
    '1': 'y',
    '2': 'start_space',
    '3': 'start_cap',
    '4': 'all_caps',
    '5': 'numeric',
}


class ResidualParallelConv(nn.Module):
    def __init__(self, insz, outsz, filtsz, activation_type='identity'):
        super(ResidualParallelConv, self).__init__()
        self.outsz = outsz
        self.convs = nn.ModuleList()
        # They only have bias on the first conv layer
        for fsz in filtsz:
            assert fsz % 2 == 1, "Only odd sized filters allowed for {}: got {}".format(self.__class__.__name__, fsz)
            pad = fsz // 2
            self.convs.append(nn.Sequential(
                nn.Conv1d(insz, outsz, fsz, padding=pad),
                pytorch_activation(activation_type)
            ))

    def forward(self, input_bct):
        output = torch.zeros((input_bct.size(0), self.outsz, input_bct.size(2)))
        for c in self.convs:
            output += c(input_bct)
        return output


class HierarchicalParallelConv(ResidualParallelConv):
    def __init__(self, insz, outsz, filtsz, activation_type='identity'):
        super(HierarchicalParallelConv, self).__init__(
            insz, outsz, filtsz, activation_type
        )
        self.conv2 = nn.Conv1d(outsz * len(filtsz), outsz, 1)

    def forward(self, input_bct):
        conv_out = []
        for conv in self.convs:
            conv_out.append(conv(input_bct))
        conv_out = torch.cat(conv_out, dim=1)
        return self.conv2(conv_out)


@register_model(task='tagger', name='sent-seg')
class SentSegTagger(TaggerModelBase):
    def __init__(self):
        super(SentSegTagger, self).__init__()

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        model = cls()
        model.lengths_key = kwargs.get('lengths_key')
        model.labels = labels
        input_sz = model.init_embed(embeddings, **kwargs)
        hsz = model.init_encoder(input_sz, **kwargs)
        print(model)
        return model

    def init_encoder(self, input_sz, **kwargs):
        hsz = kwargs['hsz']
        nlayers = kwargs.get('layers', 1)
        pdrop = kwargs.get('dropout', 0.33)
        self.rnn = LSTMEncoder(
            input_sz, hsz, 'blstm', nlayers, pdrop
        )

        filtsz = kwargs.get('filtsz', [1, 9])
        if kwargs.get('hier_conv', False):
            self.conv = HierarchicalParallelConv(input_sz, hsz, filtsz)
        else:
            self.conv = ResidualParallelConv(input_sz, hsz, filtsz)

        self.tok = nn.Linear(hsz, 1)
        self.sent = nn.Linear(hsz, 1)

        self.rnn2 = LSTMEncoder(hsz, hsz, 'blstm', 1, pdrop)
        self.tok2 = nn.Linear(hsz, 1)
        self.sent2 = nn.Linear(hsz, 1)

        self.gate_noise = nn.Dropout(kwargs.get('gate_noise', 0.02))

        self.gate_inv_temp = kwargs.get('gate_inv_temp', 0.5)

    def gate(self, x, gater):
        gate = torch.sigmoid(-gater * self.gate_inv_temp)
        noisy_gate = self.gate_noise(gate)
        inv_gate = 1 - noisy_gate
        return torch.mul(x, inv_gate)

    def encode(self, tbc, lengths):
        rnn_out = self.rnn(tbc, lengths)  # tbh

        conv_in = tbc.permute(1, 2, 0).contiguous()
        bht = self.conv(conv_in)
        res = rnn_out + bht.permute(2, 0, 1).contiguous()

        res = self.dropout(res)

        tok = self.tok(res)
        sent = self.sent(res)

        gated_res = self.gate(res, tok)
        gated_rnn_out = self.rnn2(gated_res, lengths)

        tok = tok + self.tok2(gated_rnn_out)
        sent = sent + self.sent2(gated_rnn_out)

        non_tok = F.logsigmoid(-tok)
        tok = F.logsigmoid(tok)
        non_sent = F.logsigmoid(-sent)
        sent = F.logsigmoid(sent)

        output = torch.cat([non_tok, tok + non_sent, tok + sent], dim=2)  # tb3
        return output


if __name__ == "__main__":
    import os
    data = os.path.expanduser('~/en_ewt-ud-train.conllu.sent')
    v = {
        'text': Dict1DVectorizer(fields="text"),
        'feat': FeatureDictVectorizer()
    }
    r = DocCONLLReader(v, True, True, named_fields=NAMED_FIELDS)
    vocabs = r.build_vocab([data])

