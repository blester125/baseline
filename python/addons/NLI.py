from six.moves import map, filter

import json
import codecs
import random
from collections import Counter
import mmh3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import baseline
from baseline.model import register_model
from baseline.utils import listify, Offsets
from baseline.reader import register_reader
from baseline.embeddings import register_embeddings
from baseline.vectorizers import GOVectorizer, Token1DVectorizer, register_vectorizer
from baseline.pytorch.torchy import sequence_mask, EmbeddingsContainer
from baseline.pytorch.embeddings import LookupTableEmbeddings
from baseline.pytorch.classify import ClassifierModelBase


@register_vectorizer(name='hash-oov')
class HashOOVVectorizer(Token1DVectorizer):
    def __init__(self, **kwargs):
        self.unk_size = kwargs.get('unk_size', 100)
        self.hash = kwargs.get('hash_fn', mmh3.hash)
        super(HashOOVVectorizer, self).__init__(**kwargs)

    def _next_element(self, tokens, vocab):
        for atom in self.iterable(tokens):
            value = vocab.get(atom)
            if value is None:
                print('Found an OOV')
                value = self.hash(atom) % self.unk_size + len(vocab)
            yield value


def tokens_from_parse(parse):
    return list(map(lambda t: t.rstrip(')'), filter(lambda t: t.endswith(')'), parse.split())))


@register_reader(task='classify', name='nli')
class NLIjsonlReader(object):
    def __init__(self, vectorizers, trim=True, truncate=False, **kwargs):
        super(NLIjsonlReader, self).__init__()
        self.vectorizers = {k: GOVectorizer(vectorizer) for k, vectorizer in vectorizers.items()}
        self.trim = trim
        self.trim = True
        self.truncate = truncate

    def build_vocab(self, files, **kwargs):
        texts, hypothesis, labels = self.load_examples(files)
        vocab = {k: Counter() for k in self.vectorizers.keys()}
        for text, hypo in zip(texts, hypothesis):
            for k, vectorizer in self.vectorizers.items():
                vocab[k].update(vectorizer.count(text))
                vocab[k].update(vectorizer.count(hypo))
        self.label2index = {k: i for i, k in enumerate(set(labels))}
        return vocab, self.get_labels()

    def get_labels(self):
        return {i: k for k, i in self.label2index.items()}

    def load_examples(self, files):
        texts = []
        hypothesis = []
        labels = []
        for file_name in files:
            with codecs.open(file_name, encoding='utf-8', mode='r') as f:
                for line in f:
                    example = json.loads(line)
                    label = example['gold_label']
                    if label == '-':
                        continue
                    texts.append(tokens_from_parse(example['sentence1_parse']))
                    hypothesis.append(tokens_from_parse(example['sentence2_parse']))
                    labels.append(label)
        return texts, hypothesis, labels

    def load(self, file_name, vocabs, batchsz, **kwargs):
        shuffle = kwargs.get('shuffle', False)
        sort_key = kwargs.get('sort_key', None)
        if sort_key is not None and not sort_key.endswith('_lengths'):
            sort_key += '_lengths'

        examples = []
        texts, hypothesis, labels = self.load_examples(listify(file_name))
        for text, hypo, label in zip(texts, hypothesis, labels):
            example_dict = {}
            for k, vectorizer in self.vectorizers.items():
                example_dict['prem_{}'.format(k)], lengths = vectorizer.run(text, vocabs[k])
                if lengths is not None:
                    example_dict['prem_{}_lengths'.format(k)] = lengths
                example_dict['hypo_{}'.format(k)], lengths = vectorizer.run(hypo, vocabs[k])
                if lengths is not None:
                    example_dict['hypo_{}_lengths'.format(k)] = lengths
            example_dict['y'] = self.label2index[label]
            examples.append(example_dict)
        return baseline.data.ExampleDataFeed(
            DualDictExamples(
                examples,
                do_shuffle=shuffle,
                sort_key=sort_key,
            ),
            batchsz=batchsz,
            shuffle=shuffle,
            trim=self.trim,
            truncate=self.truncate
        )


class DualDictExamples(baseline.data.DictExamples):
    """
    We can trim the premises and hypothesis separately.

    Not sure how this will translate to more features.
    """
    def __init__(self, example_list, do_shuffle=True, sort_key=None):
        self.example_list = example_list
        if do_shuffle: random.shuffle(self.example_list)
        if sort_key is not None:
            self.example_list = sorted(
                self.example_list,
                key=lambda x: x['prem_{}'.format(sort_key)]
            )
        self.sort_key = sort_key

    def _trim_batch(self, batch, keys, max_prem, max_hypo):
        if max_prem == 0: return batch
        for k in keys:
            max_ = max_prem if k.startswith('prem') else max_hypo
            if len(batch[k].shape) == 3:
                batch[k] = batch[k][:, :max_, :]
            elif len(batch[k].shape) == 2:
                batch[k] = batch[k][:, :max_]
        return batch

    def batch(self, start, batchsz, trim=False):
        ex = self.example_list[start]
        keys = ex.keys()
        batch = {}
        for k in keys:
            batch[k] = []
        sz = len(self.example_list)
        idx = start * batchsz
        max_prem_len = 0
        max_hypo_len = 0
        for i in range(batchsz):
            if idx >= sz:
                break
            ex = self.example_list[idx]
            for k in keys:
                batch[k].append(ex[k])
            # Trim all batches along the sort_key if it exists
            if trim and self.sort_key is not None:
                max_prem_len = max(max_prem_len, ex['prem_{}'.format(self.sort_key)])
                max_hypo_len = max(max_hypo_len, ex['hypo_{}'.format(self.sort_key)])
            idx += 1
        for k in keys:
            batch[k] = np.stack(batch[k])
        return self._trim_batch(batch, keys, max_prem_len, max_hypo_len) if trim else batch


def build_ffn(input_sz, hsz, nlayers, pdrop):
    proj_layers = []
    proj_layers.append(nn.Linear(input_sz, hsz))
    proj_layers.append(nn.ReLU())
    proj_layers.append(nn.Dropout(pdrop))
    for _ in range(nlayers - 1):
        proj_layers.append(nn.Linear(hsz, hsz))
        proj_layers.append(nn.ReLU())
        proj_layers.append(nn.Dropout(pdrop))
    return nn.Sequential(*proj_layers)


def decomposable_attention(prem, hypo, prem_mask, hypo_mask):
    e = torch.matmul(prem, hypo.transpose(2, 1))
    p2h = F.softmax(e.masked_fill(prem_mask == 0, 1e-4), dim=-1)
    h2p = F.softmax(e.transpose(1, 2).contiguous().masked_fill(hypo_mask == 0, 1e-4), dim=-1)
    attn_hypo = torch.matmul(p2h, hypo)
    attn_prem = torch.matmul(h2p, prem)
    return attn_prem, attn_hypo


def intra_attention(sent, mask, pos_info):
    e = torch.matmul(sent, sent.transpose(2, 1))
    e = e + pos_info
    attn = F.softmax(e.masked_fill(mask == 0, 1e-4), dim=-1)
    return torch.matmul(attn, sent)


@register_model(task='classify', name='nli')
class DecomposableAttentionClassifier(ClassifierModelBase):
    def __init__(self):
        super(DecomposableAttentionClassifier, self).__init__()
        self.embeddings = {'word': None}
        self.lengths_key = 'word_lengths'
        self.gpu = True

    @classmethod
    def create(cls, embeddings, labels, **kwargs):
        model = super(DecomposableAttentionClassifier, cls).create(embeddings, labels, **kwargs)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(kwargs.get('mu', 0), kwargs.get('std', 0.01))
        return model

    def make_input(self, batch_dict):
        example = {}
        for k in self.embeddings.keys():
            key = 'prem_{}'.format(k)
            example[key] = torch.from_numpy(batch_dict[key])
            if self.gpu: example[key] = example[key].cuda()
            key = 'hypo_{}'.format(k)
            example[key] = torch.from_numpy(batch_dict[key])
            if self.gpu: example[key] = example[key].cuda()

        if self.lengths_key is not None:
            prem_lengths = torch.from_numpy(batch_dict['prem_{}'.format(self.lengths_key)])
            if self.gpu: prem_lengths = prem_lengths.cuda()
            hypo_lengths = torch.from_numpy(batch_dict['hypo_{}'.format(self.lengths_key)])
            if self.gpu: hypo_lengths = hypo_lengths.cuda()
            example['lengths'] = (prem_lengths, hypo_lengths)

        y = batch_dict.get('y')
        if y is not None:
            y = torch.from_numpy(y)
            if self.gpu: y = y.cuda()
            example['y'] = y

        return example

    def embed(self, input_):
        prem_embeddings = []
        hypo_embeddings = []
        for k, embedding in self.embeddings.items():
            prem_embeddings.append(embedding.encode(input_['prem_{}'.format(k)]))
            hypo_embeddings.append(embedding.encode(input_['hypo_{}'.format(k)]))
        prem_embeddings = torch.cat(prem_embeddings, dim=2)
        hypo_embeddings = torch.cat(hypo_embeddings, dim=2)
        return prem_embeddings, hypo_embeddings

    def init_pool(self, input_sz, **kwargs):
        hsz = int(kwargs.get('hsz', 200))
        nlayers = int(kwargs.get('layers', 2))
        self.proj = build_ffn(input_sz, hsz, nlayers, self.pdrop)
        self.combine_proj = build_ffn(2 * hsz, hsz, nlayers, self.pdrop)
        return 2 * hsz

    def init_stacked(self, input_sz, **kwargs):
        hsz = int(kwargs.get('hsz', 200))
        nlayers = int(kwargs.get('layers', 2))
        self.stacked_layers = build_ffn(input_sz, hsz, nlayers, self.pdrop)
        return hsz

    def pool(self, embeddings, lengths):
        prem, hypo = embeddings
        prem_lengths, hypo_lengths = lengths
        prem = self.proj(prem)
        hypo = self.proj(hypo)
        prem_mask = sequence_mask(prem_lengths, prem.size(1)).unsqueeze(-1).to(prem.device)
        hypo_mask = sequence_mask(hypo_lengths, hypo.size(1)).unsqueeze(-1).to(hypo.device)
        attn_prem, attn_hypo = decomposable_attention(prem, hypo, prem_mask, hypo_mask)
        prem = self.combine(prem, attn_hypo)
        hypo = self.combine(hypo, attn_prem)
        return torch.cat([prem, hypo], dim=1)

    def combine(self, a, b):
        c = torch.cat([a, b], dim=2)
        c = self.combine_proj(c)
        return torch.sum(c, dim=1)


@register_embeddings(name='intra-attn')
class IntraSentenceAttention(LookupTableEmbeddings):
    def __init__(self, _, **kwargs):
        super(IntraSentenceAttention, self).__init__(_, **kwargs)
        self.hsz = int(kwargs.get('hsz', 200))
        nlayers = int(kwargs.get('layers', 2))
        self.bucket = int(kwargs.get('bucket', 11))
        self.pdrop = float(kwargs.get('pdrop', kwargs.get('dropout', 0.2)))
        self.proj = build_ffn(self.dsz, self.hsz, nlayers, self.pdrop)
        self.dist_emb = nn.Embedding((2 * self.bucket) + 1, 1)

    def get_dsz(self):
        return self.dsz + self.hsz

    def forward(self, x):
        raw = self.embeddings(x)
        seq_len = raw.size(1)
        seq = torch.arange(seq_len)
        pos_info = (seq.view(-1, 1) - seq + self.bucket).clamp(0, 2 * self.bucket).to(x.device)
        pos_embed = self.dist_emb(pos_info).view(-1, seq_len, seq_len)
        mask = (x != Offsets.PAD).unsqueeze(-1).to(x.device)
        attended = intra_attention(self.proj(raw), mask, pos_embed)
        return torch.cat([raw, attended], dim=2)
