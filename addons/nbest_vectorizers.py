#!/usr/bin/env python3

from collections import Counter
import numpy as np
from baseline.utils import listify
from baseline.vectorizers import (
    create_vectorizer,
    register_vectorizer,
    AbstractVectorizer,
    Token1DVectorizer,
    WordpieceVectorizer1D as Wordpiece1DVectorizer,
    BPEVectorizer1D as BPE1DVectorizer,
)


class JSONVectorizerMixin(AbstractVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get("field", "text")

    def iterable(self, example):
        return super().iterable(example[self.field])


class JSONLabelVectorizerMixin(AbstractVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get("field", "text")

    def iterable(self, example):
        return super().iterable(listify(example[self.field]))


@register_vectorizer("json-token1d")
class JSONToken1DVectorizer(JSONVectorizerMixin, Token1DVectorizer):
    pass


@register_vectorizer("json-label-token1d")
class JSONLabelToken1DVectorizer(JSONLabelVectorizerMixin, Token1DVectorizer):
    pass


class NBestJSONVectorizerMixin(AbstractVectorizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.field = kwargs.get("field", "text")
        self.mx_nbest = int(kwargs.get("mx_nbest", -1))
        self.max_n_best_seen = -1

    def reset(self):
        super().reset()
        self.max_n_best_seen = -1

    def count(self, example):
        n_best_count = Counter()
        n_best_seen = 0
        for n_best in example[self.field]:
            count = super().count(n_best)
            n_best_count.update(count)
            n_best_seen += 1
        self.max_n_best_seen = max(self.max_n_best_seen, n_best_seen)
        return n_best_count

    def run(self, example, vocab):
        if self.mx_nbest < 0:
            self.mx_nbest = self.max_n_best_seen
        vec2d = []
        lengths = []
        for i, n_best in enumerate(example[self.field]):
            if i >= self.mx_nbest:
                i -= 1
                break
            vec1d, length = super().run(n_best, vocab)
            vec2d.append(vec1d)
            lengths.append(length)
        while len(vec2d) < self.mx_nbest:
            vec2d.append(np.zeros_like(vec2d[0]))
            # Set the length to 1 to avoid problems with LSTM that try to
            # process the paddings elements of length zero. We will always
            # have the nbest mask to mask out these elements before we aggregate
            # them so we can just use a 1 without caring.
            lengths.append(1)
        n_best_length = i + 1
        vec2d = np.stack(vec2d)
        lengths = np.stack(lengths)
        return vec2d, lengths, n_best_length


@register_vectorizer(name="nbest-json-token1d")
class NBestJSONToken1DVectorizer(NBestJSONVectorizerMixin, Token1DVectorizer):
    pass


class NBestJSONVectorizerWithCLSMixin(NBestJSONVectorizerMixin):
    def run(self, example, vocab):
        if self.mx_nbest < 0:
            self.mx_nbest = self.max_n_best_seen
        vec2d = []
        lengths = []
        for i, n_best in enumerate(example[self.field]):
            if i >= self.mx_nbest:
                i -= 1
                break
            # Jump over the normal NBestJSONVectorizerMixin run method, otherwise we would
            # try to extract the field again
            vec1d, length = super(NBestJSONVectorizerMixin, self).run(n_best, vocab)
            vec2d.append(vec1d)
            lengths.append(length)
        while len(vec2d) < self.mx_nbest:
            vec1d = np.zeros_like(vec2d[0])
            vec1d[0] = vocab.get('[CLS]', 0)
            # We need to have the '[CLS]' symbol otherwise the pooling in the downstream transformers fail
            vec2d.append(vec1d)
            # Set the length to 1 to avoid problems with LSTM that try to
            # process the paddings elements of length zero. We will always
            # have the nbest mask to mask out these elements before we aggregate
            # them so we can just use a 1 without caring.
            lengths.append(1)
        n_best_length = i + 1
        vec2d = np.stack(vec2d)
        lengths = np.stack(lengths)
        return vec2d, lengths, n_best_length


@register_vectorizer(name="nbest-json-wordpiece1d")
class NBestJSONWordpiece1DVectorizer(NBestJSONVectorizerWithCLSMixin, Wordpiece1DVectorizer):
    pass


@register_vectorizer(name="nbest-json-bpe1d")
class NBestJSONBPE1DVectorizer(NBestJSONVectorizerWithCLSMixin, BPE1DVectorizer):
    pass
