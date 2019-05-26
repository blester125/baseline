from itertools import chain
from collections import deque
import numpy as np
import tensorflow as tf
from baseline.utils import Offsets, listify
from baseline.tf.tfy import get_shape_as_list
from baseline.embeddings import register_embeddings
from baseline.vectorizers import Token1DVectorizer, register_vectorizer, _token_iterator
from addons.vec_text import Text1DVectorizer
from addons.embed_elmo import ELMoHubEmbeddings


@register_vectorizer(name='gram')
class NGramVectorizer(Token1DVectorizer):
    def __init__(self, n=3, **kwargs):
        assert n % 2 != 0, "n must be odd."
        self.n = n
        self.m = n // 2
        self.buf = deque(maxlen=n)
        super(NGramVectorizer, self).__init__(**kwargs)

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen

        vec2d = np.zeros((self.mxlen, self.n), dtype=int)
        i = 0
        for i, atom in enumerate(chain(self._next_element(tokens, vocab), [Offsets.PAD] * (self.m))):
            if i - self.m == self.mxlen:
                i -= 1
                break
            self.buf.append(atom)
            if i >= self.m:
                vec2d[i - self.m, :len(self.buf)] = self.buf
        valid_length = i - self.m + 1

        if self.time_reverse:
            vec2d = vec2d[::-1]
            return vec2d, None
        return vec2d, valid_length


@register_vectorizer(name='dict-gram')
class DictNGramVectorizer(NGramVectorizer):
    def __init__(self, **kwargs):
        super(DictNGramVectorizer, self).__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '@@')

    def iterable(self, tokens):
        return _token_iterator(self, tokens)


@register_vectorizer(name='text-gram')
class TextNGramVectorizer(Text1DVectorizer):
    def __init__(self, n=3, **kwargs):
        assert n % 2 != 0, "n must be odd."
        self.n = n
        self.m = n // 2
        self.buf = deque(maxlen=n)
        super(TextNGramVectorizer, self).__init__(**kwargs)

    def run(self, tokens, vocab):
        if self.mxlen < 0:
            self.mxlen = self.max_seen

        vec2d = np.full((self.mxlen, self.n), '', dtype=np.object)
        i = 0
        for i, atom in enumerate(chain(self._next_element(tokens, vocab), [''] * (self.m))):
            if i - self.m == self.mxlen:
                i -= 1
                break
            self.buf.append(atom)
            if i >= self.m:
                vec2d[i - self.m, :len(self.buf)] = self.buf
        valid_length = i - self.m + 1

        if self.time_reverse:
            vec2d = vec2d[::-1]
            return vec2d, None
        return vec2d, valid_length


@register_vectorizer(name='dict-text-gram')
class DictTextNGramVectorizer(TextNGramVectorizer):
    def __init__(self, **kwargs):
        super(DictTextNGramVectorizer, self).__init__(**kwargs)
        self.fields = listify(kwargs.get('fields', 'text'))
        self.delim = kwargs.get('token_delim', '@@')

    def iterable(self, tokens):
        return _token_iterator(self, tokens)

import logging
LOGGER = logging.getLogger('baseline')

@register_embeddings(name='elmo-gram')
class ELMoHubGramEmbeddings(ELMoHubEmbeddings):

    @classmethod
    def create_placeholder(cls, name):
        return tf.placeholder(tf.string, [None, None, None], name=name)

    def encode(self, x=None):
        x = x if x is not None else ELMoHubGramEmbeddings.create_placeholder(self.name)
        LOGGER.warning(x.get_shape())
        B, T, N = get_shape_as_list(x)
        x = tf.reshape(x, [B * T, N])
        LOGGER.warning(x.get_shape())
        elmoed = super(ELMoHubGramEmbeddings, self).encode(x)
        LOGGER.warning(elmoed.get_shape())
        elmoed = tf.reduce_mean(elmoed, axis=1)
        LOGGER.warning(elmoed.get_shape())
        x = tf.reshape(elmoed, [B, T, self.dsz])
        LOGGER.warning(x.get_shape())
        return x



if __name__ == "__main__":
    vocab = {
        '<UNK>': 1,
        "The": 2,
        "Brown": 3,
        "Dog": 4,
        "ran": 5,
        "away": 6
    }

    tokens = ["The", "Brown", "Dog", "ran", "away"]
    n = 5
    l = 4

    v = NGramVectorizer(n=n)
    v.mxlen = l
    print(v.run(tokens, vocab))

    v = TextNGramVectorizer(n=n)
    v.mxlen = l
    print(v.run(tokens, vocab))

    tokens = [{"text": t} for t in tokens]

    v = DictNGramVectorizer(n=n)
    v.mxlen = l
    print(v.run(tokens, vocab))

    v = DictTextNGramVectorizer(n=n)
    v.mxlen = l
    print(v.run(tokens, vocab))
