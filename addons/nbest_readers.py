#!/usr/bin/env python3

import json
from itertools import chain
from collections import Counter
from operator import itemgetter
from typing import Dict, Optional, List, Tuple, Counter as CounterType
import numpy as np
from baseline.utils import read_json, listify
from baseline.reader import register_reader, _get_dir, ParallelCorpusReader, _build_vocab_for_col, _vocab_allowed, _filter_vocab
from baseline.data import ExampleDataFeed, DictExamples, Seq2SeqExamples
from baseline.vectorizers import create_vectorizer, Vectorizer
from nbest_vectorizers import JSONLabelToken1DVectorizer


@register_reader(task="classify", name="json")
class JSONReader:
    def __init__(self, vectorizers: Dict[str, Vectorizer], trim: bool = True, truncate: bool = False, mxlen: int = -1, **kwargs):
        super().__init__()
        self.vectorizers = vectorizers
        self.trim = trim
        self.truncate = truncate
        label_vectorizer_spec = kwargs.get('label_vectorizer', None)
        if label_vectorizer_spec is not None:
            self.label_vectorizer = create_vectorizer(**label_vectorizer_spec)
        else:
            self.label_vectorizer = JSONLabelToken1DVectorizer(field="y", mxlen=mxlen)
        self.labels = Counter()

    def load_examples(self, files: List[str]) -> List[Dict]:
        examples = []
        for file_name in chain(*(_get_dir(f) for f in files)):
            examples.extend(read_json(file_name))
        return examples

    def build_vocab(self, files: List[str], **kwargs) -> Tuple[Dict[str, CounterType[str]], List[str]]:
        examples = self.load_examples(files)
        vocab = {k: Counter() for k in self.vectorizers.keys()}
        for example in examples:
            for k, vectorizer in self.vectorizers.items():
                vocab[k].update(vectorizer.count(example))
            self.labels.update(self.label_vectorizer.count(example))
        self.labels = {k: i for i, k in enumerate(self.labels.keys())}
        return vocab, [k for k, _ in sorted(self.labels.items(), key=itemgetter(1))]

    def load(
        self,
        file_name: str,
        vocabs: Dict[str, Dict[str, int]],
        batchsz: int,
        shuffle: bool = False,
        sort_key: Optional[str] = None,
        suffix_length: str = "_lengths"
    ) -> ExampleDataFeed:
        if sort_key is not None and not sort_key.endswith(suffix_length):
            sort_key = f"{sort_key}{suffix_length}"

        loaded = []
        examples = self.load_examples(listify(file_name))

        for example in examples:
            example_dict = {}
            for k, vectorizer in self.vectorizers.items():
                example_dict[k], lengths = vectorizer.run(example, vocabs[k])
                if lengths is not None:
                    example_dict[f"{k}{suffix_length}"] = lengths
                y, _ = self.label_vectorizer.run(example, self.labels)
                example_dict['y'] = y.item()
                loaded.append(example_dict)
        return ExampleDataFeed(
            DictExamples(loaded, do_shuffle=shuffle, sort_key=sort_key),
            batchsz=batchsz,
            shuffle=shuffle,
            trim=self.trim,
            truncate=self.truncate,
        )


@register_reader(task="classifiy", name="jsonl")
class JSONlReader(JSONReader):
    def load_examples(self, files):
        for file_name in chain(*(_get_dir(f) for f in files)):
            with open(file_name) as f:
                for line in f:
                    line = line.rstrip("\n")
                    yield json.loads(line)


@register_reader(task="classify", name="nbest-json")
class NBestJSONReader(JSONReader):
    def load(
        self,
        file_name: str,
        vocabs: Dict[str, Dict[str, int]],
        batchsz: int,
        shuffle: bool = False,
        sort_key: Optional[str] = None,
        length_suffix: str = "_lengths"
    ) -> ExampleDataFeed:
        if sort_key is not None and not sort_key.endswith(length_suffix):
            sort_key = f"{sort_key}{length_suffix}"

        loaded = []
        examples = self.load_examples(listify(file_name))

        for example in examples:
            example_dict = {}
            for k, vectorizer in self.vectorizers.items():
                example_dict[k], lengths, n_best = vectorizer.run(example, vocabs[k])
                if lengths is not None:
                    example_dict[f"{k}{length_suffix}"] = lengths
                if n_best is not None:
                    example_dict[f"{k}{length_suffix}_nbest"] = n_best
                y, _ = self.label_vectorizer.run(example, self.labels)
                example_dict['y'] = y.item()
                loaded.append(example_dict)
        return ExampleDataFeed(
            DictExamples(loaded, do_shuffle=shuffle, sort_key=sort_key),
            batchsz=batchsz,
            shuffle=shuffle,
            trim=self.trim,
            truncate=self.truncate,
        )


@register_reader(task="seq2seq", name="json")
class JSONParallelCorpusReader(ParallelCorpusReader):
    def load_data(self, files: List[str]) -> List[Dict]:
        examples = []
        for file_name in chain(*(_get_dir(f) for f in files)):
            examples.extend(read_json(file_name))
        return examples

    def build_vocabs(self, files, **kwargs):
        vocab_file = kwargs.get("vocab_file")
        if vocab_file is not None:
            all_vects = self.src_vectorizers.copy()
            all_vects['tgt'] = self.tgt_vectorizer
            _vocab_allowed(all_vects)
            # Only read the file once
            text = _read_from_col(0, listify(vocab_file))
            src_vocab = _build_vocab_for_col(None, None, self.src_vectorizers, text=text)
            tgt_vocab = _build_vocab_for_col(None, None, {'tgt': self.tgt_vectorizer}, text=text)
            return src_vocab, tgt_vocab['tgt']

        examples = self.load_data(files)
        src_vocab = {k: Counter() for k in self.src_vectorizers.keys()}
        tgt_vocab = Counter()
        for example in examples:
            for k, vectorizer in self.src_vectorizers.items():
                src_vocab[k].update(vectorizer.count(example))
            tgt_vocab.update(self.tgt_vectorizer.count(example))

        min_f = kwargs.get("min_f", {})
        tgt_min_f = {"tgt": min_f.pop("tgt", -1)}
        src_vocab = _filter_vocab(src_vocab, min_f)
        tgt_vocab = _filter_vocab({"tgt": tgt_vocab}, min_f)["tgt"]
        return src_vocab, tgt_vocab

    def load_examples(self, tsfile, vocab1, vocab2, shuffle, sort_key):
        data = self.load_data(listify(tsfile))
        examples = []
        for d in data:
            example = {}
            for k, vectorizer in self.src_vectorizers.items():
                example[k], length = vectorizer.run(d, vocab1[k])
                if length is not None:
                    example[f"{k}_lengths"] = length
            example['tgt'], example[f'tgt_lengths'] = self.tgt_vectorizer.run(d, vocab2)
            examples.append(example)
        return Seq2SeqExamples(examples, do_shuffle=shuffle, src_sort_key=sort_key)


@register_reader(task="seq2seq", name="nbest-json")
class NBestJSONParallelCorpusReader(JSONParallelCorpusReader):
    def load_examples(self, tsfile, vocab1, vocab2, shuffle, sort_key, length_suffix="_lengths"):
        data = self.load_data(listify(tsfile))
        examples = []
        for d in data:
            example = {}
            for k, vectorizer in self.src_vectorizers.items():
                example[k], lengths, n_best = vectorizer.run(d, vocab1[k])
                if lengths is not None:
                    example[f"{k}{length_suffix}"] = lengths
                if n_best is not None:
                    example[f"{k}{length_suffix}_nbest"] = n_best
            example['tgt'], example[f'tgt_lengths'] = self.tgt_vectorizer.run(d, vocab2)
            examples.append(example)
        return NBestSeq2SeqExamples(examples, do_shuffle=shuffle, src_sort_key=sort_key)


@register_reader(task="seq2seq", name="jsonl")
class JSONlParllelCorpusReader(JSONParallelCorpusReader):
    def load_data(self, files):
        for file_name in chain(*(_get_dir(f) for f in files)):
            with open(file_name) as f:
                for line in f:
                    line = line.rstrip("\n")
                    yield json.loads(line)


@register_reader(task="seq2seq", name="nbest-jsonl")
class NBestJSONlParallelCorpusReader(JSONlParllelCorpusReader, NBestJSONParallelCorpusReader):
    pass


class NBestSeq2SeqExamples(Seq2SeqExamples):
    def __init__(self, example_list, do_shuffle=True, src_sort_key=None):
        super().__init__(example_list, do_shuffle, None)
        if src_sort_key is not None:
            self.example_list = sorted(self.example_list, key=lambda x: np.max(x[src_sort_key]))
        self.src_sort_key = src_sort_key

    def _trim_batch(self, batch, max_src_len, max_tgt_len):
        for k in batch.keys():
            max_len = max_src_len
            if k == 'tgt':
                max_len = max_tgt_len

            if max_len == 0:
                continue
            if len(batch[k].shape) == 3:
                batch[k] = batch[k][:, :, 0:max_len]
            elif len(batch[k].shape) == 4:
                batch[k] = batch[k][:, :, 0:max_src_len, :]
        return batch

    def batch(self, start, batchsz, trim=False):

        """Get a batch of data

        :param start: (``int``) The step index
        :param batchsz: (``int``) The batch size
        :param trim: (``bool``) Trim to maximum length in a batch
        :param vec_alloc: A vector allocator
        :param vec_shape: A vector shape function
        :return: batched `x` word vector, `x` character vector, batched `y` vector, `length` vector, `ids`
        """
        ex = self.example_list[start]
        keys = ex.keys()
        batch = {}

        for k in keys:
            batch[k] = []
        sz = len(self.example_list)
        idx = start * batchsz
        max_src_len = 0
        max_tgt_len = 0
        for i in range(batchsz):
            if idx >= sz:
                break

            ex = self.example_list[idx]
            for k in keys:
                batch[k].append(ex[k])

            # Trim all batches along the sort_key if it exists
            if trim and self.src_sort_key is not None:
                max_src_len = max(max_src_len, np.max(ex[self.src_sort_key]))

            if trim:
                max_tgt_len = max(max_tgt_len, np.max(ex['tgt_lengths']))

            idx += 1

        for k in keys:
            batch[k] = np.stack(batch[k])
        return self._trim_batch(batch, max_src_len, max_tgt_len) if trim else batch
