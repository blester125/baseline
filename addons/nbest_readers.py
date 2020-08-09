#!/usr/bin/env python3

import json
from itertools import chain
from operator import itemgetter
from typing import Dict, Optional, List, Tuple, Counter as CounterType
from collections import Counter
from baseline.utils import read_json, listify
from baseline.reader import register_reader, _get_dir
from baseline.data import ExampleDataFeed, DictExamples
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


@register_reader(task="classify", name="nbest-jsonl")
class NBestJSONlReader(JSONlReader, NBestJSONReader):
    pass
