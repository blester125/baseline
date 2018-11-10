import math
from collections import defaultdict
import pytest
import numpy as np
from mock import MagicMock
from baseline.utils import windowed_sort


def test_sort_lengths_match():
    in_ = np.random.randint(0, 100, size=np.random.randint(30, 100))
    gold = len(in_)
    res = windowed_sort(in_, window=np.random.randint(1, 5))
    assert len(res) == gold


def test_all_elements_in_output():
    in_ = range(np.random.randint(10, 30))
    res = windowed_sort(in_)
    for i in in_:
        assert i in res


def test_all_elements_in_input():
    in_ = range(np.random.randint(10, 30))
    res = windowed_sort(in_)
    for r in res:
        assert r in in_


def test_same_sorts_from_same_input():
    in_ = np.random.randint(0, 100, size=np.random.randint(30, 100))
    in_2 = in_.copy()
    w = np.random.randint(1, 5)
    r1 = windowed_sort(in_, w)
    r2 = windowed_sort(in_2, w)
    assert r1 == r2


def test_unchanging():
    in_ = np.random.randint(0, 100, size=np.random.randint(30, 100))
    w = np.random.randint(1, 5)
    r = windowed_sort(in_, w)
    first = r.copy()
    second = windowed_sort(r, w)
    assert first == second


def test_monotonic_buckets():
    in_ = np.random.normal(50, 10, size=100)
    in_ = list(map(int, in_))
    window = 5
    out = windowed_sort(in_, window)
    buckets = defaultdict(list)
    for x in out:
        buckets[math.floor(x / window)].append(x)
    last = []
    for i in range(len(in_)):
        if i in buckets:
            b = buckets[i]
            for x in b:
                for l in last:
                    assert x > l
            last = b


def test_buckets_in_order():
    in_ = np.random.normal(50, 10, size=100)
    in_ = list(map(int, in_))
    window = 5
    out = windowed_sort(in_, window)
    out_dict = {x: i for i, x in enumerate(out)}
    buckets = defaultdict(list)
    for x in out:
        buckets[math.floor(x / window)].append(x)
    last = []
    for i in range(len(in_)):
        if i in buckets:
            b = buckets[i]
            for x in b:
                for l in last:
                    assert out_dict[x] > out_dict[l]
            last = b


def get_dict(sort_key):
    return {sort_key: np.random.randint(1, 100)}


def test_use_key():
    sort_key = 'a'
    in_ = [get_dict(sort_key) for _ in range(np.random.randint(10, 30))]
    res = windowed_sort(in_, window=1, key=lambda x: x[sort_key])
    last = -1
    for i in res:
        assert i[sort_key] >= last
        last = i[sort_key]
