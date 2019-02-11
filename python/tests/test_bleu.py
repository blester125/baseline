import six
import string
import random
from collections import Counter
import pytest
from mock import patch, call
from hypothesis import given
import hypothesis.strategies as st
import numpy as np
from baseline.bleu import (
    n_grams,
    count_n_grams,
    find_closest,
    corpora_lengths,
    max_gold_n_gram_counts,
    count_matches,
    count_possible,
    geometric_mean,
    brevity_penalty,
    _read_references,
    _read_lines,
)


words = st.text(st.sampled_from(string.ascii_letters + string.digits + string.punctuation), min_size=1)
reference = st.lists(st.lists(words))


@st.composite
def aligned_input(draw, input_, min_value=1, max_value=51):
    """Create two lists full of `input_` that are the same length."""
    n = draw(st.integers(min_value=min_value, max_value=max_value))
    fixed_length_list = st.lists(input_, min_size=n, max_size=n)
    return draw(fixed_length_list), draw(fixed_length_list)


# Move logic of inputs into a composite function?
@given(pred_len=st.integers(10, 20), offsets=st.lists(st.integers(5, 10), min_size=2, max_size=4))
def test_find_closest_above(pred_len, offsets):
    pred_len = np.array(pred_len)
    offsets = np.array(offsets)
    input_lens = pred_len + offsets
    gold = np.min(input_lens)
    input_ = [[''] * input_len for input_len in input_lens]
    res = find_closest(pred_len, input_)
    assert res == gold


@given(st.integers(10, 20), st.lists(st.integers(5, 10), min_size=2, max_size=4))
def test_find_closest_below(p, o):
    pred_len = np.array(p)
    offsets = np.array(o)
    input_lens = pred_len - offsets
    gold = np.max(input_lens)
    input_ = [[''] * input_len for input_len in input_lens]
    res = find_closest(pred_len, input_)
    assert res == gold


@given(st.integers(10, 20), st.integers(5, 10))
def test_find_closest_tie(pred_len, offset):
    above = pred_len + offset
    below = pred_len - offset
    input_ = [[''] * input_len for input_len in (above, below)]
    gold = below
    res = find_closest(pred_len, input_)
    assert res == gold


@given(aligned_input(st.integers(2, 20)))
def test_corpora_lengths(data):
    pred_lens, gold_lens = data
    gold_pred = np.sum(pred_lens)
    gold_gold = np.sum(gold_lens)
    preds = [[''] * p for p in pred_lens]
    golds = [[[''] * g] for g in gold_lens]
    with patch('baseline.bleu.find_closest') as find_patch:
        find_patch.side_effect = gold_lens
        pred_guess, gold_guess = corpora_lengths(preds, golds)
    assert pred_guess == gold_pred
    assert gold_guess == gold_gold


@st.composite
def generate_counters(draw, prob=0.5):
    w = draw(st.lists(st.tuples(words, st.integers(min_value=2, max_value=six.MAXSIZE)), min_size=1))
    # Create a set of strings that have some max value.
    gold = Counter()
    for word, count in w:
        gold[word] = count
    counters = []
    # For each word create a distinct counter that will hold it's max value
    # With some probability add other words from the gold counter that has
    # some value that is <= the gold counts.
    for word, count in gold.items():
        counter = Counter()
        counter[word] = count
        for word2, count2 in gold.items():
            if word == word2: continue
            if np.random.rand() > prob:
                counter[word2] = count2 - np.random.randint(0, count2)
        counters.append(counter)
    random.shuffle(counters)
    return gold, counters


@given(generate_counters())
def test_max_gold_counts(data):
    gold, counters = data
    # Have the reduce work on all of these counters.
    with patch('baseline.bleu.count_n_grams') as count_mock:
        count_mock.side_effect = counters
        fake_input = [''] * len(gold)
        res = max_gold_n_gram_counts(fake_input, None)
    assert res == gold


@given(st.lists(words, min_size=1, max_size=10), st.integers(min_value=2, max_value=6))
def test_max_gold_counts_calls(input_, count):
    golds = [call(i, count) for i in input_]
    with patch('baseline.bleu.count_n_grams') as count_mock:
        _ = max_gold_n_gram_counts(input_, count)
    assert count_mock.call_args_list == golds


def test_n_grams_int():
    input_ = ['a', 'b', 'c']
    n = 3
    gold = [('a',), ('b',), ('c',), ('a', 'b'), ('b', 'c'), ('a', 'b', 'c')]
    res = n_grams(input_, n)
    assert list(res) == gold


def test_n_grams_tuple():
    input_ = ['a', 'b', 'c']
    n = (1, 3)
    gold = [('a',), ('b',), ('c',), ('a', 'b', 'c')]
    res = n_grams(input_, n)
    assert list(res) == gold


def test_geometric_mean():
    input_ = [0.82, .061, 0.22]
    gold = 0.22242765817194177
    res = geometric_mean(input_)
    np.testing.assert_allclose(res, gold)


@given(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=4))
def test_geometric_mean_whole(p):
    gold = np.power(np.prod(p), 1.0 / len(p))
    res = geometric_mean(p)
    np.testing.assert_allclose(res, gold)


@given(st.lists(st.integers(min_value=0, max_value=10)).filter(lambda x: any(y == 0 for y in x)))
def test_geometric_mean_zero(p):
    res = geometric_mean(p)
    assert res == 0.0


@given(st.lists(st.integers()).filter(lambda x: any(y < 0 for y in x)))
def test_geometric_mean_neg(p):
    res = geometric_mean(p)
    assert res == 0.0


@st.composite
def gold_less_than_preds(draw):
    p = draw(st.integers(min_value=2))
    delta = draw(st.integers(min_value=1, max_value=p - 1))
    return p, p - delta


@given(gold_less_than_preds())
def test_brevity_penalty_not_applied(data):
    p, g = data
    res, _ = brevity_penalty(p, g)
    assert res == 1.0


# This is an example of a different way to do the test above where the sample
# logic is split across the generation and the test itself. This is a quicker
# way to write the test but maybe bad style?
# @given(st.integers(min_value=2, max_value=six.MAXSIZE))
# def test_brevity_penalty_not_applied(p):
#     g = p - np.random.randint(1, p)
#     res, _ = brevity_penalty(p, g)
#     assert res == 1.0


@st.composite
def gold_more_than_preds(draw):
    p = draw(st.integers(min_value=1))
    delta = draw(st.integers(min_value=1))
    return p, p + delta


@given(gold_more_than_preds())
def test_brevity_penalty_value(data):
    p, g = data
    gold = np.exp(1 - (g / p))
    res, _ = brevity_penalty(p, g)
    np.testing.assert_allclose(res, gold)


@given(st.integers(min_value=1), st.integers(min_value=1))
def test_brevity_penalty_ratio(p, g):
    gold = p / float(g)
    _, res = brevity_penalty(p, g)
    assert res == gold


@st.composite
def generate_counters_with_known_matches(draw, prob=0.5):
    a = Counter(); b = Counter()
    n = draw(st.integers(3, 6))
    gold_counts = np.zeros(n)
    for i in range(draw(st.integers(10, 20))):
        # Generate a key in the form (i, i, i, ...).
        # This is unique because i is increasing.
        # The size is a random n_gram
        key = tuple([i] * draw(st.integers(1, n)))
        # Put the key into both counters
        if np.random.rand() > prob:
            a_v = draw(st.integers(5, 10))
            b_v = draw(st.integers(5, 10))
            a[key] = a_v
            b[key] = b_v
            gold_counts[len(key) - 1] += min(a_v, b_v)
        else:
            # Put the key randomly into one of the counters
            c = a if np.random.rand() > prob else b
            c[key] = draw(st.integers(5, 10))
    return a, b, n, gold_counts


@given(generate_counters_with_known_matches())
def test_count_matches(data):
    a, b, n, gold = data
    res = count_matches(a, b, np.zeros(n))
    np.testing.assert_equal(res, gold)


def test_count_possible():
    input_ = ['a', 'b', 'c']
    n = 3
    gold = [3, 2, 1]
    res = count_possible(input_, np.zeros(n))
    np.testing.assert_equal(res, np.array(gold))


@given(reference)
def test_read_lines(gold):
    input_ = [' '.join(g) for g in gold]
    res = _read_lines(input_, False)
    assert res == gold


@given(reference)
def test_read_lines_lowercase(gold):
    input_ = [' '.join(g) for g in gold]
    gold = [[x.lower() for x in g] for g in gold]
    res = _read_lines(input_, True)
    assert res == gold


@given(aligned_input(st.lists(words)))
def test_read_references_group_references(data):
    input1, input2 = data
    gold = [(input1[i], input2[i]) for i in range(len(input1))]
    with patch('baseline.bleu._read_lines') as read_patch:
        with patch('baseline.bleu.open'):
            read_patch.side_effect = (input1, input2)
            res = _read_references(['', ''], False)
            assert res == gold
