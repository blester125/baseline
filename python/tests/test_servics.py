import random
from copy import deepcopy
from unittest.mock import MagicMock, call, patch
import pytest
from baseline.services import Service, ClassifierService, TaggerService, EncoderDecoderService, LanguageModelService, EmbeddingsService
# import from cwd to get testing utils
from utils import rand_str


def test_sig_name_service():
    with pytest.raises(NotImplementedError):
        _ = Service.signature_name()


def test_sig_name_classifier():
    gold = "predict_text"
    sig = ClassifierService.signature_name()
    assert sig == gold


def test_task_name_classifier():
    gold = "classify"
    sig = ClassifierService.task_name()
    assert sig == gold


def test_sig_name_tagger():
    gold = "tag_text"
    sig = TaggerService.signature_name()
    assert sig == gold


def test_task_name_tagger():
    gold = "tagger"
    sig = TaggerService.task_name()
    assert sig == gold


def test_sig_name_encdec():
    gold = "suggest_text"
    sig = EncoderDecoderService.signature_name()
    assert sig == gold


def test_task_name_encdec():
    gold = "seq2seq"
    sig = EncoderDecoderService.task_name()
    assert sig == gold


def test_sig_name_lm():
    with pytest.raises(NotImplementedError):
        _ = LanguageModelService.signature_name()


def test_task_name_lm():
    gold = "lm"
    sig = LanguageModelService.task_name()
    assert sig == gold


def test_sig_name_embedding():
    gold = "embed_text"
    sig = EmbeddingsService.signature_name()
    assert sig == gold


def test_task_name_embedding():
    gold = "servable_embeddings"
    sig = EmbeddingsService.task_name()
    assert sig == gold


def test_batch_input_base_list():
    tokens = [rand_str() for _ in range(random.randint(1, 10))]
    svc = Service()
    batch = svc.batch_input(tokens)
    assert len(batch) == 1
    assert batch[0] == tokens


def test_batch_input_base_batch():
    gold_batch = [
        [rand_str() for _ in range(random.randint(1, 10))]
        for _ in range(random.randint(1, 5))
    ]
    svc = Service()
    batch = svc.batch_input(gold_batch)
    assert len(batch) == len(gold_batch)
    assert batch == gold_batch

def test_batch_input_base_string():
    tokens = [rand_str() for _ in range(random.randint(1, 10))]
    in_tokens = " ".join(tokens)
    svc = Service()
    batch = svc.batch_input(in_tokens)
    assert len(batch) == 1
    assert batch[0] == tokens


def test_batch_input_tagger_list_of_strings():
    gold = [{"text": rand_str()} for _ in range(random.randint(1, 10))]
    tokens = [t['text'] for t in gold]
    svc = TaggerService(model=MagicMock())
    batch = svc.batch_input(tokens)
    assert len(batch) == 1
    assert batch[0] == gold


def test_batch_input_tagger_batch_of_strings():
    gold = [
        [{"text": rand_str()} for _ in range(random.randint(1, 10))]
        for _ in range(random.randint(2, 5))
    ]
    tokens = [[t['text'] for t in tokens] for tokens in gold]
    svc = TaggerService(model=MagicMock())
    batch = svc.batch_input(tokens)
    assert len(batch) == len(gold)
    assert batch == gold


def test_batch_input_tagger_list_of_dict():
    gold = [{"text": rand_str()} for _ in range(random.randint(1, 10))]
    svc = TaggerService(model=MagicMock())
    batch = svc.batch_input(gold)
    assert len(batch) == 1
    assert batch[0] == gold


def test_batch_input_tagger_list_of_dict():
    gold = [
        [{"text": rand_str()} for _ in range(random.randint(1, 10))]
        for _ in range(random.randint(2, 5))
    ]
    svc = TaggerService(model=MagicMock())
    batch = svc.batch_input(gold)
    assert len(batch) == len(gold)
    assert batch == gold


def test_batch_input_tagger_string():
    gold = [{"text": rand_str()} for _ in range(random.randint(1, 10))]
    tokens = " ".join(t['text'] for t in gold)
    svc = TaggerService(model=MagicMock())
    batch = svc.batch_input(tokens)
    assert len(batch) == 1
    assert batch[0] == gold


def test_batch_input_other():
    with pytest.raises(ValueError):
        svc = TaggerService(model=MagicMock())
        batch = svc.batch_input(MagicMock())


def test_preprare_vectorizers():
    vects = {rand_str(): MagicMock() for _ in range(random.randint(1, 5))}
    svc = Service(vectorizers=vects)
    tokens = [[rand_str() for _ in range(random.randint(1, 5))] for _ in range(random.randint(1, 5))]
    svc.prepare_vectorizers(tokens)
    for vect in vects.values():
        vect.reset.assert_called_once()
        assert vect.count.call_count == len(tokens)
        for t in tokens:
            call_sig = call(t)
            assert call_sig in vect.count.call_args_list


def test_preprare_vectorizers_seq2seq():
    src_vects = {rand_str(): MagicMock() for _ in range(random.randint(1, 5))}
    tgt_vect = MagicMock()
    with patch.object(EncoderDecoderService, 'extract_tgt', return_value=({}, {})):
        svc = EncoderDecoderService()
    svc.src_vectorizers = src_vects
    svc.tgt_vectorizer = tgt_vect
    tokens = [[rand_str() for _ in range(random.randint(1, 5))] for _ in range(random.randint(1, 5))]
    svc.prepare_vectorizers(tokens)
    tgt_vect.reset.assert_not_called()
    tgt_vect.count.assert_not_called()
    for vect in src_vects.values():
        vect.reset.assert_called_once()
        assert vect.count.call_count == len(tokens)
        for t in tokens:
            call_sig = call(t)
            assert call_sig in vect.count.call_args_list


def test_encdec_extract_tgt():
    gold_src = {rand_str(): rand_str() for _ in range(random.randint(1, 10))}
    gold_src.pop('tgt', None)
    gold_tgt = rand_str()
    both = deepcopy({k: v for k, v in gold_src.items()})
    both['tgt'] = gold_tgt
    src, tgt = EncoderDecoderService.extract_tgt(both)
    assert src == gold_src
    assert tgt == gold_tgt
