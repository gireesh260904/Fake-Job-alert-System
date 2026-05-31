import numpy as np
import pytest

import predictor as pred_module


def make_predictor(monkeypatch):
    """Create a FakeJobPredictor instance without running heavy __init__.
    We monkeypatch __init__ to be a no-op and then set minimal attributes needed
    for the tests."""
    monkeypatch.setattr(pred_module.FakeJobPredictor, "__init__", lambda self, model_dir='models': None)
    p = pred_module.FakeJobPredictor()

    # Minimal methods/attributes used by is_job_description/is_job_expired
    p.clean_text = lambda x: x.lower() if x else ''
    p.get_bert_embedding = lambda text, max_length=512: np.zeros(768)

    # Minimal reference texts and embeddings (zeros are fine for tests that exercise heuristics)
    p.job_reference_texts = ["we are hiring", "responsibilities include"]
    p.non_job_texts = ["sign in to your account", "breaking news today"]
    p.non_job_content_types = ["ui", "news"]
    p.job_ref_embeddings = [np.zeros(768) for _ in p.job_reference_texts]
    p.non_job_ref_embeddings = [np.zeros(768) for _ in p.non_job_texts]

    return p


def test_meta_instruction_rejected(monkeypatch):
    p = make_predictor(monkeypatch)
    text = "81% confidence means our AI is reasonably certain about the classification even with missing information; for comparison the previous model scored lower."
    is_valid, meta = p.is_job_description(text)
    assert not is_valid
    assert isinstance(meta, dict)
    assert meta.get('reason') == 'meta_instruction' or meta.get('reason') == 'missing_keywords' or 'meta' in meta.get('content_type', '')


def test_education_info_rejected(monkeypatch):
    p = make_predictor(monkeypatch)
    text = "College admission open. The fee structure includes tuition fees, hostel fees and semester registration."
    is_valid, meta = p.is_job_description(text)
    assert not is_valid
    assert meta.get('reason') in ('education_info', 'educational_portal', 'semantic_non_job')


def test_entertainment_rejected(monkeypatch):
    p = make_predictor(monkeypatch)
    text = "George Clooney established himself as a film star with roles in many movies and received awards. He starred in several dramas and festivals." 
    is_valid, meta = p.is_job_description(text)
    assert not is_valid
    assert meta.get('reason') in ('entertainment_news', 'semantic_non_job')


def test_deadline_parsing_future_and_past(monkeypatch):
    p = make_predictor(monkeypatch)
    future = "Apply by November 30, 2099. We are hiring a developer."
    past = "Last date to apply was March 12, 2024. Join our team."

    is_expired_f, reason_f, dl_f = p.is_job_expired(future)
    assert is_expired_f is False
    assert dl_f is not None

    is_expired_p, reason_p, dl_p = p.is_job_expired(past)
    # past date should be detected as expired
    assert is_expired_p is True
    assert dl_p is not None


def test_valid_job_passes_basic_checks(monkeypatch):
    p = make_predictor(monkeypatch)
    job = "We are hiring a Data Analyst. Responsibilities include data cleaning and reporting. Apply now with resume."
    is_valid, meta = p.is_job_description(job)
    assert is_valid
    assert meta.get('content_type') == 'job'
