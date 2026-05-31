import json
import numpy as np

import app as app_module
from app import app


class DummyPredictor:
    def is_job_description(self, text):
        return True, {'content_type': 'job'}

    def predict(self, text, source=None):
        return {
            'prediction': 'Legit Job',
            'confidence': 0.91,
            'probability_legit': 0.91,
            'probability_fake': 0.09
        }

    def clean_text(self, text):
        return (text or '').lower()

    def get_bert_embedding(self, text):
        # small deterministic embedding
        return np.array([1.0, 0.5, 0.2])

    def cosine_sim(self, a, b, eps=1e-8):
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < eps or nb < eps:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    job_ref_embeddings = [np.array([1.0, 0.5, 0.2])]
    non_job_ref_embeddings = [np.array([0.0, 0.0, 0.0])]
    job_reference_texts = ['we are hiring']
    non_job_texts = ['login here']


class DummyScraper:
    def scrape_url(self, url):
        return {
            'full_text': 'We are hiring a software engineer. Apply now.',
            'source': 'example.com',
            'title': 'Software Engineer Role',
            'company': 'ExampleCo',
            'location': 'Remote'
        }


def test_predict_endpoint_with_text(monkeypatch):
    # Inject dummy predictor and scraper to avoid heavy external dependencies
    monkeypatch.setattr(app_module, 'predictor', DummyPredictor())
    monkeypatch.setattr(app_module, 'scraper', DummyScraper())

    client = app_module.app.test_client()
    resp = client.post('/predict', json={'input': 'We are hiring a backend developer.'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['prediction'] == 'Legit Job'
    assert 'confidence' in data


def test_explain_endpoint_returns_explanation_and_classification(monkeypatch):
    monkeypatch.setattr(app_module, 'predictor', DummyPredictor())
    monkeypatch.setattr(app_module, 'scraper', DummyScraper())

    client = app_module.app.test_client()
    resp = client.post('/explain', json={'input': 'We are hiring a backend developer.'})
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'explanation' in data
    # When valid, explain returns classification as well
    assert 'classification' in data
    assert data['classification']['prediction'] == 'Legit Job'
