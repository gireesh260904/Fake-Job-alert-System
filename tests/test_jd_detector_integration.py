import os
import pytest

from predictor import FakeJobPredictor


@pytest.mark.skipif(not os.path.exists('models/jd_detector.joblib'), reason='JD detector model not trained/available')
def test_predictor_uses_jd_detector_for_non_job():
    p = FakeJobPredictor()
    text = "The lenovo vs dell choice boils down to your needs; this laptop brand comparison shows design and premium power."
    is_valid, meta = p.is_job_description(text)
    assert not is_valid
    assert meta.get('reason') in ('jd_model_negative', 'education_info', 'news_article', 'entertainment_news', 'semantic_non_job', 'website_navigation')
