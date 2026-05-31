import os
from bs4 import BeautifulSoup
import pytest

from scraper import JobScraper


def load_fixture(name):
    path = os.path.join(os.path.dirname(__file__), 'fixtures', name)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def test_scrape_generic_from_fixture(monkeypatch):
    html = load_fixture('sample_job_generic.html')
    soup = BeautifulSoup(html, 'html.parser')

    scraper = JobScraper()

    # Monkeypatch _fetch_page to return our BeautifulSoup object
    monkeypatch.setattr(scraper, '_fetch_page', lambda url: soup)

    result = scraper._scrape_generic('https://example.com/job')

    assert result['error'] is None
    assert 'Senior Data Scientist' in result['full_text']
    assert 'ExampleCo' in result['full_text']


def test_scrape_structured_jsonld(monkeypatch):
    html = load_fixture('sample_job_structured.html')
    soup = BeautifulSoup(html, 'html.parser')

    scraper = JobScraper()
    monkeypatch.setattr(scraper, '_fetch_page', lambda url: soup)

    structured = scraper._extract_structured_data(soup)
    assert structured is not None
    assert 'Machine Learning Engineer' in structured['full_text']
    assert structured['source'] == 'Structured Data'
