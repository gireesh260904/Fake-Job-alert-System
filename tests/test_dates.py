from datetime import datetime
from utils.date_utils import parse_candidate_date


def test_parse_with_year_present():
    ds = "March 12, 2024"
    dt = parse_candidate_date(ds, now=datetime(2025, 1, 1))
    assert dt is not None
    assert dt.year == 2024


def test_parse_without_year_uses_current_year():
    # When no year present, prefer provided now.year
    now = datetime(2025, 6, 1)
    ds = "March 12"
    dt = parse_candidate_date(ds, now=now)
    assert dt is not None
    assert dt.year == 2025


def test_parse_without_year_next_year_fallback():
    # If the current-year parsing fails (e.g., Feb 29 on non-leap year), try next/prev years
    now = datetime(2025, 1, 1)  # 2025 is not a leap year
    ds = "February 29"
    dt = parse_candidate_date(ds, now=now)
    # February 29 should parse to 2024 or 2028 etc.; ensure we get a valid date
    assert dt is not None
    assert dt.month == 2 and dt.day == 29
