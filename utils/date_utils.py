"""Small date utilities for parsing ambiguous date strings reliably.

Functions here try to parse date fragments that may lack a year by
attempting the current year, next year, and previous year. This reduces
ambiguous parsing behavior and avoids deprecation warnings from
dateparser when a year is not provided.
"""
import re
from datetime import datetime
import dateparser


def parse_candidate_date(ds: str, now: datetime | None = None):
    """Parse a date-like string `ds` into a datetime or return None.

    If `ds` already contains a 4-digit year, parse it directly. If not,
    try appending the current year, then next year, then previous year
    (this covers most job-deadline phrases like "March 12" -> "March 12 2025").

    Returns a datetime instance or None if parsing failed.
    """
    if now is None:
        now = datetime.now()

    if not ds or not isinstance(ds, str):
        return None

    s = ds.strip()
    # If there's a 4-digit year already, try parsing directly
    if re.search(r"\b\d{4}\b", s):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                return dateparser.parse(s, settings={'DATE_ORDER': 'DMY', 'PREFER_DAY_OF_MONTH': 'first'})
        except Exception:
            return None

    # Try current year, then next year, then previous year
    for year in (now.year, now.year + 1, now.year - 1):
        candidate = f"{s} {year}"
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                dt = dateparser.parse(candidate, settings={'DATE_ORDER': 'DMY', 'PREFER_DAY_OF_MONTH': 'first'})
            if dt:
                return dt
        except Exception:
            continue

    return None
