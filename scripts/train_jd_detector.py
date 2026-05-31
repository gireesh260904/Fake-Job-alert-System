import argparse
import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


DEFAULT_TEXT_COLS = [
    'description', 'job_description', 'text', 'full_text', 'requirements', 'title'
]


NON_JOB_TEXTS = [
    # User-provided examples
    "The lenovo vs dell choice boils down to your needs, and this table makes the laptop brand comparison clear. Both brands are easy to use, with Lenovo's IdeaCentre and ThinkPad offering budget-friendly, portable designs, and Dell's XPS and OptiPlex delivering premium power and customization",
    "VIT-AP University's rankings are improving, with a recent ranking of 748th globally in Electrical and Electronic Engineering by U.S. News & World Report. The university was also recognized as the top emerging private university in India by Outlook India in 2022.",
    # Navigation / UI / portal
    "Sign in to your account browse jobs search companies create profile save jobs get alerts download app",
    "University portal student faculty staff parent alumni access academics research services digital platform institute",
    "Student login employee login parent portal transcript semester registration exam results grade report course enrollment",
    # News / entertainment
    "Breaking news today according to sources the company announced new policy trending now viral",
    "George Clooney established himself as a film star with roles in many movies and received awards.",
    # Product / e-commerce copy
    "Save big on laptops and desktops. Discover premium performance and sleek designs for work and play.",
    # Generic filler
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
]


def clean_text(s: str) -> str:
    s = str(s) if s is not None else ''
    s = s.lower()
    s = re.sub(r'http\S+|www\S+|https\S+', ' ', s)
    s = re.sub(r'\S+@\S+', ' ', s)
    s = re.sub(r'[^a-zA-Z\s.,!?]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def iter_positive_texts(csv_path: str, text_cols, min_len: int = 40, max_count: int = 20000):
    collected = 0
    for chunk in pd.read_csv(csv_path, chunksize=50000, encoding_errors='ignore'):
        col = next((c for c in text_cols if c in chunk.columns), None)
        if not col:
            continue
        for s in chunk[col].astype(str).fillna(''):
            s2 = clean_text(s)
            if len(s2) >= min_len and any(w in s2 for w in ['job', 'hiring', 'responsibilities', 'requirements', 'apply']):
                yield s2
                collected += 1
                if collected >= max_count:
                    return


def build_dataset(csv_path: str, text_cols, min_len=40, max_pos=20000):
    X_pos = list(iter_positive_texts(csv_path, text_cols, min_len=min_len, max_count=max_pos))
    y_pos = [1] * len(X_pos)

    X_neg = [clean_text(s) for s in NON_JOB_TEXTS]
    # Balance roughly via repetition if needed
    if len(X_neg) < len(X_pos):
        times = max(1, len(X_pos) // max(len(X_neg), 1))
        X_neg = (X_neg * times)[:len(X_pos)]
    y_neg = [0] * len(X_neg)

    X = X_pos + X_neg
    y = y_pos + y_neg
    return X, y


def train_and_save(csv_path: str, save_path: str, text_cols, min_len=40, max_pos=20000):
    print(f"Building dataset from: {csv_path}")
    X, y = build_dataset(csv_path, text_cols, min_len=min_len, max_pos=max_pos)
    print(f"Positives: {sum(y)}, Negatives: {len(y)-sum(y)} (Total: {len(y)})")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=50000, min_df=2)
    Xtr = vectorizer.fit_transform(X_train)
    Xte = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=None)
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    print(classification_report(y_test, y_pred, target_names=['non-jd', 'jd']))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({'vectorizer': vectorizer, 'model': clf}, save_path)
    print(f"Saved JD detector to: {save_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv-path', default='merged_job_postings.csv')
    ap.add_argument('--save-path', default=os.path.join('models', 'jd_detector.joblib'))
    ap.add_argument('--min-len', type=int, default=40)
    ap.add_argument('--max-positives', type=int, default=20000)
    ap.add_argument('--text-cols', nargs='*', default=DEFAULT_TEXT_COLS)
    args = ap.parse_args()

    train_and_save(
        csv_path=args.csv_path,
        save_path=args.save_path,
        text_cols=args.text_cols,
        min_len=args.min_len,
        max_pos=args.max_positives
    )


if __name__ == '__main__':
    main()
