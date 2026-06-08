import re
import pandas as pd
from langdetect import detect, LangDetectException

import config


_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002600-\U000026FF"
    "\U00002700-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)

_URL_PATTERN = re.compile(
    r"https?://\S+|ftp://\S+|www\.\S+|\S+\.(com|org|net|gov|edu|io|co|in|uk|us|de|fr|it|es|nl|ru|jp|cn|br|au|ca)\b",
    flags=re.IGNORECASE,
)

_MENTION_PATTERN = re.compile(r"@\w+")

_NON_ALPHA_PATTERN = re.compile(r"[^a-zA-Z0-9\s]")

_SPACE_PATTERN = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    text = _URL_PATTERN.sub("", text)
    text = _MENTION_PATTERN.sub("", text)
    text = text.replace("#", "")
    text = _EMOJI_PATTERN.sub("", text)
    text = _NON_ALPHA_PATTERN.sub("", text)
    text = _SPACE_PATTERN.sub(" ", text)
    return text.lower().strip()


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run(input_path: str, output_path: str) -> pd.DataFrame:
    input_count = None

    # Step 1: Load CSV
    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError("Input CSV must have a 'text' column")
    if "id" not in df.columns:
        df["id"] = list(range(len(df)))
    if "created_at" not in df.columns:
        df["created_at"] = ""
    input_count = len(df)
    print(f"  After load: {len(df)} posts")

    # Step 2: Drop NaN/empty text
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]
    print(f"  After drop empty text: {len(df)} posts")

    # Step 3: Remove exact duplicates (keep first)
    df = df.drop_duplicates(subset=["text"], keep="first")
    print(f"  After exact dedup: {len(df)} posts")

    # Step 4: Remove retweets
    df = df[~df["text"].astype(str).str.strip().str.startswith("RT @")]
    print(f"  After remove retweets: {len(df)} posts")

    # Step 5: Remove replies
    df = df[~df["text"].astype(str).str.strip().str.match(r"^@\w+")]
    print(f"  After remove replies: {len(df)} posts")

    # Step 6: Hashtag spam
    hashtag_counts = df["text"].astype(str).str.findall(r"#\w+").str.len()
    df = df[hashtag_counts <= config.MAX_HASHTAG_COUNT]
    print(f"  After hashtag filter: {len(df)} posts")

    # Step 7: Length filter
    df = df[df["text"].astype(str).str.strip().str.len() >= config.MIN_TEXT_LENGTH]
    print(f"  After length filter: {len(df)} posts")

    # Step 8: Language detection
    def _detect_lang(text: str):
        try:
            return detect(text)
        except LangDetectException:
            return None

    langs = df["text"].astype(str).apply(_detect_lang)
    df = df[langs.isna() | langs.isin(config.KEEP_LANGUAGES)]
    print(f"  After language filter: {len(df)} posts")

    # Step 9: Build cleaned_text
    df["cleaned_text"] = df["text"].astype(str).apply(_clean_text)

    # Step 10: Drop short cleaned_text
    df = df[df["cleaned_text"].str.len() >= config.MIN_TEXT_LENGTH]
    print(f"  After cleaned length filter: {len(df)} posts")

    # Step 11: Near-duplicate removal (Jaccard on word sets)
    # SCALABILITY NOTE: replace with MinHash/LSH at production scale (N > 10000)
    word_sets = df["cleaned_text"].apply(lambda x: set(x.split())).tolist()
    n = len(df)
    to_drop = set()
    for i in range(n):
        if i in to_drop:
            continue
        for j in range(i + 1, n):
            if j in to_drop:
                continue
            if _jaccard(word_sets[i], word_sets[j]) >= config.NEAR_DEDUP_THRESHOLD:
                to_drop.add(j)
    df = df.drop(df.index[list(to_drop)])
    df = df.reset_index(drop=True)
    print(f"  After near dedup: {len(df)} posts")

    # Step 12: Save
    df.to_csv(output_path, index=False)

    # Step 13: Summary
    output_count = len(df)
    pct = (output_count / input_count * 100) if input_count else 0.0
    print(f"Cleaning done: {input_count} -> {output_count} posts ({pct:.1f}% retained)")

    return df


if __name__ == "__main__":
    from config import RAW_DATA_PATH, CLEANED_CSV
    run(RAW_DATA_PATH, CLEANED_CSV)
