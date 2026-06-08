import math
import string
import pandas as pd
from openai import OpenAI

import config

TOP_K = 8
LLM_TOP_K = 5

_STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "are", "was", "has", "have",
    "not", "but", "from", "they", "you", "will", "been", "were", "its",
    "more", "also", "into", "can", "out", "just", "about", "like", "when",
    "than", "then", "our", "your", "their",
    "news", "breaking", "india", "now", "big", "new", "one", "two", "say",
    "says", "said", "via", "get", "got", "make", "time", "year", "day",
    "week", "today", "latest", "top", "live", "watch", "read", "update",
    "updates", "report", "reports", "amp", "tweet", "rt",
}


def _tokenize(text: str) -> list[str]:
    tokens = text.lower().split()
    tokens = [t.strip(string.punctuation) for t in tokens]
    return [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]


def run(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    cluster_labels = df["cluster_label"].unique().tolist()
    total_clusters = len(cluster_labels)

    # ---- PART 1: c-TF-IDF ----
    cluster_docs = {}
    cluster_word_counts = {}
    for clabel in cluster_labels:
        text = " ".join(df.loc[df["cluster_label"] == clabel, "cleaned_text"].astype(str))
        tokens = _tokenize(text)
        cluster_docs[clabel] = tokens
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        cluster_word_counts[clabel] = counts

    word_cluster_presence = {}
    for clabel, tokens in cluster_docs.items():
        for word in set(tokens):
            if word not in word_cluster_presence:
                word_cluster_presence[word] = set()
            word_cluster_presence[word].add(clabel)

    cluster_keywords = {}
    cluster_top5 = {}
    for clabel in cluster_labels:
        counts = cluster_word_counts[clabel]
        total_words = len(cluster_docs[clabel])
        if total_words == 0:
            cluster_keywords[clabel] = ""
            cluster_top5[clabel] = ""
            continue
        scores = {}
        for word, count in counts.items():
            tf = count / total_words
            doc_count = len(word_cluster_presence[word])
            idf = math.log(1 + total_clusters / doc_count) + 1
            scores[word] = tf * idf
        sorted_words = sorted(scores, key=scores.get, reverse=True)
        cluster_keywords[clabel] = sorted_words[:TOP_K]
        cluster_top5[clabel] = sorted_words[:LLM_TOP_K]

    # ---- PART 2: LLM Naming ----
    client = OpenAI(base_url=config.LM_STUDIO_BASE_URL, api_key=config.LM_STUDIO_API_KEY)

    cluster_titles = {}
    for clabel in cluster_labels:
        posts = df.loc[df["cluster_label"] == clabel, "cleaned_text"].tolist()
        samples = pd.Series(posts).sample(n=min(3, len(posts)), random_state=42).tolist()
        sample_lines = "\n".join(f"- {s}" for s in samples)
        top5_str = ", ".join(cluster_top5[clabel])

        prompt = (
            "You are labeling a group of social media posts that have been clustered together by an AI.\n\n"
            f"Top keywords extracted from this cluster: {top5_str}\n\n"
            f"Sample posts from this cluster:\n{sample_lines}\n\n"
            "Give this cluster a short, descriptive name (3-6 words max) that captures what these posts are about.\n"
            "Reply with ONLY the cluster name. No explanation. No punctuation at the end."
        )

        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.LLM_TEMPERATURE,
            max_tokens=32,
        )
        title = response.choices[0].message.content
        title = (title or clabel).strip().strip('"').strip("'").strip()
        cluster_titles[clabel] = title

    # ---- Merge back ----
    df["ctfidf_keywords"] = df["cluster_label"].map(
        lambda c: " | ".join(cluster_keywords.get(c, []))
    )
    df["cluster_title"] = df["cluster_label"].map(cluster_titles)

    print("Label done:")
    for clabel in cluster_labels:
        count = (df["cluster_label"] == clabel).sum()
        kw = df.loc[df["cluster_label"] == clabel, "ctfidf_keywords"].iloc[0]
        title = cluster_titles[clabel]
        print(f"  [{clabel}] -> Refined Title: {title} | keywords: {kw} | {count} posts")

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    from config import CLUSTERED_CSV, LABELLED_CSV
    run(CLUSTERED_CSV, LABELLED_CSV)
