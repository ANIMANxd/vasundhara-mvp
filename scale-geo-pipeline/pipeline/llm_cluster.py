import json
import re
import string
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

import config

BATCH_SIZE = 15
FALLBACK_CLUSTER = "Miscellaneous"


def _normalize_name(name: str) -> str:
    name = name.strip()
    name = name.translate(str.maketrans("", "", string.punctuation))
    name = re.sub(r"\s+", " ", name)
    return name.strip().title()


def _semantic_sort(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.where(norms == 0, 1, norms)
    sim = normed @ normed.T
    n = len(embeddings)
    visited = np.zeros(n, dtype=bool)
    order = [0]
    visited[0] = True
    current = 0
    for _ in range(1, n):
        row = sim[current].copy()
        row[visited] = -2.0
        nxt = int(np.argmax(row))
        order.append(nxt)
        visited[nxt] = True
        current = nxt
    return np.array(order)


def _parse_clusters(text: str, n_posts: int) -> dict[int, str]:
    def _safe_extract(data) -> dict[int, str]:
        result = {}
        for item in data:
            if isinstance(item, dict) and "id" in item and "cluster" in item:
                try:
                    result[int(item["id"])] = str(item["cluster"]).strip()
                except (ValueError, TypeError):
                    continue
        return result

    text = (text or "").strip()

    try:
        data = json.loads(text)
        if isinstance(data, list):
            parsed = _safe_extract(data)
            if parsed:
                return parsed
    except json.JSONDecodeError:
        pass

    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                parsed = _safe_extract(data)
                if parsed:
                    return parsed
        except json.JSONDecodeError:
            pass

    result = {}
    for m in re.finditer(r'"id"\s*:\s*(\d+).*?"cluster"\s*:\s*"([^"]+)"', text):
        try:
            result[int(m.group(1))] = m.group(2).strip()
        except (ValueError, IndexError):
            continue
    if result:
        return result

    return {}


def run(input_csv_path: str, input_npy_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)
    embeddings = np.load(input_npy_path)
    n = len(df)

    order = _semantic_sort(embeddings)
    df = df.iloc[order].reset_index(drop=True)

    client = OpenAI(base_url=config.LM_STUDIO_BASE_URL, api_key=config.LM_STUDIO_API_KEY)

    def call_llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        return response.choices[0].message.content

    global_clusters = []
    cluster_map = {}

    num_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_start in tqdm(range(0, n, BATCH_SIZE), total=num_batches, desc="Clustering"):
        batch_end = min(batch_start + BATCH_SIZE, n)
        batch_texts = df["cleaned_text"].iloc[batch_start:batch_end].tolist()
        n_batch = len(batch_texts)

        post_lines = "\n".join(f"{i}: {text}" for i, text in enumerate(batch_texts))
        existing = ("\n".join(f"  - {c}" for c in global_clusters)
                    if global_clusters else "(none yet — invent new ones)")

        prompt = (
            f"You are clustering a batch of {n_batch} social media posts.\n\n"
            "Existing clusters:\n"
            f"{existing}\n\n"
            "You are an aggressive consolidation engine. Your primary goal is to minimize the "
            "total number of clusters. Before creating a new cluster name, you MUST exhaustively "
            "check the 'Existing Clusters' list above. If an incoming post can fit into an "
            "existing broader category (e.g., merging 'Worst Roads in Bengaluru' or "
            "'Bengaluru Traffic Update' into an existing 'Bengaluru Traffic Chaos' cluster), "
            "you MUST assign it to that existing cluster. Only invent a new cluster name if "
            "the topic is completely alien to everything on the existing list.\n\n"
            "Rules:\n"
            "- Cluster names must be 2-5 words, concise, and descriptive.\n"
            "- Every post MUST receive a cluster name. No post goes unassigned.\n\n"
            "Reply with ONLY a valid JSON array — no other text:\n"
            '[{"id": 0, "cluster": "Cluster Name"}, {"id": 1, "cluster": "Another Cluster"}]\n\n'
            "Posts:\n"
            f"{post_lines}"
        )

        result = _parse_clusters(call_llm(prompt), n_batch)

        for i in range(n_batch):
            cname = result.get(i)
            if not cname:
                cname = FALLBACK_CLUSTER
            cname = _normalize_name(cname)
            cname = cname[:80]

            incoming_words = {w for w in cname.lower().split() if len(w) >= 3}

            matched = None
            for existing in global_clusters:
                existing_words = {w for w in existing.lower().split() if len(w) >= 3}
                if len(incoming_words & existing_words) >= 2:
                    matched = existing
                    break

            if matched:
                cname = matched
            else:
                global_clusters.append(cname)

            df_idx = batch_start + i
            cluster_map[df_idx] = cname

    # ---- Post-processing: LLM-driven label consolidation ----
    raw_labels = list(dict.fromkeys(cluster_map.values()))

    def _consolidate_clusters(labels: list[str]) -> dict[str, str]:
        label_list = "\n".join(f"{i+1}. {lbl}" for i, lbl in enumerate(labels))
        prompt = (
            f"You are consolidating a taxonomy of {len(labels)} cluster names for social media posts.\n"
            "Merge these into 6 to 8 broad, meaningful themes. "
            "Be aggressive — single-post clusters, niche topics, and anything with fewer than "
            "3 posts should be merged into the nearest broader theme. "
            "Prefer fewer, stronger clusters over many specific ones.\n"
            "Keep cluster names concise (2-5 words).\n\n"
            "Cluster names:\n"
            f"{label_list}\n\n"
            "Return ONLY a JSON object mapping each original name to its consolidated name:\n"
            '{"Original Name 1": "Broader Category", "Original Name 2": "Broader Category"}'
        )
        resp = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096,
        )
        content = resp.choices[0].message.content
        try:
            mapping = json.loads(content)
            return {k: v for k, v in mapping.items() if isinstance(k, str) and isinstance(v, str)}
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", content, re.DOTALL)
            if m:
                try:
                    mapping = json.loads(m.group(0))
                    return {k: v for k, v in mapping.items() if isinstance(k, str) and isinstance(v, str)}
                except json.JSONDecodeError:
                    pass
        return {}

    df["cluster_label"] = df.index.map(cluster_map)

    print(f"  Raw clusters before consolidation: {len(raw_labels)}")
    print("  Running consolidation pass...")

    if len(raw_labels) > 150:
        chunk_size = 100
        intermediate = []
        for i in range(0, len(raw_labels), chunk_size):
            chunk = raw_labels[i:i + chunk_size]
            chunk_map = _consolidate_clusters(chunk)
            intermediate.extend(set(chunk_map.values()))
        intermediate = list(set(intermediate))
        final_map = _consolidate_clusters(intermediate)
        full_map = {}
        for label in raw_labels:
            im = chunk_map.get(label, label)
            full_map[label] = final_map.get(im, im)
        consolidation_map = full_map
    else:
        consolidation_map = _consolidate_clusters(raw_labels)

    df["cluster_label"] = df["cluster_label"].map(
        lambda x: consolidation_map.get(x, x)
    )
    final_count = df["cluster_label"].nunique()
    print(f"  Final clusters after consolidation: {final_count}")

    unique_labels = list(dict.fromkeys(df["cluster_label"]))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    df["cluster_id"] = df["cluster_label"].map(label_to_id)

    global_clusters = list(unique_labels)
    print(f"LLM Clustering done: {len(global_clusters)} clusters found")
    for cname in global_clusters:
        count = (df["cluster_label"] == cname).sum()
        print(f"  [{cname}]: {count} posts")

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    from config import EMBEDDED_CSV, EMBEDDINGS_PATH, CLUSTERED_CSV
    run(EMBEDDED_CSV, EMBEDDINGS_PATH, CLUSTERED_CSV)
