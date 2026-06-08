import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

import config


def run(input_path: str, output_csv_path: str, output_npy_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(input_path)
    texts = df["cleaned_text"].astype(str).tolist()
    n = len(texts)

    client = OpenAI(base_url=config.LM_STUDIO_BASE_URL, api_key=config.LM_STUDIO_API_KEY)
    all_embeddings = []

    batch_size = config.EMBEDDING_BATCH_SIZE
    num_batches = (n + batch_size - 1) // batch_size

    try:
        for i in tqdm(range(0, n, batch_size), total=num_batches, desc="Embedding"):
            batch_texts = texts[i : i + batch_size]
            response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch_texts)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)
    except Exception as e:
        print(f"ERROR: Cannot connect to LM Studio at {config.LM_STUDIO_BASE_URL}. "
              "Make sure LM Studio is running and the BGE model is loaded.")
        raise

    embeddings = np.array(all_embeddings, dtype=np.float32)
    np.save(output_npy_path, embeddings)

    output_df = df[["id", "text", "cleaned_text", "created_at"]].copy()
    output_df.to_csv(output_csv_path, index=False)

    print(f"Embedding done: {n} posts, shape {embeddings.shape}")

    return output_df, embeddings


if __name__ == "__main__":
    from config import CLEANED_CSV, EMBEDDED_CSV, EMBEDDINGS_PATH
    run(CLEANED_CSV, EMBEDDED_CSV, EMBEDDINGS_PATH)
