import numpy as np
import pandas as pd
import umap

import config


def run(input_csv_path: str, input_npy_path: str, output_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv_path)
    embeddings = np.load(input_npy_path)
    n = len(df)
    orig_dim = embeddings.shape[1]

    print("Running UMAP...")
    reducer = umap.UMAP(
        n_components=config.UMAP_N_COMPONENTS,
        n_neighbors=config.UMAP_N_NEIGHBORS,
        min_dist=config.UMAP_MIN_DIST,
        metric=config.UMAP_METRIC,
        random_state=config.UMAP_RANDOM_STATE,
    )
    coords = reducer.fit_transform(embeddings)

    df["umap_x"] = coords[:, 0]
    df["umap_y"] = coords[:, 1]

    df.to_csv(output_csv_path, index=False)

    print(f"UMAP done: {n} posts reduced from {orig_dim}D -> 2D")

    return df


if __name__ == "__main__":
    from config import EMBEDDED_CSV, EMBEDDINGS_PATH, REDUCED_CSV
    run(EMBEDDED_CSV, EMBEDDINGS_PATH, REDUCED_CSV)
