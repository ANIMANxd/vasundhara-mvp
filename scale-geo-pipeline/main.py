import os
import sys
import time

from config import *
from pipeline import clean, embed, reduce, llm_cluster, label, geo_infer, export


def run_pipeline():
    pipeline_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    steps = [
        ("1/7  CLEANING",            lambda: clean.run(RAW_DATA_PATH, CLEANED_CSV)),
        ("2/7  EMBEDDING (BGE)",     lambda: embed.run(CLEANED_CSV, EMBEDDED_CSV, EMBEDDINGS_PATH)),
        ("3/7  LLM CLUSTERING",      lambda: llm_cluster.run(EMBEDDED_CSV, EMBEDDINGS_PATH, CLUSTERED_CSV)),
        ("4/7  UMAP REDUCTION",      lambda: reduce.run(CLUSTERED_CSV, EMBEDDINGS_PATH, REDUCED_CSV)),
        ("5/7  c-TF-IDF LABELLING",  lambda: label.run(REDUCED_CSV, LABELLED_CSV)),
        ("6/7  GEO INFERENCE",       lambda: geo_infer.run(LABELLED_CSV, GEO_CSV)),
        ("7/7  GEOJSON EXPORT",      lambda: export.run(GEO_CSV, GEOJSON_OUT, CLUSTERS_JSON_OUT)),
    ]

    for step_name, step_fn in steps:
        print(f"\n{'='*60}")
        print(f"STEP {step_name}")
        print('='*60)
        step_start = time.time()
        step_fn()
        step_elapsed = time.time() - step_start
        print(f"  Step time: {step_elapsed:.1f}s")

    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"  Map data:     {GEOJSON_OUT}")
    print(f"  Cluster data: {CLUSTERS_JSON_OUT}")
    print('='*60)

    total_elapsed = time.time() - pipeline_start
    mins = int(total_elapsed // 60)
    secs = int(total_elapsed % 60)
    print(f"  Total time: {mins}m {secs}s")


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as e:
        print(f"\n[PIPELINE FAILED] {type(e).__name__}: {e}")
        sys.exit(1)
