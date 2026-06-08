# SCALE-Geo Pipeline

A social media intelligence pipeline that processes raw posts through cleaning, embedding, LLM-based clustering, and geolocation inference to generate interactive visualizations.

## Overview

The pipeline transforms raw social media data into actionable geographic clusters with semantic labels:
- **Raw posts** → **Clean** → **Embed** (BGE via LM Studio) → **LLM Clustering** (Llama via LM Studio) → **UMAP Visualization** → **c-TF-IDF Labels** → **Geo-Inference** → **FastAPI** → **Leaflet.js Map**

## Architecture

- **`config.py`** — Centralized configuration: all paths, thresholds, model names
- **`pipeline/`** — Modular stages, each with a `run()` function:
  - `clean.py` — Data cleaning and preprocessing
  - `embed.py` — Generate embeddings using BGE
  - `llm_cluster.py` — Cluster embeddings using Llama
  - `reduce.py` — Dimensionality reduction with UMAP
  - `label.py` — Generate cluster labels with c-TF-IDF
  - `geo_infer.py` — Infer geographic locations
  - `export.py` — Export to GeoJSON and visualization formats
- **`main.py`** — Orchestrator that runs the full pipeline sequentially
- **`output/`** — Generated artifacts (CSVs, embeddings, GeoJSON)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model (required for NLP)
python -m spacy download en_core_web_sm
```

## Requirements

- **LM Studio** running locally at `http://localhost:1234` with:
  - BGE embedding model loaded
  - Llama LLM model loaded

Start LM Studio before running pipeline stages that require inference.

## Usage

Run the complete pipeline:

```bash
cd scale-geo-pipeline
python main.py
```

Run individual pipeline stages:

```bash
python -m pipeline.clean
python -m pipeline.embed
python -m pipeline.llm_cluster
python -m pipeline.reduce
python -m pipeline.label
python -m pipeline.geo_infer
python -m pipeline.export
```

## Key Design Principles

- **Single source of config** — No hardcoded values anywhere. All configuration in `config.py`
- **Embeddings as separate files** — Large embeddings stored in `embeddings.npy`, synced by row index with CSVs
- **Fail loudly** — Raises exceptions on errors, never silently fails
- **Scalable patterns** — All operations are O(N) or O(N log N), no O(N²) approaches
- **Comprehensive logging** — Every script logs input count, output count, and changes made

## Output Files

- **`cleaned.csv`** — Preprocessed posts
- **`embedded.csv`** — Posts with metadata; embeddings in `embeddings.npy`
- **`clustered.csv`** — Cluster assignments
- **`reduced.csv`** — UMAP reduced coordinates
- **`labelled.csv`** — Cluster labels
- **`geolocated.csv`** — Geographic inference results
- **`output.geojson`** — Final geographic features for mapping
- **`clusters.json`** — Cluster metadata and statistics
