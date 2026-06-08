# LM Studio
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"
EMBEDDING_MODEL = "text-embedding-bge-large-en-v1.5"
LLM_MODEL = "meta-llama-3.1-8b-instruct"      # change this to match your exact model name in LM Studio
EMBEDDING_DIM = 1024                  # BGE large outputs 1024 dimensions

# Paths
RAW_DATA_PATH = "../raw_tweets.csv"
OUTPUT_DIR = "output"
EMBEDDINGS_PATH = "output/embeddings.npy"
CLEANED_CSV = "output/cleaned.csv"
EMBEDDED_CSV = "output/embedded.csv"
REDUCED_CSV = "output/reduced.csv"
CLUSTERED_CSV = "output/clustered.csv"
LABELLED_CSV = "output/labelled.csv"
GEO_CSV = "output/geolocated.csv"
GEOJSON_OUT = "output/output.geojson"
CLUSTERS_JSON_OUT = "output/clusters.json"

# Cleaning
MIN_TEXT_LENGTH = 20
KEEP_LANGUAGES = ["en", "hi"]
MAX_HASHTAG_COUNT = 8
NEAR_DEDUP_THRESHOLD = 0.90

# Embedding (batch size for LM Studio API calls)
EMBEDDING_BATCH_SIZE = 32

# UMAP (2D only — for visualization)
UMAP_N_COMPONENTS = 2
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

# LLM Clustering
LLM_SAMPLE_SIZE = 50           # posts sampled for Stage 1 theme discovery
LLM_MIN_THEMES = 4
LLM_MAX_THEMES = 7
LLM_BATCH_SIZE = 20            # posts per assignment call in Stage 3
LLM_TEMPERATURE = 0.2          # low temp for deterministic output
LLM_MAX_TOKENS = 1024

# c-TF-IDF
CTFIDF_TOP_N = 5               # keywords per cluster label

# Geo
GEOCODER_USER_AGENT = "scale-geo-pipeline"
GEOCODER_SLEEP = 1.2           # Nominatim rate limit: 1 req/sec
