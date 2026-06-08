import json
import pandas as pd


def run(input_path: str, geojson_out: str, clusters_json_out: str) -> dict:
    df = pd.read_csv(input_path)

    geocoded = df[df["lat"].notna() & df["lon"].notna()]

    features = []
    for _, row in geocoded.iterrows():
        title = row.get("cluster_title")
        if title is None or not pd.notna(title):
            title = row["cluster_label"]
        props = {
            "id": str(row["id"]),
            "text": str(row["text"]),
            "created_at": str(row.get("created_at", "")),
            "cluster_label": str(row["cluster_label"]),
            "cluster_title": str(title),
            "ctfidf_keywords": str(row["ctfidf_keywords"]),
            "location_name": str(row["location_name"]),
        }
        umap_x = row.get("umap_x")
        umap_y = row.get("umap_y")
        props["umap_x"] = float(umap_x) if pd.notna(umap_x) else None
        props["umap_y"] = float(umap_y) if pd.notna(umap_y) else None

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(row["lon"]), float(row["lat"])]},
            "properties": props,
        })

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(geojson_out, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2, ensure_ascii=False)

    geocoded_ids = set(geocoded.index)
    cluster_summary = []
    for clabel, group in df.groupby("cluster_label"):
        total = len(group)
        geo_count = group.index.isin(geocoded_ids).sum()
        keywords = group["ctfidf_keywords"].iloc[0]
        title = group["cluster_title"].iloc[0] if "cluster_title" in group.columns else None
        if title is None or not pd.notna(title):
            title = clabel
        cluster_summary.append({
            "cluster_label": str(clabel),
            "cluster_title": str(title),
            "ctfidf_keywords": str(keywords),
            "total_posts": int(total),
            "geocoded_posts": int(geo_count),
        })

    cluster_summary.sort(key=lambda c: c["total_posts"], reverse=True)

    with open(clusters_json_out, "w", encoding="utf-8") as f:
        json.dump(cluster_summary, f, indent=2, ensure_ascii=False)

    n_features = len(features)
    n_clusters = len(cluster_summary)
    n_no_geo = len(df) - n_features

    print("Export done:")
    print(f"  GeoJSON features: {n_features} -> {geojson_out}")
    print(f"  Clusters:         {n_clusters} -> {clusters_json_out}")
    print(f"  Posts without geo (excluded from map): {n_no_geo}")

    return {"geojson_features": n_features, "clusters": n_clusters}


if __name__ == "__main__":
    from config import GEO_CSV, GEOJSON_OUT, CLUSTERS_JSON_OUT
    run(GEO_CSV, GEOJSON_OUT, CLUSTERS_JSON_OUT)
