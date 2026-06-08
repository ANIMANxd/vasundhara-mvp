import time
import pandas as pd
import spacy
from spacy.cli import download as spacy_download
from geopy.geocoders import Nominatim

import config


def run(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    n = len(df)

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("  Downloading en_core_web_sm (first time only)...")
        spacy_download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")

    geolocator = Nominatim(user_agent=config.GEOCODER_USER_AGENT)

    location_names = []
    lats = []
    lons = []

    geocode_cache = {}

    for _, row in df.iterrows():
        text = row["cleaned_text"]
        doc = nlp(text)
        loc = None
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                loc = ent.text
                break
        location_names.append(loc)

        if loc is not None:
            if loc in geocode_cache:
                lat, lon = geocode_cache[loc]
            else:
                try:
                    result = geolocator.geocode(loc)
                    if result:
                        lat, lon = result.latitude, result.longitude
                    else:
                        lat, lon = None, None
                except Exception:
                    lat, lon = None, None
                geocode_cache[loc] = (lat, lon)
                time.sleep(config.GEOCODER_SLEEP)
            lats.append(lat)
            lons.append(lon)
        else:
            lats.append(None)
            lons.append(None)

    df["location_name"] = location_names
    df["lat"] = lats
    df["lon"] = lons

    with_loc = df["location_name"].notna().sum()
    with_geo = df["lat"].notna().sum()

    print("Geo inference done:")
    print(f"  Total posts:            {n}")
    print(f"  Posts with location:    {with_loc}")
    print(f"  Posts geocoded (lat/lon): {with_geo}")
    unique_locations = len(geocode_cache)
    print(f"  Unique locations geocoded: {unique_locations} (cache saved {with_loc - unique_locations} API calls)")

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    from config import LABELLED_CSV, GEO_CSV
    run(LABELLED_CSV, GEO_CSV)
