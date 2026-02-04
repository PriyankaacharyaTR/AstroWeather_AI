# from astroquery.jplhorizons import Horizons
# import pandas as pd

# # EXACT order from your training dataset
# FEATURE_ORDER = [
#     "moon_X", "moon_Y", "moon_Z", "moon_VX", "moon_VY", "moon_VZ",
#     "sun_X", "sun_Y", "sun_Z", "sun_VX", "sun_VY", "sun_VZ",
#     "jupiter_X", "jupiter_Y", "jupiter_Z", "jupiter_VX", "jupiter_VY", "jupiter_VZ",
#     "venus_X", "venus_Y", "venus_Z", "venus_VX", "venus_VY", "venus_VZ"
# ]

# # Conversion factor: 1 AU to KM
# AU_TO_KM = 149597870.7

# def get_planet_features(date_str):
#     # JPL Horizons time format
#     start_time = f"{date_str} 00:00"
#     stop_time = f"{date_str} 00:01"

#     bodies = {
#         "sun": "10",
#         "moon": "301",
#         "jupiter": "5", 
#         "venus": "299"
#     }

#     data = {}

#     for name, code in bodies.items():
#         # location="500@399" = Earth Center (Geocentric)
#         # This matches your inspection report showing Sun ~63M+ KM away
#         obj = Horizons(
#             id=code,
#             location="500@399",
#             epochs={"start": start_time, "stop": stop_time, "step": "1d"}
#         ).vectors()

#         row = obj.to_pandas().iloc[0]

#         # Convert AU to KM and AU/Day to KM/Day
#         data[f"{name}_X"]  = float(row["x"]) * AU_TO_KM
#         data[f"{name}_Y"]  = float(row["y"]) * AU_TO_KM
#         data[f"{name}_Z"]  = float(row["z"]) * AU_TO_KM
#         data[f"{name}_VX"] = float(row["vx"]) * AU_TO_KM
#         data[f"{name}_VY"] = float(row["vy"]) * AU_TO_KM
#         data[f"{name}_VZ"] = float(row["vz"]) * AU_TO_KM

#     # Return DataFrame with the correct order
#     df = pd.DataFrame([data])[FEATURE_ORDER]
#     return df



import requests
import pandas as pd
import io
import json
import os

# Load feature columns from JSON
FEATURE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'feature_columns.json')
with open(FEATURE_CONFIG_PATH, 'r') as f:
    FEATURE_CONFIG = json.load(f)

FEATURE_ORDER = FEATURE_CONFIG['feature_columns']
PLANETARY_BODIES = FEATURE_CONFIG['planetary_bodies']
TARGET_COLUMNS = FEATURE_CONFIG['target_columns']

def fetch_nasa_vector(obj_id, date_str):
    """Calls JPL Horizons API for a single body."""
    BASE_URL = "https://ssd.jpl.nasa.gov/api/horizons.api"
    
    params = {
        "format": "json",
        "COMMAND": f"'{obj_id}'",
        "OBJ_DATA": "NO",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "VECTORS",
        "CENTER": "'500@399'",        # Earth Geocenter (matches your training)
        "START_TIME": f"'{date_str}'",
        "STOP_TIME": f"'{date_str} 00:01'",
        "STEP_SIZE": "'1 d'",
        "OUT_UNITS": "'KM-D'",        # Crucial: Returns KM and KM/Day
        "CSV_FORMAT": "YES"
    }
    
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    # Extract the CSV portion from the JSON response
    # The actual vector data is between $$SOE and $$EOE markers
    result_str = data.get("result", "")
    csv_part = result_str.split("$$SOE")[1].split("$$EOE")[0].strip()
    
    # Column names for Horizons vector output:
    # JDTDB, Calendar Date, X, Y, Z, VX, VY, VZ
    df = pd.read_csv(io.StringIO(csv_part), header=None)
    return {
        "X": float(df.iloc[0, 2]),
        "Y": float(df.iloc[0, 3]),
        "Z": float(df.iloc[0, 4]),
        "VX": float(df.iloc[0, 5]),
        "VY": float(df.iloc[0, 6]),
        "VZ": float(df.iloc[0, 7]),
    }

def get_planet_features(date_str):
    """
    Fetch planetary vectors from NASA JPL Horizons and build feature row
    matching the training dataset structure.
    """
    all_data = {}
    planetary_vectors = {}
    
    # Fetch planetary vectors for each body
    for name, code in PLANETARY_BODIES.items():
        vec = fetch_nasa_vector(code, date_str)
        all_data[f"{name}_X"] = vec["X"]
        all_data[f"{name}_Y"] = vec["Y"]
        all_data[f"{name}_Z"] = vec["Z"]
        all_data[f"{name}_VX"] = vec["VX"]
        all_data[f"{name}_VY"] = vec["VY"]
        all_data[f"{name}_VZ"] = vec["VZ"]
        planetary_vectors[name] = vec
    
    # Add date-derived features
    date = pd.to_datetime(date_str)
    all_data["Year"] = date.year
    all_data["Month"] = date.month
    all_data["Day"] = date.day
    all_data["DayOfYear"] = date.dayofyear
    all_data["WeekOfYear"] = int(date.isocalendar().week)
    
    # Return DataFrame with correct column order matching training
    df = pd.DataFrame([all_data])[FEATURE_ORDER]
    return df, planetary_vectors


def get_planet_features_simple(date_str):
    """
    Simple version that returns just the DataFrame (for backward compatibility).
    """
    df, _ = get_planet_features(date_str)
    return df