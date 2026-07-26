import pandas as pd
import math
import numpy as np

SF_LAT = 37.7749
SF_LON = -122.4194

LA_LAT = 34.0522
LA_LON = -118.2437

SAC_LAT = 38.5816
SAC_LON = -121.4944

def add_features(housedf):
    housedf=housedf.copy()

    housedf["rooms_per_household"] = (
        housedf["total_rooms"] / housedf["households"]
    )

    housedf["bedrooms_per_room"] = (
        housedf["total_bedrooms"] / housedf["total_rooms"]
    )

    housedf["population_per_household"] = (
        housedf["population"] / housedf["households"]
    )
    #major cities only increase r^2 by 0.02
    housedf["distance_sf"] = np.sqrt(
        (housedf["latitude"] - SF_LAT)**2 +
        (housedf["longitude"] - SF_LON)**2
    )

    housedf["distance_la"] = np.sqrt(
        (housedf["latitude"] - LA_LAT)**2 +
        (housedf["longitude"] - LA_LON)**2
    )

    housedf["distance_sac"] = np.sqrt(
        (housedf["latitude"] - SAC_LAT)**2 +
        (housedf["longitude"] - SAC_LON)**2
    )

    housedf["income_squared"] = housedf["median_income"] ** 2

    housedf["log_income"] = np.log1p(housedf["median_income"])

    housedf["income_x_rooms"] = (
        housedf["median_income"] *
        housedf["rooms_per_household"]
    )

    housedf["income_x_age"] = (
        housedf["median_income"] *
        housedf["housing_median_age"]
    )

    housedf["newness_score"] = (
        1 / (housedf["housing_median_age"] + 1)
    )

    housedf = pd.get_dummies(
        housedf,
        columns=["ocean_proximity"],
        #drop_first=True
    )

    return housedf

#housedf.to_csv("housing_with_features.csv", index=False)
