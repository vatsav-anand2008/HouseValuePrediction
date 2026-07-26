from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

'''preprocessor = ColumnTransformer(
    transformers=[
        (
            "ocean",
            OneHotEncoder(handle_unknown="ignore"),
            ["ocean_proximity"]
        )
    ],
    remainder="passthrough"
)'''

pipeline = Pipeline([
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42
    ))
])