import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

from pipeline import pipeline
from addfeatures import add_features

DATA_PATH = Path(__file__).parent / "data" / "1553768847-housing.csv"
df=pd.read_csv(DATA_PATH)

df = add_features(df)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#print(list(X_train))

pipeline.fit(X_train, y_train)

#MODEL_PATH = Path(__file__).parent / "models"
joblib.dump(pipeline,"models/finalized_random_forest.pkl")

print("Model saved.")