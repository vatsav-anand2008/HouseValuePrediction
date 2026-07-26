import pandas as pd
import joblib

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from addfeatures import add_features

df = pd.read_csv("../data/1553768847-housing.csv")

df = add_features(df)

X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

model = joblib.load(
    "models/random_forest.pkl"
)

pred = model.predict(X)

print("R²:", r2_score(y, pred))
print("MAE:", mean_absolute_error(y, pred))
print("RMSE:", mean_squared_error(y, pred)**0.5)
