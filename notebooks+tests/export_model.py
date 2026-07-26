from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib

housedf=pd.read_csv('../data/housing_with_features.csv').dropna()

X = housedf.drop("median_house_value", axis=1)
y = housedf["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model=RandomForestRegressor(n_estimators=500,random_state=42)
model.fit(X_train,y_train)
joblib.dump(model,'housing_random_forest.pkl')
