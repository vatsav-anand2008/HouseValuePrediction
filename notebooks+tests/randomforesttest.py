from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd

housedf=pd.read_csv('housing_with_features.csv').dropna()

X = housedf.drop("median_house_value", axis=1)
y = housedf["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

result={}

for x in range(100,1000,100):
    model=RandomForestRegressor(n_estimators=x,random_state=42)
    model.fit(X_train,y_train)

    preds=model.predict(X_test)

    r2=r2_score(y_test,preds)
    result[str(x)]=r2

print(result)