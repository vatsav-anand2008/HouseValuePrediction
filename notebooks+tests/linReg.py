import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import math
import statistics
from itertools import combinations

hdf=pd.read_csv('../data/housing_with_features.csv').dropna()
features=list(hdf)
features.remove('median_house_value')
maxR=-1
bestCombo=None

for i in range(1,len(features)+1):
    for combo in combinations(features,i):
        x=hdf[list(combo)]
        y=hdf['median_house_value']

        model=LinearRegression()
        model.fit(x,y)

        preds=model.predict(x)
        r2=r2_score(y,preds)

        if r2>maxR:
            maxR=r2
            bestCombo=combo
print(f'Best Combo: {bestCombo}')
print(f'Best R: {maxR}')
