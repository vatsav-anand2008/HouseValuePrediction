import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import math
import statistics

housedf=pd.read_csv('1553768847-housing.csv')
columns=list(housedf)
categories=housedf['ocean_proximity'].unique()
#print(categories[:-1])
rAvgs={}
for col in columns[:-2]:
    rList=[]
    for category in categories:
        subset=housedf[housedf['ocean_proximity']==category]
        subset=subset[[col, 'median_house_value']].dropna()
        m,b=np.polyfit(subset[col],subset['median_house_value'],1)

        r=math.sqrt(r2_score(subset['median_house_value'],[m*i+b for i in subset[col]]))
        rList.append(r)
    rAvgs[col]=statistics.mean(rList)
    print(f'{col}, done')
print(rAvgs)