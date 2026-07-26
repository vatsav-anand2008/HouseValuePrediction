#import sys
#print(sys.executable)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
import math

housedf=pd.read_csv('1553768847-housing.csv')
categories=housedf['ocean_proximity'].unique()

for category in categories:
    plt.figure(figsize=(15,10))
    subset=housedf[housedf['ocean_proximity'] == category]
    ratio=[x/y for x,y in zip(subset['total_rooms'],subset['population'])]
    plt.scatter(ratio,subset['median_house_value'],alpha=0.3,label=category.lower())

    m,b=np.polyfit(ratio,subset['median_house_value'],1)
    plt.plot(ratio,[m*i+b for i in ratio])

    r=math.sqrt(r2_score(subset['median_house_value'],[m*i+b for i in ratio]))
    plt.text(1,1,f'r={r:.3f}, r^2={r**2:.3f}',fontsize=13)

    plt.xlabel('Total Rooms/Population')
    plt.ylabel('Median House Value')
    plt.title(category)
    plt.legend()

plt.show()
plt.pause(1)