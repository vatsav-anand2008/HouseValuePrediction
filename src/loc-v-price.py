import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(15,10))

housedf=pd.read_csv('1553768847-housing.csv')
inland=housedf[housedf['ocean_proximity']=='INLAND']['median_house_value']
nearBay=housedf[housedf['ocean_proximity']=='NEAR BAY']['median_house_value']
nearOcean=housedf[housedf['ocean_proximity']=='NEAR OCEAN']['median_house_value']
hourOcean=housedf[housedf['ocean_proximity']=='<1H OCEAN']['median_house_value']
island=housedf[housedf['ocean_proximity']=='ISLAND']['median_house_value']

plt.boxplot([inland,nearBay,nearOcean,hourOcean,island],labels=['Inland','Near Bay','Near Ocean','<1 Hour from Ocean','Island'])
plt.title('House Prices Across Different Areas')
plt.ylabel('Median House Price')
plt.show()