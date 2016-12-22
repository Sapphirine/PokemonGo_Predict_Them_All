
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from matplotlib import animation
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:

# read data
df = pd.read_csv('/Users/ysz/Documents/bda_hw3/300k.csv', low_memory=False)
type_id=pd.read_csv('/Users/ysz/Documents/bda_hw3/type_id.csv')

# build additional features that are requested in making maps
id_type={}
for i in range(type_id.shape[0]):
    id_type[type_id['id'][i]]=type_id['type'][i]
    
def id_to_water(x):
    return int('water' in id_type[x['pokemonId']])

df['water_type']=df.apply(id_to_water,axis=1)


# In[3]:

# split the data by water/non-water
water = df.loc[(df.water_type == 1)]
no_water = df.loc[(df.water_type == 0)]


# In[5]:

# draw the map
plt.figure(1, figsize=(20,10))
m1 = Basemap(projection='merc',
             llcrnrlat=-60,
             urcrnrlat=65,
             llcrnrlon=-180,
             urcrnrlon=180,
             lat_ts=0,
             resolution='c')

m1.fillcontinents(color='#191919',lake_color='#0093D2') 
m1.drawmapboundary(fill_color='#000000')              
m1.drawcountries(linewidth=0.1, color="w")

# Plot the data
x, y = m1(no_water.longitude.tolist(), no_water.latitude.tolist())
m1.scatter(x,y, s=3, c="#00D285", lw=0, alpha=1, zorder=5)
plt.title("Non-water Pokemon Appearence")
plt.show()


# In[11]:

water.shape


# In[12]:

df.shape


# In[ ]:



