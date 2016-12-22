
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np


# In[2]:

df=pd.read_csv('/Users/ysz/Documents/bda_hw3/300k.csv')
df=df.drop(df.columns[56:207],axis=1)  ##drop cooc after making vis
df=df.drop(df.columns[4:12,],axis=1)  ##drop weird id
df=df.drop(df.columns[8:11],axis=1)  ##drop day/month/year

##make categorical features into dummy columns
cat_columns=['appearedTimeOfDay','appearedDayOfWeek','terrainType','city','continent','weather','weatherIcon']
for cat in cat_columns: 
    dummies = pd.get_dummies(df[cat]).rename(columns=lambda x: cat+'_' + str(x))
    df = pd.concat([df, dummies], axis=1)
    df.drop([cat], inplace=True, axis=1)
##drop some other columns that shouldn't matter
to_drop=['class','appearedLocalTime','appearedMinute','sunriseMinute','sunsetMinute']
for col in to_drop:
    df=df.drop(col, 1)

##transform True/False values into 1/0 in the dataframe
tf_col=[]
for col in df.columns:
    if str(df.loc[0][col]) in ['True','False']:
        tf_col.append(col)
for col in tf_col:
    df[col]=list(map(lambda x:int(x),df[col]))


##create features that describe each pokemon is water/normal type or not
type_id=pd.read_csv('/Users/ysz/Documents/bda_hw3/type_id.csv')

id_type={}
for i in range(type_id.shape[0]):
    id_type[type_id['id'][i]]=type_id['type'][i]   

def id_to_water(x):
    return int('water' in id_type[x['pokemonId']])
df['water_type']=df.apply(id_to_water,axis=1)

def id_to_normal(x):
    return int('normal' in id_type[x['pokemonId']])
df['normal_type']=df.apply(id_to_normal,axis=1)

#create 2 new features, 'to_evolve' and 'evolved'
evolve=pd.read_csv('/Users/ysz/Documents/bda_hw3/evol.csv')
id_to_evolve={}
for i in range(evolve.shape[0]):
    words=evolve['evol'][i]
    if words[0:5]=='Costs':
        #print 'Costs'
        id_to_evolve[i+1]=1
    else:
        id_to_evolve[i+1]=0
#id_to_evolve
id_evolved={}
for i in range(evolve.shape[0]):
    words=evolve['evol'][i]
    if words[0:7]=='Evolved' or 'No evolution' in words:
        #print 'Costs'
        id_evolved[i+1]=1
    else:
        id_evolved[i+1]=0
#id_evolved
def id_to_toevolve(x):
    return int(id_to_evolve[x['pokemonId']])
df['to_evolve']=df.apply(id_to_toevolve,axis=1)
def id_to_evolved(x):
    return int(id_evolved[x['pokemonId']])
df['evolved']=df.apply(id_to_evolved,axis=1)


n_train=int(0.7*df.shape[0]);n_train

def qmark_to_float(x):
    if type(x['pokestopDistanceKm'])!= float and type(x['pokestopDistanceKm'])!= int:
        try:
            return float(x['pokestopDistanceKm'])
        except:
            return 0
    else:
        return x['pokestopDistanceKm']

df['pokestopDistanceKm']=df.apply(qmark_to_float,axis=1)


# In[3]:

# scale the continuous columns
from sklearn import preprocessing

X = df.ix[:, 1:203].as_matrix()
y = df.ix[:,203].as_matrix()
to_scale_list = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 26]
X1 = preprocessing.scale(X[:, to_scale_list], axis = 0)
X_scale = np.concatenate((X1, X[:, list(set(range(0, 202)) - set(to_scale_list))]), axis = 1)


# In[4]:

# do KNN
from sklearn.neighbors import KNeighborsClassifier

K = [2, 3, 5, 8, 10, 15, 20, 30, 40, 50, 75, 100]
P = [1, 2]
scores = []
    for p in P:
        tmp = []
        for k in K:
            model = KNeighborsClassifier(n_neighbors = k, p = p)
            model.fit(X_scale, y)
            score = model.score(X_scale, y)
            tmp.append(score)
            print(score)
        scores.append(tmp)

