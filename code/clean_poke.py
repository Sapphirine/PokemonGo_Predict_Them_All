import pandas as pd
import numpy as np

df=pd.read_csv('300k.csv')
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
    df[col]=map(lambda x:int(x),df[col])


##create binary vector to see if each pokemon is of a specific type or not
type_id=pd.read_csv('type_id.csv')

id_type={}
for i in range(type_id.shape[0]):
    id_type[type_id['id'][i]]=type_id['type'][i]   

def id_to_water(x):
    return int('water' in id_type[x['pokemonId']])
df['water_type']=df.apply(id_to_water,axis=1)
def id_to_normal(x):
    return int('normal' in id_type[x['pokemonId']])
df['normal_type']=df.apply(id_to_normal,axis=1)
	print 'normal'

#create 2 new features, 'to_evolve' and 'evolved'
evolve=pd.read_csv('evol.csv')
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

#create training set and test set on X
X_train=df[df.columns[1:202]][0:n_train].as_matrix().astype(float)
X_test=df[df.columns[1:202]][n_train:].as_matrix().astype(float)

#sanity check
X_train.shape
X_test.shape

##create all the labels Y
water_train=df['water_type'][0:n_train].as_matrix()
water_test=df['water_type'][n_train:].as_matrix()
normal_train=df['normal_type'][0:n_train].as_matrix()
normal_test=df['normal_type'][n_train:].as_matrix()
to_evolve_train=df['to_evolve'][0:n_train].as_matrix()
to_evolve_test=df['to_evolve'][n_train:].as_matrix()
evolved_train=df['evolved'][0:n_train].as_matrix()
evolved_test=df['evolved'][n_train:].as_matrix()

#K-Nearest-Neighbor
from sklearn.ensemble import RandomForestClassifier

normal_accuracy={}
for n in [1,2,3,4,5,6,7,8,9,10]:
    normal_accuracy_test[n]=[]
    for i in [5,10,15,20,25,30]:
        #import time
        #start = time.time()
        rfc = RandomForestClassifier(n_estimators=n,max_depth=i,max_features='log2')
        rfc.fit(X_train,normal_train)
        end=time.time()
        #print end-start
        accuracy=rfc.score(X_test,normal_test)
        print n,i,round(accuracy*100,3)
        normal_accuracy_test[n].append(accuracy)


##Spark & Logistic Regression

import os
os.environ["SPARK_HOME"] = "/Users/tianhaolu/Documents/spark-2.0.1-bin-hadoop2.7"
conf = (SparkConf().setMaster('local').setAppName('a'))
sc = SparkContext(conf=conf)


from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf

data = sc.textFile("df_spark_water.csv")
header = data.first() #extract header
data = data.filter(lambda x: x != header)   #filter out header

def join_space(line):
    return reduce(lambda x,y:x+" "+y,line)

data=data.map(lambda x: x.split(",")).map(lambda x:x[1:]).map(join_space)
##data.first()


def parsePoint(line):
    values = [float(x) for x in line.split(' ')]
    return LabeledPoint(values[0], values[1:])

parsedData = data.map(parsePoint)
parsedData.first()

training, test = parsedData.randomSplit([0.7, 0.3])

model = LogisticRegressionWithLBFGS.train(training)

train_labelsAndPreds = training.map(lambda p: (p.label, model.predict(p.features)))
trainErr = train_labelsAndPreds.filter(lambda (v, p): v != p).count() / float(training.count())
print("Training Error = " + str(trainErr), 'train size=',training.count())
test_labelsAndPreds = test.map(lambda p: (p.label, model.predict(p.features)))
testErr = test_labelsAndPreds.filter(lambda (v, p): v != p).count() / float(test.count())
print("Testing Error = " + str(trainErr), 'test size=',test.count())


##for further visualization purpose we downloaded the 311 dataset##
df_NY=df[df['city_New_York'].isin([1])]
NY_2015=pd.read_csv('311_2015.csv')
NY_2016=pd.read_csv('311_2015.csv')
NY_poke_occ={}
##zoom into NY, build a dictionary with each pokemon's occurences in NYC
for a in df_NY['pokemonId']:
    if a not in NY_poke_occ.keys():
        NY_poke_occ[a]=1
    else:
        NY_poke_occ[a]+=1
NY_poke_occ

##create custom rule: if <100 rare; between 100 and 1000 usual; >1000 common
def id_to_rare(x):
    if NY_poke_occ[x['pokemonId']]<100:
        return 'rare'
    elif NY_poke_occ[x['pokemonId']]>=100 and NY_poke_occ[x['pokemonId']]<=1000:
        return 'usual'
    else:
        return 'common'

df_NY['rareness']=df_NY.apply(id_to_rare,axis=1)

df_NY['ones']=1
df_NY.groupby(['appearedHour']).size()

NY_2015=NY_2015[['Created Date','Complaint Type','Longitude','Latitude']]
NY_2016=NY_2016[['Created Date','Complaint Type','Longitude','Latitude']]


##transform the specific hours from 311 dataset to the morning/afternoon/evening/night segment as the Pokemon data does, for consistency
def get_hour(x):
    if x['Created Date'][-2:] not in (['AM','PM']):
        return 0
    else:
        hour=int(x['Created Date'].split(' ')[1][0:2])
        ap=x['Created Date'][-2:]
        if hour>=7 and hour<= 12 and ap=='AM':
            return 'morning'
        elif hour>=1 and hour<=4 and ap=='PM':
            return 'afternoon'
        elif hour>=5 and hour<=7 and ap=='PM':
            return 'evening'
        else:
            return 'night'
 
##finally export it to CSV and will import into R for visualization  
NY_2015['time']=NY_2015.apply(get_hour,axis=1)
NY_2016['time']=NY_2016.apply(get_hour,axis=1)

NY_2015.to_csv('NY_2015.csv')
NY_2016.to_csv('NY_2016.csv')
df_NY.to_csv('df_NY.csv')

