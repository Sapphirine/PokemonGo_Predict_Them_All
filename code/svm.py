import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

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

def qmark_to_float(x):
    if type(x['pokestopDistanceKm'])!= float and type(x['pokestopDistanceKm'])!= int:
        try:
            return float(x['pokestopDistanceKm'])
        except:
            return 0
    else:
        return x['pokestopDistanceKm']

df['pokestopDistanceKm']=df.apply(qmark_to_float,axis=1)

type_id=pd.read_csv('type_id.csv')

def is_that_type(types, tp):
    that_type = {}
    for i in range(0, len(types)):
        if (tp in types[i]):
            that_type[i] = 1.0
        else:
            that_type[i] = 0.0
    return that_type
        
def predict_accuracy(df, type_id, tp, kernel_name, train_size):
    is_type = is_that_type(type_id['type'], tp)

    ## build objects
    df_y = []
    for i in df[df.columns[0]]:
        df_y.append(is_type[int(i)])
    
    ## build variables
    df_x = df[df.columns[1:202]][0:].as_matrix().astype(float)
    
    ## training dataset
    df_svc = svm.SVC(kernel=kernel_name)
    df_svc.fit(df_x[0:train_size], df_y[0:train_size])
    
    ## predicting training set
    df_y_predicted_self = df_svc.predict(df_x[0:train_size])
    count_self = 0.0
    for i in range(0, len(df_y_predicted_self)):
        if df_y_predicted_self[i] == df_y[i]:
            count_self += 1.0
    
    ## predicting dataset
    df_y_predicted = df_svc.predict(df_x[train_size:(train_size * 10)])
    count = 0.0
    for i in range(0, len(df_y_predicted)):
        if df_y_predicted[i] == df_y[train_size + i]:
            count += 1.0
    print 'done' + tp + '-' + kernel_name + '-' + str(train_size), count_self / len(df_y_predicted_self), count / len(df_y_predicted)
    return [tp + '-' + kernel_name, train_size, count_self / len(df_y_predicted_self), count / len(df_y_predicted)]

sample_types = ['normal', 'water', 'grass']
sample_size = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

rbfs = []
sigmoids = []
for i in sample_types:
    for num in sample_size:
        rbfs.append(predict_accuracy(df, type_id, i, 'rbf', num))
        sigmoids.append(predict_accuracy(df, type_id, i, 'sigmoid', num))

print rbfs
print sigmoids

normal_rbf = [i[3] for i in rbfs[0:8]]
normal_sig = [i[3] for i in sigmoids[0:8]]
water_rbf = [i[3] for i in rbfs[8:16]]
water_sig = [i[3] for i in sigmoids[8:16]]
grass_rbf = [i[3] for i in rbfs[16:24]]
grass_sig = [i[3] for i in sigmoids[16:24]]

myfile = open('svm_result.csv', 'wb')
myfile.write('type,kernel,train_size,accuracy')
x = 0
for j in sample_size:
    myfile.write('\n')
    myfile.write('normal,rbf,' + str(j) + ',' + str(normal_rbf[x]))
    myfile.write('\n')
    myfile.write('normal,sigmoid,' + str(j) + ',' + str(normal_sig[x]))
    myfile.write('\n')
    myfile.write('water,rbf,' + str(j) + ',' + str(water_rbf[x]))
    myfile.write('\n')
    myfile.write('water,sigmoid,' + str(j) + ',' + str(water_sig[x]))
    myfile.write('\n')
    myfile.write('grass,rbf,' + str(j) + ',' + str(grass_rbf[x]))
    myfile.write('\n')
    myfile.write('grass,sigmoid,' + str(j) + ',' + str(grass_sig[x]))
    x += 1
myfile.close()
