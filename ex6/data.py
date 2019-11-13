import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing , svm
from sklearn.model_selection  import cross_validate
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
# print(df.head())
# print( type (df) )

forecast_col = 'Adj. Close'
df.fillna(-99999,inplace=True)

forecast_out = int( math.ceil(0.1*len(df) ) )

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
# print(df.head())
X=np.array(df.drop( ['label'],1 ) )
y=np.array(df['label'])

X = preprocessing.scale(X)
X = X[:-forecast_out+1]
df.dropna(inplace=True)
y = np.array(df['label'])

# print(len(X),len(y))

X_train,X_test,y_train,y_test = cross_validate.train_test_split(X,y,test_size=0.2)
clf = svm.SV()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)