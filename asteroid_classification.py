import pandas as pd 
import numpy as np

import sklearn 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('/content/nasa.csv')

cols = list(df.columns)
dtypes = list(df.dtypes)

type_dict = {cols[i]:dtypes[i] for i in range(len(cols))}

corrs = df.corr()
corrs = list(corrs['Hazardous'])
corrs_dict = {cols[i]:corrs[i] for i in range(len(corrs))}

info_dict = {cols[i]:{'Type':type_dict[cols[i]], "Correlation":corrs_dict[cols[i]] if cols[i] in corrs_dict else "nan"} for i in range(len(cols))}

to_drop = ['Neo Reference ID',
 'Name',  'Est Dia in KM(min)',
 'Est Dia in KM(max)', 'Est Dia in Miles(min)',
 'Est Dia in Miles(max)',
 'Est Dia in Feet(min)',
 'Est Dia in Feet(max)',
 'Relative Velocity km per hr', 'Miss Dist.(miles)', 'Orbit ID', 'Orbiting Body',  'Miss Dist.(miles)', 'Equinox']

df_train = df.drop(to_drop, axis = 1)

df_true = []
df_false = []

for i in range(len(df)):
  if df_train['Hazardous'].iloc[i] == True:
    df_true.append(df_train.iloc[i])
  else:
    df_false.append(df_train.iloc[i])

df_true = pd.DataFrame(df_true)
df_false = pd.DataFrame(df_false)

df_false_keep = df_false.sample(755)

df_train_final = df_true.append(df_false_keep)

from sklearn.utils import shuffle

df_train_final = shuffle(df_train_final)

y = df_train_final["Hazardous"]
X = df_train_final.drop(['Hazardous'], axis = 1)

X = X.drop(['Close Approach Date'], axis = 1)

X['Est Dia in M Avg'] = X[['Est Dia in M(min)', 'Est Dia in M(max)']].mean(axis = 1)

X = X.drop(['Est Dia in M(min)', 'Est Dia in M(max)'], axis = 1)

X = X.drop(['Orbit Determination Date'], axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

