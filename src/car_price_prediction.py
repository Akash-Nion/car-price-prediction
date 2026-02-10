# Car Price Prediction Project
# Converted from Jupyter Notebook

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns

# -----------------------------

df = pd.read_csv('../data/cars.csv')
df

# -----------------------------

df.drop(columns=['Unnamed: 0','normalized-losses'],inplace=True)

# -----------------------------

symbol_pattern = r'[^\w\s]'
for column in df.columns:
    column_text = ' '.join(df[column].astype(str))
    symbols_found = re.findall(symbol_pattern, column_text)
    if symbols_found:
        print(f"Symbols found in column '{column}': {symbols_found}")
    else:
        print(f"No symbols found in column '{column}'")

# -----------------------------

mask = df.applymap(lambda x: '?' in str(x))
df[mask] = np.nan

# -----------------------------

df['price'] = df['price'].astype(float)

# -----------------------------

df['horsepower'] = df['horsepower'].astype(float)
df['peak-rpm'] = df['peak-rpm'].astype(float)
df['stroke'] = df['stroke'].astype(float)
df['bore'] = df['bore'].astype(float)

# -----------------------------

df.info()

# -----------------------------

df.isnull().sum()

# -----------------------------

df['bore'].fillna(int(df['bore'].mean()), inplace=True)
df['stroke'].fillna(int(df['stroke'].mean()), inplace=True)
df['horsepower'].fillna(int(df['horsepower'].mean()), inplace=True)
df['peak-rpm'].fillna(int(df['peak-rpm'].mean()), inplace=True)
df['price'].fillna(int(df['price'].mean()), inplace=True)

# -----------------------------

df['num-of-doors'].ffill(inplace=True)

# -----------------------------

df.isnull().sum()

# -----------------------------

print(df.duplicated().sum())
print(df.shape)

# -----------------------------

import seaborn as sns
sns.distplot(df['price'])

# -----------------------------

import matplotlib.pyplot as plt
sns.barplot(x=df['make'],y=df['price'])
plt.xticks(rotation='vertical')
plt.show()

# -----------------------------

sns.displot(df['horsepower'])

# -----------------------------

sns.scatterplot(x=df['horsepower'],y=df['price'])

# -----------------------------

x = df.drop('price', axis = 1)
x

# -----------------------------

y = df['price']
y

# -----------------------------

from sklearn.preprocessing import LabelEncoder
le_x = LabelEncoder()

# -----------------------------

x = x.apply(le_x.fit_transform)


# -----------------------------

x

# -----------------------------

DX = x
DX['price'] = y
DX

# -----------------------------

sns.heatmap(DX.corr());

# -----------------------------

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
FEATURES = SelectKBest (score_func= f_classif)
FEATURES.fit(x,y)

# -----------------------------

FEATURES.scores_

# -----------------------------

score_col = pd.DataFrame(data = FEATURES.scores_,columns = ['score'])
score_col

# -----------------------------

Name_col=pd.DataFrame(data = x.columns,columns = ['Features'])
Name_col

# -----------------------------

Features_score=pd.concat([Name_col, score_col], axis=1)
Features_score

# -----------------------------

Features_score.nlargest(10,'score')

# -----------------------------

from sklearn.linear_model import LinearRegression

# -----------------------------

from sklearn.metrics import r2_score,mean_absolute_error

# -----------------------------

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(x,y,test_size=0.20, random_state=4)
x_train.shape
x_test.shape

# -----------------------------

lr = LinearRegression()

lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
print("r2 Score ",r2_score(y_test,y_pred))
print("MAE Score ",mean_absolute_error(y_test,y_pred))

# -----------------------------

lr.score(x_test,y_test)

# -----------------------------

