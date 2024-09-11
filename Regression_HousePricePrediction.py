  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

  
df = pd.read_csv("Data_House_price_prediction.csv")

df.head()

  
df.info()

  
df.shape

  
df.isnull().sum()

  
df.nunique()

  
df.describe()

  
# df_corr = df.corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(df_corr, annot=True, cmap='coolwarm')    #check seaborn off. doc, annot=True -> values no, mela varum
# plt.title('Correlation Matrix')
# plt.show()

  
df.columns

  
columns_to_remove = ['date','yr_renovated', 'street', 'city','statezip', 'country']
df = df.drop(columns=columns_to_remove)

  
df.head()

  
df_corr = df.corr()                           #myself giving heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm')    #check seaborn off. doc, annot=True -> values no, mela varum
plt.title('Correlation Matrix')
plt.show()

  
#finding outlier using z-score
import scipy.stats as stas
z_scores = stas.zscore(df)     #standard deviation sa ithu find panidu
threshold = 3
print("Size before removing outliers",df.shape)
outlier_df = df[(z_scores>threshold).any(axis=1)]
df = df[(z_scores<=threshold).all(axis=1)]
print("Size after removing outliers",df.shape)

  
outlier_df.head()    #these are outliers and deleted


  
#feature preprocessing -> Normalisation or standardization
#standardization -> giving same unit scaling
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

scalar.fit(df)

df_scaled = pd.DataFrame(scalar.transform(df), columns=df.columns)



  
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

  
X = df.drop('price',axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25,random_state=42)


  
print(f"'X_train{X_train.shape}")
print(f"'X_test{X_test.shape}")

  
models =[
    ("RandomForestRegressor:",RandomForestRegressor()),
    ("DecisionTreeRegressor:",DecisionTreeRegressor()),
    ("LinearRegression:",LinearRegression()),
    ("KNeighborsRegressor:",KNeighborsRegressor()),
]

  
from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error

  
#using this models 1.train  2.test 
for name,model in models:
    print(name)
    print()
    model.fit(X_train,y_train)
    y_pred =model.predict(X_test)
    print("Mean squared error:",mean_squared_error(y_test,y_pred))
    print('\n')
    print("Mean Absolute error:",mean_absolute_error(y_test,y_pred))
    print('\n')
    print("R-squared:",r2_score(y_test,y_pred))
    print('\n')


