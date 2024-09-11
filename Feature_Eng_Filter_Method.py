  
import pandas as pd


  
#sampleDataset for Loan approval
data = {
    'age':[25,30,20,22],
    'income':[25000,50000,48000,15000],
    'loan_approval':[1,0,1,0]
}
df = pd.DataFrame(data)
df

  
#Find corr
df.corr()

  
#finding corr for target col

correlation = df.corr()['loan_approval']
correlation

  
sort_features = correlation.sort_values(ascending=False)
sort_features

  
from sklearn.datasets import load_iris

  
#Currently it is data only not dataset
data =load_iris()
data

  
from sklearn.feature_selection import SelectKBest, f_classif


  
x,y = data.data,data.target

  
df = pd.DataFrame(data.data,columns=data.feature_names)
df['target'] = data.target
df

  
#different types of scoring fun() -> 1.Classificarion  2. Regression
#this is classification 
k_best =SelectKBest(score_func=f_classif,k=2)

  
x_new =k_best.fit_transform(x,y)

  
seleted_indices = k_best.get_support(indices=True)

  
seleted_Features =df.columns[seleted_indices]
seleted_Features

  



